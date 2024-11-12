# Licensed under the MIT license.

import sys

sys.path.append(".")

import numpy as np, os, random, json, math, wandb
from tqdm import trange
from typing import List, Dict, Tuple
from copy import deepcopy
import re

try:
    from rapidfuzz import fuzz, process
except:
    pass

from models.IO_System import IO_System
from common.utils import write_jsonl
from eval_src.Evaluator import Evaluator, GSM8KEvaluator
from MCTS_backbone import MCTS_Searcher, MCTS_Node
from run_src.rstar_utils import (
    Node_Type,
    GeneratorError,
    reach_terminal_ost_step,
    concat_ost_steps,
    make_hint,
    print_tree_from_root,
    stochastic_find_best_solution,
    make_funchead_and_docstring,
)
from prompts.prompt import (
    ost_prompt,
    rephrase_prompt,
)


def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)


class Generator:
    """Generator generates children nodes"""

    def __init__(self, args, tokenizer, model, evaluator: Evaluator) -> None:
        self.io = IO_System(args, tokenizer, model)
        self.evaluator = evaluator

        self.num_subquestions = args.num_subquestions
        # NOTE 默认为3, 在 generate ost step 里生成 3 个回复序列, 每个回复序列生成一个子节点
        self.num_a1_steps = args.num_a1_steps
        # NOTE 设置生成的 max token
        self.max_tokens = 1024
        self.enable_potential_score = args.enable_potential_score
        self.mcts_num_last_votes = args.mcts_num_last_votes  # 默认是 32

    # 从output_list中选择出现次数最多的answer和对应的completion
    def _get_most_likely_answer(self, io_output_list: List[str]) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        if len(io_output_list) == 1:
            most_confident_answer_full_completion = io_output_list[0]
            confidence = 1
        else:
            _, most_confident_answer_full_completion, _, confidence = (
                self.evaluator.find_most_confident_answer(io_output_list)
            )
            assert confidence > 0

        return most_confident_answer_full_completion, confidence

    # NOTE 生成 impl.
    def _generate_impl(
        self,
        requirement: str,
        paraphrased: bool,
        num_return: int,
        func_head: str,
        test_case: str,
        hint: str = None,
    ):
        # fewshot_cot_prompt = (
        #     self.fewshot_cot_prompt
        #     if not paraphrased
        #     else self.fewshot_cot_prompt_rephrased
        # )
        # question += "\n\n" + hint if hint is not None else ""
        # io_input = self.fewshot_cot_config["prompt_template"].format(
        #     examples=fewshot_cot_prompt, instruction=question
        # )
        funchead_and_docstring = make_funchead_and_docstring(
            requirement, func_head, test_case
        )
        io_input = f"""
        You are a Python assistant. Implement a Python function based on the given function head, docstring, and hint.
[Function head and docstring]:
{funchead_and_docstring}
[Hint]
{hint}
[Function implementation]
"""
        io_output_list = self.io.generate(
            model_input=io_input,
            num_return=num_return,
            max_tokens=self.max_tokens,
            stop_tokens=["[Function head and docstring]:"],
        )
        cleaned_io_output_list = [
            io_output.strip() for io_output in io_output_list
        ]  #! cleaning
        return io_input, cleaned_io_output_list

    # NOTE 直接生成答案
    def generate_direct_answers(
        self,
        user_requirement: str,
        paraphrased: bool,
        hint: str,
        func_head: str,
        test_case: str,
    ):

        direct_answer_list, value_list = [], []
        num_return = self.mcts_num_last_votes  # 默认为32
        io_input, cleaned_io_output_list = self._generate_impl(
            requirement=user_requirement,
            paraphrased=paraphrased,
            num_return=num_return,
            hint=hint,
            func_head=func_head,
            test_case=test_case,
        )

        try:
            # 选择出现次数最多的答案返回, 这个答案次数的占比即 value
            most_likely_answer, likelihood = self._get_most_likely_answer(
                cleaned_io_output_list
            )
        except Exception as e:
            raise GeneratorError(
                source="generate direct answer from: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )

        direct_answer_list.append(most_likely_answer)
        value_list.append(likelihood)

        return direct_answer_list, value_list

    # NOTE 重述 docstring
    def generate_rephrased_requirement(self, user_question: str):
        rephrased_user_requirement_list = []
        io_input = f"""
{rephrase_prompt}

Original requirement: 
{user_question}
Rephrased requirement:
"""
        io_output = self.io.generate(
            model_input=io_input,
            max_tokens=1024,
            num_return=1,
            stop_tokens=["\n\n", "Original requirement:"],
        )[0]
        rephrased_user_requirement_list.append(io_output)

        return rephrased_user_requirement_list

    # NOTE 给出之前生成的单步思考, 生成下一步思考
    def generate_ost_step(
        self,
        requirement: str,
        solution_trace: Dict[int, Dict[str, str]],
        func_head: str,
        test_case: str,
    ):
        funchead_and_docstring = make_funchead_and_docstring(
            requirement, func_head, test_case
        )
        idx = func_head.find("(")
        func_name = func_head[4:idx]
        ost_step_list = []
        #  也是一步一步提出来的
        existing_ost_steps, next_ost_step_id = concat_ost_steps(solution_trace)
        io_input = f"""
{ost_prompt}
[Function haed and docstring]
{funchead_and_docstring}
[Step to implement]
To implement the {func_name} function, we need to follow these steps:
{existing_ost_steps}Step{next_ost_step_id}:
"""
        io_output_list = self.io.generate(
            model_input=io_input,
            max_tokens=256,
            num_return=self.num_a1_steps,
            stop_tokens=["\n", "\n\n", f"Step{next_ost_step_id + 1}:"],
        )
        ost_step_list = [io_output.strip() for io_output in io_output_list]

        return ost_step_list


class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        # 直接把整个样本传进来
        task: str,
        verbose: bool = False,
        node_value: float = None,
        generator: Generator = None,
        user_requirement: str = None,
        max_depth_allowed: int = None,
        rephrased_requirement: str = None,  # rephrase后的要求
        direct_answer: str = None,
        ost_step: str = None,
    ) -> None:
        super().__init__()

        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []
        self.depth = depth
        self.node_type = node_type
        self.node_value = node_value
        self.direct_answer = direct_answer
        self.ost_step = ost_step
        self.task = task

        if parent is None:
            self.verbose = verbose
            self.user_requirement = user_requirement  # 即每个样本的要求
            self.generator = generator
            self.question_index = generator.question_index
            self.max_depth_allowed = max_depth_allowed

            code = task["code"]
            func_name = re.search(r"def (.+?)\(", code).group(1)
            self.func_name = func_name
        else:
            self.verbose = parent.verbose
            self.user_requirement = parent.user_requirement
            self.generator = parent.generator
            self.question_index = parent.generator.question_index
            self.max_depth_allowed = parent.max_depth_allowed
            self.func_name = parent.func_name

        if node_type is Node_Type.USER_QUESTION:
            self.paraphrased = False
        elif node_type is Node_Type.REPHRASED_USER_QUESTION:
            # 是否重述过用户需求
            self.paraphrased = True
            self.user_requirement = rephrased_requirement
        else:
            assert parent is not None
            self.paraphrased = parent.paraphrased
            # 和父节点的requirement保持一致
            self.user_requirement = parent.user_requirement

        # 记录 ost 步数
        # TODO 一般不能大于 8? 拿几个样本去 gpt 那里试一下, 小模型的思考步骤应该比那个多个一两步
        if parent is None:  # root
            self.ost_step_counter = 0
        else:
            if node_type is Node_Type.OST_STEP:
                self.ost_step_counter = parent.ost_step_counter + 1
            else:
                self.ost_step_counter = parent.ost_step_counter

        # NOTE 更新推理路径
        if parent is None:
            # 0 表示当前这是第 0 个 subquestion
            self.solution_trace: Dict[int, Dict[str, str]] = {
                0: {"user_requirement": user_requirement, "ost_step": {}}
            }
        else:
            self.solution_trace = deepcopy(parent.solution_trace)
            if node_type is Node_Type.REPHRASED_USER_QUESTION:
                self.solution_trace[0]["user_requirement"] = rephrased_requirement
            elif node_type is Node_Type.OST_STEP:
                # solution_trace[0]["ost_step"] 也是一个 dict, key 是思考的步数
                self.solution_trace[0]["ost_step"][self.ost_step_counter] = ost_step

            pass

    # TODO 这个只是输出相关, 等有时间了再来考虑
    # def __str__(self) -> str:
    #     type2str = {
    #         Node_Type.USER_QUESTION: "U",
    #         Node_Type.REPHRASED_USER_QUESTION: "RU",
    #         Node_Type.DIRECT_ANSWER: "DA",
    #         Node_Type.SUBQUESTION: "SQ",
    #         Node_Type.RE_SUBANSWER: "RS",
    #         Node_Type.OST_STEP: "TS",
    #     }
    #     return f"{type2str[self.node_type]}-{self.id}"

    def _create_children(self):
        # NOTE 直接生成答案
        def do_action_generate_direct_answers():
            verbose_print(
                f"---- Generating direct answers for node {self.id}...", self.verbose
            )

            if self.node_type == Node_Type.OST_STEP:
                hint = make_hint(self.solution_trace, self.node_type, self.func_name)
            else:
                hint = None

            code = self.task["code"]
            func_head = re.search(r"def .+?:", code).group(0)
            test_case = self.task["test_list"][0][7:]

            (direct_answer_list, value_list) = self.generator.generate_direct_answers(
                user_requirement=self.user_requirement,
                paraphrased=self.paraphrased,
                hint=hint,
                func_head=func_head,
                test_case=test_case,
            )
            for direct_answer, value in zip(direct_answer_list, value_list):
                if np.isnan(value) or value <= 0:
                    breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.DIRECT_ANSWER,
                        #  value 即 node 的 reward, 计算方式为出现次数最多的答案次数占总次数的比例
                        node_value=value,
                        direct_answer=direct_answer,
                    )
                )

        # NOTE 重述用户的需求
        def do_action_generate_rephrased_user_requirement():
            verbose_print(
                f"---- Generating rephrased user question for node {self.id}...",
                self.verbose,
            )

            #! ACTION: generate paraphrased question for the root question
            rephrased_user_requirement_list = (
                self.generator.generate_rephrased_requirement(
                    user_question=self.user_requirement
                )
            )
            for rephrased_user_requirement in rephrased_user_requirement_list:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.REPHRASED_USER_QUESTION,
                        rephrased_requirement=rephrased_user_requirement,
                    )
                )

        # NOTE 生成单步思考
        def do_action_generate_ost_step(parent_is_subquestion=False):
            verbose_print(
                f"---- Generating one-step thought steps for node {self.id}...",
                self.verbose,
            )

            #! ACTION: generate one-step thought step
            ost_step_list = self.generator.generate_ost_step(
                requirement=self.user_requirement,
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
                parent_is_subquestion=parent_is_subquestion,
            )
            for ost_step in ost_step_list:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.OST_STEP,
                        ost_step=ost_step,
                    )
                )

        # create children
        # NOTE 规定了每种类型的节点可以创造什么类型的子节点
        if self.node_type is Node_Type.USER_QUESTION:
            do_action_generate_ost_step()
            do_action_generate_direct_answers()
            do_action_generate_rephrased_user_requirement()
        elif self.node_type is Node_Type.REPHRASED_USER_QUESTION:
            do_action_generate_ost_step()
            do_action_generate_direct_answers()
        elif self.node_type is Node_Type.DIRECT_ANSWER:
            raise ValueError("DIRECT_ANSWER node cannot create children!!")
        elif self.node_type is Node_Type.OST_STEP:
            do_action_generate_ost_step()
            do_action_generate_direct_answers()
            # XXX 在单步思考中途修改 requirement, solution trace 需要更改?
            do_action_generate_rephrased_user_requirement()

        assert self.children
        return self.children

    # 有效的叶结点是子问题类型(回答完了)和直接回答类型
    def is_valid_leaf_node(self):
        self.node_type is Node_Type.DIRECT_ANSWER

    # 有效的solution node是子问题节点(回答完了), 单步思考节点(回答完了), 或直接回答节点
    def is_valid_solution_node(self):
        return (
            self.node_type is Node_Type.OST_STEP
            and reach_terminal_ost_step(self.ost_step)
        ) or self.node_type is Node_Type.DIRECT_ANSWER

    def find_children(self, rollout_id: int):
        self.children = self.children or self._create_children()
        for child in self.children:
            child.set_rollout_id(rollout_id)
        assert self.children
        return self.children

    def is_terminal(self):
        # return self.depth >= self.max_depth_allowed or self.is_valid_leaf_node()
        # XXX github上有人说加入ost判定后会稍微增加性能, 测试一下
        return self.depth >= self.max_depth_allowed or self.is_valid_solution_node()

    def calculate_reward(self):
        if self.is_valid_leaf_node():
            assert self.node_value is not None, breakpoint()
            return self.node_value
        else:
            return 0

    def skip_backprop(self):
        return (
            self.node_type is Node_Type.USER_QUESTION
            or self.node_type is Node_Type.REPHRASED_USER_QUESTION
        )


def search_for_answers(
    args, user_question: str, question_id: int, gt_answer: str, generator: Generator
):
    verbose_print(
        f"********************* Searching for answers to question {question_id} ********************* ",
        args.verbose,
    )

    #! build an MCTS searcher
    mcts_searcher = MCTS_Searcher(
        exploration_weight=args.mcts_exploration_weight,
        weight_scheduler=args.mcts_weight_scheduler,
        num_rollouts=args.num_rollouts,
        discount=args.mcts_discount_factor,
        verbose=args.verbose,
    )

    #! build the MCTS tree
    root_node = Reasoning_MCTS_Node(
        parent=None,
        depth=0,
        node_type=Node_Type.USER_QUESTION,
        verbose=args.verbose,
        generator=generator,
        disable_a5=args.disable_a5,
        user_requirement=user_question,
        max_depth_allowed=args.max_depth_allowed,
        disable_a1=args.disable_a1,
        enable_potential_score=args.enable_potential_score,
    )

    model_solutions = []
    model_all_solutions = []
    model_rollout_nodes = []
    # 进行指定次数次 rollout
    for i in (pbar := trange(args.num_rollouts, disable=True, position=0)):
        rollout_node = mcts_searcher.do_rollout(root_node, i)
        model_rollout_nodes.append(rollout_node)

        # 每次 rollout 找出 best_solution 和 所有 solution
        _, best_solution, _, chosen_node, all_solution_nodes, all_solutions = (
            stochastic_find_best_solution(
                root_node,
                generator.evaluator,
                enable_potential_score=args.enable_potential_score,
            )
        )
        model_solutions.append(best_solution)
        model_all_solutions.append(all_solutions)

        if args.save_tree:
            with open(
                os.path.join(
                    args.answer_sheets_dir,
                    f"Question {question_id:04d} - Rollout {i}.tree",
                ),
                "w",
            ) as f:
                print_tree_from_root(
                    mcts_searcher=mcts_searcher,
                    rollout_id=i,
                    root_node=root_node,
                    chosen_node=chosen_node,
                    file=f,
                )

    # NOTE 记录最终整个树里所有的 solution
    path1 = os.path.join(
        args.answer_sheets_dir, f"Task_id_{question_id}_all_solutions.jsonl"
    )
    all_solutions = [
        {"trace": node.solution_trace, "rollout_id": node.rollout_id}
        for node in all_solution_nodes
    ]
    write_jsonl(path1, all_solutions, append=True)

    # NOTE 记录每次simulate的路径中最后的节点
    # XXX 不知道为什么在 eval 的时候会把这两个文件的 json 加在一起, 这样会让其中一些答案重复从而数量增多, 影响到选择最终答案, 但是我还是生成出来
    # TODO 记得 do eval 的时候不要用这个函数
    path2 = os.path.join(
        args.answer_sheets_dir, f"Task_id_{question_id}_last_node_per_simulate.json"
    )
    last_node_per_simulate = []
    for i, node in enumerate(model_rollout_nodes):
        last_node_per_simulate.append({"trace": node.solution_trace, "rollout_id": i})
    write_jsonl(path2, last_node_per_simulate, append=True)

    return model_solutions, i, model_all_solutions
