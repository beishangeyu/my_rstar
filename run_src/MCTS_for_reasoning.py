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
    reach_terminal_subquestion,
    reach_terminal_ost_step,
    concat_subqs_and_subas,
    concat_ost_steps,
    concat_subqs_subas_as_ost_steps,
    make_hint,
    make_response_prefix,
    split_user_question,
    print_tree_from_root,
    find_valid_solution_nodes,
    find_best_solution,
    stochastic_find_best_solution,
    make_prompt,
)
from prompts.prompt import (
    ost_prompt,
    ost_stop_token,
    rephrase_prompt,
    rephrase_stop_token,
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
        self.num_a1_steps = args.num_a1_steps  # 默认为3
        self.num_votes = args.num_votes  # 默认是 32
        # NOTE 设置生成的 max token
        self.max_tokens = 1024
        self.enable_potential_score = args.enable_potential_score

        self.mcts_num_last_votes = args.mcts_num_last_votes

        with open(args.decompose_template_path, "r") as f:
            decompose_template = json.load(f)
            self.question_index = decompose_template["index"]

        self.decompose_prompt = read_txt(args.decompose_prompt_path)
        self.fewshot_cot_prompt = read_txt(args.fewshot_cot_prompt_path)
        self.fewshot_cot_config = read_json(args.fewshot_cot_config_path)

        if not args.disable_a1:  # A1: Propose an one-step thought.
            self.fewshot_ost_prompt = ost_prompt
            self.fewshot_ost_config = read_json(args.fewshot_ost_config_path)

        if not args.disable_a5:  # A5: Rephrase the question/sub-question.
            self.rephrasing_prompt_template = rephrase_prompt
            self.decompose_prompt_rephrased = read_txt(
                args.decompose_prompt_rephrased_path
            )
            self.fewshot_cot_prompt_rephrased = read_txt(
                args.fewshot_cot_prompt_rephrased_path
            )
            self.fewshot_ost_prompt_rephrased = read_txt(
                args.fewshot_ost_prompt_rephrased_path
            )

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

        # TODO make prompt 或许可以根据类型的不同构造不同的 input
        io_input = make_prompt(
            requirement=requirement,
            hint=hint,
            func_head=func_head,
            test_case=test_case,
        )
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
        is_ost: bool,
        func_head: str,
        test_case: str,
    ):

        # TODO 父节点类型不同, prompt 格式也不一样
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
            # TODO 这里的答案需要修改吗? 因为有多行
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
        rephrased_user_question_list = []
        io_input = f"""
{rephrase_prompt}

Original requirement: 
{user_question}
Rephrased requirement:
"""
        # TODO 直接用模型的回答去替换本来的prompt
        io_output = self.io.generate(
            model_input=io_input,
            max_tokens=1024,
            num_return=1,
            stop_tokens=rephrase_stop_token,
        )[0]
        rephrased_user_question_list.append(io_output)

        return rephrased_user_question_list

    # NOTE 给出之前生成的单步思考, 生成下一步思考
    def generate_ost_step(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
        parent_is_subquestion: bool,
        task: str,
    ):
        ost_step_list = []
        #  也是一步一步提出来的
        existing_ost_steps, next_ost_step_id = concat_ost_steps(solution_trace)
        io_input = f"""
{ost_prompt}
[function haed and docstring]
{task}
[step to implement]
{existing_ost_steps}
Step{next_ost_step_id}:
"""
        io_output_list = self.io.generate(
            model_input=io_input,
            max_tokens=256,
            num_return=self.num_a1_steps,  # 默认生成3个回复, 每个回复生成一个子节点
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
        # --- For instantiating root node ---
        node_value: float = None,
        generator: Generator = None,
        disable_a5: bool = None,
        user_requirement: str = None,
        max_depth_allowed: int = None,
        disable_a1: bool = None,
        # -----------------------------------
        # --- rephrase之后的用户需求  ---
        rephrased_requirement: str = None,
        # ------------------------------------------------------
        expected_answer: str = None,
        # --- For instantiating DIRECT_ANSWER node ---
        direct_answer: str = None,
        # --------------------------------------------
        # --- For instantiating OST_STEP node ---
        ost_step: str = None,
    ) -> None:
        """params:
        subquestion: the node is proposing a new subquestion
        subanswer: the answer corresponding to the new subquestion the node proposed
        re_subanswer: the node is proposing a new subanswer to the parent's subquestion
        """
        super().__init__()

        #! sanity checks
        try:
            assert depth is not None
            assert node_type is not None
            if node_value is not None:
                assert node_value > 0, breakpoint()

            if node_type is Node_Type.USER_QUESTION:
                assert depth == 0
                assert all(
                    attr is None
                    for attr in [
                        parent,
                        node_value,
                        rephrased_requirement,
                        direct_answer,
                        ost_step,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [
                        generator,
                        disable_a5,
                        user_requirement,
                        expected_answer,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
            elif node_type is Node_Type.REPHRASED_USER_QUESTION:
                assert depth == 1
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        generator,
                        disable_a5,
                        user_requirement,
                        expected_answer,
                        direct_answer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, rephrased_requirement])
            elif node_type is Node_Type.DIRECT_ANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        user_requirement,
                        expected_answer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(
                    attr is not None for attr in [parent, node_value, direct_answer]
                )
            elif node_type is Node_Type.OST_STEP:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        generator,
                        disable_a5,
                        user_requirement,
                        rephrased_requirement,
                        expected_answer,
                        direct_answer,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, ost_step])
        except AssertionError:
            print(f"Instantiating node with type {node_type} failed!")
            breakpoint()
            exit()

        #! attributes
        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []
        self.depth = depth
        self.node_type = node_type
        self.node_value = node_value
        self.direct_answer = direct_answer
        self.ost_step = ost_step
        self.task = task

        # root node
        if parent is None:
            self.verbose = verbose
            self.user_requirement = user_requirement  # 即每个样本的要求
            self.expected_answer = expected_answer
            self.generator = generator
            self.disable_a5 = disable_a5
            self.question_index = generator.question_index
            self.max_depth_allowed = max_depth_allowed
            self.disable_a1 = disable_a1
        else:
            self.verbose = parent.verbose
            self.user_requirement = parent.user_requirement
            self.expected_answer = parent.expected_answer
            self.generator = parent.generator
            self.disable_a5 = parent.disable_a5
            self.question_index = parent.generator.question_index
            self.max_depth_allowed = parent.max_depth_allowed
            self.disable_a1 = parent.disable_a1

        #! keep track of paraphrasing
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

        # 记录从根节点到当前节点的推理路径
        # TODO 随着action set扩大, 或许需要修改, 但是最好还是以dict形式, 方便组合
        if parent is None:  # root
            # assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace: Dict[int, Dict[str, str]] = {
                0: {"user_requirement": user_requirement, "ost_step": {}}
            }  # 这是一个 dict形式的树, ost_step也是一个dict, key是第几步, value就是具体内容
        else:
            # assert self.node_type is not Node_Type.USER_QUESTION
            # deepcopy parent的, 防止修改影响到之前的node
            self.solution_trace = deepcopy(parent.solution_trace)

            if node_type is Node_Type.REPHRASED_USER_QUESTION:
                # 直接更换成重述后的
                self.solution_trace[0]["user_question"] = rephrased_requirement
            elif node_type is Node_Type.OST_STEP:
                # self.solution_trace[self.subquestion_counter]["ost_step"][self.ost_step_counter] = ost_step
                # TODO 因为当前动作没含有 subquestion, 第一个直接取 0 即可, 后续或需要扩展
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

            if (
                self.node_type is not Node_Type.USER_QUESTION
                and self.node_type is not Node_Type.REPHRASED_USER_QUESTION
            ):
                hint = make_hint(self.solution_trace, self.node_type)
            else:
                hint = None
                code = self.task["code"]
                func_head = re.search(r"def .+?:", code).group(0)
                test_case = self.task["test_list"][0][7:]

            # TODO user question 应该组合
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
                        # NOTE value 即 node 的 reward, 计算方式为出现次数最多的答案次数占总次数的比例
                        node_value=value,
                        direct_answer=direct_answer,
                    )
                )

        # NOTE 重述用户的需求
        def do_action_generate_rephrased_user_question():
            verbose_print(
                f"---- Generating rephrased user question for node {self.id}...",
                self.verbose,
            )

            #! ACTION: generate paraphrased question for the root question
            rephrased_user_question_list = (
                self.generator.generate_rephrased_requirement(
                    user_question=self.user_requirement
                )
            )
            # TODO 用新生成的需求替换掉原来的 docstring
            for rephrased_user_question in rephrased_user_question_list:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.REPHRASED_USER_QUESTION,
                        rephrased_requirement=rephrased_user_question,
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
                user_question=self.user_requirement,
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

        #! create children
        if self.node_type is Node_Type.USER_QUESTION:
            # A1: Propose an one-step thought.
            if not self.disable_a1:
                do_action_generate_ost_step()

            # A2: Propose the remaining thought steps
            do_action_generate_direct_answers()

            # A5: Rephrase the question/sub-question.
            if not self.disable_a5:
                do_action_generate_rephrased_user_question()
        elif self.node_type is Node_Type.REPHRASED_USER_QUESTION:
            # A1: Propose an one-step thought.
            if not self.disable_a1:
                do_action_generate_ost_step()

            # A2: Propose the remaining thought steps
            do_action_generate_direct_answers()
        elif self.node_type is Node_Type.DIRECT_ANSWER:
            raise ValueError("DIRECT_ANSWER node cannot create children!!")
        # 先生成单步思考之后生成答案
        elif self.node_type is Node_Type.OST_STEP:
            do_action_generate_ost_step()
            do_action_generate_direct_answers()

        assert self.children
        return self.children

    # 有效的叶结点是子问题类型(回答完了)和直接回答类型
    def is_valid_leaf_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        self.node_type is Node_Type.DIRECT_ANSWER

    # 有效的solution node是子问题节点(回答完了), 单步思考节点(回答完了), 或直接回答节点
    def is_valid_solution_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type or OST_STEP type
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
        expected_answer=gt_answer,
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
    path2 = os.path.join(
        args.answer_sheets_dir, f"Task_id_{question_id}_last_node_per_simulate.json"
    )
    last_node_per_simulate = []
    for i, node in enumerate(model_rollout_nodes):
        last_node_per_simulate.append({"trace": node.solution_trace, "rollout_id": i})
    write_jsonl(path2, last_node_per_simulate, append=True)

    return model_solutions, i, model_all_solutions
