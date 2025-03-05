# Licensed under the MIT license.

import sys

sys.path.append(".")

import numpy as np, os
from typing import List, Dict, Tuple
from copy import deepcopy
import re


from models.IO_System import IO_System
from common.utils import write_jsonl
from run_src.Evaluator import Evaluator
from MCTS_backbone import MCTS_Searcher, MCTS_Node
from rstar_utils import (
    Node_Type,
    GeneratorError,
    get_nodetype,
    concat_ost_steps,
    concat_cpd_steps,
    make_cpd_hint,
    make_ost_hint,
    stochastic_find_best_solution,
    make_funchead_and_docstring,
    extract_answer_from_model_completion,
)
from prompt import (
    ost_prompt,
    rephrase_prompt,
    direct_answer_prompt,
    direct_answer_no_hints_prompt,
    cpd_prompt,
    cpd_final_answer_prompt,
)


def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)


# WARNING 打印输入
is_debug = True


def print_input_to_model(s: str):
    if is_debug:
        print(f"---- Input to model:\n{s}")
        print("---- End of input")


def print_output_from_model(s: str):
    if is_debug:
        print(f"---- Output from model:\n{s}")
        print("---- End of output")


def print_hint(hint: str):
    if is_debug:
        print(f"---- Hint:\n{hint}")
        print("---- End of hint")


class Generator:
    """Generator generates children nodes"""

    def __init__(self, args, tokenizer, model, evaluator: Evaluator) -> None:
        self.io = IO_System(args, tokenizer, model)
        self.evaluator = evaluator
        self.num_a1_steps = args.num_a1_steps  # 默认为3
        self.num_subquestions = args.num_subquestions  # 默认是3
        self.mcts_num_last_votes = args.mcts_num_last_votes  # 默认是 32

    # 从output_list中选择出现次数最多的answer和对应的completion
    def _get_most_likely_answer(self, io_output_list: List[str]) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        if len(io_output_list) == 1:
            most_confident_answer_full_completion = io_output_list[0]
            confidence = 1
        else:
            _, most_confident_answer_full_completion, confidence = (
                self.evaluator.find_most_confident_answer(io_output_list)
            )
            assert confidence > 0

        return most_confident_answer_full_completion, confidence

    # 生成 impl.
    def _generate_impl(
        self,
        requirement: str,
        num_return: int,
        func_head: str,
        test_case: str,
        is_cpd_type: bool,
        hint: str = None,
    ):
        funchead_and_docstring = make_funchead_and_docstring(
            requirement, func_head, test_case
        )
        if is_cpd_type:
            io_input = f"""{cpd_final_answer_prompt.strip()}

### Python programming problem:
{funchead_and_docstring.strip()}

### Thinking Steps:
{hint.strip()}
"""

        elif hint:
            io_input = f"""{direct_answer_prompt.strip()}

### Python programming problem:
{funchead_and_docstring.strip()}

### Hints:
{hint.strip()}
"""

        elif not hint:
            io_input = f"""{direct_answer_no_hints_prompt.strip()}

### Python programming problem:
{funchead_and_docstring.strip()}
"""

        io_input += "\n" + "### Function implementation:\n"
        io_output_list = self.io.generate(
            model_input=io_input,
            num_return=num_return,
            max_tokens=600,
            stop_tokens=["### ", "You are a Python assistant. "],
            top_p=0.95,
            top_k=10,
            temperature=0.8,
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]
        cleaned_io_output_list = [
            extract_answer_from_model_completion(io) for io in cleaned_io_output_list
        ]
        print_input_to_model(io_input)
        for io_output in cleaned_io_output_list:
            print_output_from_model(io_output)

        return io_input, cleaned_io_output_list

    # 直接生成答案, value=出现次数最多的答案次数/总次数
    def generate_direct_answers(
        self,
        user_requirement: str,
        hint: str,
        func_head: str,
        test_case: str,
        is_cpd_type: bool,
    ):

        direct_answer_list, value_list = [], []
        num_return = self.mcts_num_last_votes
        io_input, cleaned_io_output_list = self._generate_impl(
            requirement=user_requirement,
            num_return=num_return,
            hint=hint,
            func_head=func_head,
            test_case=test_case,
            is_cpd_type=is_cpd_type,
        )

        try:
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

    # 重述 docstring
    def generate_rephrased_requirement(self, user_requirement: str):
        rephrased_user_requirement_list = []
        io_input = f"""{rephrase_prompt.strip()}

Original requirement: 
{user_requirement.strip()}
Rephrased requirement:
"""
        io_output = self.io.generate(
            model_input=io_input,
            max_tokens=128,
            num_return=1,
            stop_tokens=[
                "\n\n\n",
                "Original requirement:",
                "You are an AI assistant ",
            ],
        )[0]
        rephrased_user_requirement_list.append(io_output.strip())

        print_input_to_model(io_input)
        print_output_from_model(io_output)

        return rephrased_user_requirement_list

    # 给出之前生成的单步思考, 生成下一步思考
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
        io_input = f"""{ost_prompt.strip()}

### Python programming problem:
{funchead_and_docstring.strip()}

### Step to implement:
"""
        if len(existing_ost_steps) > 0:
            io_input += existing_ost_steps.strip() + "\n"
        io_output_list = self.io.generate(
            model_input=io_input,
            max_tokens=200,  # 单步思考不需要那么多token
            num_return=self.num_a1_steps,
            stop_tokens=[
                "### ",
                f"{next_ost_step_id + 1}. ",
                "You are a Python assistant. ",
                "def ",
            ],
        )
        # 这得到的多个step是并行关系
        ost_step_list = [io_output.strip() for io_output in io_output_list]
        # 过滤掉空的
        ost_step_list = [step for step in ost_step_list if len(step) > 0]

        print_input_to_model(io_input)
        for step in ost_step_list:
            print_output_from_model(step)
        return ost_step_list

    # 一次性生成剩下所有的 cot steps 而不是一步一步来
    def gene_remain_steps(
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
        io_input = f"""{ost_prompt.strip()}

### Python programming problem:
{funchead_and_docstring.strip()}

### Step to implement:
"""
        # 这里如果直接加进去, 如果还没有ost step, 就会变成一个空行, 模型会从下一行开始, 这个空行就会影响模型输出
        if len(existing_ost_steps) > 0:
            io_input += existing_ost_steps.strip() + "\n"
        io_output_list = self.io.generate(
            model_input=io_input,
            max_tokens=600,
            num_return=1,  # 这个动作只会生成1个子节点, 单步则会生成3个
            stop_tokens=[
                "\n\n\n",
                "### ",
                "You are a Python assistant.",
                "def ",
            ],
        )
        # 这得到的多个step是承接关系
        ost_step_list = re.findall(
            r"(\d+\..+?)(?=\d+\.|$)", io_output_list[0].strip(), re.DOTALL
        )
        # 清理每个部分的空白，但保留序号
        ost_step_list = [step.strip() for step in ost_step_list]
        ost_step_list = [step for step in ost_step_list if len(step) > 0]

        print_input_to_model(io_input)
        print_output_from_model(io_output_list[0].strip())
        for step in ost_step_list:
            print_output_from_model(step)
        return ost_step_list

    def gene_cpd_step(
        self,
        requirement: str,
        solution_trace: Dict[int, Dict[str, str]],
        func_head: str,
        test_case: str,
    ) -> List[str]:
        paradim = [
            "1. Input Analysis: What is the input of the function? What are the possible input types or ranges? Are there any special cases (e.g., empty input, invalid input) to consider?",
            "2. Output Definition: What does the function need to return? What is the type of the return value? Are there any specific formats or conditions?",
            "3. Function Decomposition: What smaller steps can the target functionality be broken into?",
            "4. Boundary Conditions: What boundary cases need to be handled? How do we ensure the function works correctly in these cases?",
        ]
        funchead_and_docstring = make_funchead_and_docstring(
            requirement, func_head, test_case
        )
        cpd_step_list = []
        #  也是一步一步提出来的
        existing_cpd_steps, next_cpd_id = concat_cpd_steps(solution_trace)
        io_input = f"""{cpd_prompt.strip()}

### Python programming problem:
{funchead_and_docstring.strip()}

### Thinking Steps:
"""
        if len(existing_cpd_steps) > 0:
            io_input += existing_cpd_steps.strip() + "\n"
        io_input += f"{paradim[next_cpd_id-1]}\n"
        io_output_list = self.io.generate(
            model_input=io_input,
            max_tokens=200,  # 单步思考不需要那么多token
            num_return=3,  # NOTE 同ost的个数
            stop_tokens=[
                "### ",
                f"{next_cpd_id+1}. ",
                "You are a Python assistant. ",
                "def ",
            ],
        )
        # 这得到的多个step是并行关系
        cpd_step_list = [io_output.strip() for io_output in io_output_list]
        # 过滤掉空的
        cpd_step_list = [step for step in cpd_step_list if len(step) > 0]
        print_input_to_model(io_input)
        for step in cpd_step_list:
            print_output_from_model(step)

        return cpd_step_list


class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        # 直接把整个样本传进来
        task: Dict = None,
        task_id: int = None,
        verbose: bool = False,
        node_value: float = None,
        generator: Generator = None,
        user_requirement: str = None,
        max_depth_allowed: int = None,
        rephrased_requirement: str = None,  # rephrase后的要求
        direct_answer: str = None,
        step_list: List[str] = None,
        is_gen_remaining: bool = None,
        disable_gene_remain_ost: bool = None,
        disable_gene_remain_subq: bool = None,
        cpd_answer: str = None,
    ) -> None:
        super().__init__()

        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []

        self.node_type = node_type
        self.node_value = node_value
        self.direct_answer = direct_answer
        self.ost_step = step_list
        self.depth = depth

        if parent is None:
            self.verbose = verbose
            self.user_requirement = user_requirement  # 即每个样本的要求
            self.generator = generator
            self.max_depth_allowed = max_depth_allowed
            code = task["code"]
            func_name = re.search(r"def (.+?)\(", code).group(1)
            self.func_name = func_name
            self.task = task
            self.task_id = task_id
            self.is_gen_remaining = False
            self.disable_gene_remain_ost = disable_gene_remain_ost
            self.disable_gene_remain_subq = disable_gene_remain_subq
        else:
            self.verbose = parent.verbose
            self.user_requirement = parent.user_requirement
            self.generator = parent.generator
            self.max_depth_allowed = parent.max_depth_allowed
            self.func_name = parent.func_name
            self.task = parent.task
            self.task_id = parent.task_id
            self.disable_gene_remain_ost = parent.disable_gene_remain_ost
            self.disable_gene_remain_subq = parent.disable_gene_remain_subq
            if is_gen_remaining is not None:
                self.is_gen_remaining = is_gen_remaining
            else:
                self.is_gen_remaining = parent.is_gen_remaining

        # 是否重述过用户需求
        if node_type is Node_Type.USER_QUESTION:
            self.paraphrased = False
        elif node_type is Node_Type.REPHRASED_USER_QUESTION:
            self.paraphrased = True
            self.user_requirement = rephrased_requirement
        else:
            assert parent is not None
            self.paraphrased = parent.paraphrased
            # 和父节点的requirement保持一致
            self.user_requirement = parent.user_requirement

        # 记录 ost 步数
        if parent is None:  # root
            self.ost_step_counter = 0
        # 如果节点是ost类型, 可能有多步, 记录trace时增加计数
        else:
            self.ost_step_counter = parent.ost_step_counter

        # cpd思考步数
        if parent is None:
            self.cpd_counter = 0
        else:
            self.cpd_counter = parent.cpd_counter

        # 更新推理路径
        self.stop_num_subq = 100
        self.stop_num_ost = 100
        if parent is None:
            self.solution_trace: Dict[int, Dict[str, str]] = {
                0: {
                    "user_requirement": user_requirement,
                    "ost_step": {},
                    "cpd_step": {},
                }
            }
        else:
            self.solution_trace = deepcopy(parent.solution_trace)

            if node_type is Node_Type.REPHRASED_USER_QUESTION:
                self.solution_trace[0]["user_requirement"] = rephrased_requirement

            elif node_type is Node_Type.CPD:
                self.cpd_counter += 1
                self.solution_trace[0]["cpd_step"][self.cpd_counter] = cpd_answer

            elif node_type is Node_Type.OST_STEP:
                # solution_trace[0]["ost_step"] 也是一个 dict, key 是思考的步数
                for ost_step in step_list:
                    self.ost_step_counter += 1
                    # 设置阈值, 超过的直接丢掉
                    if self.ost_step_counter > self.stop_num_ost:
                        break
                    self.solution_trace[0]["ost_step"][self.ost_step_counter] = ost_step

            elif node_type is Node_Type.DIRECT_ANSWER:
                self.solution_trace[0]["direct_answer"] = {
                    "text": direct_answer,
                    "value": node_value,  # 即这个答案的置信度, 是所有答案中这个答案出现次数的占比
                }
            pass

    def _create_children(self):
        # 直接生成答案
        def do_action_generate_direct_answers():
            verbose_print(
                f"---- Generating direct answers for node {self.id}...", self.verbose
            )

            # NOTE cpd 类型的跟 direct answer 类型的不一样
            if self.ost_step_counter > 0:
                hint = make_ost_hint(self.solution_trace)
            elif self.cpd_counter > 0:
                hint = make_cpd_hint(self.solution_trace)
            else:
                hint = ""

            code = self.task["code"]
            func_head = re.search(r"def .+?:", code).group(0)
            test_case = self.task["test_list"][0][7:]

            (direct_answer_list, value_list) = self.generator.generate_direct_answers(
                user_requirement=self.user_requirement,
                hint=hint,
                func_head=func_head,
                test_case=test_case,
                is_cpd_type=True if self.cpd_counter > 0 else False,
            )
            for direct_answer, value in zip(direct_answer_list, value_list):
                if np.isnan(value) or value <= 0:
                    breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.DIRECT_ANSWER,
                        node_value=value,
                        direct_answer=direct_answer,
                    )
                )

        # 重述用户的需求
        def do_action_generate_rephrased_user_requirement():
            verbose_print(
                f"---- Generating rephrased user question for node {self.id}...",
                self.verbose,
            )

            #! ACTION: generate paraphrased question for the root question
            rephrased_user_requirement_list = (
                self.generator.generate_rephrased_requirement(
                    user_requirement=self.user_requirement
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

        # 生成单步思考
        def do_action_generate_ost_step():
            verbose_print(
                f"---- Generating one-step thought steps for node {self.id}...",
                self.verbose,
            )
            code = self.task["code"]
            func_head = re.search(r"def .+?:", code).group(0)
            test_case = self.task["test_list"][0][7:]

            ost_step_list = self.generator.generate_ost_step(
                requirement=self.user_requirement,
                solution_trace=self.solution_trace,
                func_head=func_head,
                test_case=test_case,
            )
            for ost_step in ost_step_list:
                # 如果 ost step 不为空才添加
                if len(ost_step) > 0:
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.OST_STEP,
                            step_list=[ost_step],
                        )
                    )

        # 生成剩下所有的思考步骤
        def do_action_generate_remain_steps():
            verbose_print(
                f"---- Generating remain one-step thought steps for node {self.id}...",
                self.verbose,
            )
            code = self.task["code"]
            func_head = re.search(r"def .+?:", code).group(0)
            test_case = self.task["test_list"][0][7:]

            ost_step_list = self.generator.gene_remain_steps(
                requirement=self.user_requirement,
                solution_trace=self.solution_trace,
                func_head=func_head,
                test_case=test_case,
            )
            # 如果 ost step 不为空才添加
            if len(ost_step_list) > 0:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.OST_STEP,
                        step_list=ost_step_list,
                        is_gen_remaining=False,
                    )
                )

        # 生成cpd思考步骤
        def do_action_generate_cpd():
            verbose_print(f"---- Generating CPD for node {self.id}...", self.verbose)
            code = self.task["code"]
            func_head = re.search(r"def .+?:", code).group(0)
            test_case = self.task["test_list"][0][7:]
            cpd_step_list = self.generator.gene_cpd_step(
                requirement=self.user_requirement,
                solution_trace=self.solution_trace,
                func_head=func_head,
                test_case=test_case,
            )
            for cpd_step in cpd_step_list:
                if len(cpd_step) > 0:
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.CPD,
                            cpd_answer=cpd_step,
                        )
                    )

        # 规定了每种类型的节点可以创造什么类型的子节点
        # NOTE 对于生成ost节点, 可以尽早剪枝加速运行
        last_ost_step = ""
        if len(self.solution_trace[0]["ost_step"]):
            last_ost_step = self.solution_trace[0]["ost_step"].get(
                self.ost_step_counter
            )
        if self.node_type is Node_Type.USER_QUESTION:
            do_action_generate_rephrased_user_requirement()
            do_action_generate_direct_answers()
            do_action_generate_ost_step()
            do_action_generate_remain_steps()
            do_action_generate_cpd()

        elif self.node_type is Node_Type.REPHRASED_USER_QUESTION:
            # NOTE 如果父节点是 cpd 但是还没有考虑完, 这里就会直接生成 direct answer 跳过限制, 所以要判定一下
            if self.parent.node_type != Node_Type.CPD or self.cpd_counter == 4:
                do_action_generate_direct_answers()

            # 同一条路径下ost和cpd二选一
            if self.cpd_counter == 0 and (
                not last_ost_step or "Return " not in last_ost_step
            ):
                do_action_generate_ost_step()
                if not self.is_gen_remaining and not self.disable_gene_remain_ost:
                    do_action_generate_remain_steps()
            if self.ost_step_counter == 0 and self.cpd_counter < 4:
                do_action_generate_cpd()

        elif self.node_type is Node_Type.OST_STEP:
            # 同一条路径上只会rephrase一次
            if not self.paraphrased:
                do_action_generate_rephrased_user_requirement()

            do_action_generate_direct_answers()
            if (
                self.ost_step_counter < self.stop_num_ost
                and "Return " not in last_ost_step
            ):
                do_action_generate_ost_step()

                if not self.is_gen_remaining and not self.disable_gene_remain_ost:
                    do_action_generate_remain_steps()

        # NOTE cpd 节点可以做的动作
        elif self.node_type is Node_Type.CPD:
            if not self.paraphrased:
                do_action_generate_rephrased_user_requirement()
            # 考虑完全之后才回答
            if self.cpd_counter < 4:
                do_action_generate_cpd()
            else:
                do_action_generate_direct_answers()
        elif self.node_type is Node_Type.DIRECT_ANSWER:
            raise ValueError("DIRECT_ANSWER node cannot create children!!")
        assert self.children
        return self.children

    # 有效的叶结点是 direct answer
    def is_valid_leaf_node(self):
        self.node_type is Node_Type.DIRECT_ANSWER

    # 有效的 solution node 只会是 direct answer (由于 ost 到了最后会停下来, 还是由 direct answer 生成回复)
    def is_valid_solution_node(self):
        return self.node_type is Node_Type.DIRECT_ANSWER

    def find_children(self, rollout_id: int):
        self.children = self.children or self._create_children()
        for child in self.children:
            child.set_rollout_id(rollout_id)
        assert self.children
        return self.children

    def is_terminal(self):
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
    args, user_question: str, task_id: int, generator: Generator, task: Dict
):
    verbose_print(
        f"********************* Searching for answers to question {task_id} ********************* ",
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
        user_requirement=user_question,
        max_depth_allowed=args.max_depth_allowed,
        task=task,
        task_id=task_id,
        disable_gene_remain_ost=args.disable_gene_remain_ost,
        disable_gene_remain_subq=args.disable_gene_remain_subq,
    )

    model_solutions = []
    model_all_solutions = []
    model_rollout_nodes = []
    # 进行指定次数次 rollout
    for i in range(args.num_rollouts):
        rollout_node = mcts_searcher.do_rollout(root_node, i)

        # 每次 rollout 找出 best_solution 和 所有 solution
        _, best_solution, _, chosen_node, all_solution_nodes, all_solutions = (
            stochastic_find_best_solution(
                root_node,
                generator.evaluator,
            )
        )
        model_solutions.append(best_solution)
        model_all_solutions.append(all_solutions)

    # 记录最终整个树里所有的 solution
    path1 = os.path.join(args.gene_result, f"Task_id_{task_id}_all_solutions.jsonl")
    all_solution_nodes_ = [
        {
            "trace": node.solution_trace,
            "rollout_id": node.rollout_id,
            "type": get_nodetype(node),
        }
        for node in all_solution_nodes
    ]
    write_jsonl(path1, all_solution_nodes_)

    return model_solutions, i, model_all_solutions
