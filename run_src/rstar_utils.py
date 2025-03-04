# Licensed under the MIT license.
import sys

sys.path.append(".")
from enum import Enum, unique
import math
from typing import Dict, Tuple
import math
import re


@unique
class Node_Type(Enum):
    USER_QUESTION = "USER_QUESTION"
    REPHRASED_USER_QUESTION = "REPHRASED_USER_QUESTION"
    DIRECT_ANSWER = "DIRECT_ANSWER"
    OST_STEP = "OST_STEP"
    SUBQUESTION = "SUBQUESTION"
    CPD = "Code_Paradigm_Decompose"


def get_nodetype(Reasoning_MCTS_Node):
    if Reasoning_MCTS_Node is None:
        return None
    elif Reasoning_MCTS_Node.node_type is Node_Type.USER_QUESTION:
        return "USER_QUESTION"
    elif Reasoning_MCTS_Node.node_type is Node_Type.REPHRASED_USER_QUESTION:
        return "REPHRASED_USER_QUESTION"
    elif Reasoning_MCTS_Node.node_type is Node_Type.OST_STEP:
        return "OST_STEP"
    elif Reasoning_MCTS_Node.node_type is Node_Type.SUBQUESTION:
        return "SUBQUESTION"
    elif Reasoning_MCTS_Node.node_type is Node_Type.DIRECT_ANSWER:
        return "DIRECT_ANSWER"


class GeneratorError(Exception):
    def __init__(self, source, io_input, io_output_list) -> None:
        super().__init__()

        self.source = source
        self.io_input = io_input
        self.io_output_list = io_output_list


def remove_comments(code: str) -> str:
    # 分离多行字符串和单行注释的逻辑
    result = []
    lines = code.splitlines()
    in_multiline = False
    multiline_delimiter = None

    for line in lines:
        stripped_line = line.strip()

        # 检测是否进入或退出多行字符串/注释
        if in_multiline:
            if stripped_line.endswith(multiline_delimiter):
                in_multiline = False
            continue
        elif stripped_line.startswith('"""') or stripped_line.startswith("'''"):
            multiline_delimiter = stripped_line[:3]
            if not stripped_line.endswith(multiline_delimiter):  # 未在同一行结束
                in_multiline = True
            continue
        elif stripped_line.startswith("#"):  # 单行注释
            continue

        # 移除行内注释，但保留字符串中的 #
        if "#" in line:
            # 简单检查是否在字符串中（不完美，但适用于大多数情况）
            if not (line.count('"') % 2 == 1 or line.count("'") % 2 == 1):
                line = re.sub(r"#.*$", "", line, flags=re.MULTILINE)

        # 只保留非空行
        if stripped_line:
            result.append(line)

    # 合并结果，保留单个换行符
    return "\n".join(result).strip()


def extract_answer_from_model_completion(completion: str) -> str:
    # 找到第一个 def 的位置
    start_idx = completion.find("def ")
    if start_idx == -1:
        first_function = ""

    # 找到下一个 def 的位置（如果有）
    next_def_idx = completion.find("def ", start_idx + 1)
    if next_def_idx == -1:
        first_function = completion[start_idx:].strip()
    else:
        # 取第一个 def 到下一个 def 之前的内容
        first_function = completion[start_idx:next_def_idx].strip()
    if not first_function:
        return ""
    # 去除注释
    return remove_comments(first_function)


def concat_ost_steps(solution_trace: Dict[int, Dict[str, str]]) -> Tuple[str, int]:
    last_tuple_recording = list(solution_trace.values())[0]  # 取出最后一个 kv pair
    assert "ost_step" in last_tuple_recording.keys()
    if len(last_tuple_recording["ost_step"]) > 0:
        solution_trace_str = ""

        for step_id, step_text in last_tuple_recording["ost_step"].items():
            solution_trace_str += step_text.strip() + "\n"
        return solution_trace_str, step_id + 1
    else:
        # 还没有 ost step
        return "", 1


# 连接已有的cpd steps
def concat_cpd_steps(solution_trace: Dict[int, Dict[str, str]]) -> Tuple[str, int]:

    paradim = [
        "1. Input Analysis: What is the input of the function? What are the possible input types or ranges? Are there any special cases (e.g., empty input, invalid input) to consider?",
        "2. Output Definition: What does the function need to return? What is the type of the return value? Are there any specific formats or conditions?",
        "3. Function Decomposition: What smaller steps can the target functionality be broken into?",
        "4. Boundary Conditions: What boundary cases need to be handled? How do we ensure the function works correctly in these cases?",
    ]
    last_tuple_recording = list(solution_trace.values())[0]  # 取出最后一个 kv pair
    assert "cpd_step" in last_tuple_recording.keys()
    if len(last_tuple_recording["cpd_step"]) > 0:
        solution_trace_str = ""
        for step_id, step_text in last_tuple_recording["cpd_step"].items():

            solution_trace_str += (
                paradim[step_id - 1] + "\n" + "    " + step_text.strip() + "\n"
            )
        return solution_trace_str, step_id + 1
    else:
        # 还没有 ost step
        return "", 1


# disc 时在 mask 前要先把solution trace拼接起来
def concat_solution_trace(
    solution_trace: Dict[int, Dict[str, str]],
) -> Tuple[str, str, str, float]:
    reward_value = 0.0

    question_trace = list(solution_trace.values())
    main_question = question_trace[0]
    requirement = main_question["user_requirement"]

    ost_step = main_question["ost_step"]  # type = dict[int, str]
    cpd_step = main_question["cpd_step"]  # type = dict[int, str]
    if len(ost_step) > 0:
        hints = "### Hints:\n"
        steps_list = list(main_question["ost_step"].values())
        steps = "\n".join(steps_list)
        hints += steps.strip() + "\n\n"
    elif len(cpd_step) > 0:
        paradim = [
            "1. Input Analysis: What is the input of the function? What are the possible input types or ranges? Are there any special cases (e.g., empty input, invalid input) to consider?",
            "2. Output Definition: What does the function need to return? What is the type of the return value? Are there any specific formats or conditions?",
            "3. Function Decomposition: What smaller steps can the target functionality be broken into?",
            "4. Boundary Conditions: What boundary cases need to be handled? How do we ensure the function works correctly in these cases?",
        ]
        hints = " ### Thinking Steps:\n"
        steps_list = list(main_question["cpd_step"].values())
        for i in range(0, 4):
            hints += paradim[i] + "\n" + "    " + steps_list[i].strip() + "\n"
    else:
        hints = ""

    # NOTE 只给推理路径, 防止mask的时候把答案mask掉不好处理
    solution_trace = hints

    final_step = main_question["direct_answer"][
        "text"
    ]  # 就把main question的trace取出来就好
    reward_value = (
        main_question["direct_answer"]["value"]
        if "value" in main_question["direct_answer"]
        else 0.0
    )
    return (
        requirement.strip(),
        solution_trace.strip(),
        final_step.strip(),
        min(0, reward_value) + 1,
    )


# 对 solution trace 进行随机遮蔽
def mask_solution_trace(
    solution_trace_str: str,
    num_return: int,
    left_boundary: float,  # 最少留下left_boundary, 即如果left_boundary=0.2, 则至少留下20%的字符串
    right_boundary: float,  # 最多留下right_boundary, 即如果right_boundary=0.8, 则最多留下80%的字符串
) -> list[str]:
    # opasdjifpoaisdfjpoasidfjapsodifj, num_return: 4, left: 0.2, right: 0.8
    # return: opasd, opasdjifp, opasdjifpoaisdfj, opasdjifpoaisdfjpoasidfjaps
    if num_return == 1:
        interval = 0
    else:
        assert num_return > 1
        assert (
            right_boundary >= left_boundary
        ), f"right_boundary: {right_boundary} < left_boundary: {left_boundary}"
        # 每个前缀字符串之间的比例间隔
        interval = (right_boundary - left_boundary) / (num_return - 1)

    if not solution_trace_str:
        return ["" for _ in range(num_return)]
    words_in_solution_trace = solution_trace_str.split(" ")
    ost_len = len(words_in_solution_trace)
    masked_solution_traces = []
    for i in range(num_return):
        prefix_part_ratio = left_boundary + i * interval
        prefix_part_num_words = math.ceil(ost_len * prefix_part_ratio)
        prefix_part_str = " ".join(words_in_solution_trace[:prefix_part_num_words])
        masked_solution_traces.append(prefix_part_str.strip())

    return masked_solution_traces


# 把solution trace结合成hint
def make_ost_hint(
    solution_trace: Dict[int, Dict[str, str]],  # 只有第一个dict是有用的
) -> str:

    # 这里的 hint 不用加上 '### Hints', 因为后边有了
    hint = ""

    step_list = [step.strip() for step in list(solution_trace[0]["ost_step"].values())]
    if step_list:
        hint += "\n".join(step_list) + "\n"

    return hint.strip()


# 构建cpd类型hint
def make_cpd_hint(
    solution_trace: Dict[int, Dict[str, str]],
) -> str:

    # TODO 考虑加入一次性回答完的动作
    # 目前只一步一步来
    step_list = [step for step in list(solution_trace[0]["cpd_step"].values())]

    # NOTE 目前规定思考完整才能回答
    assert (
        len(step_list) == 4
    ), f"cpd_steps should be equal to 4, but now is {len(step_list)}"

    hint = f"""
1. Input Analysis: What is the input of the function? What are the possible input types or ranges? Are there any special cases (e.g., empty input, invalid input) to consider?
    {step_list[0].strip()}
2. Output Definition: What does the function need to return? What is the type of the return value? Are there any specific formats or conditions?
    {step_list[1].strip()}
3. Function Decomposition: What smaller steps can the target functionality be broken into?
    {step_list[2].strip()}
4. Boundary Conditions: What boundary cases need to be handled? How do we ensure the function works correctly in these cases?
    {step_list[3].strip()}
"""
    return hint.strip()


def make_funchead_and_docstring(
    requirement: str, func_head: str, test_case: str
) -> str:
    # 处理多行requirement
    tmp = requirement.split("\n")
    requirement = tmp[0] + "\n" + "\n".join("    " + s for s in tmp[1:])
    s = f"""
{func_head.strip()}
    '''
    {requirement.strip()}
    for example:
    {test_case.strip()}
    '''
"""
    return s.strip()


def find_valid_solution_nodes(root_node):
    valid_solution_nodes = []

    def recursion(node):
        if node.is_valid_solution_node():
            valid_solution_nodes.append(node)
            return

        if not node.children:  #! no children
            return

        for child in node.children:
            recursion(child)

    recursion(root_node)

    return valid_solution_nodes


def find_best_solution(root_node, evaluator):
    # todo: what strategy do we use to select best node?
    """The function finds the best solution from the solution nodes in the MCTS tree.
    Return: top answer, top solution, confidence of the top answer, the corresponding node of the answer, all solution nodes
    """
    solution_nodes = find_valid_solution_nodes(root_node)

    if len(solution_nodes) == 0:
        return None, None

    def extract_solution_from_node(node):
        if node.node_type is Node_Type.SUBQUESTION:
            return node.subanswer
        elif node.node_type is Node_Type.DIRECT_ANSWER:
            return node.direct_answer
        else:
            return None

    solutions = [extract_solution_from_node(node) for node in solution_nodes]

    top_answer, top_completion, top_completion_id, top_confidence = (
        evaluator.find_most_confident_answer(solutions)
    )
    return (
        top_answer,
        top_completion,
        top_confidence,
        solution_nodes[top_completion_id],
        solution_nodes,
    )


def stochastic_find_best_solution(
    root_node,
    evaluator,
):
    # todo: what strategy do we use to select best node?
    """The function finds the best solution from the solution nodes in the MCTS tree.
    Return: top answer, top solution, confidence of the top answer, the corresponding node of the answer, all solution nodes
    """
    solution_nodes = find_valid_solution_nodes(root_node)

    if len(solution_nodes) == 0:
        return None, None

    def extract_solution_from_node(node):
        if node.node_type is Node_Type.DIRECT_ANSWER:
            return node.direct_answer
        else:
            return None

    solutions = [extract_solution_from_node(node) for node in solution_nodes]

    top_answer, top_completion, top_completion_id, top_confidence = (
        evaluator.stochastic_find_most_confident_answer(completions=solutions)
    )
    return (
        top_answer,
        top_completion,
        top_confidence,
        solution_nodes[top_completion_id],
        solution_nodes,
        solutions,
    )
