# Licensed under the MIT license.
import sys

sys.path.append(".")
from enum import Enum, unique
import re
import math
from typing import Dict, Tuple
import math
from run_src import Evaluator
from run_src.prompt import ost_prompt


@unique
class Node_Type(Enum):
    USER_QUESTION = "USER_QUESTION"
    REPHRASED_USER_QUESTION = "REPHRASED_USER_QUESTION"
    DIRECT_ANSWER = "DIRECT_ANSWER"
    OST_STEP = "OST_STEP"


def get_nodetype(Reasoning_MCTS_Node):
    if Reasoning_MCTS_Node is None:
        return None
    elif Reasoning_MCTS_Node.node_type is Node_Type.USER_QUESTION:
        return "USER_QUESTION"
    elif Reasoning_MCTS_Node.node_type is Node_Type.REPHRASED_USER_QUESTION:
        return "REPHRASED_USER_QUESTION"
    elif Reasoning_MCTS_Node.node_type is Node_Type.OST_STEP:
        return "OST_STEP"
    elif Reasoning_MCTS_Node.node_type is Node_Type.DIRECT_ANSWER:
        return "DIRECT_ANSWER"


class GeneratorError(Exception):
    def __init__(self, source, io_input, io_output_list) -> None:
        super().__init__()

        self.source = source
        self.io_input = io_input
        self.io_output_list = io_output_list


def split_user_question(user_question: str):
    user_question = user_question.strip().rstrip(".")
    last_period_id = user_question.rfind(".")
    assert last_period_id < len(user_question) - 1
    user_question_context = user_question[: last_period_id + 1].strip()
    user_question_problem = user_question[last_period_id + 1 :].strip()
    return user_question_context, user_question_problem


def reach_terminal_ost_step(ost_step: str):
    assert ost_step is not None

    return "answer is" in ost_step.lower()


def concat_ost_steps(solution_trace: Dict[int, Dict[str, str]]) -> Tuple[str, int]:
    """Return: concatenated one-step thought steps, next one-step thought step id"""
    last_tuple = list(solution_trace.items())[-1]  # 取出最后一个 kv pair
    last_tuple_id, last_tuple_recording = last_tuple[0], last_tuple[1]
    assert "ost_step" in last_tuple_recording.keys()
    if len(last_tuple_recording["ost_step"]) > 0:
        solution_trace_str = ""

        for step_id, step_text in last_tuple_recording["ost_step"].items():
            solution_trace_str += f"Step{step_id}: " + step_text + "\n"
        return solution_trace_str, step_id + 1
    else:
        # 还没有 ost step
        return "", 1


def concat_solution_trace(
    solution_trace: Dict[int, Dict[str, str]],
    func_name: str,
):
    solution_trace_str = ""
    final_step_str = ""
    end_node_type = None
    reward_value = 0.0

    for item_idx, solution_step in enumerate(solution_trace.items()):
        if item_idx == 0:
            # 没有 ost step 只有 direct answer
            solution_step = solution_step[1]
            if not solution_step["ost_step"]:
                direct_answer = solution_step["direct_answer"]["text"].strip()
                solution_trace_str = f"[Function implementation]\n{direct_answer}"
                final_step_str = solution_step["direct_answer"]["text"].strip()
                reward_value = (
                    solution_step["direct_answer"]["value"]
                    if "value" in solution_step["direct_answer"]
                    else 0.0
                )
                end_node_type = Node_Type.DIRECT_ANSWER
                break
            # 存在 ost step
            elif (
                len(solution_step["ost_step"]) > 0
                and "direct_answer" in solution_step.keys()
            ):
                solution_trace_str = f"[Step to implement]\nTo implement the {func_name} function, we can follow these steps:"  # 这一段是不应该被mask的
                for step_id, step_text in solution_step["ost_step"].items():
                    solution_trace_str += f"Step{step_id}: " + step_text.strip() + "\n"
                solution_trace_str += "\n"
                solution_trace_str += "[Function implementation]\n"
                solution_trace_str += solution_step["direct_answer"]["text"].strip()
                final_step_str = solution_step["direct_answer"]["text"].strip()
                reward_value = (
                    solution_step["direct_answer"]["value"]
                    if "value" in solution_step["direct_answer"]
                    else 0.0
                )
                end_node_type = Node_Type.DIRECT_ANSWER
                break

    return (
        solution_trace_str.strip(),  # NOTE strip 会去掉换行符, 记得加上
        final_step_str.strip(),
        end_node_type,
        min(0, reward_value) + 1,
    )


# NOTE 对 solution trace 进行随机遮蔽
def mask_solution_trace(
    solution_trace_str: str,
    num_return: int,
    left_boundary: float,
    right_boundary: float,
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

    words_in_solution_trace = solution_trace_str.split(" ")
    ost_len = len(words_in_solution_trace)
    masked_solution_traces = []
    for i in range(num_return):
        prefix_part_ratio = left_boundary + i * interval
        prefix_part_num_words = math.ceil(ost_len * prefix_part_ratio)
        prefix_part_str = " ".join(words_in_solution_trace[:prefix_part_num_words])
        masked_solution_traces.append(prefix_part_str)

    return masked_solution_traces


# NOTE 把solution trace结合成hint
def make_hint(
    solution_trace: Dict[int, Dict[str, str]],  # 只有第一个dict是有用的
    node_type: Node_Type,
    func_name: str,
    new_ost_step=None,
) -> str:
    # 这个函数只被 direct answer 调用, 利用过往的ost step来生成hint
    hint = f"To implement the {func_name} function, we need to follow these steps:"
    # 取出solution_trace中最后一个key value pair
    last_tuple = list(solution_trace.items())[-1]
    last_tuple_recording = last_tuple[1]
    assert last_tuple_recording["ost_step"]
    for step_id, step_text in last_tuple_recording["ost_step"].items():
        hint += step_text + "\n"
    if new_ost_step is not None:
        hint += new_ost_step + "\n"

    return hint.strip()


def make_funchead_and_docstring(
    requirement: str, func_head: str, test_case: str
) -> str:
    s = f"""
{func_head}
    '''
    {requirement}
    for example:
    {test_case}
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


if __name__ == "__main__":
    from run_src.prompt import *

    ssss = ost_prompt
    tl = mask_solution_trace(ssss, 5, 0.2, 0.8)
    for t in tl:
        print(t)
        print(8 * "*")
