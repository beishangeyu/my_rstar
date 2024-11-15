# Licensed under the MIT license.

from enum import Enum, unique
import re
import math
from typing import Dict, Tuple
import math
from eval_src import Evaluator


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


# TODO 这里的concat要更改
def concat_solution_trace(solution_trace: Dict[int, Dict[str, str]]):
    """Note that the solution trace might be subqs-subas and also one-step thought steps."""
    solution_trace_str = ""
    final_step_str = ""
    end_node_type = None
    reward_value = 0.0

    for item_idx, (subq_id, solution_step) in enumerate(solution_trace.items()):
        if item_idx == 0:
            if (
                len(solution_step["ost_step"]) == 0
                and "direct_answer" in solution_step.keys()
            ):
                solution_trace_str += solution_step["direct_answer"]["text"].strip()
                final_step_str = solution_step["direct_answer"]["text"].strip()
                reward_value = (
                    solution_step["direct_answer"]["value"]
                    if "value" in solution_step["direct_answer"]
                    else 0.0
                )
                end_node_type = Node_Type.DIRECT_ANSWER
                break
            elif (
                len(solution_step["ost_step"]) > 0
                and "direct_answer" in solution_step.keys()
            ):
                for step_id, step_text in solution_step["ost_step"].items():
                    solution_trace_str += step_text.strip() + " "
                solution_trace_str += "Now we can answer the question: "
                solution_trace_str += solution_step["direct_answer"]["text"].strip()
                final_step_str = solution_step["direct_answer"]["text"].strip()
                reward_value = (
                    solution_step["direct_answer"]["value"]
                    if "value" in solution_step["direct_answer"]
                    else 0.0
                )
                end_node_type = Node_Type.DIRECT_ANSWER
                break
            elif (
                len(solution_step["ost_step"]) > 0
                and "direct_answer" not in solution_step.keys()
            ):
                final_step_str = None
                for i, (step_id, step_text) in enumerate(
                    solution_step["ost_step"].items()
                ):
                    solution_trace_str += step_text.strip() + " "
                    if i == len(solution_step["ost_step"].items()) - 1:
                        final_step_str = step_text.strip()
                        reward_value = 0.0
                solution_trace_str = solution_trace_str.strip()
                end_node_type = Node_Type.OST_STEP
            else:
                continue
        elif 0 < item_idx < len(solution_trace) - 1:
            intermediate_step = (
                solution_step["subanswer"]["text"].split("The answer is")[0].strip()
            )
            solution_trace_str += intermediate_step + " "
            # concat trace for one-step thought step after subquestion
            if (
                len(solution_step["ost_step"]) > 0
                and "direct_answer" in solution_step.keys()
            ):
                for step_id, step_text in solution_step["ost_step"].items():
                    solution_trace_str += step_text.strip() + " "
                solution_trace_str += "Now we can answer the question: "
                solution_trace_str += solution_step["direct_answer"]["text"].strip()
                final_step_str = solution_step["direct_answer"]["text"].strip()
                reward_value = (
                    solution_step["direct_answer"]["value"]
                    if "value" in solution_step["direct_answer"]
                    else 0.0
                )
                end_node_type = Node_Type.DIRECT_ANSWER
                break
            elif (
                len(solution_step["ost_step"]) > 0
                and "direct_answer" not in solution_step.keys()
            ):
                final_step_str = None
                for i, (step_id, step_text) in enumerate(
                    solution_step["ost_step"].items()
                ):
                    solution_trace_str += step_text.strip() + " "
                    if i == len(solution_step["ost_step"].items()) - 1:
                        final_step_str = step_text.strip()
                        reward_value = 0.0
                solution_trace_str = solution_trace_str.strip()
                end_node_type = Node_Type.OST_STEP
        elif item_idx == len(solution_trace) - 1:
            assert item_idx > 0
            # 1. subq-suba
            if (
                "subanswer" in solution_step.keys()
                and len(solution_step["ost_step"]) == 0
                and "direct_answer" not in solution_step.keys()
            ):
                solution_trace_str += "Now we can answer the question: "
                solution_trace_str += solution_step["subanswer"]["text"].strip()
                final_step_str = solution_step["subanswer"]["text"].strip()
                reward_value = (
                    solution_step["subanswer"]["value"]
                    if "value" in solution_step["subanswer"]
                    else 0.0
                )
                end_node_type = Node_Type.SUBQUESTION
                break
            # 2. subq-suba-ost
            elif (
                "subanswer" in solution_step.keys()
                and len(solution_step["ost_step"]) > 0
                and "direct_answer" not in solution_step.keys()
            ):
                intermediate_step = (
                    solution_step["subanswer"]["text"].split("The answer is")[0].strip()
                )
                solution_trace_str += intermediate_step + " "
                final_step_str = None
                for i, (step_id, step_text) in enumerate(
                    solution_step["ost_step"].items()
                ):
                    solution_trace_str += step_text.strip() + " "
                    if i == len(solution_step["ost_step"].items()) - 1:
                        final_step_str = step_text.strip()
                        reward_value = 0.0
                solution_trace_str = solution_trace_str.strip()
                end_node_type = Node_Type.OST_STEP
            # 3. subq-suba-ost-diranswer
            elif (
                "subanswer" in solution_step.keys()
                and len(solution_step["ost_step"]) > 0
                and "direct_answer" in solution_step.keys()
            ):
                intermediate_step = (
                    solution_step["subanswer"]["text"].split("The answer is")[0].strip()
                )
                solution_trace_str += intermediate_step + " "
                for step_id, step_text in solution_step["ost_step"].items():
                    solution_trace_str += step_text.strip() + " "
                solution_trace_str += "Now we can answer the question: "
                solution_trace_str += solution_step["direct_answer"]["text"].strip()
                final_step_str = solution_step["direct_answer"]["text"].strip()
                reward_value = (
                    solution_step["direct_answer"]["value"]
                    if "value" in solution_step["direct_answer"]
                    else 0.0
                )
                end_node_type = Node_Type.DIRECT_ANSWER
                break
            # 4. subq-suba-diranswer
            elif (
                "subanswer" in solution_step.keys()
                and len(solution_step["ost_step"]) == 0
                and "direct_answer" in solution_step.keys()
            ):
                intermediate_step = (
                    solution_step["subanswer"]["text"].split("The answer is")[0].strip()
                )
                solution_trace_str += intermediate_step + " "
                solution_trace_str += "Now we can answer the question: "
                solution_trace_str += solution_step["direct_answer"]["text"].strip()
                final_step_str = solution_step["direct_answer"]["text"].strip()
                reward_value = (
                    solution_step["direct_answer"]["value"]
                    if "value" in solution_step["direct_answer"]
                    else 0.0
                )
                end_node_type = Node_Type.DIRECT_ANSWER
                break
            # 5. diranswer
            elif "direct_answer" in solution_step.keys():
                assert len(solution_step["ost_step"]) == 0
                assert "subanswer" not in solution_step.keys()

                solution_trace_str += "Now we can answer the question: "
                solution_trace_str += solution_step["direct_answer"]["text"].strip()
                final_step_str = solution_step["direct_answer"]["text"].strip()
                reward_value = (
                    solution_step["direct_answer"]["value"]
                    if "value" in solution_step["direct_answer"]
                    else 0.0
                )
                end_node_type = Node_Type.DIRECT_ANSWER
                break
            else:
                import pdb

                pdb.set_trace()

    solution_trace_str = solution_trace_str.replace("Let's think step by step. ", "")
    solution_trace_str = "Let's think step by step. " + solution_trace_str

    return (
        solution_trace_str.strip(),
        final_step_str.strip(),
        end_node_type,
        min(0, reward_value) + 1,
    )


def concat_rap_solution_trace(solution_trace: str):
    solution_trace_list = solution_trace.split("\n")
    answer_list = []
    for item in solution_trace_list:
        if item.startswith("Answer"):
            item = re.sub(r"Answer \d+\.\d+: ", "", item)
            final_step = item
            item = re.sub(r" The answer is \d+\.", "", item)
            answer_list.append(item)
    return " ".join(answer_list).strip(), final_step


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
        interval = (right_boundary - left_boundary) / (num_return - 1)

    words_in_solution_trace = solution_trace_str.split(" ")
    ost_len = len(words_in_solution_trace)
    # Mask the solution trace string from least to most
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
