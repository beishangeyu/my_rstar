# Licensed under the MIT license.

import sys

sys.path.append(".")
from common.utils import write_jsonl, read_jsonl, load_dataset
from eval_src.Evaluator import *

import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
from argparse import ArgumentParser


# TODO 传入 List[Dict] 类型, 需要修改逻辑?
def extract_trace(data_item: List[Dict], num_votes: int):
    res = []
    for item in data_item:
        trace = item["trace"]["0"]
        rollout_id = item["rollout_id"]
        if num_votes != -1 and rollout_id >= num_votes:
            continue
        if "direct_answer" in trace:
            res.append(trace["direct_answer"]["text"])
    return res


def extract_completions(data_item):
    res = []
    for item in data_item:
        res.append(data_item[item]["model_solution"])
    return res


def eval_single_item(
    task_id: int,
    gene_result_dir: str,
    dataset_name: str,
    test_list: List[str],
    evaluator: Evaluator,
    num_votes=-1,
) -> dict:
    data_item = {}
    solution_candidates = read_jsonl(
        os.path.join(gene_result_dir, f"Task_id_{task_id}_all_solutions.jsonl")
    )

    solution_candidates = extract_trace(solution_candidates, num_votes)
    model_answer, _, _, _ = evaluator.find_most_confident_answer(solution_candidates)
    result = evaluator.chect_correctness(model_answer, dataset_name, test_list)
    # TODO 多添加几个 key?
    data_item["task_id"] = task_id
    data_item["correct"] = result
    data_item["predict_answer"] = model_answer

    return data_item


def eval_exp(
    gene_result: str,
    dataset_name: str,
    eval_result: str,
    num_votes: int = -1,
    model_ckpts=str,
):
    dataset_path = f"./data/{args.dataset_name}.jsonl"
    dataset = load_dataset(read_jsonl(dataset_path))
    evaluator = PythonEvaluator()
    gene_result_dir = os.path.join(gene_result, f"{dataset_name}", f"{model_ckpts}")

    data_list = []
    for item in dataset:
        task_id = item["task_id"]
        test_list = item["test_list"]
        dta = eval_single_item(
            task_id, gene_result_dir, dataset_name, test_list, evaluator, num_votes
        )
        data_list.append(dta)

    # Calculate accuracy
    accuracy = sum([item["correct"] for item in data_list]) / len(data_list)
    print(f"accuracy: {accuracy}")

    eval_result_dir = os.path.join(eval_result, f"{dataset_name}", f"{model_ckpts}")
    os.makedirs(eval_result_dir, exist_ok=True)
    write_jsonl(os.path.join(eval_result_dir, "eval_results.jsonl"), data_list)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_votes", type=int, default=-1)
    args = parser.parse_args()

    eval_exp(
        args.gene_result,
        args.dataset_name,
        args.eval_result,
        args.num_votes,
        args.model_ckpts,
    )

# if __name__ == "__main__":
#     path = "./gene_result/mbpp_Mistral-7B-v0.1/Mistral-7B-v0.1/Task_id_7_all_solutions.jsonl"
#     data = read_jsonl(path)
#     for it in data:
#         print(it)
#     print(extract_trace(data_item=data, num_votes=-1))
