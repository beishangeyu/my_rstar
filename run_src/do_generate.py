# Licensed under the MIT license.
import sys

sys.path.append(".")

from eval_src.Evaluator import *
from MCTS_for_reasoning import Generator, search_for_answers
from run_src.rstar_utils import GeneratorError
from common.arguments import get_parser, post_process_args, save_args
from common.utils import fix_seeds, read_jsonl, write_jsonl, load_dataset
import os
import json
import time
from tqdm import tqdm


def main(args):
    fix_seeds(args.seed)

    args.local_rank, args.world_size = 0, 1

    dataset_path = f"./data/{args.dataset_name}.jsonl"
    dataset = load_dataset(read_jsonl(dataset_path))
    evaluator = PythonEvaluator()

    tokenizer, model = None, None
    if args.api == "vllm":
        from models.vLLM_API import load_vLLM_model

        tokenizer, model = load_vLLM_model(
            args.model_ckpt, args.seed, args.tensor_parallel_size, args.half_precision
        )
    generator = Generator(args, tokenizer, model, evaluator)

    num_tested = 0
    start_time = time.time()

    for i, data_item in enumerate(dataset):
        print(f"---------------------- Curent task id: {i} -------------------------")

        problem_id, problem, gt_solution = (
            data_item["id"],
            data_item["problem"],
            data_item["solution"],
        )
        gt_answer = data_item["code"]

        js = {
            "id": problem_id,
            "problem": problem,
            "model_completion": None,
            "model_answer": None,
            "all_model_completions": {},
            "gold_solution": gt_solution,
            "gold_answer": gt_answer,
        }

        model_solutions, stopping_id, model_all_solutions = [], -1, []

        # try:
        model_solutions, stopping_id, model_all_solutions = search_for_answers(
            args=args,
            user_question=problem,
            question_id=i,
            gt_answer=gt_solution,
            generator=generator,
        )
        # except GeneratorError as e:
        #     print(e)
        #     js["generator_error"] = {
        #         "source": e.source,
        #         "io_input": e.io_input,
        #         "io_output_list": e.io_output_list,
        #     }
        # except Exception as e:
        #     print(e)
        #     js["other_error"] = {"text": str(e)}

        num_tested += 1

        # TODO 更改保存文件的方式
        with open(
            os.path.join(args.answer_sheets_dir, f"Question {i:04d} - Answer.json"), "w"
        ) as f:
            json.dump(js, f)

        with open(
            os.path.join(args.run_outputs_dir, "intermediate_result.txt"), "w"
        ) as f:
            f.write(
                f"Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}\n"
            )
            f.write(
                f"Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}\n"
            )

    end_time = time.time()

    print(
        f"==> Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}"
    )
    print(
        f"==> Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}"
    )
    print(
        f"==> Total time: {end_time-start_time:.2f}s, Avg time: {(end_time-start_time)/(num_tested):.2f}s"
    )

    with open(os.path.join(args.run_outputs_dir, "final_result.txt"), "w") as f:
        f.write(
            f"Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}\n"
        )
        f.write(
            f"Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}\n"
        )
        f.write(
            f"Total time: {end_time-start_time:.2f}s, Avg time: {(end_time-start_time)/(num_tested):.2f}s\n"
        )


if __name__ == "__main__":
    #! -------------------------------- Arguments --------------------------------
    parser = get_parser()

    # 数据集的名称
    parser.add_argument("--dataset_name", type=str)

    parser.add_argument("--num_rollouts", type=int, default=15)
    parser.add_argument("--num_votes", type=int, default=10)
    parser.add_argument("--max_depth_allowed", type=int, default=5)

    # MCTS
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument(
        "--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const"
    )
    parser.add_argument("--mcts_num_last_votes", type=int, default=None)
    parser.add_argument("--save_tree", action="store_true")

    # Action1: Propose an one-step thought.
    parser.add_argument("--num_a1_steps", type=int, default=None)

    # Paraphrasing
    parser.add_argument("--modify_prompts_for_rephrasing", action="store_true")

    args = parser.parse_args()

    # NOTE 一次generate中模型回复的次数
    if args.mcts_num_last_votes is None:
        args.mcts_num_last_votes = 32

    # NOTE 采用一次action1生成的子节点数量
    if not args.disable_a1:
        if args.num_a1_steps is None:
            args.num_a1_steps = 3

    args = post_process_args(args)
    print(args)
    save_args(args)
    main(args)
