# Licensed under the MIT license.
import sys

sys.path.append(".")

from Evaluator import *
from MCTS_for_reasoning import Generator, search_for_answers
from common.arguments import get_parser, post_process_args, save_args
from common.utils import fix_seeds, read_jsonl, load_dataset, enumerate_resume


def main(args):
    fix_seeds(args.seed)

    args.local_rank, args.world_size = 0, 1

    dataset_path = f"./data/{args.dataset_name}.jsonl"
    dataset = load_dataset(read_jsonl(dataset_path))
    evaluator = PythonEvaluator(
        device=args.evaluator_device,
        threshold=args.evaluator_threshold,
        disable_clone_detector=args.disable_clone_detector,
        disable_mutual_vote=args.disable_mutual_vote,
    )

    tokenizer, model = None, None
    if args.api == "vllm":
        from models.vLLM_API import load_vLLM_model

        tokenizer, model = load_vLLM_model(
            args.model_ckpt,
            args.seed,
            args.tensor_parallel_size,
            args.half_precision,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    generator = Generator(args, tokenizer, model, evaluator)

    num_tested = 0
    for i, data_item in enumerate_resume(dataset, args.gene_result):
        problem_id, problem = data_item["task_id"], data_item["adv_text"]
        if args.dataset_name == "mbpp":
            problem = data_item["text"]
        if args.dataset_name == "humaneval_modi":
            strs = data_item["text"].split("====")
            problem = "".join(strs)
            strs = problem.split("\n")
            if len(strs) > 1:
                problem = strs[0] + "\n" + "\n".join("    " + s for s in strs[1:])
        # for debug
        # if problem_id < 17:
        #     continue
        # if problem_id > 17:
        #     break

        model_solutions, stopping_id, model_all_solutions = [], -1, []
        model_solutions, stopping_id, model_all_solutions = search_for_answers(
            args=args,
            user_question=problem,
            generator=generator,
            task=data_item,
            task_id=problem_id,
        )

        num_tested += 1


if __name__ == "__main__":
    parser = get_parser()

    parser.add_argument("--num_rollouts", type=int, default=15)
    # 每次会生成几个subq类型的子节点
    parser.add_argument(
        "--num_subquestions",
        type=int,
        default=3,
    )
    parser.add_argument("--max_depth_allowed", type=int, default=11)

    # MCTS
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    # 探索权重
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    # 动态调整探索权重
    parser.add_argument(
        "--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const"
    )
    # 在 generate direct answer 的时候模型生成的序列个数
    parser.add_argument("--mcts_num_last_votes", type=int, default=32)
    parser.add_argument("--save_tree", action="store_true")
    # 每次会生成几个ost类型子节点
    parser.add_argument("--num_a1_steps", type=int, default=3)
    # 这是第几个结果
    parser.add_argument("--result_iteration", type=int, required=True)

    args = parser.parse_args()

    args = post_process_args(args)
    print(args)
    save_args(args)
    main(args)
