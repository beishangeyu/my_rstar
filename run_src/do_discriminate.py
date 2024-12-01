# Licensed under the MIT license.
import sys

sys.path.append(".")

from datetime import datetime
from copy import deepcopy
from collections import defaultdict
from argparse import ArgumentParser
from models.vLLM_API import load_vLLM_model, generate_with_vLLM_model
from run_src.rstar_utils import (
    concat_solution_trace,
    mask_solution_trace,
    make_funchead_and_docstring,
)
from run_src.Evaluator import *
from common.utils import (
    fix_seeds,
    write_jsonl,
    read_jsonl,
    load_dataset,
    enumerate_resume,
)
from common.arguments import get_parser, post_process_args, save_args
import os
import json
from tqdm import tqdm
from prompt import ost_prompt


# NOTE 封装一个trace和它所有的masked trace
class Candidate:
    def __init__(
        self,
        solution_trace,
        masked_solution_trace_list,
        final_step,
        final_answer,
        id,
        freq=1,
        trace_reward=1.0,
        c_type="default",
    ):
        self.solution_trace = solution_trace
        self.masked_solution_trace_list = masked_solution_trace_list
        self.final_step = final_step
        self.final_answer = final_answer
        self.id = id
        self.freq = freq
        self.trace_reward = trace_reward
        self.c_type = c_type

    def __str__(self):
        return f"Candidate {self.id}: {self.final_answer}"

    def to_dict(self):
        return {
            "solution_trace": self.solution_trace,
            "masked_solution_trace_list": self.masked_solution_trace_list,
            "final_step": self.final_step,
            "final_answer": self.final_answer,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            solution_trace=data["solution_trace"],
            masked_solution_trace_list=data["masked_solution_trace_list"],
            final_step=data["final_step"],
            final_answer=data["final_answer"],
            id=data["id"],
        )


# 用于把相同answer的candidate分在一起, 返回answer的confidence和出现次数
def group_candidates_by_answer(candidates: list[Candidate], evaluator, criteria="freq"):
    """Return answer2candidates, answer2confidence, answer2cnt."""
    answer2candidates = {}  # 记录每个answer
    answer2confidence = defaultdict(float)  # 记录每个answer的confidence
    answer2cnt = defaultdict(int)  # 记录每个answer的出现次数

    # 遍历同一个 task id 的所有 trace 的所有的 masked trace
    for c in candidates:
        has_existed = False  # 表示是否已经记录相同的answer

        # 如果answer2candidates不为空, 就比较其中所有的answer和当前candidate的answer是否相等
        # 即如果当前这个answer被记录了, 就把这个candidate加到这个answer对应的list中
        for existing_answer in answer2candidates.keys():
            # 如果存在一个answer和当前candidate的answer相等
            if evaluator.check_answers_equiv(c.final_answer, existing_answer):
                has_existed = True
                # 是在把相同answer的candidate放在一起
                answer2candidates[str(existing_answer)].extend([c] * c.freq)
                # NOTE 默认以 original trace 的 reward 作为 confidence
                # XXX 如果改成 freq 就和 do eval 一样了
                answer2confidence[str(existing_answer)] += (
                    c.trace_reward if criteria == "reward" else c.freq
                )
                answer2cnt[str(existing_answer)] += c.freq
                break

        # 如果当前这个answer还没被记录, 就记录
        if not has_existed:
            if str(c.final_answer) in answer2candidates:
                # 这个candidate出现了几次,就往里边加几个
                answer2candidates[str(c.final_answer)].extend([c] * c.freq)
            else:
                answer2candidates[str(c.final_answer)] = [c] * c.freq
            answer2confidence[str(c.final_answer)] += (
                c.trace_reward if criteria == "reward" else c.freq
            )
            answer2cnt[str(c.final_answer)] += c.freq

    assert all(
        answer2cnt[ans] == len(answer2candidates[ans]) for ans in answer2cnt.keys()
    )
    assert float(sum([candidate.trace_reward for candidate in candidates])) == float(
        sum([answer2confidence[ans] for ans in answer2confidence.keys()])
    )

    # 即 all_solution.jsonl 中的 trace 个数
    candidates_count = sum([candidate.freq for candidate in candidates])
    for ans in answer2confidence.keys():
        answer2confidence[ans] /= candidates_count

    return answer2candidates, answer2confidence, answer2cnt


class Discriminator:
    def __init__(self, args, evaluator, discriminate_out_dir):
        self.args = args
        self.evaluator = evaluator
        self.disc_out_dir = discriminate_out_dir

    # 过滤掉没有答案的
    def _filter_none(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates = [c for c in candidates if c.final_answer is not None]
        return candidates

    # 过滤掉答案长度大于100的
    def _filter_long(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates = [c for c in candidates if len(c.final_answer) <= 100]
        return candidates

    def _filter_reasoning_consistency(
        self,
        gen_model,
        funchead_and_docstring: str,
        candidates: list[Candidate],
    ) -> list[Candidate]:

        assert all(
            len(c.masked_solution_trace_list) == self.args.num_masked_solution_traces
            for c in candidates
            if c.c_type == "default"
        )
        consistent_candidates = []
        # NOTE 这里所有的 candidate 都对应同一个 task_id
        for c in candidates:
            completion_list = []
            for masked_solution_trace in c.masked_solution_trace_list:
                # 补上问题
                masked_solution_trace = (
                    f"[Function haed and docstring]\n{funchead_and_docstring}\n\n"
                    + masked_solution_trace
                )
                # 如果有 ost step, 就补上 ost 的 prompt
                if "[Step to implement]" in c.solution_trace:
                    masked_solution_trace = ost_prompt + "\n" + masked_solution_trace
                completions = self._gen_func(
                    gen_model=gen_model,
                    gen_input=masked_solution_trace,
                    temperature=self.args.rc_temperature,
                    n=self.args.rc_n_completions,  # NOTE 生成 rc_n_completions 个补全答案
                    max_tokens=1024,
                    stop_tokens=[
                        "[[Function haed and docstring]]",
                        "You are a Python assistant.",
                    ],
                )
                completion_list.append(completions)
            # 把 list of str 展开成 list of str
            completion_list = [
                c for cps in completion_list for c in cps if c is not None
            ]
            answer_list = [
                self.evaluator.extract_answer_from_model_completion(completion)
                for completion in completion_list
            ]

            num_consistent = 0
            candidate_count = len(completion_list)
            # 多数投票策略
            if self.args.rc_mode == "maj":
                answer = self.evaluator.find_most_confident_answer(completion_list)
                if self.evaluator.check_answers_equiv(c.final_answer, answer):
                    consistent_candidates.append(c)
            else:
                # 把candidate和discriminator的补全答案一个个比较, 如果相等, num_consistent就加1
                for answer in answer_list:
                    if self.evaluator.check_answers_equiv(c.final_answer, answer):
                        num_consistent += 1

                # 只要有一个补全答案跟本来的答案一样, 这个 candidate 就有效
                if self.args.rc_mode == "loose":
                    if num_consistent > 0:
                        consistent_candidates.append(c)
                # 过半的的补全答案和本来的答案一样, 这个 candidate 才有效
                elif self.args.rc_mode == "mid":
                    if num_consistent >= candidate_count // 2:
                        consistent_candidates.append(c)
                # 所有的补全答案和本来的答案一样, 这个 candidate 才有效
                elif self.args.rc_mode == "strict":
                    if num_consistent == candidate_count:
                        consistent_candidates.append(c)

        # 返回所有达到一致性要求的candidate
        return consistent_candidates

    def _gen_func(
        self,
        gen_model,
        gen_input,
        temperature: float,
        n: int = 1,
        max_tokens: int = 768,
        stop_tokens=None,
    ):
        if temperature == 0.0:
            n = 1

        response = generate_with_vLLM_model(
            model=gen_model,
            input=gen_input,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            stop=stop_tokens,
        )
        if n == 1:
            if isinstance(gen_input, str):
                return response[0].outputs[0].text
            elif isinstance(gen_input, list):
                return [r.outputs[0].text for r in response]
        elif n > 1:
            if isinstance(gen_input, str):
                return [o.text for o in response[0].outputs]
            elif isinstance(gen_input, list):
                return [[o.text for o in r.outputs] for r in response]

    def _calculate_scores(
        self,
        unfiltered_candidates: list[Candidate],
        filtered_candidates: list[Candidate],
    ) -> dict:
        # 获取一致性满足要求的candidate对应的answer的confidence和出现次数
        _, filtered_answer2confidence, filtered_answer2cnt = group_candidates_by_answer(
            filtered_candidates, self.evaluator, self.args.rc_criteria
        )
        print(f"==> Confidence: {filtered_answer2confidence}")
        # 只满足非空和长度要求的candidate的answer的出现次数
        _, _, unfiltered_answer2cnt = group_candidates_by_answer(
            unfiltered_candidates, self.evaluator, self.args.rc_criteria
        )

        filtered_answer2survival_rate = {}
        for filtered_ans in filtered_answer2cnt.keys():
            has_existed = False
            # 检查是否过滤后的答案是否跟未过滤中的某个答案相等
            # 如果是, 存活率 = 过滤后的出现次数 / 未过滤的出现次数
            # 如果不是, 存活率是0
            for unfiltered_ans in unfiltered_answer2cnt.keys():
                if self.evaluator.check_answers_equiv(filtered_ans, unfiltered_ans):
                    has_existed = True
                    filtered_answer2survival_rate[filtered_ans] = (
                        filtered_answer2cnt[filtered_ans]
                        / unfiltered_answer2cnt[unfiltered_ans]
                    )
                    break
            if not has_existed:
                filtered_answer2survival_rate[filtered_ans] = 0.0

        print(f"==> Survival rates: {filtered_answer2survival_rate}")

        # 计算得分
        # 每个answer的得分 = 存活率 + confidence
        filtered_answer2score = {}
        for filtered_ans in filtered_answer2confidence.keys():
            has_existed = False
            for unfiltered_ans in unfiltered_answer2cnt.keys():
                if self.evaluator.check_answers_equiv(filtered_ans, unfiltered_ans):
                    has_existed = True
                    filtered_answer2score[filtered_ans] = (
                        filtered_answer2confidence[filtered_ans]
                        + filtered_answer2survival_rate[filtered_ans]
                    )
                    break
            if not has_existed:
                filtered_answer2score[filtered_ans] = 0.0

        print(f"==> Scores: {filtered_answer2score}")

        return filtered_answer2score

    # 从"非空且长度合规的candidate"还有"在此基础上符合一致性要求的candidate"中选出winner
    def _find_winner_filtered(
        self,
        unfiltered_candidates: list[Candidate],
        filtered_candidates: list[Candidate],
    ) -> Candidate:
        # 如果没有一致性达到要求的candidate, 就从prefiltered的candidate中选最好(出现次数最多)的那个作为winner
        if len(filtered_candidates) == 0:
            answer2candidates, answer2confidence, _ = group_candidates_by_answer(
                unfiltered_candidates, self.evaluator, self.args.rc_criteria
            )
            most_confident_answer = max(
                answer2confidence.keys(), key=lambda x: answer2confidence[x]
            )
            winner = answer2candidates[most_confident_answer][0]
            print(f"==> Winner answer: {most_confident_answer}\n")
        # 如果只有一个达到一致性要求的candidate, 直接把这个选成winner
        elif len(filtered_candidates) == 1:
            winner = filtered_candidates[0]
            print(f"==> Winner answer: {winner.final_answer}\n")
        # 如果所有的达到一致性要求的candidate的answer都和user question的标准答案不一样, winner为none
        elif not any(
            self.evaluator.chect_correctness(c.final_answer)
            for c in filtered_candidates
        ):
            winner = None
            print(f"==> Winner answer: None")
        # 如果达到一致性要求的candidate不止一个, 且在达到一致性要求的candidate中存在和标准答案一样的
        # 计算所有answer中分数最高的, 然后选那个answer对应的第一个candidate作为winner
        else:
            # 计算每个answer的分数(多个candidate的answer可能一样)
            filtered_answer2score = self._calculate_scores(
                unfiltered_candidates, filtered_candidates
            )
            # 选出分数最高的answer
            winner_answer = max(
                filtered_answer2score.keys(), key=lambda x: filtered_answer2score[x]
            )
            print(f"==> Winner answer: {winner_answer}")
            # next会返回第一个符合标准的元素
            winner = next(
                c
                for c in filtered_candidates
                if self.evaluator.check_answers_equiv(c.final_answer, winner_answer)
            )

        return winner


class MajorityVoteDiscriminator(Discriminator):
    def __init__(self, args, evaluator, discriminate_out_dir):
        super().__init__(args, evaluator, discriminate_out_dir)
        self.tokenizer, self.model = None, None
        if self.args.api == "vllm":
            if args.max_model_len > 0:
                self.tokenizer, self.model = load_vLLM_model(
                    args.model_ckpt,
                    args.seed,
                    max_num_seqs=args.max_num_seqs,
                    tensor_parallel_size=args.tensor_parallel_size,
                    max_model_len=args.max_model_len,
                )
            else:
                self.tokenizer, self.model = load_vLLM_model(
                    args.model_ckpt,
                    args.seed,
                    max_num_seqs=args.max_num_seqs,
                    tensor_parallel_size=args.tensor_parallel_size,
                )

    def select(
        self, candidates: list[Candidate], funchead_and_docstring: str
    ) -> Candidate:

        unfiltered_candidates = candidates
        print(
            f"==> Unfiltered answers: {[c.final_answer for c in unfiltered_candidates]}"
        )
        # candidate: [1, 2, 3, 4, 5, None, paosdifjpsod]
        # 先把没有答案的和答案太长的过滤掉
        prefiltered_candidates = self._filter_none(candidates)
        # prefiltered_candidates: [1, 2, 3, 4, 5]
        print(
            f"==> Pre-filtered answers: {[c.final_answer for c in prefiltered_candidates]}"
        )
        # 过滤掉一致性不够的答案
        # TODO 这里的 problem 应该用的是 trace 中的, 可能 rephrase 过, 可以优化吗?
        filtered_candidates = self._filter_reasoning_consistency(
            self.model, funchead_and_docstring, prefiltered_candidates
        )
        # filtered_candidates: [1, 2, 3]
        print(
            f"==> RC-filtered answers: {[c.final_answer for c in filtered_candidates]}"
        )
        return self._find_winner_filtered(prefiltered_candidates, filtered_candidates)


def main():
    parser = get_parser()
    parser.add_argument("--threshold", type=float, default=0.999)
    # vLLM
    parser.add_argument("--max_num_seqs", type=int, default=256)
    # NOTE mask 的最小值和最大值
    parser.add_argument("--mask_left_boundary", type=float, default=0.2)
    parser.add_argument("--mask_right_boundary", type=float, default=0.5)
    # NOTE 对一个 solution trace 要生成几个 masked trace
    parser.add_argument("--num_masked_solution_traces", type=int, default=4)
    # NOTE fillter consistency 的严格程度, loose: 有一个一样就行, mid: 大于一半一样就行, strict: 全一样才行
    parser.add_argument(
        "--rc_mode", type=str, default="mid", choices=["loose", "mid", "strict", "maj"]
    )
    parser.add_argument("--rc_temperature", type=float, default=1.0)
    # NOTE 对一个 masked trace 要生成几个补全答案
    parser.add_argument("--rc_n_completions", type=int, default=1)
    # NOTE group candidates by answer 时 confidence 的评判标准, 如果是 freq 就跟 do eval 一样了, 没意义
    parser.add_argument(
        "--rc_criteria", type=str, default="reward", choices=["freq", "reward"]
    )
    # NOTE do generate 结果的存放路径
    parser.add_argument("--gene_result_name", type=str)
    args = parser.parse_args()

    fix_seeds(args.seed)

    gene_result_dir = os.path.join(
        args.gene_result, args.dataset_name, args.gene_result_name
    )
    # NOTE discriminate 结果的存放路径
    model_name = args.model_ckpt.split("/")[-1]
    discriminate_out_dir = os.path.join(
        args.disc_result, args.gene_result_name, f"{model_name}"
    )
    os.makedirs(discriminate_out_dir, exist_ok=True)

    # 记录当前的args
    recording = vars(args)
    with open(os.path.join(discriminate_out_dir, "summary.json"), "w") as f:
        json.dump(recording, f, indent=4)

    evaluator = PythonEvaluator()
    discriminator = MajorityVoteDiscriminator(args, evaluator, discriminate_out_dir)

    #! ------ Select winner candidate for each example ------

    data_path = f"./data/{args.dataset_name}.jsonl"
    dataset = load_dataset(read_jsonl(data_path))
    num_correct, num_correct_majvote, num_correct_limit, num_tested = 0, 0, 0, 0
    # 遍历每个 task_id
    total_num_candidates = 0
    for i, item in enumerate_resume(dataset, discriminate_out_dir):
        task_id = item["task_id"]
        path = os.path.join(
            gene_result_dir,
            f"Task_id_{task_id}_all_solutions.jsonl",
        )
        solution_traces = read_jsonl(path)

        test_list = item["test_list"]
        code = item["code"]
        func_name = re.search(r"def (.+?)\(", code).group(1)
        func_head = re.search(r"def .+?:", code).group(0)
        test_case = item["test_list"][0][7:]
        requirement = item["text"]
        funchead_and_docstring = make_funchead_and_docstring(
            requirement, func_head, test_case
        )
        # XXX select的时候的problem就是requirement, 纠结的是如果有rephrased要不要用rephrased的

        all_candidates = []
        solution_trace_dic = {}
        # 遍历同一个 task id 下所有的 solution trace, 将它们添加到的 dict 中
        for id, it in enumerate(solution_traces):
            # 把 solution trace 组合起来, 添加到 dict 中
            solution_trace, final_step, _, reward = concat_solution_trace(
                it["trace"], func_name
            )
            if solution_trace in solution_trace_dic:
                solution_trace_dic[solution_trace]["freq"] = (
                    solution_trace_dic[solution_trace]["freq"] + 1
                )
                solution_trace_dic[solution_trace]["reward"] = (
                    solution_trace_dic[solution_trace]["reward"] + reward
                )
                if len(solution_trace_dic[solution_trace]["final_step"]) < len(
                    final_step
                ):
                    solution_trace_dic[solution_trace]["final_step"] = final_step
            else:
                solution_trace_dic[solution_trace] = {
                    "freq": 1,
                    "reward": reward,
                    "final_step": final_step,
                }

        # 遍历所有 solution trace, 对他们进行 mask, 一个 trace 对应一个 Candidate
        for solution_trace in solution_trace_dic.keys():
            final_step = solution_trace_dic[solution_trace]["final_step"]
            # freq 是一条 solution trace 在 all solution traces 中的出现次数
            trace_freq = solution_trace_dic[solution_trace]["freq"]
            # reward 是一条 solution trace 在模型生成的多个序列中的出现次数
            trace_reward = solution_trace_dic[solution_trace]["reward"]

            masked_solution_trace_list = mask_solution_trace(
                solution_trace,
                num_return=args.num_masked_solution_traces,
                left_boundary=args.mask_left_boundary,
                right_boundary=args.mask_right_boundary,
            )
            final_answer = evaluator.extract_answer_from_model_completion(final_step)
            # 一个 Candidate 记录同一条 trace 的所有 masked trace
            candidate = Candidate(
                solution_trace,
                deepcopy(masked_solution_trace_list),
                final_step,
                final_answer,
                id,  # all solution 中第几条 trace
                trace_freq,
                trace_reward,
            )
            all_candidates.append(candidate)

        # 将所有的 candidate 按照 answer 划分
        answer2candidates, answer2confidence, _ = group_candidates_by_answer(
            all_candidates, evaluator, args.rc_criteria
        )
        # 选出 confidence 最高的 answer
        most_confident_answer = max(
            answer2candidates.keys(), key=lambda x: answer2confidence[x]
        )
        highest_confidence = answer2confidence[most_confident_answer]
        assert highest_confidence > 0
        candidates = all_candidates
        # len(candidates) 也是当前 task id 下 trace 种类数
        total_num_candidates += len(candidates)
        # 如果所有 trace 的 answer 都不对, 就不继续找了
        if not any(
            evaluator.check_correctness(ans, args.dataset_name, test_list)
            for ans in answer2candidates.keys()
        ):
            print("Well, no correct answer in candidates. Skipping...")
            winner_answer = ""
        # 否则选出 confidence 最高的看看对不对
        else:
            # 如果最高confidence大于阈值, 直接选对应answer作为winner
            if highest_confidence > args.threshold:
                print("You are very confident. Skipping...")
                winner_answer = most_confident_answer
            # 否则调用select
            else:
                winner_candidate = discriminator.select(
                    candidates,
                    funchead_and_docstring,
                )
                # 如果选出了 winner, 则 answer 是 winner 的 final answer
                if winner_candidate is not None:
                    winner_answer = winner_candidate.final_answer
                # 否则也只能老实使用最高confidence的answer
                else:
                    winner_answer = most_confident_answer
        # -------------------------------
        # winner answer 是经过补全之后选出来的, most_confident_answer 只是计算原 trace 的 confidence 之后选的, 不涉及一致性
        # 判别 winner answer 是否正确
        correct = evaluator.check_correctness(
            winner_answer, args.dataset_name, test_list
        )
        # 判别最高置信度answer是否正确
        correct_majvote = evaluator.check_correctness(
            most_confident_answer, args.dataset_name, test_list
        )
        # 在所有 answer 里边是否有正确的 (对应于 winner answer 不对但是别的 answer 对了的情况)
        correct_limit = (
            1
            if any(
                evaluator.check_correctness(ans, args.dataset_name, test_list)
                for ans in answer2candidates.keys()
            )
            else 0
        )
        print(f"==> Correct: {correct}")
        # 保存数据
        temp_recording = {}
        temp_recording.update(
            {
                "task_id": task_id,
                "correct": correct,
                "correct_majvote": correct_majvote,
                "correct_limit": correct_limit,
            }
        )
        result_path = os.path.join(discriminate_out_dir, f"Task_result.jsonl")
        write_jsonl(result_path, [temp_recording], append=True)
        num_correct += int(correct)
        num_correct_majvote += int(correct_majvote)
        num_correct_limit += int(correct_limit)
        num_tested += 1

        info = f"Acc: {num_correct / num_tested:.4f}; Majority vote acc: {num_correct_majvote / num_tested:.4f}; Limit acc: {num_correct_limit / num_tested:.4f}"
        print(info)

    print(
        f"Accuracy: {num_correct / num_tested:.4f}; Majority vote accuracy: {num_correct_majvote / num_tested:.4f}; Limit accuracy: {num_correct_limit / num_tested:.4f}"
    )
    path = os.path.join(discriminate_out_dir, f"summary.json")
    recording.update(
        {
            "num_correct": num_correct,
            "num_correct_majvote": num_correct_majvote,
            "num_correct_limit": num_correct_limit,
            "num_tested": num_tested,
            "accuracy": num_correct / num_tested,
            "majority_vote_accuracy": num_correct_majvote / num_tested,
            "limit_accuracy": num_correct_limit / num_tested,
            "avg_num_candidates": total_num_candidates / num_tested,
        }
    )
    with open(path, "w") as f:
        json.dump(recording, f, indent=4)

    print(f"Recording: \n{recording}")


if __name__ == "__main__":
    main()
