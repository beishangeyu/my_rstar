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
from eval_src.Evaluator import *
from common.utils import fix_seeds, write_jsonl, read_jsonl, load_dataset
from common.arguments import get_parser
import os
import json
from tqdm import tqdm


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
                # 默认以出现次数作为confidence
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

    candidates_count = sum([candidate.freq for candidate in candidates])
    for ans in answer2confidence.keys():
        answer2confidence[
            ans
        ] /= candidates_count  # 当前answer的confidence是当前answer的出现次数 / 所有answer的出现次数

    return answer2candidates, answer2confidence, answer2cnt


class Discriminator:
    def __init__(self, args, evaluator):
        self.args = args
        self.evaluator = evaluator

        # XXX 这里的fewshot_config跟generate之间的是一样的?
        self.fewshot_config = read_json(args.fewshot_config_path)
        self.fewshot_template = self.fewshot_config["prompt_template"]
        self.stop_tokens = self.fewshot_config["stop_tokens"]

        self.fewshot_prompt = read_txt(args.fewshot_prompt_path)

    # 过滤掉没有答案的
    def _filter_none(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates = [c for c in candidates if c.final_answer is not None]
        return candidates

    # 过滤掉答案长度大于100的
    def _filter_long(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates = [c for c in candidates if len(c.final_answer) <= 100]
        return candidates

    def _filter_reasoning_consistency(
        self, gen_model, problem: str, candidates: list[Candidate], aux={}
    ) -> list[Candidate]:
        problem_id = aux["problem_id"]
        file_idx = aux["file_idx"]

        prompt_template = self.fewshot_template
        fewshot_examples = self.fewshot_prompt
        stop_tokens = self.stop_tokens

        assert all(
            len(c.masked_solution_trace_list) == self.args.num_masked_solution_traces
            for c in candidates
            if c.c_type == "default"
        )
        gen_input_list = []
        ground_truth_list = []
        c_completion_num_list = []
        for c in candidates:
            # 对一个mask solution trace补全rc_n_completions次
            for masked_solution_trace in c.masked_solution_trace_list:
                for _ in range(self.args.rc_n_completions):
                    gen_input_list.append(
                        prompt_template.format(
                            examples=fewshot_examples, instruction=problem
                        )
                        + masked_solution_trace
                    )
                    # 这里以candidates的final_answer作为标准答案
                    ground_truth_list.append(c.final_answer)
            c_completion_num_list.append(
                len(c.masked_solution_trace_list) * self.args.rc_n_completions
            )

        # XXX 对每个candidate会被mask多次(每次长度不同?)
        """gen_input_list:
        [c1_mask1, c1_mask2, ..., c2_mask1, c2_mask2, ..., ......, ct_mask1, ct_mask2, ...]
        """

        # Manually split into batches
        batch_size = self.args.max_num_seqs // self.args.rc_n_completions // 2
        gen_output_list = []
        # 按批量处理
        for start_idx in range(0, len(gen_input_list), batch_size):
            end_idx = start_idx + batch_size
            sub_gen_input_list = gen_input_list[start_idx:end_idx]
            sub_gen_output_list = self._gen_func(
                gen_model=gen_model,
                gen_input=sub_gen_input_list,
                temperature=self.args.rc_temperature,
                n=1,
                max_tokens=512,
                stop_tokens=stop_tokens + ["\n"],
            )
            # extend函数不会将sub_gen_output_list作为一个元素加入, 而是和gen_output_list合并
            gen_output_list.extend(sub_gen_output_list)

        with open(
            os.path.join(
                self.args.discriminate_results_dir, f"problem-{problem_id}.json"
            ),
            "w",
        ) as f:
            js = {
                "problem_id": problem_id,
                "file_idx": file_idx,
                "gen_output_list": gen_output_list,
            }
            json.dump(js, f)

        # gen_output_list 长这样, 按照不同的mask来划分
        """gen_output_list:
        [[c1_mask1_o1, c1_mask1_o2, ...], [c1_mask2_o1, c1_mask2_o2, ...], ..., [ct_mask1_o1, ct_mask1_o2, ...], [ct_mask2_o1, ct_mask2_o2, ...], ...]
        """

        # 把gen_output_list展开
        if all(isinstance(item, list) for item in gen_output_list):
            completion_list = []
            for n_completions in gen_output_list:
                for completion in n_completions:
                    completion_list.append(completion)
            assert len(
                completion_list
            ) == self.args.rc_n_completions * self.args.num_masked_solution_traces * len(
                candidates
            )
            # 每个被mask的推理轨迹会让模型补全多次
            candidate_group_size = (
                self.args.rc_n_completions * self.args.num_masked_solution_traces
            )
        elif all(isinstance(item, str) for item in gen_output_list):
            completion_list = gen_output_list
            candidate_group_size = self.args.num_masked_solution_traces

        answer_list = [
            self.evaluator.extract_answer_from_model_completion(completion)
            for completion in completion_list
        ]

        count = 0
        completion_group_list = []
        answer_group_list = []
        gt_group_list = []
        # 再按照不同的candidate划分成多个list, list of list中每个list对应一个candidate
        # 每个list中包含不同的mask的补全结果
        for num in c_completion_num_list:
            completion_group_list.append(completion_list[count : count + num])
            answer_group_list.append(answer_list[count : count + num])
            gt_group_list.append(ground_truth_list[count : count + num])
            count += num
        assert count == len(completion_list) == len(answer_list)

        consistent_candidates = []

        # 每个candidate对应一个group_list
        for c, completion_group, answer_group, gt_answer in zip(
            candidates, completion_group_list, answer_group_list, gt_group_list
        ):
            candidate_group_size = len(c.masked_solution_trace_list)
            num_consistent = 0
            if self.args.rc_mode == "maj":
                answer = self.evaluator.find_most_confident_answer(completion_group)[0]
                if self.evaluator.check_answers_equiv(gt_answer[-1], answer):
                    consistent_candidates.append(c)
            else:
                # 把candidate和discriminator的补全答案一个个比较, 如果相等, num_consistent就加1
                for answer, gt_a in zip(answer_group, gt_answer):
                    if self.evaluator.check_answers_equiv(gt_a, answer):
                        num_consistent += 1

                # 三种不同的策略, num_consistent的阈值不同, 一致的数量超过阈值的时候, 认为由generator生成的这个candidate是有效的
                if self.args.rc_mode == "loose":
                    if num_consistent > 0:
                        consistent_candidates.append(c)
                elif self.args.rc_mode == "mid":
                    if num_consistent >= candidate_group_size // 2:
                        consistent_candidates.append(c)
                elif self.args.rc_mode == "strict":
                    if num_consistent == candidate_group_size:
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

    # XXX 是怎么计算分数的?
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
        gt_answer: str = None,
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
            self.evaluator.check_answers_equiv(c.final_answer, gt_answer)
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
    def __init__(self, args, evaluator):
        super().__init__(args, evaluator)
        self.tokenizer, self.model = None, None
        if self.args.api == "vllm":
            self.tokenizer, self.model = load_vLLM_model(
                args.model_ckpt, args.seed, max_num_seqs=args.max_num_seqs
            )

    def select(
        self, problem: str, candidates: list[Candidate], gt_answer: str = None, aux={}
    ) -> Candidate:
        print(f"==> Ground truth answer: {gt_answer}")

        unfiltered_candidates = candidates
        print(
            f"==> Unfiltered answers: {[c.final_answer for c in unfiltered_candidates]}"
        )
        # candidate: [1, 2, 3, 4, 5, None, paosdifjpsod]
        # 先把没有答案的和答案太长的过滤掉
        prefiltered_candidates = self._filter_none(candidates)
        prefiltered_candidates = self._filter_long(prefiltered_candidates)
        # prefiltered_candidates: [1, 2, 3, 4, 5]
        print(
            f"==> Pre-filtered answers: {[c.final_answer for c in prefiltered_candidates]}"
        )
        # 过滤掉一致性不够的答案
        filtered_candidates = self._filter_reasoning_consistency(
            self.model, problem, prefiltered_candidates, aux
        )
        # filtered_candidates: [1, 2, 3]
        print(
            f"==> RC-filtered answers: {[c.final_answer for c in filtered_candidates]}"
        )
        return self._find_winner_filtered(
            prefiltered_candidates, filtered_candidates, gt_answer
        )


def main():
    parser = get_parser()
    parser = ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.999)
    # vLLM
    parser.add_argument("--max_num_seqs", type=int, default=256)
    # For multi-choice
    parser.add_argument(
        "--multi_choice_prompt_type",
        type=str,
        default=None,
        choices=["fewshot", "instruct"],
    )
    # For reasoning consistency
    # NOTE mask比例最少占多少和最多占多少
    parser.add_argument("--mask_left_boundary", type=float, default=0.2)
    parser.add_argument("--mask_right_boundary", type=float, default=0.5)
    # NOTE 对一条路径要生成多少条mask路径
    parser.add_argument("--num_masked_solution_traces", type=int, default=4)
    parser.add_argument(
        "--rc_mode", type=str, default="mid", choices=["loose", "mid", "strict", "maj"]
    )
    parser.add_argument("--rc_temperature", type=float, default=1.0)
    parser.add_argument("--rc_n_completions", type=int, default=1)
    parser.add_argument(
        "--rc_criteria", type=str, default="reward", choices=["freq", "reward"]
    )
    args = parser.parse_args()

    # TODO 改成我的 prompt
    args.fewshot_config_path = os.path.join(
        "prompts", args.dataset_name, "fewshot_cot", "fewshot_cot_config.json"
    )
    args.fewshot_prompt_path = os.path.join(
        "prompts", args.dataset_name, "fewshot_cot", "fewshot_cot_prompt.txt"
    )

    fix_seeds(args.seed)
    print(args)

    gene_result_dir = os.path.join(
        args.gene_result, f"{args.dataset_name}", f"{args.model_ckpt}"
    )
    # NOTE discriminate 结果的存放路径
    discriminate_out_dir = os.path.join(
        args.disc_result, f"{args.dataset_name}", f"{args.model_ckpt}"
    )
    os.makedirs(discriminate_out_dir, exist_ok=True)

    # 记录当前的args
    recording_file = os.path.join(discriminate_out_dir, "args.jsonl")
    recording = vars(args)

    evaluator = PythonEvaluator
    discriminator = MajorityVoteDiscriminator(args, evaluator)

    #! ------ Select winner candidate for each example ------

    num_correct, num_correct_majvote, num_correct_limit, num_tested = 0, 0, 0, 0
    data_path = f"./data/mbpp_Mistral-7B-v0.1.jsonl"
    dataset = load_dataset(read_jsonl(data_path))
    # 遍历每个 task_id
    total_num_candidates = 0
    for item in dataset:
        task_id = item["task_id"]
        path = os.path.join(
            args.gene_result,
            args.dataset_name,
            f"Task_id_{task_id}_all_solutions.jsonl",
        )
        solution_traces = read_jsonl(path)

        code = item["code"]
        func_head = re.search(r"def .+?:", code).group(0)
        func_name = re.search(r"def (.+?)\(", code).group(1)
        test_case = item["test_list"][0][7:]

        all_candidates = []
        solution_trace_dic = {}
        # 遍历同一个 task id 下所有的 solution trace
        for i, it in enumerate(solution_traces):
            requirement = it["trace"]["user_requirement"]
            funchead_and_docstring = make_funchead_and_docstring(
                requirement, func_head, test_case
            )
            solution_trace, final_step, _, reward = concat_solution_trace(
                it, funchead_and_docstring, func_name
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

    for file_idx, answer_js_file in enumerate(answer_sheet_json_files):
        print(
            f"\n[Processing file {file_idx}; Total number of files: {len(answer_sheet_json_files)}]\n"
        )
        try:
            answer_js = read_json(answer_js_file)
        except:
            continue

        try:
            problem = answer_js["problem"]
            # assert problem_id == answer_js["id"]
            gold_answer = answer_js["gold_answer"]
        except:
            pass

        trace_js = read_json(
            answer_js_file.replace("Answer", "Final Solutions")
        ) + read_json(answer_js_file.replace("Answer", "Rollout Solutions"))
        if args.cutoff_rollout > -1:
            trace_js = [s for s in trace_js if s["rollout_id"] <= args.cutoff_rollout]

        # ------ Collect all_candidates, answer2candidates answer2confidence ------
        all_candidates = []
        solution_trace_dic = {}
        for id, s in enumerate(trace_js):
            trace = s["trace"] if "trace" in s else s
            solution_trace, final_step, _, reward = concat_solution_trace(trace)
            # 遍历所有的 solution trace,
            # 记录相同的 trace 的出现次数, 累积 reward, 取最长的 final step 作为这个solution trace的final step
            # XXX 为什么要选最长的???
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

        for solution_trace in solution_trace_dic.keys():
            final_step = solution_trace_dic[solution_trace]["final_step"]
            trace_freq = solution_trace_dic[solution_trace]["freq"]
            trace_reward = solution_trace_dic[solution_trace]["reward"]

            masked_solution_trace_list = mask_solution_trace(
                solution_trace,
                num_return=args.num_masked_solution_traces,
                left_boundary=args.mask_left_boundary,
                right_boundary=args.mask_right_boundary,
            )
            final_answer = evaluator.extract_answer_from_model_completion(final_step)
            candidate = Candidate(
                solution_trace,
                deepcopy(masked_solution_trace_list),
                final_step,
                final_answer,
                id,
                trace_freq,
                trace_reward,
            )
            all_candidates.append(candidate)

        answer2candidates, answer2confidence, _ = group_candidates_by_answer(
            all_candidates, evaluator, args.rc_criteria
        )
        most_confident_answer = max(
            answer2candidates.keys(), key=lambda x: answer2confidence[x]
        )
        highest_confidence = answer2confidence[most_confident_answer]
        assert highest_confidence > 0
        # -------------------------------------------------------------------------

        # candidates = [cands[0] for _, cands in answer2candidates.items()]   #! representative
        candidates = all_candidates  # ! exhaustive
        total_num_candidates += len(candidates)

        # ------ Get winner answer ------
        if not any(
            evaluator.check_answers_equiv(ans, gold_answer)
            for ans in answer2candidates.keys()
        ):
            # In this case, we know that there is no correct answer in the candidates
            print("Well, no correct answer in candidates. Skipping...")
            winner_answer = ""
        else:
            if highest_confidence > args.threshold:
                print("You are very confident. Skipping...")
                winner_answer = most_confident_answer
            else:
                winner_candidate = discriminator.select(
                    problem,
                    candidates,
                    gt_answer=gold_answer,
                    aux={"file_idx": file_idx, "problem_id": problem_id},
                )
                if winner_candidate is not None:
                    winner_answer = winner_candidate.final_answer
                else:
                    winner_answer = most_confident_answer
        # -------------------------------
        correct = evaluator.check_answers_equiv(winner_answer, gold_answer)
        correct_majvote = evaluator.check_answers_equiv(
            most_confident_answer, gold_answer
        )
        correct_limit = (
            1
            if any(
                evaluator.check_answers_equiv(ans, gold_answer)
                for ans in answer2candidates.keys()
            )
            else 0
        )
        print(f"==> Correct: {correct}")
        try:
            with open(
                os.path.join(
                    args.discriminate_results_dir, f"problem-{problem_id}.json"
                ),
                "r",
            ) as f:
                temp_recording = json.load(f)
        except:
            temp_recording = {}
        temp_recording.update(
            {
                "correct": correct,
                "correct_majvote": correct_majvote,
                "correct_limit": correct_limit,
            }
        )
        with open(
            os.path.join(args.discriminate_results_dir, f"problem-{problem_id}.json"),
            "w",
        ) as f:
            json.dump(temp_recording, f, indent=4)
        num_correct += int(correct)
        num_correct_majvote += int(correct_majvote)
        num_correct_limit += int(correct_limit)
        num_tested += 1

        info = f"Acc: {num_correct / num_tested:.4f}; Majority vote acc: {num_correct_majvote / num_tested:.4f}; Limit acc: {num_correct_limit / num_tested:.4f}"
        print(info)
        pbar.set_description(info, refresh=True)

        pbar.update(1)
    #! --------------------------------------------------------

    print(
        f"Accuracy: {num_correct / num_tested:.4f}; Majority vote accuracy: {num_correct_majvote / num_tested:.4f}; Limit accuracy: {num_correct_limit / num_tested:.4f}"
    )

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

    print(f"Recording: \n{recording}")

    with open(recording_file, "w") as f:
        json.dump(recording, f, indent=4)


if __name__ == "__main__":
    main()
