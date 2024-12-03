# Licensed under the MIT license.
import os, json, re
from typing import List, Dict, Tuple
from collections import defaultdict
import random
from fuzzywuzzy import fuzz, process
from threading import Thread


class Evaluator:
    def __init__(self) -> None:
        self.answer_marker = "answer is"

    def _is_number(self, s) -> Tuple[bool, str]:
        try:
            res = float(s)
            return True, str(res)
        except:
            pass
        try:
            import unicodedata

            res = unicodedata.numeric(s)
            return True, str(res)
        except:
            pass
        return False, None

    def validate_completion(self, completion: str) -> bool:
        if self.answer_marker.lower() in completion.lower():
            return True

        return False

    def isolate_answer(self, text: str):
        if text is None:
            return None

        assert isinstance(text, str)
        text = text.lower()
        split_ans = text.split(self.answer_marker.lower())
        if len(split_ans) > 1:
            ans = split_ans[-1].replace(":", "").strip()
            extract_ans_temp = ans.split(".\n")[0].strip()
            if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == ".":
                extract_ans = extract_ans_temp[0:-1]
            else:
                extract_ans = extract_ans_temp
            extract_ans = extract_ans.strip().strip("\n")
            return extract_ans
        else:
            return text

    # 找到出现次数最多的answer, 提供它对应的第一个completion和他在所有completion中的index, 以及confidence
    def find_most_confident_answer(self, completions: List[str]):
        """Returns the most confident answer, its completion, its id in the input list, and its confidence."""
        if completions is None or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        answer2ids = defaultdict(list)
        # 把相同answer的completion放在一起
        for id, c in enumerate(completions):
            model_answer = self.extract_answer_from_model_completion(c)
            has_existed = False
            for existing_answer in answer2completions.keys():
                if self.check_answers_equiv(model_answer, existing_answer):
                    if model_answer == existing_answer:
                        has_existed = True
                    answer2completions[existing_answer].append(c)
                    answer2ids[existing_answer].append(id)
            if not has_existed:
                answer2completions[model_answer].append(c)
                answer2ids[model_answer].append(id)

        assert len(answer2completions.keys()) > 0, "There are no valid completions."
        sum_num = 0
        for x in answer2completions.keys():
            sum_num += len(answer2completions[x])
        # TODO 打印一下所有answer的占比, 记得删掉
        print("*" * 10 + "Print answer count" + "*" * 10)
        for answer in answer2completions.keys():
            print(f"count: {len(answer2completions[answer])} / {sum_num}")
        print("*" * 30)
        # -------------------------------------------
        most_confident_answer = max(
            answer2completions.keys(), key=lambda x: len(answer2completions[x])
        )
        assert (
            len(answer2completions[most_confident_answer]) > 0
        ), "There are no completions for the most confident answer."
        # confidence 是这个 answer 占总 completions 的比例
        confidence = len(answer2completions[most_confident_answer]) / sum_num
        assert confidence > 0
        return (
            most_confident_answer,
            # 选择该出现次数最多的answer的第一个completion
            answer2completions[most_confident_answer][0],
            answer2ids[most_confident_answer][0],
            confidence,  # 该answer出现的次数 / 总的completion数
        )

    def stochastic_select_answer(
        self, completion2score, answer2completions, completions
    ):
        answer2score = {}
        answer_counts = {}
        for completion, score in completion2score.items():
            answer = self.extract_answer_from_model_completion(completion)
            if answer in answer2score:
                answer2score[answer] += score
                answer_counts[answer] += 1
            else:
                answer2score[answer] = score
                answer_counts[answer] = 1

        for answer in answer2score:
            answer2score[answer] /= answer_counts[answer]

        top_answers = sorted(answer2score.items(), key=lambda x: x[1], reverse=True)[:1]
        answers, scores = zip(*top_answers)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            selected_answer = random.choices(answers, weights=probabilities, k=1)[0]
        except:
            selected_answer = random.choices(answers, k=1)[0]

        most_confident_completion = answer2completions[selected_answer][0]
        completion_index = completions.index(most_confident_completion)
        confidence = answer2score[selected_answer]

        return selected_answer, most_confident_completion, completion_index, confidence

    def stochastic_calculate_completion_scores(self, answer2completions):
        completion2count = {}
        for answer, comps in answer2completions.items():
            count = len(comps)
            for comp in comps:
                completion2count[comp] = count

        completion2score = {}
        for idx, comp in enumerate(completion2count.keys()):
            weight = 1
            score = weight * completion2count[comp]
            completion2score[comp] = score
        return completion2score

    def stochastic_select_response(self, completion2score, completions):
        sorted_completions = sorted(
            completion2score.items(), key=lambda x: x[1], reverse=True
        )[:1]
        completions, scores = zip(*sorted_completions)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            sampled_completion = random.choices(
                completions, weights=probabilities, k=1
            )[0]
        except:
            sampled_completion = random.choices(completions, k=1)[0]
        confidence = completion2score[sampled_completion]
        most_confident_answer = self.extract_answer_from_model_completion(
            sampled_completion
        )
        id_of_most_confident = completions.index(sampled_completion)
        return (
            most_confident_answer,
            sampled_completion,
            id_of_most_confident,
            confidence,
        )

    def stochastic_find_most_confident_answer(
        self,
        completions: List[str],
    ):

        if not completions or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        for idx, comp in enumerate(completions):
            try:
                answer = self.extract_answer_from_model_completion(comp)
                answer2completions[answer].append(comp)
            except:
                continue

        if not answer2completions:
            return None, None, None, None

        completion2score = self.stochastic_calculate_completion_scores(
            answer2completions
        )

        most_confident_answer, sampled_completion, id_of_most_confident, confidence = (
            self.stochastic_select_response(completion2score, completions)
        )
        return (
            most_confident_answer,
            sampled_completion,
            id_of_most_confident,
            confidence,
        )

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        raise NotImplementedError

    def extract_answer_from_gold_solution(self, solution: str) -> str:
        raise NotImplementedError

    def extract_answer_from_model_completion(self, completion: str) -> str:
        raise NotImplementedError

    def check_correctness(self, code: str, dataset_name: str, test_list: List[str]):
        raise NotImplementedError


class PythonEvaluator(Evaluator):
    def __init__(self, device: str = "cuda:0"):
        super().__init__()
        from transformers import pipeline

        # NOTE 加载 code clone 工具
        self.pipe = pipeline(
            model="Lazyhope/python-clone-detection",
            trust_remote_code=True,
            device=device,
        )

    # 比较两个函数是否相等
    def check_answers_equiv(self, answer_a: str, answer_b: str):
        # NOTE 使用 code clone 工具判断代码是否相同
        is_clone = self.pipe((answer_a, answer_b))
        # TODO 测试一下阈值
        if is_clone[True] > 0.9:
            return True
        else:
            return False

    def extract_answer_from_model_completion(self, completion: str) -> str:
        return remove_comments(completion)

    def test_mbpp(self, test_list, code, timeout=5):
        test_list_code = "\n".join(test_list)
        try:
            template = f"{code}\n{test_list_code}\n"
            function_with_timeout(exec, (template, globals()), timeout)
            return 1
        except:
            return 0

    def check_correctness(self, code: str, dataset_name: str, test_list: List[str]):
        if "mbpp" in dataset_name:
            return self.test_mbpp(test_list, code, timeout=5)
        else:
            pass


# 扩展线程类, 为了捕获子线程的异常以及设置超时, 防止代码死循环
class PropagatingThread(Thread):
    def run(self):
        self.exc = None  # 存储异常
        try:
            self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        super(PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        return self.ret


def function_with_timeout(func, args, timeout):
    thread = PropagatingThread(target=func, args=args)
    thread.start()
    thread.join(timeout)  # 只等待 timeout 时间

    # is_alive 返回 true 表示线程还在运行, 即超时
    if thread.is_alive():
        raise TimeoutError()


# 去除注释和空行
def remove_comments(code: str) -> str:
    pattern = r"(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\'|#.*?$)"
    # 去除注释
    code = re.sub(pattern, "", code, flags=re.MULTILINE | re.DOTALL)
    # 去除代码内空行和前后空行
    return re.sub(r"\n\s*\n", "\n", code).strip()
