# Licensed under the MIT license.
import os, json, re
from typing import List, Dict, Tuple
from collections import defaultdict
import random
from fuzzywuzzy import fuzz, process


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
    def find_most_confident_answer(
        self, completions: List[str], prior_weights: List[float] = None
    ):
        """Returns the most confident answer, its completion, its id in the input list, and its confidence."""
        if completions is None or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        answer2ids = defaultdict(list)
        # 把相同answer的completion放在一起
        for id, c in enumerate(completions):
            try:
                # TODO 对于代码而言 code 即 answer, 不用再取出来, 但是要把注释去掉
                model_answer = self.extract_answer_from_model_completion(c)
                has_existed = False
                for existing_answer in answer2completions.keys():
                    if self.check_answers_equiv(model_answer, existing_answer):
                        assert not has_existed
                        has_existed = True
                        answer2completions[existing_answer].append(c)
                        answer2ids[existing_answer].append(id)
                if not has_existed:
                    answer2completions[model_answer].append(c)
                    answer2ids[model_answer].append(id)
            except:
                pass

        assert len(answer2completions.keys()) > 0, "There are no valid completions."
        if prior_weights is not None:
            assert len(completions) == len(prior_weights)
            completion2count = {}
            for answer, answer_completions in answer2completions.items():
                count = len(answer_completions)
                for answer_completion in answer_completions:
                    completion2count[answer_completion] = count

            completion2score = {}
            for id, (completion, count) in enumerate(completion2count.items()):
                prior_weight = prior_weights[id]
                score = prior_weight * (count / len(completions))
                completion2score[completion] = score

            most_confident_completion = max(
                completion2score.keys(), key=lambda x: completion2score[x]
            )

            return (
                self.extract_answer_from_model_completion(most_confident_completion),
                most_confident_completion,
                completions.index(most_confident_completion),
                completion2score[most_confident_completion],
            )
        else:
            most_confident_answer = max(
                answer2completions.keys(), key=lambda x: len(answer2completions[x])
            )
            assert (
                len(answer2completions[most_confident_answer]) > 0
            ), "There are no completions for the most confident answer."
            confidence = len(answer2completions[most_confident_answer]) / len(
                completions
            )
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

    def stochastic_calculate_completion_scores(self, prior_weights, answer2completions):
        completion2count = {}
        for answer, comps in answer2completions.items():
            count = len(comps)
            for comp in comps:
                completion2count[comp] = count

        completion2score = {}
        for idx, comp in enumerate(completion2count.keys()):
            weight = prior_weights[idx] if prior_weights is not None else 1
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
        prior_weights: List[float] = None,
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
            prior_weights, answer2completions
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


class PythonEvaluator(Evaluator):
    # 比较两个函数是否相等
    def check_answers_equiv(self, answer_a: str, answer_b: str):
        return answer_a == answer_b

    # TODO 即 task 的 code 部分
    def extract_answer_from_gold_solution(self, solution: str) -> str:
        return self.isolate_answer(solution)

    def extract_answer_from_model_completion(self, completion: str) -> str:
        return remove_comments(completion)


# 去除注释和空行
def remove_comments(code: str) -> str:
    pattern = r"(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\'|#.*?$)"
    # 去除注释
    code = re.sub(pattern, "", code, flags=re.MULTILINE | re.DOTALL)
    # 去除代码内空行和前后空行
    return re.sub(r"\n\s*\n", "\n", code).strip()


if __name__ == "__main__":
    code = """
def count_differences(str1: str, str2: str) -> int:
    \"\"\"
    计算两个字符串中不同字符的个数。
    
    参数:
    str1 (str): 第一个字符串
    str2 (str): 第二个字符串
    
    返回:
    int: 两个字符串中不同字符的个数。如果字符串长度不同，只比较最短长度的字符。

    示例:
    >>> count_differences("abcde", "abfde")
    1
    >>> count_differences("hello", "hallo")
    2
    \"\"\"
    # 取两个字符串中较小的长度，以防止长度不一致时超出索引范围
    min_length = min(len(str1), len(str2))

    # 初始化计数器，用于记录不同字符的数量
    difference_count = 0

    # 遍历每个字符，逐一比较两字符串的字符是否相同
    for i in range(min_length):
        if str1[i] != str2[i]:  # 如果字符不同
            difference_count += 1  # 计数器加一

    # 返回两个字符串不同字符的个数
    return difference_count
"""
    print(remove_comments(code=code))
