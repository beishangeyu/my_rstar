from vllm import LLM, SamplingParams
import os

input = """
I will provide a main question. Please break it down into several sub-questions and answer each sub-question one by one in the order, without skipping any.

Question: Write a Python function to count the number of vowels in a given string.
Break it down into sub-questions:
sub-question1: What defines a vowel in the context of this problem?
Answer to sub-question1: Vowels are typically defined as the characters a, e, i, o, u (case-insensitive). Need to decide whether to include uppercase letters (e.g., A, E) as valid vowels.
sub-question2: How to iterate through the input string and check each character?
Answer to sub-question2: Loop through each character in the string and determine if it matches any of the predefined vowels. Use a counter variable to track the total number of vowels found.
sub-question3: How to handle case sensitivity?
Answer to sub-question3: Convert the input string to lowercase (or uppercase) before checking vowels, or explicitly check both lowercase and uppercase versions of vowels.
sub-question4: What edge cases should be considered?
Answer to sub-question4: Empty strings, strings with no vowels, strings with mixed characters (letters, symbols, numbers), and strings containing uppercase vowels (e.g., "AEIOU").

Question: Write a function to sort a given matrix in ascending order according to the sum of its rows.
"""
# sub-question1: Understand the Problem and Input Format. What is the input to the function? (e.g., a string) What is the expected output? (e.g., an integer indicating the number of vowels)
# sub-question2: Identify Vowels. How do you determine if a character is a vowel? (e.g., using a set or list of vowels)
# sub-question3: Count Vowels. How can you count the number of vowels in the given string?
# sub-question4: Define the Function. How do you structure the function in Python? What parameters does the function take? What should the function return?

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
prompts = [f"{input}"]
sampling_params = SamplingParams(
    temperature=0.8, top_p=0.95, max_tokens=1024, stop=["Question: "]
)
llm = LLM(model="mistralai/Mistral-7B-v0.1", gpu_memory_utilization=0.95)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(
        f"Prompt: {prompt}\n=============================\nGenerated text:\n{generated_text.strip()}"
    )
