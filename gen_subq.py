from vllm import LLM, SamplingParams
import os

input = """
Given a question, please decompose it into sub-questions.

Question 1: Write a function to sort a given matrix in ascending order according to the sum of its rows.
Break it down into sub-questions:
1. Understand the Problem and Input Format. What is the input to the function? (e.g., a matrix represented as a list of lists) What is the input to the function? (e.g., a matrix represented as a list of lists) What is the expected output? (e.g., a sorted matrix based on the sum of its rows) Are there any constraints on the matrix size or values?  
2. Calculate the Sum of Each Row. How do you calculate the sum of elements in a single row of the matrix? How can you compute the sum for all rows in the matrix?  
3. Sort the Matrix Based on Row Sums. How do you associate each row with its sum? How can you sort the rows of the matrix in ascending order based on their sums?
4. Define the Function. How do you structure the function in Python? What parameters does the function take? What should the function return?  


Question 2: Write a python function to check whether the first and last characters of a given string are equal or not.
Break it down sub-questions:
1.Understand the Problem and Input Format. What is the input to the function? (e.g., a string). What is the expected output? (e.g., a boolean value indicating whether the first and last characters are equal)
2.Access the First and Last Characters of the String. How can you retrieve the first character of the string?. How can you retrieve the last character of the string?
3.Compare the First and Last Characters. How do you compare two characters in Python? What should the function return if they are equal? What should it return if they are not equal?
4.Define the Function. How do you structure the function in Python? What parameters does the function take? What should the function return?

Question 3: Write a Python function to count the number of vowels in a given string.
"""

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
prompts = [f"{input}"]
sampling_params = SamplingParams(
    temperature=0.8, top_p=0.95, max_tokens=256, stop=["\n\n\n", "Question 3"]
)
llm = LLM(model="mistralai/Mistral-7B-v0.1", gpu_memory_utilization=0.95)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(
        f"Prompt: {prompt}\n=============================\nGenerated text: {generated_text}"
    )
