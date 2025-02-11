from vllm import LLM, SamplingParams
import os

input = """
I will provide a main question. Please break it down into several sub-questions and answer each sub-question one by one in the order, without skipping any.

Question: Write a function to sort a given matrix in ascending order according to the sum of its rows.
Break it down into sub-questions:
sub-question1: Understand the Problem and Input Format. What is the input to the function? (e.g., a matrix represented as a list of lists) What is the input to the function? (e.g., a matrix represented as a list of lists) What is the expected output? (e.g., a sorted matrix based on the sum of its rows) Are there any constraints on the matrix size or values?  
Answer to sub-question1: The input is a matrix represented as a list of lists. The output is the same matrix, but sorted by the sum of each row in ascending order. There are no specific size constraints, but the matrix typically has rows of the same length.
sub-question2: Calculate the Sum of Each Row. How do you calculate the sum of elements in a single row of the matrix? How can you compute the sum for all rows in the matrix?  
Answer to sub-question2: To calculate the sum of each row, use Python's sum() function. To get the sum for all rows, iterate through the matrix and apply sum() to each row.
sub-question3: Sort the Matrix Based on Row Sums. How do you associate each row with its sum? How can you sort the rows of the matrix in ascending order based on their sums?
Answer to sub-question3: Pair each row with its sum. Use Python's sorted() function to sort these pairs based on the row sum. Once sorted, extract the rows back into a new matrix.
sub-question4: Define the Function. How do you structure the function in Python? What parameters does the function take? What should the function return?  
Answer to sub-question4: The function should take the matrix as input, compute row sums, sort the rows by sum, and return the sorted matrix.

Question: Write a Python function to count the number of vowels in a given string.
Break it down into sub-questions:
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
