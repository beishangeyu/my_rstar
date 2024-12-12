# TODO 遇到看不懂的先不要管, 先测试目前动作能否有效, 无效再考虑弄懂那些看不懂的动作

# NOTE 提出单步思考
# TODO 或许要告诉模型在可以生成函数的时候要结尾 Implement
ost_prompt = """
You are a Python assistant. You are given function head and its docstring. Provide the full implementation of the following function.

[Function haed and docstring]
def max_aggregate(stdata):
    '''
    Write a function to calculate the maximum aggregate from the list of tuples.
    for example:
    max_aggregate([('Juan Whelan',90),('Sabah Colley',88),('Peter Nichols',7),('Juan Whelan',122),('Sabah Colley',84)])==('Juan Whelan', 212)
    '''
    
[Step to implement]
To implement the max_aggregate function, we need to follow these steps:
Step1: Understand the Input. The function takes a list of tuples (stdata). Each tuple represents a set of data points (e.g., scores, measurements).
Step2: Define the Aggregate Calculation. Determine how to calculate the "aggregate" from each tuple. This could mean summing the values in the tuple, finding the maximum value, or some other form of aggregation.
Step3: Initialize a Variable to Track the Maximum. Before iterating through the list, initialize a variable to store the maximum aggregate value found.
Step4: Iterate Through the List of Tuples. For each tuple in the list, calculate the aggregate using the defined method.
Step5: Update the Maximum Aggregate. Compare the current aggregate with the stored maximum, and update it if the current aggregate is greater.
Step6: Return the Maximum Aggregate. After iterating through all tuples, return the maximum aggregate value.
Step7: Implement the function

[Function implementation]
def max_aggregate(stdata):
    '''
    Write a function to calculate the maximum aggregate from the list of tuples.
    for example:
    max_aggregate([('Juan Whelan',90),('Sabah Colley',88),('Peter Nichols',7),('Juan Whelan',122),('Sabah Colley',84)])==('Juan Whelan', 212)
    '''
    max_agg = float('-inf')  # Start with the smallest possible value
    
    for data in stdata:
        # Calculate the aggregate for the current tuple (e.g., sum)
        current_agg = sum(data)  # Change this if another aggregation is needed
        
        # Update max_agg if the current aggregate is greater
        if current_agg > max_agg:
            max_agg = current_agg
            
    return max_agg
    
[Function head and docstring]
def find_Odd_Pair(A,N):
    '''
    Write a python function to count the pairs with xor as an odd number.
    '''

[Step to implement]
To implement the find_Odd_Pair function, we can follow these steps:
Step1: Understand the Input. The function takes a list A and an integer N, where N is the number of elements in the list A.
Step2: Understand XOR Properties. The XOR of two numbers is odd if and only if one number is odd and the other is even. Therefore, we need to count how many odd and even numbers are present in the list.
Step3: Count Odd and Even Numbers. Initialize two counters: one for odd numbers and one for even numbers. Iterate through the list to populate these counters.
Step4: Calculate Pairs. The number of valid pairs (one odd, one even) can be calculated by multiplying the count of odd numbers by the count of even numbers.
Step5: Return the Count. Finally, return the total count of such pairs.
Step6: Implement the function

[Function implementation]
def find_Odd_Pair(A, N):
    '''
    Write a python function to count the pairs with xor as an odd number.
    '''
    odd_count = 0
    even_count = 0
    
    # Count odd and even numbers
    for num in A:
        if num % 2 == 0:
            even_count += 1
        else:
            odd_count += 1
            
    # The number of odd pairs is the product of odd_count and even_count
    return odd_count * even_count
"""


# NOTE 重述用户的要求
rephrase_prompt = """
You are an AI assistant to help me rephrase the requirement.

Original requirement:
Write a python function to check whether the first and last characters of a given string are equal or not.
Rephrased requirement:
Write a Python function to check if the first and last characters of a given string are equal.

Original requirement:
Writing a python function to unearth the first recurrent nature in a given chain
Rephrased requirement:
Write a Python function to find the first recurrent element in a given sequence.

Original requirement:
Write a function to count the same pair in two given lists usage map function.
Rephrased requirement:
Write a Python function using map to count the number of matching pairs in two given lists.
"""

# 对 prompt 进行测试
if __name__ == "__main__":
    import sys

    sys.path.append(".")  # 这个路径好像是以命令行的路径为准而不是这个文件所在的路径
    from models.vLLM_API import *

    # 设置使用的显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # model_ckpt = "microsoft/Phi-3-mini-4k-instruct"
    model_ckpt = "mistralai/Mistral-7B-v0.1"

    tokenizer, model = load_vLLM_model(
        model_ckpt,
        seed=42,
        tensor_parallel_size=1,
        half_precision=False,
        max_model_len=30000,
        gpu_memory_utilization=0.95,
    )
    #     input = (
    #         ost_prompt
    #         + """
    # [Function head and docstring]
    # def difference(n):
    # '''
    # Write a python function to find the difference between sum of cubes of first n natural numbers and the sum of first n natural numbers.
    # for example:
    # difference(3) == 30
    # '''
    # [Step to implement]
    # """
    #     )
    # input = rephrase_prompt + """
    # Original requirement: Writing a python function to left rotating the bits of a afforded number
    # Rephrased requirement:
    # """
    input = (
        ost_prompt
        + """
[Function haed and docstring]

def find_char_long(text):
    '''
    Write a Python function using regex to find all words in a string that have a length of at least 4 characters.
    for example:
    find_char_long('Please move back to stream') == ['Please', 'move', 'back', 'stream']
    '''


[Step to implement]
To implement the find_char_long function, we need to follow these steps:
Step1: Understand the Input. The function takes a string (text).
"""
    )
    # last_output = ""
    # for i in range(10):
    #     input += last_output
    #     output = generate_with_vLLM_model(model, input, stop=ost_stop_token)
    #     last_output = output[0].outputs[0].text
    #     print(output[0].outputs[0].text)
    resp = generate_with_vLLM_model(
        model,
        input,
        n=10,
        stop=[
            "\n",
            "\n\n",
            f"Step3",
            "[Function head and docstring]",
            "[Function implementation]",
            "[Step to implement]",
            "You are a Python assistant.",
        ],
        max_tokens=1024,
    )
    output_list = [o.text.strip()[7:] for o in resp[0].outputs]
    for it in output_list:
        print(it)
