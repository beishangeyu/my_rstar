# NOTE 提出单步思考
# TODO 对于 Mistral-7B-v0.1, ost 的 stop token 应该是 [function haed and docstring]
ost_prompt = """
You are a Python assistant. You are given function head and its docstring. Provide the full implementation of the following function.

[function haed and docstring]
def max_aggregate(stdata):
    '''
    Write a function to calculate the maximum aggregate from the list of tuples.
    for example:
    max_aggregate([('Juan Whelan',90),('Sabah Colley',88),('Peter Nichols',7),('Juan Whelan',122),('Sabah Colley',84)])==('Juan Whelan', 212)
    '''
    
[step to implement]:
To implement the max_aggregate function, we need to follow these steps:
Step1: Understand the Input. The function takes a list of tuples (stdata). Each tuple represents a set of data points (e.g., scores, measurements).
Step2: Define the Aggregate Calculation. Determine how to calculate the "aggregate" from each tuple. This could mean summing the values in the tuple, finding the maximum value, or some other form of aggregation.
Step3: Initialize a Variable to Track the Maximum. Before iterating through the list, initialize a variable to store the maximum aggregate value found.
Step4: Iterate Through the List of Tuples. For each tuple in the list, calculate the aggregate using the defined method.
Step5: Update the Maximum Aggregate. Compare the current aggregate with the stored maximum, and update it if the current aggregate is greater.
Step6: Return the Maximum Aggregate. After iterating through all tuples, return the maximum aggregate value.

[function implementation]
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
    
[function head and docstring]
def find_Odd_Pair(A,N):
    '''
    Write a python function to count the pairs with xor as an odd number.
    '''

[step to implement]
To implement the find_Odd_Pair function, we can follow these steps:

Step1: Understand the Input. The function takes a list A and an integer N, where N is the number of elements in the list A.
Step2: Understand XOR Properties. The XOR of two numbers is odd if and only if one number is odd and the other is even. Therefore, we need to count how many odd and even numbers are present in the list.
Step3: Count Odd and Even Numbers. Initialize two counters: one for odd numbers and one for even numbers. Iterate through the list to populate these counters.
Step4: Calculate Pairs. The number of valid pairs (one odd, one even) can be calculated by multiplying the count of odd numbers by the count of even numbers.
Step5: Return the Count. Finally, return the total count of such pairs.

[function implementation]
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
# XXX 初步测试 ost prompt 下模型的行为符合预期



# TODO 重述用户的要求
# TODO 提出下一个子问题并回答
# TODO 直接回答问题

# 对 prompt 进行测试
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from models.vLLM_API import *

    os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 可以这样子设置可见显卡
    model_ckpt = "mistralai/Mistral-7B-v0.1"
    tokenizer, model = load_vLLM_model(
        model_ckpt, seed=42, tensor_parallel_size=1, half_precision=False
    )
#     input = ost_prompt + """
# [function head and docstring]
# def difference(n):
# '''
# Write a python function to find the difference between sum of cubes of first n natural numbers and the sum of first n natural numbers.
# for example:
# difference(3) == 30
# '''
# [step to implement]
# """
    output = generate_with_vLLM_model(model, input)
    print(output[0].outputs[0].text)
    