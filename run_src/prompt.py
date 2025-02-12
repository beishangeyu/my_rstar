# TODO 遇到看不懂的先不要管, 先测试目前动作能否有效, 无效再考虑弄懂那些看不懂的动作

# NOTE 提出单步思考
# TODO 或许要告诉模型在可以生成函数的时候要结尾 Implement
ost_prompt = """
You are a Python assistant. You are given function head and its docstring. Provide the full implementation of the following function.

### Function haed and docstring
def max_aggregate(stdata):
    '''
    Write a function to calculate the maximum aggregate from the list of tuples.
    for example:
    max_aggregate([('Juan Whelan',90),('Sabah Colley',88),('Peter Nichols',7),('Juan Whelan',122),('Sabah Colley',84)])==('Juan Whelan', 212)
    '''
    
### Step to implement
To implement the max_aggregate function, we need to follow these steps:
Step1: Understand the Input. The function takes a list of tuples (stdata). Each tuple represents a set of data points (e.g., scores, measurements).
Step2: Define the Aggregate Calculation. Determine how to calculate the "aggregate" from each tuple. This could mean summing the values in the tuple, finding the maximum value, or some other form of aggregation.
Step3: Initialize a Variable to Track the Maximum. Before iterating through the list, initialize a variable to store the maximum aggregate value found.
Step4: Iterate Through the List of Tuples. For each tuple in the list, calculate the aggregate using the defined method.
Step5: Update the Maximum Aggregate. Compare the current aggregate with the stored maximum, and update it if the current aggregate is greater.
Step6: Return the Maximum Aggregate. After iterating through all tuples, return the maximum aggregate value.
Step7: Implement the function
"""

disc_prompt = """
You are a Python assistant. You are given function head and its docstring. Provide the full implementation of the following function.

### Function haed and docstring
def max_aggregate(stdata):
    '''
    Write a function to calculate the maximum aggregate from the list of tuples.
    for example:
    max_aggregate([('Juan Whelan',90),('Sabah Colley',88),('Peter Nichols',7),('Juan Whelan',122),('Sabah Colley',84)])==('Juan Whelan', 212)
    '''
    
### Step to implement
Understand the Input. The function takes a list of tuples (stdata). Each tuple represents a set of data points (e.g., scores, measurements).
Define the Aggregate Calculation. Determine how to calculate the "aggregate" from each tuple. This could mean summing the values in the tuple, finding the maximum value, or some other form of aggregation.
Initialize a Variable to Track the Maximum. Before iterating through the list, initialize a variable to store the maximum aggregate value found.
Iterate Through the List of Tuples. For each tuple in the list, calculate the aggregate using the defined method.
Update the Maximum Aggregate. Compare the current aggregate with the stored maximum, and update it if the current aggregate is greater.
Return the Maximum Aggregate. After iterating through all tuples, return the maximum aggregate value.
Implement the function

### Function Implementation
def max_aggregate(stdata):
    aggregates = {}
    for name, score in stdata:
        if name in aggregates:
            aggregates[name] += score
        else:
            aggregates[name] = score

    max_name = max(aggregates, key=aggregates.get)
    max_score = aggregates[max_name]

    return (max_name, max_score)
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
    pass
