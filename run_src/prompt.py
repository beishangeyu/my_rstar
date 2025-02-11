# 提出单步思考
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


# 重述用户的要求
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

# 分解问题 回答子问题
gene_subq_suba_prompt = """
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
"""

# 一个个回答子问题
# 暂时不需要
gen_suba_prompt = """
I will provide a main question and break it down into several sub-questions. Please answer each sub-question one by one in the order, without skipping any.

Question: Write a function to sort a given matrix in ascending order according to the sum of its rows.
Break it down into sub-questions:
sub-question1. Understand the Problem and Input Format. What is the input to the function? (e.g., a matrix represented as a list of lists) What is the input to the function? (e.g., a matrix represented as a list of lists) What is the expected output? (e.g., a sorted matrix based on the sum of its rows) Are there any constraints on the matrix size or values?  
sub-question2. Calculate the Sum of Each Row. How do you calculate the sum of elements in a single row of the matrix? How can you compute the sum for all rows in the matrix?  
sub-question3. Sort the Matrix Based on Row Sums. How do you associate each row with its sum? How can you sort the rows of the matrix in ascending order based on their sums?
sub-question4. Define the Function. How do you structure the function in Python? What parameters does the function take? What should the function return?  

Answer to sub-question1: Understand the Problem. The input is a matrix represented as a list of lists. The output is the same matrix, but sorted by the sum of each row in ascending order. There are no specific size constraints, but the matrix typically has rows of the same length.
Answer to sub-question2: Calculate Row Sums. To calculate the sum of each row, use Python's sum() function. To get the sum for all rows, iterate through the matrix and apply sum() to each row.
Answer to sub-question3: Sort the Matrix. Pair each row with its sum. Use Python's sorted() function to sort these pairs based on the row sum. Once sorted, extract the rows back into a new matrix.
Answer to sub-question4: Define the Function. The function should take the matrix as input, compute row sums, sort the rows by sum, and return the sorted matrix.
"""
# 对 prompt 进行测试
if __name__ == "__main__":
    pass
