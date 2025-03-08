# BUG 这里要仔细看看

direct_answer_no_hints_prompt = """You are a Python assistant. Solve the given Python programming problem.  
ONLY provide the simplest possible solution using one or a few lines of Python code.  
Do not define functions. No explanations or comments. 

### Python programming problem:
Concatenate elements of a list 'x' of multiple integers to a single integer

### Solution:
sum(d * 10 ** i for i, d in enumerate(x[::-1]))

### Python programming problem:
get the average of a list values for each key in dictionary `d`

### Solution:
[(i, sum(j) / len(j)) for i, j in list(d.items())]
"""

cpd_final_answer_prompt = """You are a Python assistant. 
I will give you some hints, and you need to use them to generate a concise code snippet to solve a Python programming problem. 
Output only one line of Python code. No multi-line code or explanations allowed.

### Python programming problem:
Concatenate elements of a list 'x' of multiple integers to form a single integer.

### Thinking Steps:
1. Input Analysis:
   The input is a list 'x' of non-negative integers, e.g., [1, 2, 3]. Special cases include an empty list ([]) or a single-element list ([5]).
2. Output Definition:
   The output is a single integer formed by concatenating all elements in order, e.g., [1, 2, 3] → 123.
3. Approach:
   Convert each integer to a string using map(), join them into one string, and convert back to an integer.
4. Boundary Conditions:
   Return 0 for an empty list; return the element itself for a single-element list.
5. One-Line Requirement:
   The solution must be written as a single line of code.

### Snippet:
int(''.join(map(str, x)))
"""

direct_answer_prompt = """You are a Python assistant. 
I will give you some hints, and you need to use them to generate a concise code snippet to solve a Python programming problem. 
Output only one line of Python code. No multi-line code or explanations allowed.

### Python programming problem:
Count the pairs with xor as an odd number in list A with the length of N.

### Hints:
1. Understand the Input. 
    The input consists of a list `A` and an integer `N`, where `N` is the length of `A`. 
    The task is to count the number of pairs `(i, j)` where `A[i] ^ A[j]` is an odd number.
2. Understand XOR Properties. 
    The XOR of two numbers is odd if and only if one number is odd and the other is even. 
    This is because an odd number has a binary last bit of 1, while an even number has a last bit of 0, making their XOR result’s last bit 1 (odd). 
    Thus, the problem reduces to counting the odd and even numbers in `A` and multiplying these counts.
3. Calculate Pairs. 
    Count the number of odd and even numbers in the list `A`. 
    The total number of valid pairs is the product of these two counts.
4. Return the Result in one line of code. 
    Use the `sum()` function to count odd numbers in `A`, and derive the even count using `N`, then compute the product in a single line of code.
    
### Snippet:
sum(1 for i in range(N) for j in range(i + 1, N) if (A[i] ^ A[j]) % 2 != 0)
"""

ost_prompt = """You are a Python assistant. 
I will give you some hints, and you need to use them to generate a concise code snippet to solve a Python programming problem. 
Only ONE line of code can be generated.

### Python programming problem:
Count the pairs with xor as an odd number in list A with the length of N.

### Hints:
1. Understand the Input. 
    The input consists of a list `A` and an integer `N`, where `N` is the length of `A`. 
    The task is to count the number of pairs `(i, j)` where `A[i] ^ A[j]` is an odd number.
2. Understand XOR Properties. 
    The XOR of two numbers is odd if and only if one number is odd and the other is even. 
    This is because an odd number has a binary last bit of 1, while an even number has a last bit of 0, making their XOR result’s last bit 1 (odd). 
    Thus, the problem reduces to counting the odd and even numbers in `A` and multiplying these counts.
3. Calculate Pairs.
    Count the number of odd and even numbers in the list `A`. 
    The total number of valid pairs is the product of these two counts.
4. Return the Result in One Line.
    Use the `sum()` function to count odd numbers in `A`, and derive the even count using `N`, then compute the product in a single line of code.
"""


rephrase_prompt = """You are a Python expert. I will provide a Python programming problem, and your task is to rewrite it in a clearer and more structured manner to enhance readability and comprehension.

Requirements:
1. Clearly define the input format, expected output, and goal of the problem.
2. Consider boundary conditions and relevant edge cases.
3. Ensure the restated problem is well-structured, detailed, and follows a standardized format.
4. Output the refined problem description in a professional and well-formatted manner.

### Original Programming Problem:
convert a list of integers ‘x’ into a single integer.

### Refined Problem Description:
You are given a list of integers called 'x'. 
The goal is to take all the individual integers in this list and combine them in such a way that they form a single integer. 
This involves concatenating the numbers together while preserving their original order in the list.
The output should be an integer. 
The boundary conditions include an empty list should return 0 and list with a single element should return that element as an integer.
"""

cpd_prompt = """
You are a Python assistant. 
Approach and decompose programming problems from a specific perspective.

### Python programming problem:
Convert a list of integers into a single integer

### Thinking Steps:
1. Input Analysis: What is the input of the function? What are the possible input types or ranges? Are there any special cases (e.g., empty input, invalid input) to consider?
    The input is a list of integers, e.g., [1, 2, 3] or [5, 0, 7].
    The input is a list containing integers (int).
    The list can be empty ([]), contain a single integer ([5]), or multiple integers.
    Integers can be positive, negative, or zero, e.g., [-1, 2, -3].
    Assume the input is a valid list of integers, with no non-integer elements (e.g., strings or floats). The problem doesn’t specify handling invalid inputs, so we assume valid input.
2. Output Definition: What does the function need to return? What is the type of the return value? Are there any specific formats or conditions?
    The function returns a single integer, e.g., [1, 2, 3] returns 123, [5, 0, 7] returns 507.
    The return value is an integer (int).
    The output is the integers from the list concatenated in order, e.g., [1, 2, 3] becomes 123, not a sum like 6.
3. Function Decomposition: What smaller steps can the target functionality be broken into?
    1. Convert each integer in the list to a string (since concatenation requires strings).
    2. Concatenate all strings into a single string.
    3. Convert the concatenated string back to an integer.
4. Boundary Conditions: What boundary cases need to be handled? How do we ensure the function works correctly in these cases?
    Empty list ([]): ''.join() returns '', but int('') raises a ValueError.
    Single-element list ([5]): Should return the integer itself, e.g., 5.
    Negative numbers ([-1, 2]): Should handle the sign correctly, e.g., -12.
    Zeros ([0, 0, 1] or [1, 0]): Should preserve zeros, e.g., 001 (i.e., 1) or 10.
"""
