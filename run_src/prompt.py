# WARNING 许多prompt都指明时python专家, 如果要用别的语言要更改
direct_answer_no_hints_prompt = """You are a Python assistant. Solve the given Python programming problem. ONLY provide the function implementation code.

### Python programming problem:
def find_Odd_Pair(A,N):
    '''
    Write a python function to count the pairs with xor as an odd number.
    '''

### Function implementation:
def find_Odd_Pair(A, N):
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if (A[i] ^ A[j]) % 2 != 0:
                count += 1
    return count
"""

cpd_final_answer_prompt = """You are a Python assistant. I will give you some hints, and you need to use them to solve the Python programming problem. ONLY provide the function implementation code.

### Python programming problem:
def has_unique_chars(s):
    '''
    Write a Python function to determine whether a string contains only unique characters (i.e., no duplicates).
    for example:
    assert has_unique_chars("abc")==True
    '''
    
### Thinking Steps:
1. Input Analysis: What is the input of the function? What are the possible input types or ranges? Are there any special cases (e.g., empty input, invalid input) to consider?
    The input is a string, which can contain letters, numbers, spaces, or special characters.
    The input is of type str in Python. 
    Possible cases include: An empty string, A single-character string, A string with mixed characters (e.g., "abc123", "a b", "a#a").
2. Output Definition: What does the function need to return? What is the type of the return value? Are there any specific formats or conditions?
    The function returns a boolean value (True or False) indicating whether all characters in the string are unique (no duplicates).
    The return value is of type bool. True means no character appears more than once; False means at least one character is repeated.
3. Function Decomposition: What smaller steps can the target functionality be broken into?
    1. Take the input string.
    2. Check each character to see if it appears more than once.
    3. Return True if all characters are unique, False otherwise.
4. Boundary Conditions: What boundary cases need to be handled? How do we ensure the function works correctly in these cases?
    Empty string: Should return True (no characters, thus no duplicates).
    Single character: Should return True (one character is always unique).
    String with spaces or special characters: Should treat them as regular characters (e.g., "a b" has a space).
    Using a set to track unique characters handles all these cases naturally, as it considers spaces and special characters as distinct elements.   

### Function implementation:
def has_unique_chars(s):
    return len(s) == len(set(s))
"""

direct_answer_prompt = """You are a Python assistant. I will give you some hints, and you need to use them to solve a Python programming problem. ONLY provide the function implementation code.

### Python programming problem:
def find_Odd_Pair(A,N):
    '''
    Write a python function to count the pairs with xor as an odd number.
    '''

### Hints:
1. Understand the Input. 
    The function takes a list `A` and an integer `N`, where `N` is the number of elements in the list `A`.
    We need to count the number of pairs `(i, j)` such that `A[i] ^ A[j]` results in an odd number.  
2. Understand XOR Properties. 
    XOR of two numbers is odd if and only if one number is odd and the other is even.
    This means we need to determine how many odd and even numbers exist in `A`.
3. Count Odd and Even Numbers. 
    Initialize two counters: one for odd numbers and one for even numbers.  
    Iterate through the list `A` and count the occurrences of odd and even numbers.  
4. Calculate Pairs. 
    A valid pair consists of one odd and one even number.  
    The number of such pairs can be found by multiplying the count of odd numbers by the count of even numbers. 
5. Return the Result  
    Return the computed number of valid pairs.  

### Function implementation:
def find_Odd_Pair(A, N):
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if (A[i] ^ A[j]) % 2 != 0:
                count += 1
    return count
"""


# 提出单步思考
ost_prompt = """You are a Python assistant. You are given a python programming problem. Your task is to analyze the problem and generate a structured step-by-step thought process to solve it. 
    
### Python programming problem:
def find_Odd_Pair(A,N):
    '''
    Write a python function to count the pairs with xor as an odd number.
    '''

### Step to implement:
1. Understand the Input. 
    The function takes a list `A` and an integer `N`, where `N` is the number of elements in the list `A`.
    We need to count the number of pairs `(i, j)` such that `A[i] ^ A[j]` results in an odd number.  
2. Understand XOR Properties. 
    XOR of two numbers is odd if and only if one number is odd and the other is even.
    This means we need to determine how many odd and even numbers exist in `A`.
3. Count Odd and Even Numbers. 
    Initialize two counters: one for odd numbers and one for even numbers.  
    Iterate through the list `A` and count the occurrences of odd and even numbers.  
4. Calculate Pairs. 
    A valid pair consists of one odd and one even number.  
    The number of such pairs can be found by multiplying the count of odd numbers by the count of even numbers. 
5. Return the Result  
    Return the computed number of valid pairs.  
"""


# 重述用户的要求
rephrase_prompt = """You are a Python expert. I will provide a Python programming problem, and your task is to rewrite it in a clearer and more structured manner to enhance readability and comprehension.

Requirements:
1. Clearly define the input format, expected output, and goal of the problem.
2. Consider boundary conditions and relevant edge cases.
3. Ensure the restated problem is well-structured, detailed, and follows a standardized format.
4. Output the refined problem description in a professional and well-formatted manner.

### Original Programming Problem:
Write a function to find all words which are at least 4 characters long in a string by using regex.

### Refined Problem Description:
You are given a string 's' containing words separated by whitespace.
Find all words that are at least 4 characters long using regex.
Return a list of matching words in their original order.
Ignore punctuation and special characters when identifying words.
If no words meet the criteria, return an empty list.
"""


# 代码范式分解的prompt
cpd_prompt = """You are a Python assistant. Approach and decompose programming problems from a specific perspective.

### Python programming problem:
def has_unique_chars(s):
    '''
    Write a Python function to determine whether a string contains only unique characters (i.e., no duplicates).
    for example:
    assert has_unique_chars("abc")==True
    '''

### Thinking Steps:
1. Input Analysis: What is the input of the function? What are the possible input types or ranges? Are there any special cases (e.g., empty input, invalid input) to consider?
    The input is a string, which can contain letters, numbers, spaces, or special characters.
    The input is of type str in Python. 
    Possible cases include: An empty string, A single-character string, A string with mixed characters (e.g., "abc123", "a b", "a#a").
2. Output Definition: What does the function need to return? What is the type of the return value? Are there any specific formats or conditions?
    The function returns a boolean value (True or False) indicating whether all characters in the string are unique (no duplicates).
    The return value is of type bool. True means no character appears more than once; False means at least one character is repeated.
3. Function Decomposition: What smaller steps can the target functionality be broken into?
    1. Take the input string.
    2. Check each character to see if it appears more than once.
    3. Return True if all characters are unique, False otherwise.
4. Boundary Conditions: What boundary cases need to be handled? How do we ensure the function works correctly in these cases?
    Empty string: Should return True (no characters, thus no duplicates).
    Single character: Should return True (one character is always unique).
    String with spaces or special characters: Should treat them as regular characters (e.g., "a b" has a space).
    Using a set to track unique characters handles all these cases naturally, as it considers spaces and special characters as distinct elements.
"""
