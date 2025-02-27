direct_answer_prompt = """As a Python expert, generate only the function implementation code based strictly on the provided signature, docstring and hints, without any explanations, tests or additional text.    

### Function signature and docstring
def find_Odd_Pair(A,N):
    '''
    Write a python function to count the pairs with xor as an odd number.
    '''

### Hints
Understand the Input. The function takes a list A and an integer N, where N is the number of elements in the list A.
Understand XOR Properties. The XOR of two numbers is odd if and only if one number is odd and the other is even. Therefore, we need to count how many odd and even numbers are present in the list.
Count Odd and Even Numbers. Initialize two counters: one for odd numbers and one for even numbers. Iterate through the list to populate these counters.
Calculate Pairs. The number of valid pairs (one odd, one even) can be calculated by multiplying the count of odd numbers by the count of even numbers.
Return the Count. Finally, return the total count of such pairs.
Implement the function

### Function implementation
def find_Odd_Pair(A, N):
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if (A[i] ^ A[j]) % 2 != 0:
                count += 1
    return count
"""


# 提出单步思考
ost_prompt = """You are a Python assistant. You are given function signature and docstring. Provide the full implementation of the following function.
    
### Function signature and docstring
def find_Odd_Pair(A,N):
    '''
    Write a python function to count the pairs with xor as an odd number.
    '''

### Step to implement
To implement the find_Odd_Pair function, we can follow these steps:
Step1: Understand the Input. The function takes a list A and an integer N, where N is the number of elements in the list A.
Step2: Understand XOR Properties. The XOR of two numbers is odd if and only if one number is odd and the other is even. Therefore, we need to count how many odd and even numbers are present in the list.
Step3: Count Odd and Even Numbers. Initialize two counters: one for odd numbers and one for even numbers. Iterate through the list to populate these counters.
Step4: Calculate Pairs. The number of valid pairs (one odd, one even) can be calculated by multiplying the count of odd numbers by the count of even numbers.
Step5: Return the Count. Finally, return the total count of such pairs.
Step6: Implement the function
"""


# 重述用户的要求
rephrase_prompt = """You are an AI assistant to help me rephrase the requirement.

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
gene_subq_suba_prompt = """I will provide a main question. Please break it down into several sub-questions and answer each sub-question one by one in the order, without skipping any.

Question: 
Write a Python function to count the number of vowels in a given string.
Break it down into sub-questions:
Sub-question1: What defines a vowel in the context of this problem?
Answer to sub-question1: Vowels are typically defined as the characters a, e, i, o, u (case-insensitive). Need to decide whether to include uppercase letters (e.g., A, E) as valid vowels.
Sub-question2: How to iterate through the input string and check each character?
Answer to sub-question2: Loop through each character in the string and determine if it matches any of the predefined vowels. Use a counter variable to track the total number of vowels found.
Sub-question3: How to handle case sensitivity?
Answer to sub-question3: Convert the input string to lowercase (or uppercase) before checking vowels, or explicitly check both lowercase and uppercase versions of vowels.
Sub-question4: What edge cases should be considered?
Answer to sub-question4: Empty strings, strings with no vowels, strings with mixed characters (letters, symbols, numbers), and strings containing uppercase vowels (e.g., "AEIOU").
"""

# TODO 让模型生成测试样例, 这里的
gene_testcase_prompt = """Generate 10 test cases for a Python function. Each test case must start with 'assert' and be provided consecutively, without any additional separators or explanations.

### Python function:
def is_prime_v2(n):
    if n <= 1:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    return all(n % i != 0 for i in range(3, int(n**0.5) + 1, 2))
### Test cases:
"""