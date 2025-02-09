from vllm import LLM, SamplingParams
import os

input = """
Generate 10 test cases for a Python function. Each test case must start with 'assert' and be provided consecutively, without any additional separators or explanations.

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

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
prompts = [f"{input}"]
sampling_params = SamplingParams(
    temperature=0.8, top_p=0.95, max_tokens=256, stop=["def"]
)
llm = LLM(model="mistralai/Mistral-7B-v0.1", gpu_memory_utilization=0.95)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}\nGenerated text: {generated_text}")
