"""
Qwen3 sanity test.
Goal: confirm the model loads, Metal GPU accelerates it,
and it produces coherent output. No tool-calling yet.
"""

from llama_cpp import Llama
import time

MODEL_PATH = "./models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

print("Loading Qwen3-4B-Instruct-2507...")
print("(Watch for 'Metal' in the logs below — that confirms GPU acceleration)")
print("-" * 60)

start = time.time()

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,           # same context window you used for Gemma
    n_gpu_layers=-1,      # -1 = offload all layers to Metal GPU
    verbose=True,         # show the loading logs so we can see Metal kick in
)

load_time = time.time() - start
print("-" * 60)
print(f"Model loaded in {load_time:.1f} seconds.\n")

# Simple test prompt — Qwen3 uses the standard chat template
prompt = "What is the capital of France? Answer in one sentence."

print(f"Prompt: {prompt}\n")
print("Generating response...\n")

gen_start = time.time()

response = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=100,
    temperature=0.3,
)

gen_time = time.time() - gen_start

answer = response["choices"][0]["message"]["content"]
tokens_used = response["usage"]["completion_tokens"]
tokens_per_sec = tokens_used / gen_time if gen_time > 0 else 0

print("-" * 60)
print(f"Response: {answer}")
print("-" * 60)
print(f"Generation time: {gen_time:.2f}s")
print(f"Tokens generated: {tokens_used}")
print(f"Speed: {tokens_per_sec:.1f} tokens/sec")