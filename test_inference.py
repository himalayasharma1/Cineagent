from llama_cpp import Llama

print("Loading model onto Metal GPU...")

llm = Llama(
    model_path="models/mistral-7b-instruct-q4.gguf",
    n_ctx=2048,        # Context window — how much text it can hold in mind at once
    n_gpu_layers=-1,   # -1 = put ALL layers on Metal GPU (the key M5 flag)
    verbose=False      # Suppress the internal loading logs
)

print("Model loaded. Asking a cinema question...")

response = llm(
    "[INST] In one sentence, what is film noir? [/INST]",
    max_tokens=80,
    temperature=0.3,   # Low = more factual, less creative
    stop=["[INST]"]    # Stop generating when it tries to start a new prompt
)

print("\n--- Mistral says ---")
print(response["choices"][0]["text"].strip())
print("\n--- Stats ---")
print(f"Tokens generated: {response['usage']['completion_tokens']}")
