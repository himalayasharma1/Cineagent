"""
Qwen3 tool-calling sanity test.
Goal: verify the model emits structured tool-call JSON when given a tool definition,
instead of free-form text describing what it would do.
"""

from llama_cpp import Llama
import json
import re

MODEL_PATH = "./models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

print("Loading Qwen3...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_gpu_layers=-1,
    verbose=False,  # quieter this time, we don't need the load logs
)

print("Model loaded.\n")

# ---------------------------------------------------------------
# Define a fake tool. This uses OpenAI's function-calling schema,
# which llama-cpp-python supports for compatible models like Qwen3.
# ---------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city, e.g. 'Tokyo'"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# ---------------------------------------------------------------
# A prompt that SHOULD trigger the tool.
# If Qwen3's tool-calling works, it will return a tool_calls array.
# If it doesn't work, it will return free-form text.
# ---------------------------------------------------------------
messages = [
    {"role": "user", "content": "What's the weather like in Tokyo right now?"}
]

print("Prompt: What's the weather like in Tokyo right now?")
print("Expected: a structured tool call to get_weather with city='Tokyo'\n")
print("-" * 60)

response = llm.create_chat_completion(
    messages=messages,
    tools=tools,
    tool_choice="auto",   # let the model decide whether to call a tool
    temperature=0.3,
    max_tokens=300,
)

# ---------------------------------------------------------------
# Inspect the response. We want to see tool_calls in the message.
# ---------------------------------------------------------------
message = response["choices"][0]["message"]

print("Full message returned by Qwen3:")
print(json.dumps(message, indent=2))
print("-" * 60)


# ---------------------------------------------------------------
# Qwen3 emits tool calls as <tool_call>...</tool_call> tags inside
# the content field. We parse them ourselves with a simple regex.
# ---------------------------------------------------------------
content = message.get("content", "")

tool_call_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
matches = tool_call_pattern.findall(content)

if matches:
    print(f"\n✅ SUCCESS: Found {len(matches)} tool call(s) in Qwen3's response.\n")
    for i, raw_json in enumerate(matches, 1):
        print(f"  Tool call #{i} (raw): {raw_json}")
        try:
            parsed = json.loads(raw_json)
            print(f"  Tool name: {parsed['name']}")
            print(f"  Arguments: {parsed['arguments']}")
        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON parse error: {e}")
        except KeyError as e:
            print(f"  ⚠️  Missing expected field: {e}")
else:
    print("\n❌ No tool calls found in response.")
    print(f"Content: {content[:300]}")