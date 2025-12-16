import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# 1. Load variables from the .env file into the environment
load_dotenv()

# 2. Retrieve the API key securely
hf_token = os.getenv("HF_API_TOKEN")

if not hf_token:
    raise ValueError("API key not found. Please check your .env file.")

# 3. Initialize the Client
# You can specify a model here, or leave it empty to use a recommended default.
# We are using 'HuggingFaceH4/zephyr-7b-beta' as a reliable free-tier example.
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=hf_token
)

def chat_with_llm(prompt):
    """
    Sends a chat prompt to the Hugging Face Inference API.
    """
    try:
        # 4. Send the request
        # 'stream=True' allows us to print tokens as they arrive, like ChatGPT
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            stream=True
        )

        print(f"User: {prompt}")
        print("Bot: ", end="", flush=True)

        # 5. Process the streamed response
        for chunk in response:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
        print("\n")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    user_input = "Write a haiku about Python coding."
    chat_with_llm(user_input)