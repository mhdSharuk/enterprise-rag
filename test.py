import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

token = os.getenv("GITHUB_PAT")
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-5"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
    model=model
)

print(response.choices[0].message.content)

