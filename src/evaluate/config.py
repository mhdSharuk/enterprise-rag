import os
from langfuse import Langfuse
from pathlib import Path
from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM

env_path = Path(os.getcwd()) / ".env"
load_dotenv(dotenv_path=env_path, override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EVALUATION_MODEL = os.getenv("EVALUATION_MODEL")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL")

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
 
TEST_SAMPLE = 10


class GroqLLM(DeepEvalBaseLLM):
    def __init__(self, model_name=EVALUATION_MODEL):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=GROQ_BASE_URL,
            api_key=GROQ_API_KEY
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)
