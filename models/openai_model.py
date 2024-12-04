from .base_model import BaseModel
import openai
import os

class OpenAIModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.client = openai.OpenAI()

    def perform_step(self, conversation_history):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=conversation_history
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in OpenAIModel: {e}")
            return ""

    def initialize_conversation(self, system_prompt):
        if self.model_name in ["o1-preview-2024-09-12",
                               "o1-mini-2024-09-12"]:
            return [{"role": "user", "content": system_prompt}]
        else:
            return [{"role": "system", "content": system_prompt}]
