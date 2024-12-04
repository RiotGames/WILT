import os
from .base_model import BaseModel
from groq import Groq


class GroqModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        groq_api_key = os.environ["GROQ_API_KEY"]
        self.client = Groq(api_key=groq_api_key)

    def perform_step(self, conversation_history):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=conversation_history
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in GroqModel: {e}")
            return ""

    def initialize_conversation(self, system_prompt):
        return [{"role": "system", "content": system_prompt}]
