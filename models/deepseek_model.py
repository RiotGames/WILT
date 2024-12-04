from .base_model import BaseModel
import openai
import os

class DeepseekModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        deepseek_api_key = os.environ['DEEPSEEK_API_KEY']
        self.client = openai.OpenAI(api_key=deepseek_api_key,
                                    base_url="https://api.deepseek.com")

    def perform_step(self, conversation_history):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=conversation_history
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in DeepseekModel: {e}")
            return ""

    def initialize_conversation(self, system_prompt):
        return [{"role": "system", "content": system_prompt}]
