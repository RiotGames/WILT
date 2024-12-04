from .base_model import BaseModel
from mistralai import Mistral
import os

class MistralModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.model_name = model_name
        mistral_api_key = os.environ['MISTRAL_API_KEY']
        self.client = Mistral(api_key=mistral_api_key)
        
    def perform_step(self, conversation_history):
        try:
            response = self.client.chat.complete(
                model=self.model_name,
                messages=conversation_history
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in MistralModel: {e}")
            return ""

    def initialize_conversation(self, system_prompt):
        return [{"role": "system", "content": system_prompt}]
