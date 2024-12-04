from .base_model import BaseModel
import google.generativeai as genai
from google.generativeai.types import RequestOptions
from google.api_core import retry
import os

class GeminiModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model_name)

    def perform_step(self, conversation_history):
        try:
            gemini_messages = [
                {"role": "model" if msg["role"] == "assistant" else msg["role"],
                 "parts": [msg["content"]]}
                for msg in conversation_history[:-1]
            ]

            chat = self.model.start_chat(history=gemini_messages)
            
            response = chat.send_message(conversation_history[-1]["content"],
                                         request_options=RequestOptions(
                                             retry=retry.Retry(initial=10, multiplier=2,
                                                               maximum=60, timeout=300)
                                         ))

            return response.text
        
        except Exception as e:
            print(f"Error in GeminiModel: {e}")
            return ""

    def initialize_conversation(self, system_prompt):
        return [{"role": "user", "content": system_prompt}]
