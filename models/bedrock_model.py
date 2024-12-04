from .base_model import BaseModel
import os
import time
import json
import boto3
from botocore.exceptions import ClientError


class BedrockModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.model = model_name
        self.client = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")

    def perform_step(self, conversation_history, retry_count=0):
        try:
            bedrock_messages = [
                {
                    "role": "assistant" if msg["role"] == "assistant" else msg["role"],
                    "content": [{"text": msg["content"]}]
                }
                for msg in conversation_history
            ]

            response = self.client.converse(modelId=self.model,
                                            messages=bedrock_messages,
                                            inferenceConfig={
                                                "maxTokens": 512,
                                                "temperature": 0.5,
                                                "topP": 0.9
                                            },)

            return response["output"]["message"]["content"][0]["text"]
        
        except Exception as e:
            print(f"Error in BedrockModel: {e}")
            time.sleep(30)
            if retry_count >= 3:
                return ""
            else:
                print(f"attempting retry {retry_count+1}...")
                return self.perform_step(conversation_history, retry_count+1)

    def initialize_conversation(self, system_prompt):
        return [{"role": "user", "content": system_prompt}]
