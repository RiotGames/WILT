from .base_model import BaseModel
import llama_cpp

class LLamaCPPModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.client = llama_cpp.Llama(
            model_path="models/local/Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf",
            n_ctx=4096,
            n_gpu_layers=-1,
            chat_format="llama-3",
            verbose=False,
        )

    def perform_step(self, conversation_history):
        try:
            response = self.client.create_chat_completion(
                model=self.model_name,
                messages=conversation_history
            )
            return response['choices'][0]["message"]["content"]
        except Exception as e:
            print(f"Error in LlamaCPPModel: {e}")
            return ""

    def initialize_conversation(self, system_prompt):
        return [{"role": "system", "content": system_prompt}]
