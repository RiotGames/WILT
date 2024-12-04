from .openai_model import OpenAIModel
from .groq_model import GroqModel
from .anthropic_model import AnthropicModel
from .deepseek_model import DeepseekModel
from .hermes_model import LocalModel
from .phi_model import PhiModel
from .llama_model import LLamaCPPModel
from .gemini_model import GeminiModel
from .mistral_model import MistralModel
from .bedrock_model import BedrockModel
from .openrouter_model import OpenrouterModel

class ModelFactory:
    @staticmethod
    def create_model(model_name):
        if model_name in ["llama3-70b-8192",
                          "llama-3.1-70b-versatile",
                          "llama-3.1-8b-instant",
                          "llama3-8b-8192",
                          "gemma-7b-it",
                          "gemma2-9b-it",
                          "mixtral-8x7b-32768"]:
            return GroqModel(model_name)
        elif model_name in ["meta.llama3-1-405b-instruct-v1:0",
                            "meta.llama3-1-8b-instruct-v1:0"]:
            return BedrockModel(model_name)
        elif model_name in ["gemini-1.5-flash",
                            "gemini-1.5-pro",
                            "gemini-1.5-pro-exp-0801",
                            "gemini-1.5-pro-exp-0827",
                            "gemini-1.5-flash-exp-0827",
                            "gemini-1.5-flash-8b-exp-0827"]:
            return GeminiModel(model_name)
        elif model_name in ["cpp-llama3-8b"]:
            return LLamaCPPModel(model_name)
        elif model_name in ["mistral-large-2407",
                            "open-mistral-nemo"]:
            return MistralModel(model_name)
        elif model_name in ["chatgpt-4o-latest",
                            "gpt-4-turbo",
                            "gpt-3.5-turbo",
                            "gpt-4o-2024-05-13",
                            "gpt-4o-2024-08-06",
                            "gpt-4o-mini",
                            "o1-preview-2024-09-12",
                            "o1-mini-2024-09-12"]:
            return OpenAIModel(model_name)
        elif model_name in ["claude-3-5-sonnet-20240620",
                            "claude-3-haiku-20240307"]:
            return AnthropicModel(model_name)
        elif model_name in ["deepseek-chat", "deepseek-coder"]:
            return DeepseekModel(model_name)
        elif model_name in ["NousResearch/Hermes-2-Theta-Llama-3-8B"]:
            return LocalModel(model_name)
        elif model_name in ["microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-small-8k-instruct"]:
            return PhiModel(model_name)
        elif model_name in ["qwen/qwen-2.5-coder-32b-instruct"]:
            return OpenrouterModel(model_name)
        else:
            raise ValueError(f"Unknown model: {model_name}")
