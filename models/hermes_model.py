from .base_model import BaseModel
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import guidance

from guidance import gen, select

class LocalModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        model_obj = LlamaForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            load_in_4bit=False,
            trust_remote_code=True,
            use_flash_attention_2=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)        
        self.model = guidance.models.Transformers(model_obj, self.tokenizer, echo=False)

    def perform_step(self, conversation_history):

        ## add result from previous test case here
        @guidance
        def context(lm, prev):
            newline = "\n"
            lm += f"""
            <|im_start|>user
            {prev}
            <|im_end|>
            """
            return lm

        if conversation_history[-1]['role'] == "user":
            self.model += context(conversation_history[-1]['content'])

        ## add next test case here
        @guidance
        def guide(lm):
            newline = '\n'
            lm += "<|im_start|>assistant\n"
            lm += f"Reasoning: {gen('logic', stop=[newline, 'Test Case:', 'Final Guess:'])}\n"
            lm += select(["Test Case: ```",
                          "Final Guess: ```"])
            lm += gen("selection", stop="`")
            lm += "```\n<|im_end|>\n"
            return lm

        self.model += guide()
        response = str(self.model)

        start_tag = "<|im_start|>assistant"
        end_tag = "<|im_end|>"
        start_index = response.rfind(start_tag)
        end_index = response.rfind(end_tag)
    
        if start_index != -1 and end_index != -1 and start_index < end_index:
            extracted_response = response[start_index + len(start_tag):end_index].strip()
            return extracted_response
        else:
            return ""

    def process_chat(self, conversation_history):
        msgs = []
        for m in conversation_history:
            msgs.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        return '\n'.join(msgs)

    def initialize_conversation(self, system_prompt):
        sys = [{"role": "system", "content": system_prompt}]
        chat = self.process_chat(sys)
        self.model += chat
        
        return sys
    
