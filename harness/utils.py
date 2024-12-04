import openai
import pydantic
import os
from typing import Literal

class LLMJudgment(pydantic.BaseModel):
    reasoning: str
    judgment: Literal["Correct",
                      "Missing Edge Case",
                      "Near-Miss Magnitudes",
                      "Completely Different"]

class HarnessUtils:
    @staticmethod
    def calculate_set_size(rule, inclusive_int_range=(1, 100)):
        """
        NOTE(thwu): Only use this for rules that check for set membership
        """
        set_size = 0
        num_args = rule.__code__.co_argcount
        for i in range(inclusive_int_range[0], inclusive_int_range[1]):
            if num_args == 1:
                if rule(i):
                    set_size += 1
            else:
                if rule(i,i,i):
                    set_size += 1
        return set_size
    
    @staticmethod
    def calculate_all_set_sizes(rules, inclusive_int_range=(1, 100)):
        """
        NOTE(thwu): Only use this for rules that check for set membership
        """
        return {i: HarnessUtils.calculate_set_size(rules[i], inclusive_int_range) for i in rules.keys()}


    def judge(self, gt, guess):
        """
        NOTE(Eryk): used to score responses for near-misses
        """
        openai.api_key = os.environ["OPENAI_API_KEY"]
        client = openai.OpenAI()

        with open('./prompts/judge_prompt.txt', 'r') as f:
            system_prompt = f.read()

        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"First function: {gt}\n\nSecond function: {guess}"}
            ],
            response_format=LLMJudgment
        )

        return response.choices[0].message.parsed.judgment
