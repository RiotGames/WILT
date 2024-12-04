"""
This script analyzes a folder of conversation history jsons, applying different
metrics of complexity to try and gauge which LLMs follow Occam's Razor.
"""

from harness.test_cases import TESTS_FULL
import inspect
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def calculate_set_inclusion_score(guessed_lambda, rule):
    random_inputs = [tuple(random.uniform(-200, 200)
                            for _ in range(3)) for _ in range(10000)]
    grid_inputs = []

    for coords in itertools.product(range(-20, 21), repeat=3):
        grid_inputs.append(coords)

    test_inputs = random_inputs + grid_inputs
    
    try:
        guessed_set = set(inp for inp in test_inputs if guessed_lambda(*inp))
        rule_set = set(inp for inp in test_inputs if rule(*inp))

        if guessed_set.issubset(rule_set) or rule_set.issubset(guessed_set):
            return len(guessed_set) / len(rule_set)
    except:
        return None
    return None

def count_num_ops(rulestr):
    operations = [
        "and", 
        "or", 
        "not", 
        "**", 
        "==", 
        "!=", 
        "<=", 
        ">=",
        ">>",
        "<<",
        "//", 
        "+", 
        "-", 
        "/", 
        "*",
        "=",
        "%" ,
        "&",
        "|",
        "~",
        "^",
        "<",
        ">"
    ]
    operation_count = 0

    # Sort the operations list by length in descending order to handle longer operations first
    operations.sort(key=len, reverse=True)

    # Loop through each operation and count occurrences in the string
    for op in operations:
        count = rulestr.count(op)
        operation_count += count
        rulestr = rulestr.replace(op, '')
    
    return operation_count

def main():
    # Basically convert the keys to ints
    tests = {}
    incorrects = {}
    set_inclusion_scores = {}
    guess_str_len = {}
    num_ops = {}

    for i in range(1,len(TESTS_FULL)+1):
        tests[i] = TESTS_FULL[f"{i}"]
        incorrects[i] = [("Correct Rule", inspect.getsource(tests[i]).strip())]


    for json_filename in os.listdir("full_results"): # ["mistral-large-2407_full_context.json"]:
        set_inclusion_scores[json_filename] = []
        guess_str_len[json_filename] = []
        num_ops[json_filename] = []
        

        with open(f"./full_results/{json_filename}", "r") as file:
            test_convos = json.load(file)

        for convo in test_convos:
            test_index = convo["test_index"]
            if test_index not in incorrects:
                incorrects[test_index] = []
            conv_hist = convo["conversation_history"]
            if "Sorry" in conv_hist[-1]["content"] or "Congratulations" in conv_hist[-1]["content"]:
                if "Final Guess: ```" in conv_hist[-2]["content"]:
                    wrong_rule = conv_hist[-2]["content"]
                    wrong_rule = wrong_rule[wrong_rule.index("Final Guess: ```") + 16:]
                    wrong_rule = wrong_rule[: wrong_rule.index("```")]

                    try:
                        set_inclusion_score = calculate_set_inclusion_score(eval(wrong_rule), tests[test_index])
                    except:
                        set_inclusion_score = None
                    if set_inclusion_score is not None:
                        set_inclusion_scores[json_filename].append(set_inclusion_score)
                    num_op = count_num_ops(wrong_rule)
                    num_ops[json_filename].append(num_op)
                    str_len = len(wrong_rule) - 2
                    guess_str_len[json_filename].append(str_len)

                    incorrects[test_index].append((json_filename, wrong_rule, set_inclusion_score, num_op, str_len))

    for i in incorrects:
        print(i)
        for j in incorrects[i]:
            print(f"    {j}")

    models_ordered_as_in_paper = [
        'claude-3-5-sonnet-20240620_full_context.json',
        'o1-mini-2024-09-12_full_context.json',
        'o1-preview-2024-09-12_full_context.json',
        'chatgpt-4o-latest_full_context.json',
        'mistral-large-2407_full_context.json',
        'gpt-4o-2024-08-06_full_context.json',
        'meta.llama3-1-405b-instruct-v1_0_full_context.json',
        'gemini-1.5-flash-8b-exp-0827_full_context.json',
        'llama3-70b-8192_full_context.json',
        'deepseek-chat-v2.5_full_context.json',
        'gpt-4o-mini_full_context.json',
        'gemini-1.5-pro_full_context.json',
        'gemini-1.5-flash_full_context.json',
        'deepseek-coder_full_context.json',
        'deepseek-chat_full_context.json',
        'meta.llama3-1-8b-instruct-v1_0_full_context.json',
        'open-mistral-nemo_full_context.json',
        'claude-3-haiku-20240307_full_context.json',
        'gemini-1.5-flash-exp-0827_full_context.json',
        'gemma2-9b-it_full_context.json',
    ]

    for k, v in set_inclusion_scores.items():
        set_inclusion_scores[k] = np.median(np.array(v))
    sorted_set_inclusion_scores = sorted(set_inclusion_scores.items(), key=lambda x: x[1])
    print("Set Inclusion Scores")
    for model in models_ordered_as_in_paper:
        print(model,set_inclusion_scores[model])
    # for score in sorted_set_inclusion_scores:
    #     print(score)

    categories = [i[0] for i in sorted_set_inclusion_scores]
    values = [i[1] for i in sorted_set_inclusion_scores]
    plt.barh(categories, values)
    plt.title('Set Inclusion Score')
    plt.tight_layout()
    plt.show()

    for k, v in num_ops.items():
        num_ops[k] = np.median(np.array(v))
    sorted_num_ops = sorted(num_ops.items(), key=lambda x: x[1])
    print("Num Ops")
    # for score in sorted_num_ops:
    #     print(score)
    for model in models_ordered_as_in_paper:
        print(model, num_ops[model])

    categories = [i[0] for i in sorted_num_ops]
    values = [i[1] for i in sorted_num_ops]
    plt.barh(categories, values)
    plt.title('Num Ops Score')
    plt.tight_layout()
    plt.show()

    for k, v in guess_str_len.items():
        guess_str_len[k] = np.median(np.array(v))
    sorted_guess_str_len = sorted(guess_str_len.items(), key=lambda x: x[1])
    print("Guess Str Len")
    # for score in sorted_guess_str_len:
    #     print(score)
    for model in models_ordered_as_in_paper:
        print(model, guess_str_len[model])

    categories = [i[0] for i in sorted_guess_str_len]
    values = [i[1] for i in sorted_guess_str_len]
    plt.barh(categories, values)
    plt.title('Guess Str Len')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

        

