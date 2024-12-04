"""
This script analyzes a folder of conversation history jsons, applying different
metrics of complexity to try and gauge which LLMs follow Occam's Razor.
"""

from analysis import complexity_analysis
from harness.test_cases import TESTS_FULL
import inspect
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns

def main():
    filename_to_model_name = {
        "o1-mini-2024-09-12" : "o1-mini",
        "o1-preview-2024-09-12" : "o1-preview",
        "claude-3-5-sonnet-20240620" : "claude 3.5 sonnet",
        "chatgpt-4o-latest" : "chatgpt-4o-latest", 
        "mistral-large-2407" : "mistral-large-2407",
        "deepseek-chat-v2.5": "deepseek-chat-v2.5",
        "gpt-4o-mini" : "gpt 4o mini", 
    }
    filenames = list(filename_to_model_name.keys())
    modelnames = list(filename_to_model_name.values())
    modelnames.append("Average")
    num_models = len(filename_to_model_name)
    # Basically convert the keys to ints
    tests = {}
    incorrects = {}
    set_inclusion_scores = np.zeros((num_models, num_models))
    guess_str_len = np.zeros((num_models, num_models))
    num_ops = np.zeros((num_models, num_models))

    for i in range(1,len(TESTS_FULL)+1):
        tests[i] = TESTS_FULL[f"{i}"]
        rulestr = inspect.getsource(tests[i]).strip()
        incorrects[i] = [("Correct Rule", rulestr)]


    for json_filename in os.listdir("testswaps"):
        base_model = json_filename[:json_filename.index("_test_swap")]
        swap_model = json_filename[json_filename.index("_test_swap")+11:json_filename.index("_full_context.json")]
        base_model_idx = filenames.index(base_model)
        swap_model_idx = filenames.index(swap_model)

        set_inclusion_scores_for_json = []
        guess_str_len_for_json = []
        num_ops_for_json = []

        with open(f"./testswaps/{json_filename}", "r") as file:
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
                        set_inclusion_score = complexity_analysis.calculate_set_inclusion_score(
                            eval(wrong_rule), tests[int(test_index)]
                        )
                    except:
                        set_inclusion_score = None
                    if set_inclusion_score is not None:
                        set_inclusion_scores_for_json.append(set_inclusion_score)

                    num_op = complexity_analysis.count_num_ops(wrong_rule)
                    num_ops_for_json.append(num_op)
                    str_len = len(wrong_rule) - 2
                    guess_str_len_for_json.append(str_len)

                    incorrects[test_index].append((json_filename, wrong_rule, set_inclusion_score, num_op, str_len))

        set_inclusion_scores[base_model_idx, swap_model_idx] = np.median(np.array(set_inclusion_scores_for_json))
        num_ops[base_model_idx, swap_model_idx] = np.median(np.array(num_ops_for_json))
        guess_str_len[base_model_idx, swap_model_idx] = np.median(np.array(guess_str_len_for_json))

    for i in incorrects:
        print(i)
        for j in incorrects[i]:
            print(f"    {j}")


    # Define font sizes
    title_fontsize = 24
    tick_label_fontsize = 18
    label_fontsize = 16
    annotation_fontsize = 18

    plt.figure(figsize=(12, 10))
    
    col_mean = np.mean(set_inclusion_scores, axis=0)
    row_mean = np.concatenate((np.mean(set_inclusion_scores, axis=1),np.full(1, np.nan)), axis=0)
    row_mean = np.expand_dims(row_mean, axis=1)
    set_inclusion_scores = np.vstack((set_inclusion_scores, col_mean))
    set_inclusion_scores = np.hstack((set_inclusion_scores, row_mean))
    ax = sns.heatmap(
        set_inclusion_scores,
        annot=True,
        cmap='YlOrRd',
        fmt='.2f',
        cbar_kws={'label': 'Value'},
        annot_kws={'size': annotation_fontsize},
        linewidths=.5,
        linecolor='gray',
        xticklabels=modelnames,
        yticklabels=modelnames
    )
    num_rows, num_cols = set_inclusion_scores.shape
    # Draw a thicker horizontal line before the last row (averages)
    ax.axhline(num_rows - 1, color='black', linewidth=2)
    # Draw a thicker vertical line before the last column (averages)
    ax.axvline(num_cols - 1, color='black', linewidth=2)
    # Draw the last missing gray lines
    ax.axhline(num_rows, color='gray', linewidth=1)
    ax.axvline(num_cols, color='gray', linewidth=1)

    plt.title('Set Inclusion Scores', fontsize=title_fontsize)
    plt.xlabel('Model used for tests', fontsize=label_fontsize)
    plt.ylabel('Model used for inference', fontsize=label_fontsize)
    plt.xticks(rotation=45, ha='right', fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 10))
    col_mean = np.mean(num_ops, axis=0)
    row_mean = np.concatenate((np.mean(num_ops, axis=1),np.full(1, np.nan)), axis=0)
    row_mean = np.expand_dims(row_mean, axis=1)
    num_ops = np.vstack((num_ops, col_mean))
    num_ops = np.hstack((num_ops, row_mean))
    ax = sns.heatmap(
        num_ops,
        annot=True,
        cmap='YlOrRd',
        fmt='.2f',
        cbar_kws={'label': 'Value'},
        annot_kws={'size': annotation_fontsize},
        linewidths=.5,
        linecolor='gray',
        xticklabels=modelnames,
        yticklabels=modelnames
    )
    num_rows, num_cols = num_ops.shape
    # Draw a thicker horizontal line before the last row (averages)
    ax.axhline(num_rows - 1, color='black', linewidth=2)
    # Draw a thicker vertical line before the last column (averages)
    ax.axvline(num_cols - 1, color='black', linewidth=2)
    # Draw the last missing gray lines
    ax.axhline(num_rows, color='gray', linewidth=1)
    ax.axvline(num_cols, color='gray', linewidth=1)

    plt.title('Num Ops', fontsize=title_fontsize)
    plt.xlabel('Model used for tests')
    plt.ylabel('Model used for inference')
    plt.xticks(rotation=45, ha='right', fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 10))
    col_mean = np.mean(guess_str_len, axis=0)
    row_mean = np.concatenate((np.mean(guess_str_len, axis=1),np.full(1, np.nan)), axis=0)
    row_mean = np.expand_dims(row_mean, axis=1)
    guess_str_len = np.vstack((guess_str_len, col_mean))
    guess_str_len = np.hstack((guess_str_len, row_mean))
    ax = sns.heatmap(
        guess_str_len,
        annot=True,
        cmap='YlOrRd',
        fmt='.2f',
        cbar_kws={'label': 'Value'},
        annot_kws={'size': annotation_fontsize},
        linewidths=.5,
        linecolor='gray',
        xticklabels=modelnames,
        yticklabels=modelnames
    )
    num_rows, num_cols = guess_str_len.shape
    # Draw a thicker horizontal line before the last row (averages)
    ax.axhline(num_rows - 1, color='black', linewidth=2)
    # Draw a thicker vertical line before the last column (averages)
    ax.axvline(num_cols - 1, color='black', linewidth=2)
    # Draw the last missing gray lines
    ax.axhline(num_rows, color='gray', linewidth=1)
    ax.axvline(num_cols, color='gray', linewidth=1)

    plt.title('Guess Str Len', fontsize=title_fontsize)
    plt.xlabel('Model used for tests')
    plt.ylabel('Model used for inference')
    plt.xticks(rotation=45, ha='right', fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()