import os
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt

def combine_json_files(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith("_full_context.json")]
    
    # Sort the files to ensure consistent ordering
    json_files.sort()
    
    output = []
    model_stats = defaultdict(lambda: {'total_repeats': 0, 'novelties': [], 'min_novelties': [], 'turn_novelties': defaultdict(list)})
    
    for file in json_files:
        # Extract the prefix (filename without "_full_context.json")
        prefix = re.sub(r'_full_context\.json$', '', file)
        
        # Read the JSON file
        with open(os.path.join(directory, file), 'r') as f:
            data = json.load(f)
        
        # Sort the data by test_index
        sorted_data = sorted(data, key=lambda x: x['test_index'])
        file_output = f"{prefix}\n{'=' * len(prefix)}\n\n"
        
        for testcase in sorted_data:
            repeats = testcase.get('repeats', 'N/A')
            novelty = testcase.get('novelty', 'N/A')
            
            file_output += f"Test Index: {testcase['test_index']}\n"
            file_output += f"Repeats: {repeats}\n"
            file_output += f"Novelty: {novelty}\n\n"
            
            # Update model statistics
            if isinstance(repeats, (int, float)):
                model_stats[prefix]['total_repeats'] += repeats
            if isinstance(novelty, list):
                try:
                    model_stats[prefix]['novelties'].extend(novelty)
                    model_stats[prefix]['min_novelties'].append(min(novelty))
                    for turn, nov in enumerate(novelty):
                        model_stats[prefix]['turn_novelties'][turn].append(nov)
                except:
                    print(f"{prefix} has at least one missing novelty vector")
                    model_stats[prefix]['min_novelties'].append(-1)
        
        output.append(file_output)
    
    # Calculate and add summary statistics
    summary = "Summary Statistics\n==================\n\n"
    for model, stats in model_stats.items():
        total_repeats = stats['total_repeats']
        avg_novelty = sum(stats['novelties']) / len(stats['novelties']) if stats['novelties'] else 'N/A'
        avg_min_novelty = sum(stats['min_novelties']) / len(stats['min_novelties']) if stats['min_novelties'] else 'N/A'
        summary += f"{model}:\n"
        summary += f"  Total Repeats: {total_repeats}\n"
        summary += f"  Average Novelty: {avg_novelty:.2f}\n"
        summary += f"  Average Minimum Novelty: {avg_min_novelty:.2f}\n\n"
    
    output.append(summary)
    
    # Write the combined output to a single file
    with open('combined_output_stats.txt', 'w') as f:
        f.write('\n'.join(output))
    
    return model_stats

def plot_novelty_by_turn(model_stats):
    models_to_plot = ['claude-3-5-sonnet-20240620', 'o1-mini-2024-09-12', 'o1-preview-2024-09-12', 'mistral-large-2407', 'deepseek-chat-v2.5', 'chatgpt-4o-latest', 'gpt-4o-mini']
    plt.figure(figsize=(12, 6))
    
    for model in models_to_plot:
        if model in model_stats:
            turns = sorted(model_stats[model]['turn_novelties'].keys())
            avg_novelties = [sum(model_stats[model]['turn_novelties'][turn]) / len(model_stats[model]['turn_novelties'][turn]) for turn in turns]
            plt.plot(turns, avg_novelties, label=model, marker='o')            
    
    plt.xlabel('Turn')
    plt.ylabel('Average Novelty')
    plt.title('Average Novelty by Turn for Different Models')
    plt.legend()
    plt.grid(True)
    plt.savefig('novelty_by_turn.png')
    plt.close()

# Usage
directory = '../results'
model_stats = combine_json_files(directory)
plot_novelty_by_turn(model_stats)
