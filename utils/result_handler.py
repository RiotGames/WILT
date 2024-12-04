import os
import csv
import json

def sanitize_model_name(model_name):
    return model_name.replace("/", "__")

def dump_results(model_name, accuracy, avg_guesses, points, full_context, split):
    model_name = sanitize_model_name(model_name)
    qualifier = f"_{split}" if split != "full" else ""        
    os.makedirs('./results', exist_ok=True)
    
    # Dump CSV results
    csv_file = f'./results/{model_name}{qualifier}_results.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['accuracy', 'avg_guesses', 'points'])
        writer.writerow([accuracy, avg_guesses, points])
    
    # Dump full context as JSON
    json_file = f'./results/{model_name}{qualifier}_full_context.json'
    with open(json_file, 'w') as f:
        json.dump(full_context, f, indent=2)

    print(f"Results saved to {csv_file} and {json_file}")
