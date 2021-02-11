import os
import json

input_dir = 'raw_ai_benchmark_list.json'
output_dir = 'benchmark2score.json'

with open(input_dir, 'r') as f:
    raw_list = json.load(f)
    benchmark2score = {}
    for device in raw_list:
        model = device['Model']
        score = int(device['AI_Score'].strip().split()[0])
        assert isinstance(score, int)
        benchmark2score[model] = score
    
    with open(output_dir, 'w') as of:
        json.dump(benchmark2score, of, indent=2)
