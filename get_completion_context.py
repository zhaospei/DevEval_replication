import json
import os
from pathlib import Path
from tqdm import tqdm

SOURCE_CODE_ROOT = '/drive2/tuandung/DevEval/Source_Code'

datasets = []
with open('/drive2/tuandung/DevEval/data_clean.jsonl') as f:
    for line in f:
        data = json.loads(line)
        datasets.append(data)

completion_dataset = []

for data in tqdm(datasets):
    completion_path = Path(data['completion_path'])
    completion_path = os.path.join(SOURCE_CODE_ROOT, completion_path)
    
    with open(completion_path, 'r') as f:
        file_lines = f.readlines()
    # write the new completion file
    sos, eos = data['body_position'][0]-1, data['body_position'][1]
    s_sos, s_eos = data['signature_position'][0]-1, data['signature_position'][1]
    contexts_above = ''.join(file_lines[:s_sos])
    contexts_below = ''.join(file_lines[eos:])
    input_code = ''.join(file_lines[s_sos:sos])
    ground_truth = ''.join(file_lines[sos:eos])
    function_name = data['namespace'].split('.')[-1]
    
    sample = {
        'namespace': data['namespace'],
        'completion_path': data['completion_path'],
        'contexts_above': contexts_above,
        'contexts_below': contexts_below,
        'input_code': input_code,
        'function_name': function_name,
        'ground_truth': ground_truth,
        'indent': data['indent']
    }
    completion_dataset.append(sample)

with open('/drive2/tuandung/DevEval/completion_dataset.jsonl', 'w') as f:
    for sample in completion_dataset:
        f.write(json.dumps(sample) + '\n')