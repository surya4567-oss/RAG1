import json
import pandas as pd
from pathlib import Path

df_d = pd.read_csv('data/defects_rows.csv').set_index('defect_id')
df_t = pd.read_csv('data/testcases.csv').set_index('test_case_id')
df_e = pd.read_csv('data/test_execution.csv').set_index('execution_id')

with open('vanillarag_evaluation_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Checking Q051 - Q065 (Metadata Filtering)...")
for item in data[50:65]:
    qid = item['id']
    q = item['query']
    exp = item['expected_answer']
    # Check if exp is in df_d, df_t, df_e
    row = None
    if exp in df_d.index:
        row = df_d.loc[exp].to_dict()
        print(f"{qid}: Defect {exp} -> {row['module']}, {row['severity']}, {row['priority']}, {row['status']}, {row['assigned_to']}, {row['reported_by']}")
    elif exp in df_t.index:
        row = df_t.loc[exp].to_dict()
        print(f"{qid}: TestCase {exp} -> {row['module']}, {row['priority']}, {row['status']}, {row['requirement_id']}")
    elif exp in df_e.index:
        row = df_e.loc[exp].to_dict()
        print(f"{qid}: Execution {exp} -> {row['execution_date']}, {row['browser']}, {row['operating_system']}, {row['status']}, {row['build_number']}")
    else:
        print(f"{qid}: Expected {exp} NOT FOUND in index!")

