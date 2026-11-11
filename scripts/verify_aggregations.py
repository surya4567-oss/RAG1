import json
import pandas as pd
from pypdf import PdfReader

df_d = pd.read_csv('data/defects_rows.csv')
df_t = pd.read_csv('data/testcases.csv')
df_e = pd.read_csv('data/test_execution.csv')

print("Q106 total defects:", len(df_d))
print("Q107 total test cases:", len(df_t))
print("Q108 total test executions:", len(df_e))

with open('data/screenshots/ocr_batch_001.json') as f:
    ocr_001 = json.load(f)
print("Q109 ocr_batch_001 items:", len(ocr_001))

with open('data/failure_logs/logs_001.txt') as f:
    logs_001 = f.read()
log_blocks = [b for b in logs_001.split('======================================================================') if b.strip()]
print("Q110 logs_001 blocks:", len(log_blocks))

pdf = PdfReader('data/SRS.pdf')
print("Q111 SRS pages:", len(pdf.pages))

print("Q112 defects 'New':", (df_d['status'] == 'New').sum())
print("Q113 defects 'Closed':", (df_d['status'] == 'Closed').sum())
print("Q114 defects 'In Progress':", (df_d['status'] == 'In Progress').sum())
print("Q115 defects severity 'Critical':", (df_d['severity'] == 'Critical').sum())
print("Q116 defects severity 'Major':", (df_d['severity'] == 'Major').sum())
print("Q117 executions 'Passed':", (df_e['status'] == 'Passed').sum())
print("Q118 executions 'Failed':", (df_e['status'] == 'Failed').sum())
print("Q119 testcases 'POS' module:", (df_t['module'] == 'POS').sum())
print("Q120 defects 'Security' bug_type:", (df_d['bug_type'] == 'Security').sum())

