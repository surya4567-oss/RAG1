import json
import pandas as pd

df_d = pd.read_csv('data/defects_rows.csv').set_index('defect_id')
df_t = pd.read_csv('data/testcases.csv').set_index('test_case_id')
df_e = pd.read_csv('data/test_execution.csv').set_index('execution_id')

print("--- Comparison (Q121-Q130) ---")
print("Q121 BUG-000001 sev:", df_d.loc['BUG-000001', 'severity'], "| BUG-000003 sev:", df_d.loc['BUG-000003', 'severity'])
print("Q122 EXEC-000001 sec:", df_e.loc['EXEC-000001', 'execution_time_seconds'], "| EXEC-000002 sec:", df_e.loc['EXEC-000002', 'execution_time_seconds'])
print("Q123 TC-000001 pri:", df_t.loc['TC-000001', 'priority'], "| TC-000002 pri:", df_t.loc['TC-000002', 'priority'])
print("Q124 EXEC-000001 status:", df_e.loc['EXEC-000001', 'status'], "| EXEC-000006 status:", df_e.loc['EXEC-000006', 'status'])
print("Q125 BUG-000001 pri:", df_d.loc['BUG-000001', 'priority'], "| BUG-000005 pri:", df_d.loc['BUG-000005', 'priority'])
print("Q126 EXEC-000006 browser:", df_e.loc['EXEC-000006', 'browser'], "| EXEC-000007 browser:", df_e.loc['EXEC-000007', 'browser'])
print("Q127 TC-000003 status:", df_t.loc['TC-000003', 'status'], "| TC-000006 status:", df_t.loc['TC-000006', 'status'])
print("Q128 EXEC-000002 OS:", df_e.loc['EXEC-000002', 'operating_system'], "| EXEC-000003 OS:", df_e.loc['EXEC-000003', 'operating_system'])
print("Q129 BUG-000004 status:", df_d.loc['BUG-000004', 'status'], "| BUG-000007 status:", df_d.loc['BUG-000007', 'status'])
print("Q130 EXEC-000007 sec:", df_e.loc['EXEC-000007', 'execution_time_seconds'], "| EXEC-000009 sec:", df_e.loc['EXEC-000009', 'execution_time_seconds'])

print("\n--- Temporal (Q131-Q140) ---")
print("Q131 BUG-000001 created/resolved:", df_d.loc['BUG-000001', ['created_date', 'resolved_date']].to_dict())
print("Q132 EXEC-000002 date:", df_e.loc['EXEC-000002', 'execution_date'])
print("Q133 EXEC-000001 date:", df_e.loc['EXEC-000001', 'execution_date'], "| EXEC-000005 date:", df_e.loc['EXEC-000005', 'execution_date'])
print("Q136 TC-000001 created_date:", df_t.loc['TC-000001', 'created_date'])
print("Q137 EXEC-000004 date:", df_e.loc['EXEC-000004', 'execution_date'], "| EXEC-000006 date:", df_e.loc['EXEC-000006', 'execution_date'])
print("Q140 BUG-000004 status/resolved:", df_d.loc['BUG-000004', ['status', 'resolved_date']].to_dict())

print("\n--- OCR Checks (Q141-Q150) ---")
with open('data/screenshots/ocr_batch_001.json') as f:
    b1 = json.load(f)
with open('data/screenshots/ocr_batch_002.json') as f:
    b2 = json.load(f)

for x in b1:
    if x.get('defect_id') == 'BUG-001585':
        print("Q141 BUG-001585 buttons:", x.get('ui_buttons'))
        print("Q144 BUG-001585 menus:", x.get('active_menus'))
        print("Q145 BUG-001585 warning:", repr(x.get('warning_message')))
        print("Q146 BUG-001585 error:", repr(x.get('error_message')))
        print("Q147 BUG-001585 user_action:", repr(x.get('user_action')))
        break

for x in b1:
    if x.get('execution_id') == 'EXEC-017509':
        print("Q142 EXEC-017509 screen_name:", x.get('screen_name'))
        break

for x in b1:
    if x.get('defect_id') == 'BUG-003851':
        print("Q143 BUG-003851 dialog:", x.get('dialog_box'))
        break

for x in b2:
    if x.get('execution_id') == 'EXEC-002301':
        print("Q148 EXEC-002301 screen_name:", x.get('screen_name'))
    if x.get('defect_id') == 'BUG-008442':
        print("Q149 BUG-008442 dialog:", x.get('dialog_box'))
    if x.get('execution_id') == 'EXEC-006588':
        print("Q150 EXEC-006588 buttons:", x.get('ui_buttons'))

