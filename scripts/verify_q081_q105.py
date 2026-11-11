import json
import pandas as pd

df_d = pd.read_csv('data/defects_rows.csv').set_index('defect_id')
df_t = pd.read_csv('data/testcases.csv').set_index('test_case_id')
df_e = pd.read_csv('data/test_execution.csv').set_index('execution_id')

with open('vanillarag_evaluation_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Checking Multi-hop Queries (Q081 - Q105)...")
for item in data[80:105]:
    qid = item['id']
    q = item['query']
    exp = item['expected_answer']

    print(f"\n{qid}: {q}")
    print(f"  Expected in JSON: {exp}")

    # Check Q081: defect BUG-000001 linked test run status
    if "BUG-000001" in q and "execution" in q.lower():
        eid = df_d.loc['BUG-000001', 'linked_execution']
        st = df_e.loc[eid, 'status'] if eid in df_e.index else 'NOT_FOUND'
        print(f"  Actual in Data:   linked_execution={eid}, status={st}")

    # Check Q082: defect BUG-000001 linked test case requirement ID
    elif "BUG-000001" in q and "requirement" in q.lower():
        tcid = df_d.loc['BUG-000001', 'linked_test_case']
        req = df_t.loc[tcid, 'requirement_id'] if tcid in df_t.index else 'NOT_FOUND'
        print(f"  Actual in Data:   linked_test_case={tcid}, requirement_id={req}")

    # Check Q083: defect BUG-000002 linked execution browser
    elif "BUG-000002" in q and "browser" in q.lower():
        eid = df_d.loc['BUG-000002', 'linked_execution']
        br = df_e.loc[eid, 'browser'] if eid in df_e.index else 'NOT_FOUND'
        print(f"  Actual in Data:   linked_execution={eid}, browser={br}")

    # Check Q084: defect BUG-000003 linked test case module
    elif "BUG-000003" in q and "module" in q.lower():
        tcid = df_d.loc['BUG-000003', 'linked_test_case']
        mod = df_t.loc[tcid, 'module'] if tcid in df_t.index else 'NOT_FOUND'
        print(f"  Actual in Data:   linked_test_case={tcid}, module={mod}")

    # Check Q085: defect BUG-000004 linked execution time
    elif "BUG-000004" in q and "time" in q.lower():
        eid = df_d.loc['BUG-000004', 'linked_execution']
        sec = df_e.loc[eid, 'execution_time_seconds'] if eid in df_e.index else 'NOT_FOUND'
        print(f"  Actual in Data:   linked_execution={eid}, execution_time={sec}s")

    # Check Q086: defect BUG-000005 linked test case priority
    elif "BUG-000005" in q and "priority" in q.lower():
        tcid = df_d.loc['BUG-000005', 'linked_test_case']
        pri = df_t.loc[tcid, 'priority'] if tcid in df_t.index else 'NOT_FOUND'
        print(f"  Actual in Data:   linked_test_case={tcid}, priority={pri}")

    # Check Q087: defect BUG-000006 linked execution OS
    elif "BUG-000006" in q and "operating system" in q.lower():
        eid = df_d.loc['BUG-000006', 'linked_execution']
        os_val = df_e.loc[eid, 'operating_system'] if eid in df_e.index else 'NOT_FOUND'
        print(f"  Actual in Data:   linked_execution={eid}, operating_system={os_val}")

    # Check Q088: defect BUG-000007 linked test case requirement ID
    elif "BUG-000007" in q and "requirement" in q.lower():
        tcid = df_d.loc['BUG-000007', 'linked_test_case']
        req = df_t.loc[tcid, 'requirement_id'] if tcid in df_t.index else 'NOT_FOUND'
        print(f"  Actual in Data:   linked_test_case={tcid}, requirement_id={req}")

    # Check Q089: defect BUG-000008 linked execution build
    elif "BUG-000008" in q and "build" in q.lower():
        eid = df_d.loc['BUG-000008', 'linked_execution']
        bld = df_e.loc[eid, 'build_number'] if eid in df_e.index else 'NOT_FOUND'
        print(f"  Actual in Data:   linked_execution={eid}, build_number={bld}")

    # Check Q090: defect BUG-000009 linked test case module
    elif "BUG-000009" in q and "module" in q.lower():
        tcid = df_d.loc['BUG-000009', 'linked_test_case']
        mod = df_t.loc[tcid, 'module'] if tcid in df_t.index else 'NOT_FOUND'
        print(f"  Actual in Data:   linked_test_case={tcid}, module={mod}")

    # Check Q091: EXEC-012473 title
    elif "EXEC-012473" in q and "title" in q.lower():
        # Find which defect has linked_execution == EXEC-012473
        m = df_d[df_d['linked_execution'] == 'EXEC-012473']
        if len(m):
            print(f"  Actual in Data:   defect={m.index[0]}, title={m.iloc[0]['title']}")

    # Check Q092: EXEC-001114 title
    elif "EXEC-001114" in q and "title" in q.lower():
        m = df_d[df_d['linked_execution'] == 'EXEC-001114']
        if len(m):
            print(f"  Actual in Data:   defect={m.index[0]}, title={m.iloc[0]['title']}")

    # Check Q095: EXEC-000001 requirement ID
    elif "EXEC-000001" in q and "requirement" in q.lower():
        tcid = df_e.loc['EXEC-000001', 'test_case_id']
        req = df_t.loc[tcid, 'requirement_id'] if tcid in df_t.index else 'NOT_FOUND'
        print(f"  Actual in Data:   test_case_id={tcid}, requirement_id={req}")

    # Check Q096: EXEC-000002 requirement ID
    elif "EXEC-000002" in q and "requirement" in q.lower():
        tcid = df_e.loc['EXEC-000002', 'test_case_id']
        req = df_t.loc[tcid, 'requirement_id'] if tcid in df_t.index else 'NOT_FOUND'
        print(f"  Actual in Data:   test_case_id={tcid}, requirement_id={req}")

    # Check Q097: EXEC-000003 lifecycle status
    elif "EXEC-000003" in q and "status" in q.lower():
        tcid = df_e.loc['EXEC-000003', 'test_case_id']
        st = df_t.loc[tcid, 'status'] if tcid in df_t.index else 'NOT_FOUND'
        print(f"  Actual in Data:   test_case_id={tcid}, status={st}")

    # Check Q098: EXEC-000004 automation status
    elif "EXEC-000004" in q and "automation" in q.lower():
        tcid = df_e.loc['EXEC-000004', 'test_case_id']
        st = df_t.loc[tcid, 'automation_status'] if tcid in df_t.index else 'NOT_FOUND'
        print(f"  Actual in Data:   test_case_id={tcid}, automation_status={st}")

    # Check Q099: TC-003557 title of defect
    elif "TC-003557" in q and "defect" in q.lower():
        m = df_d[df_d['linked_test_case'] == 'TC-003557']
        if len(m):
            print(f"  Actual in Data:   defect={m.index[0]}, title={m.iloc[0]['title']}")

    # Check Q100: TC-008797 severity of defect
    elif "TC-008797" in q and "defect" in q.lower():
        m = df_d[df_d['linked_test_case'] == 'TC-008797']
        if len(m):
            print(f"  Actual in Data:   defect={m.index[0]}, severity={m.iloc[0]['severity']}")

