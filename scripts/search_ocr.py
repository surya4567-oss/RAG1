import json
from pathlib import Path

ocr_files = sorted(Path("data/screenshots").glob("*.json"))

def search(term):
    print(f"\nSearching for {term}:")
    for f in ocr_files:
        with open(f, 'r') as jf:
            items = json.load(jf)
            for idx, it in enumerate(items):
                # Convert to string and search
                s = json.dumps(it)
                if term in s:
                    print(f"  Found in {f.name} [index {idx}]:")
                    print(f"    execution_id: {it.get('execution_id')}")
                    print(f"    defect_id:    {it.get('defect_id')}")
                    print(f"    screen_name:  {it.get('screen_name')}")
                    print(f"    dialog_box:   {it.get('dialog_box')}")
                    print(f"    ui_buttons:   {it.get('ui_buttons')}")
                    print(f"    active_menus: {it.get('active_menus')}")
                    print(f"    warning:      {it.get('warning_message')}")
                    print(f"    error:        {it.get('error_message')}")
                    print(f"    user_action:  {it.get('user_action')}")

for t in ["BUG-001585", "BUG-003851", "BUG-008442", "EXEC-006588", "EXEC-017509", "EXEC-002249"]:
    search(t)

