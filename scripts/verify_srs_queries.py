from pypdf import PdfReader

reader = PdfReader("data/SRS.pdf")
print("Total pages:", len(reader.pages))

text = ""
for i, page in enumerate(reader.pages[:10]):
    text += f"\n--- Page {i+1} ---\n" + page.extract_text()

# Check Q001, Q002, Q003
print("Page 1 snippet:\n", text[:1000])

# Search for specific terms
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

terms = [
    "ODOO-ERP-SRS-001",
    "Baseline draft",
    "11 October 2026",
    "Supply chain",
    "Commercial",
    "Auditor",
    "Lead-to-cash",
    "Procure-to-pay",
    "Version 0.1",
    "11 Oct 2026",
]
for t in terms:
    print(f"Term '{t}' in SRS: {t in full_text}")

