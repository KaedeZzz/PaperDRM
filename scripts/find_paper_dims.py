"""Search Detailed Paper Info for specific stock IDs and return folio width (mm)."""
import openpyxl

XLSX = r"data\manuscript_db\Master Paper Spreadsheet.xlsx"

# stock IDs matching our folio images (from TP summary sheet)
# format: (serial_name, stock_id, shelfmark, folio)
TARGETS = [
    ("Ee5-22_f328r",  "0061a", "Ee.5.22",  "328r"),
    ("Ff2-6_f140r",   "0069a", "Ff.2.6",   "140r"),
    ("Ff4-9_f42r",    "0070b", "Ff.4.9",   "42r"),
    ("Ff4-15_f24r",   "0076b", "Ff.4.15",  "24r"),
    ("Hh2-10_f24r",   "0095a", "Hh.2.10",  "24r"),
    ("Hh2-12_f190r",  "0136b", "Hh.2.12",  "190r"),
    ("Ii3-8_f135v",   "0155b", "Ii.3.8",   "135v"),
    ("Kk1-5_f5v",     "235a",  "Kk.1.5 pts5-6", "5v"),
    ("Kk1-5_f9v",     "235b",  "Kk.1.5 pts5-6", "9v"),
]

wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
ws = wb["Detailed Paper Info"]
rows = list(ws.iter_rows(values_only=True))
header = [str(c).strip() if c is not None else "" for c in rows[0]]

# print column indices for height/width
print("Header cols:")
for i, h in enumerate(header[:25]):
    print(f"  [{i}] {h}")
print()

target_ids = {t[1]: t for t in TARGETS}

print(f"{'Serial':<22} {'Stock':>8} {'Height':>8} {'Width':>8}  Shelfmark / locus-range")
print("-"*80)
for row in rows[1:]:
    stock_id = str(row[0]).strip() if row[0] is not None else ""
    if stock_id in target_ids:
        name, sid, shelfmark, folio = target_ids[stock_id]
        height = row[11]
        width  = row[12]
        locus_from = row[7]
        locus_to   = row[8]
        print(f"{name:<22} {sid:>8} {str(height):>8} {str(width):>8}  {shelfmark} {locus_from}–{locus_to}")
