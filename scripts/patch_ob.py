import json

nb = json.load(open("notebooks/flight_predictor.ipynb", encoding="utf-8"))

old = 'df_b = df_book.copy()\ndf_b = pd.get_dummies'
new = ('df_b = df_book.copy()\n'
       '# month column contains full names (e.g. "July") -- convert to integers 1-12\n'
       'MONTH_MAP = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,\n'
       '             "July":7,"August":8,"September":9,"October":10,"November":11,"December":12}\n'
       'df_b["month"] = df_b["month"].map(MONTH_MAP)\n'
       'df_b = pd.get_dummies')

for cell in nb["cells"]:
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if old in src:
        cell["source"] = src.replace(old, new)
        print("Patched overbooking prep cell.")
        break

with open("notebooks/flight_predictor.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Saved.")
