import json

with open('ML_project.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        code.append("".join(cell['source']))

with open('ml_code.py', 'w', encoding='utf-8') as f:
    f.write("\n\n".join(code))
