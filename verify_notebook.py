"""
verify_notebook.py — 重放执行 churn_analysis.ipynb 并断言 0 错误
用途：CI 中验证 Notebook 端到端可复现；本地也可用于检查改动是否破坏 Notebook
"""
import nbformat
from nbclient import NotebookClient

nb = nbformat.read("churn_analysis.ipynb", as_version=4)
client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

errors = [
    o
    for c in nb.cells
    if c.cell_type == "code"
    for o in c.get("outputs", [])
    if o.get("output_type") == "error"
]
n_code = sum(1 for c in nb.cells if c.cell_type == "code")
assert not errors, f"{len(errors)} 个 cell 执行失败: {errors[0].get('ename')}"
print(f"OK: {n_code} 个 code cell 全部执行成功，0 错误")
