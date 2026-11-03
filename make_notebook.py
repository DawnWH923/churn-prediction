"""
make_notebook.py — 从 churn_prediction.py 生成讲解版 Notebook 并真实执行验证
产物：churn_analysis.ipynb（所有 cell 均已执行，输出内嵌）
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# 电商用户流失预测与分群分析

**流程**：数据加载 → 划分与标准化 → 三模型对比（AUC）→ XGBoost 详解 → 特征重要性 → K-Means 分群 → 高风险清单输出

> 数据说明：模拟数据集（`generate_data.py` 生成 5000 条），分布假设贴近真实电商业务；
> 替换 `data/ecommerce_churn.csv` 为同结构真实数据即可复用全部建模代码。""")

code("""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.cluster import KMeans

df = pd.read_csv("data/ecommerce_churn.csv")
df.head()""")

md("## 1. 数据概览\n\n流失率约 9%，类别不平衡——这就是后文用 **AUC 而非准确率** 评估的原因：全预测\"不流失\"准确率也有 91%，但毫无业务价值。")

code("""X = df.drop(columns=["user_id", "churn"])
y = df["churn"]

print(f"样本数 {len(df)}，流失率 {y.mean():.1%}")
df.describe().round(2)""")

md("## 2. 划分训练/测试集 + 标准化\n\n80/20 划分；标准化对逻辑回归（基于梯度）必要，对树模型无影响但统一处理无害。`fit_transform` 只在训练集上 fit，防止测试集信息泄漏。")

code("""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)""")

md("## 3. 三模型对比\n\n| 模型 | 原理 | 特点 |\n|---|---|---|\n| 逻辑回归 | 线性概率模型 | 快、可解释、只能学线性关系 |\n| 随机森林 | 多树投票（Bagging） | 抗过拟合、能学非线性 |\n| XGBoost | 梯度提升（Boosting） | 逐棵纠错，表格数据王者 |")

code("""models = {
    "逻辑回归": LogisticRegression(max_iter=1000),
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
    print(f"{name}: AUC = {auc:.3f}")""")

md("> 模拟数据线性关系强，逻辑回归 AUC 最高属正常现象；真实电商数据非线性更强，XGBoost 通常反超。工程上 XGBoost 生态最成熟（特征重要性、调参、部署工具链完善），作为最终模型。")

md("## 4. XGBoost 详细评估\n\n重点看**召回率（recall）**：漏掉一个要流失的用户 = 少一次挽留机会，业务上召回率比精确率更重要。")

code("""best = XGBClassifier(eval_metric="logloss", random_state=42)
best.fit(X_train_s, y_train)
print(classification_report(y_test, best.predict(X_test_s)))""")

md("## 5. 特征重要性\n\nTop 特征与业务直觉一致：越久没来（recency 大）、登录越少，越可能流失。")

code("""imp = pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.round(3).to_string())""")

md("## 6. K-Means 用户分群（K=4）\n\nK=4 由业务解释性确定：高价值活跃 / 高价值沉默 / 价格敏感 / 低价值流失，每类对应不同运营策略。")

code("""km = KMeans(n_clusters=4, n_init=10, random_state=42)
df["cluster"] = km.fit_predict(X)
summary = df.groupby("cluster")[["recency", "frequency", "monetary", "login_freq", "churn"]].mean()
summary.round(2)""")

md("## 7. 高风险用户清单输出\n\n按流失概率降序导出，供运营定向发召回优惠券；线上落地时对比发券组与对照组的回流率即可算 ROI。")

code("""df["churn_prob"] = best.predict_proba(sc.transform(X))[:, 1]
df[["user_id", "recency", "frequency", "monetary", "churn", "churn_prob"]].to_csv(
    "predictions.csv", index=False, encoding="utf-8-sig")
print(f"高流失风险用户（概率>0.5）：{(df['churn_prob'] > 0.5).sum()} 人，已导出 predictions.csv")""")

nb.cells = cells  # 挂载 cell 列表（漏掉会写出空 notebook）
nbf.write(nb, "churn_analysis.ipynb")
print("notebook generated")

# ---- 真实执行验证（nbclient 内核跑全部 cell，输出内嵌） ----
import nbformat
from nbclient import NotebookClient

nb2 = nbformat.read("churn_analysis.ipynb", as_version=4)
client = NotebookClient(nb2, timeout=300, kernel_name="python3")
client.execute()
nbformat.write(nb2, "churn_analysis.ipynb")
n_exec = sum(1 for c in nb2.cells if c.cell_type == "code")
print(f"notebook executed OK ({n_exec} code cells)")
