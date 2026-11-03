"""
churn_prediction.py — 电商用户流失预测
对比逻辑回归、随机森林、XGBoost，并用 K-Means 做用户分群
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.cluster import KMeans

# 1. 读取数据
df = pd.read_csv("data/ecommerce_churn.csv")
X = df.drop(columns=["user_id", "churn"])
y = df["churn"]

# 2. 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)

# 3. 对比三个模型
print("=" * 50)
print("模型对比（AUC）")
print("=" * 50)
models = {
    "逻辑回归": LogisticRegression(max_iter=1000),
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, pred)
    print(f"{name}: AUC = {auc:.3f}")

# 4. XGBoost 详细报告
best = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
best.fit(X_train_s, y_train)
pred = best.predict(X_test_s)
print("\n" + "=" * 50)
print("XGBoost 分类报告")
print("=" * 50)
print(classification_report(y_test, pred))

# 5. 特征重要性
print("=" * 50)
print("特征重要性（Top 5）")
print("=" * 50)
imp = pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head())

# 6. K-Means 用户分群
print("\n" + "=" * 50)
print("K-Means 用户分群（4类）")
print("=" * 50)
km = KMeans(n_clusters=4, n_init=10, random_state=42)
df["cluster"] = km.fit_predict(X)
summary = df.groupby("cluster")[["recency", "frequency", "monetary", "login_freq", "churn"]].mean()
print(summary.round(2))

# 7. 输出高风险用户清单
risk_users = df[df["churn"] == 1].sort_values("recency", ascending=False)
print(f"\n高流失风险用户数：{len(risk_users)}")
print("已保存预测结果到 predictions.csv")
df["churn_prob"] = best.predict_proba(sc.transform(X))[:, 1]
df[["user_id", "recency", "frequency", "monetary", "churn", "churn_prob"]].to_csv(
    "predictions.csv", index=False, encoding="utf-8-sig"
)
