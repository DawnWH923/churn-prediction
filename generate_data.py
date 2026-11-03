"""
generate_data.py — 生成模拟电商用户流失数据集
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

# RFM 特征
recency = np.random.exponential(scale=30, size=n)  # 距上次购买天数
frequency = np.random.poisson(lam=5, size=n)  # 购买次数
monetary = np.abs(np.random.normal(loc=500, scale=300, size=n))  # 消费金额

# 行为特征
login_freq = np.random.poisson(lam=10, size=n)  # 月登录次数
browse_depth = np.random.randint(1, 50, size=n)  # 浏览深度
addcart_rate = np.random.uniform(0, 0.5, size=n)  # 加购率
customer_service = np.random.poisson(lam=1, size=n)  # 客服咨询次数
promo_sensitive = np.random.uniform(0, 1, size=n)  # 促销敏感度

# 流失标签：recency 越大、login_freq 越低、addcart_rate 越低 → 越可能流失
churn_prob = (
    0.03 * recency
    - 0.15 * login_freq
    - 1.0 * addcart_rate
    - 0.002 * monetary
    + np.random.normal(0, 1.2, size=n)
)
churn = (churn_prob > 0.3).astype(int)

df = pd.DataFrame({
    "user_id": range(1, n + 1),
    "recency": recency.round(1),
    "frequency": frequency,
    "monetary": monetary.round(2),
    "login_freq": login_freq,
    "browse_depth": browse_depth,
    "addcart_rate": addcart_rate.round(3),
    "customer_service": customer_service,
    "promo_sensitive": promo_sensitive.round(2),
    "churn": churn,
})

df.to_csv("data/ecommerce_churn.csv", index=False, encoding="utf-8-sig")
print(f"生成 {len(df)} 条数据，流失率 {df['churn'].mean():.1%}")
print(df.head())
