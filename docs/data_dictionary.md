# 数据字典（data/ecommerce_churn.csv）

来源：`generate_data.py`，`np.random.seed(42)`，5000 行。分布假设贴近真实电商业务（README「数据说明」）。

| 字段 | 类型 | 分布/含义 |
|------|------|----------|
| user_id | int | 用户编号 1-5000 |
| recency | float | 距上次购买天数，指数分布 λ=30 |
| frequency | int | 购买次数，泊松分布 λ=5 |
| monetary | float | 累计消费金额，截断正态 μ=500 σ=300 |
| login_freq | int | 月登录次数，泊松分布 λ=10 |
| browse_depth | int | 浏览深度，均匀 1-50 |
| addcart_rate | float | 加购率，均匀 0-0.5 |
| customer_service | int | 客服咨询次数，泊松分布 λ=1 |
| promo_sensitive | float | 促销敏感度，均匀 0-1 |
| churn | int | **流失标签（构造逻辑见下）** |

## 流失标签构造

```python
churn_prob = 0.03*recency - 0.15*login_freq - 1.0*addcart_rate - 0.002*monetary + N(0, 1.2)
churn = churn_prob > 0.3
```

业务含义：**越久没买、登录越少、加购越少 → 越可能流失**，消费金额有轻微保护作用，再叠加随机噪声。替换真实数据时保持"标签=业务口径"这一原则即可，建模代码无需改动。
