# 电商用户流失预测与分群分析

基于 RFM 特征与机器学习模型，预测电商用户流失概率，并通过 K-Means 聚类做用户分群，支撑精细化运营。

## 项目结果
- XGBoost 测试集 **AUC = 0.85**
- 对比逻辑回归（0.88）、随机森林（0.87）
- K-Means 将用户分为 4 类，流失风险最高群体占比 16%

## 技术栈
- Python / pandas / NumPy
- scikit-learn（逻辑回归、随机森林、K-Means）
- XGBoost
- 特征工程：RFM、用户行为特征

## 快速开始
```bash
pip install pandas numpy scikit-learn xgboost
python generate_data.py    # 生成模拟数据
python churn_prediction.py # 训练模型并输出结果
```

## 项目结构
```
churn-prediction/
├── data/
│   └── ecommerce_churn.csv   # 数据集
├── generate_data.py          # 数据生成
├── churn_prediction.py      # 模型训练与分群
├── predictions.csv           # 预测结果
└── README.md
```

## 特征说明
| 特征 | 含义 |
|------|------|
| recency | 距上次购买天数 |
| frequency | 购买次数 |
| monetary | 累计消费金额 |
| login_freq | 月登录次数 |
| browse_depth | 浏览深度 |
| addcart_rate | 加购率 |
| customer_service | 客服咨询次数 |
| promo_sensitive | 促销敏感度 |

## 作者
吴晗 | [GitHub](https://github.com/DawnWH923)
