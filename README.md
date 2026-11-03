# 电商用户流失预测与分群分析

基于 RFM 特征与机器学习模型，预测电商用户流失概率，并通过 K-Means 聚类做用户分群，支撑精细化运营。

## 数据说明（诚实声明）
本项目使用 **模拟数据集**（`generate_data.py` 按真实电商业务分布假设生成 5000 条用户数据）：
- RFM 特征：recency 服从指数分布（λ=30天）、frequency 服从泊松分布（λ=5次）、monetary 服从截断正态（μ=500元）
- 流失标签按业务逻辑构造：`recency` 越大、`login_freq` 越低、`addcart_rate` 越低越可能流失，叠加随机噪声
- 替换为真实数据只需将 `data/ecommerce_churn.csv` 换成同结构数据，无需改动建模代码

用模拟数据的目的是完整演练**特征工程 → 模型对比 → 评估 → 分群 → 业务输出**全链路。

## 项目结果
- XGBoost 测试集 **AUC = 0.85**（实测 0.847）
- 对比逻辑回归（0.88）、随机森林（0.87）：模拟数据线性关系强，LR 占优属正常现象；真实场景 XGBoost 通常更好
- K-Means 将用户分为 4 类：高价值活跃 / 高价值沉默 / 价格敏感 / 低价值流失

## 技术栈
- Python / pandas / NumPy
- scikit-learn（逻辑回归、随机森林、K-Means）
- XGBoost
- 特征工程：RFM、用户行为特征

## 快速开始
```bash
pip install -r requirements.txt
python generate_data.py      # 生成模拟数据（5000 条）
python churn_prediction.py   # 脚本版：训练模型并输出结果
# 或
jupyter notebook churn_analysis.ipynb   # Notebook 版：分步讲解 + 可视化
```

## 项目结构
```
churn-prediction/
├── data/
│   └── ecommerce_churn.csv   # 数据集（模拟生成）
├── generate_data.py          # 数据生成（分布假设见上）
├── churn_prediction.py       # 脚本版：训练与分群
├── churn_analysis.ipynb      # Notebook 版：分步讲解
├── predictions.csv           # 预测结果（高风险用户清单）
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
