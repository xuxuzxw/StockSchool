# AI策略系统设计文档

## 1. 系统概述

AI策略系统是StockSchool的核心智能决策模块，扮演"校长"角色，通过机器学习模型学习最优因子权重组合，提供个性化投资策略建议。系统具备模型解释性、策略评估和实时预测能力。

### 1.1 设计目标

- **智能化决策**：基于AI模型自动学习因子权重，提供数据驱动的投资建议
- **个性化服务**：支持用户自定义投资偏好，提供个性化策略定制
- **可解释性**：提供模型决策过程的详细解释，满足合规要求
- **高性能**：支持实时预测和高并发访问
- **可扩展性**：模块化设计，支持功能扩展和水平扩展

### 1.2 核心功能

1. AI模型训练和管理
2. 因子权重学习机制
3. 股票评分和排名系统
4. 个性化策略定制
5. 模型解释性分析
6. 策略回测和评估
7. 在线预测服务
8. 模型监控和管理

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        API Gateway                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                    Load Balancer                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                  FastAPI 服务集群                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ Prediction  │ │ Strategy    │ │ Backtest    │           │
│  │ Service     │ │ Service     │ │ Service     │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                    Redis 缓存层                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                 PostgreSQL 数据库                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Celery 任务队列                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ Model       │ │ Monitor     │ │ Backtest    │           │
│  │ Training    │ │ Tasks       │ │ Tasks       │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心模块

#### 2.2.1 AIModelManager (AI模型管理器)
- **功能**：模型训练、版本管理、A/B测试
- **基础**：扩展现有的ModelTrainingPipeline
- **新增功能**：
  - 模型版本控制和回滚
  - 自动重训练触发
  - 模型性能对比
  - A/B测试框架

#### 2.2.2 FactorWeightEngine (因子权重引擎)
- **功能**：学习和管理因子权重
- **核心算法**：
  - SHAP值计算
  - 排列重要性
  - 动态权重调整
  - 权重稳定性控制

#### 2.2.3 StockScoringEngine (股票评分引擎)
- **功能**：股票评分和排名
- **特性**：
  - 基于因子值和权重计算综合评分
  - 支持行业中性化调整
  - 实时评分更新
  - 历史评分追踪

#### 2.2.4 StrategyCustomizer (策略定制器)
- **功能**：个性化策略定制
- **特性**：
  - 自然语言偏好解析
  - 量化参数转换
  - 策略配置管理
  - 风险偏好适配

#### 2.2.5 ModelExplainer (模型解释器)
- **功能**：模型解释性分析
- **工具集成**：
  - SHAP (SHapley Additive exPlanations)
  - LIME (Local Interpretable Model-agnostic Explanations)
  - 特征重要性分析
  - 决策路径可视化

#### 2.2.6 BacktestEngine (回测引擎)
- **功能**：策略回测和评估
- **特性**：
  - 多种回测模式
  - 风险收益指标计算
  - 归因分析
  - 交易成本考虑

#### 2.2.7 PredictionService (预测服务)
- **功能**：实时预测API
- **特性**：
  - 高并发支持
  - 结果缓存
  - 批量预测
  - 热更新支持

#### 2.2.8 ModelMonitor (模型监控器)
- **功能**：模型性能监控
- **特性**：
  - 性能漂移检测
  - 自动告警
  - 重训练触发
  - 监控指标记录

## 3. 数据库设计

### 3.1 新增表结构

#### 3.1.1 ai_models (AI模型表)
```sql
CREATE TABLE ai_models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    file_path TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    metrics JSONB,
    hyperparameters JSONB,
    feature_columns TEXT[],
    training_start_date DATE,
    training_end_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(50),
    description TEXT
);
```

#### 3.1.2 factor_weights (因子权重表)
```sql
CREATE TABLE factor_weights (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ai_models(id),
    factor_name VARCHAR(100) NOT NULL,
    weight DECIMAL(10, 6) NOT NULL,
    importance_score DECIMAL(10, 6),
    calculation_method VARCHAR(50),
    date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 3.1.3 stock_scores (股票评分表)
```sql
CREATE TABLE stock_scores (
    id SERIAL PRIMARY KEY,
    stock_code VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    model_id INTEGER REFERENCES ai_models(id),
    raw_score DECIMAL(10, 6) NOT NULL,
    normalized_score DECIMAL(10, 6) NOT NULL,
    rank_overall INTEGER,
    rank_industry INTEGER,
    industry_code VARCHAR(20),
    prediction_confidence DECIMAL(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 3.1.4 user_strategies (用户策略表)
```sql
CREATE TABLE user_strategies (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    description TEXT,
    preferences JSONB NOT NULL,
    parameters JSONB NOT NULL,
    model_id INTEGER REFERENCES ai_models(id),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 3.1.5 backtest_results (回测结果表)
```sql
CREATE TABLE backtest_results (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES user_strategies(id),
    backtest_name VARCHAR(100) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(15, 2),
    final_capital DECIMAL(15, 2),
    total_return DECIMAL(10, 6),
    annual_return DECIMAL(10, 6),
    max_drawdown DECIMAL(10, 6),
    sharpe_ratio DECIMAL(10, 6),
    volatility DECIMAL(10, 6),
    win_rate DECIMAL(5, 4),
    trade_count INTEGER,
    metrics JSONB,
    trades JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 3.1.6 model_monitoring (模型监控表)
```sql
CREATE TABLE model_monitoring (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ai_models(id),
    monitor_date DATE NOT NULL,
    accuracy_score DECIMAL(10, 6),
    drift_score DECIMAL(10, 6),
    prediction_count INTEGER,
    error_rate DECIMAL(5, 4),
    latency_avg DECIMAL(10, 3),
    status VARCHAR(20),
    alerts JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3.2 索引设计
```sql
-- ai_models 索引
CREATE INDEX idx_ai_models_status ON ai_models(status);
CREATE INDEX idx_ai_models_type_version ON ai_models(model_type, version);

-- factor_weights 索引
CREATE INDEX idx_factor_weights_model_date ON factor_weights(model_id, date);
CREATE INDEX idx_factor_weights_factor_name ON factor_weights(factor_name);

-- stock_scores 索引
CREATE INDEX idx_stock_scores_stock_date ON stock_scores(stock_code, trade_date);
CREATE INDEX idx_stock_scores_date_rank ON stock_scores(trade_date, rank_overall);
CREATE INDEX idx_stock_scores_model_date ON stock_scores(model_id, trade_date);

-- user_strategies 索引
CREATE INDEX idx_user_strategies_user_status ON user_strategies(user_id, status);

-- backtest_results 索引
CREATE INDEX idx_backtest_results_strategy ON backtest_results(strategy_id);
CREATE INDEX idx_backtest_results_date_range ON backtest_results(start_date, end_date);

-- model_monitoring 索引
CREATE INDEX idx_model_monitoring_model_date ON model_monitoring(model_id, monitor_date);
```

## 4. API设计

### 4.1 RESTful API 端点

#### 4.1.1 模型管理 API
```
GET    /api/v1/models                    # 获取模型列表
POST   /api/v1/models                    # 创建新模型
GET    /api/v1/models/{model_id}         # 获取模型详情
PUT    /api/v1/models/{model_id}         # 更新模型
DELETE /api/v1/models/{model_id}         # 删除模型
POST   /api/v1/models/{model_id}/train   # 训练模型
POST   /api/v1/models/{model_id}/deploy  # 部署模型
```

#### 4.1.2 预测服务 API
```
POST   /api/v1/predict/single            # 单只股票预测
POST   /api/v1/predict/batch             # 批量股票预测
GET    /api/v1/predict/scores            # 获取股票评分
GET    /api/v1/predict/rankings          # 获取股票排名
```

#### 4.1.3 策略管理 API
```
GET    /api/v1/strategies                # 获取策略列表
POST   /api/v1/strategies                # 创建策略
GET    /api/v1/strategies/{strategy_id}  # 获取策略详情
PUT    /api/v1/strategies/{strategy_id}  # 更新策略
DELETE /api/v1/strategies/{strategy_id} # 删除策略
```

#### 4.1.4 回测服务 API
```
POST   /api/v1/backtest                  # 执行回测
GET    /api/v1/backtest/{backtest_id}    # 获取回测结果
GET    /api/v1/backtest/history          # 获取回测历史
```

#### 4.1.5 模型解释 API
```
GET    /api/v1/explain/model/{model_id}  # 模型全局解释
POST   /api/v1/explain/prediction        # 预测结果解释
GET    /api/v1/explain/features          # 特征重要性
```

### 4.2 API 响应格式
```json
{
  "code": 200,
  "message": "success",
  "data": {},
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "uuid"
}
```

## 5. 技术栈

### 5.1 核心技术
- **编程语言**：Python 3.11+
- **Web框架**：FastAPI
- **机器学习**：scikit-learn, LightGBM, XGBoost
- **模型解释**：SHAP, LIME
- **数据处理**：pandas, numpy
- **数据库**：PostgreSQL
- **缓存**：Redis
- **任务队列**：Celery
- **容器化**：Docker

### 5.2 监控和运维
- **监控**：Prometheus + Grafana
- **日志**：structlog
- **配置管理**：pydantic-settings
- **API文档**：OpenAPI/Swagger

## 6. 实施计划

### 6.1 Phase 1: 核心功能 (4周)
1. **Week 1-2**: 扩展AIModelManager，实现模型版本管理
2. **Week 2-3**: 实现FactorWeightEngine，支持SHAP值计算
3. **Week 3-4**: 开发StockScoringEngine和PredictionService
4. **Week 4**: 创建基础API接口和数据库表

### 6.2 Phase 2: 高级功能 (4周)
1. **Week 5-6**: 实现ModelExplainer，提供模型解释功能
2. **Week 6-7**: 开发BacktestEngine，支持策略回测
3. **Week 7-8**: 创建StrategyCustomizer，支持个性化定制
4. **Week 8**: 完善API接口和前端集成

### 6.3 Phase 3: 运维优化 (2周)
1. **Week 9**: 实现ModelMonitor，完善监控告警
2. **Week 10**: 性能优化，文档完善，测试覆盖

## 7. 风险和挑战

### 7.1 技术风险
- **模型性能**：需要持续监控和优化模型准确性
- **数据质量**：依赖高质量的因子数据
- **系统性能**：需要优化预测服务的响应时间

### 7.2 业务风险
- **市场变化**：需要适应市场环境的变化
- **合规要求**：需要满足金融监管的合规要求
- **用户接受度**：需要提供易用的界面和解释

### 7.3 缓解措施
- 建立完善的测试和监控体系
- 实施渐进式部署和A/B测试
- 提供详细的模型解释和风险提示
- 建立用户反馈和持续改进机制

## 8. 总结

AI策略系统将为StockSchool提供强大的智能决策能力，通过模块化设计确保系统的可扩展性和可维护性。系统将分阶段实施，优先实现核心预测功能，然后逐步完善高级功能和运维能力。