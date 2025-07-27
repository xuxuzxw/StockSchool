# StockSchool 量化投研系统

## 项目简介

StockSchool是一个基于Python的量化投研系统，集成了数据获取、因子计算、策略评估、性能监控和告警等功能。本系统采用模块化设计，支持高度可配置化，适用于量化投资研究和策略开发。

## 主要特性

### 数据获取与同步
- **多数据源支持**: 集成Tushare和AkShare数据接口
- **全面数据覆盖**: 股票基本信息、日线行情、财务数据、市场情绪数据
- **增量同步**: 支持增量更新和断点续传
- **数据质量控制**: 自动数据清洗、异常值检测和缺失值处理

### 因子计算引擎
- **技术因子**: 20+种技术指标，包括动量、趋势、波动率、成交量因子
- **基本面因子**: 盈利能力、成长性、杠杆、流动性等财务指标
- **因子预处理**: 去极值、标准化、中性化等预处理功能
- **高性能计算**: 支持批量处理和并行计算

### AI模型训练与预测
- **多模型支持**: 线性回归、随机森林、XGBoost、LightGBM、神经网络
- **训练流水线**: 完整的模型训练、验证、保存流程
- **每日预测**: 自动化的每日股票预测脚本
- **模型解释**: 支持SHAP值计算和特征重要性分析
- **批量预测**: 支持历史时间段的批量预测和回测

### 全流程自动化调度 (v1.1.6)
- **日度工作流**: 完整的从数据同步到AI预测的自动化流程
- **任务编排**: 基于Celery的分布式任务调度系统
- **质量保证**: 全面的数据质量检查和流程监控
- **容错机制**: 完善的错误处理和任务重试机制
- **性能监控**: 实时任务状态监控和性能基线建立

### 系统特性
- **特征工程**: 完整的特征商店系统，支持特征定义、存储和检索
- **策略评估**: 15+种策略评估指标，包括收益率、风险指标、绩效比率等
- **监控告警**: 实时性能监控和多渠道告警系统
- **Web API**: 基于FastAPI的RESTful API接口
- **高度可配置**: 所有关键参数支持配置文件动态加载

## 技术栈

- **后端框架**: FastAPI
- **数据库**: MySQL/SQLite
- **数据源**: Tushare Pro API, AkShare
- **机器学习**: scikit-learn, XGBoost, LightGBM, SHAP
- **数据处理**: pandas, numpy
- **技术指标**: talib, 自研指标库
- **配置管理**: PyYAML
- **任务调度**: Celery分布式任务队列
- **日志系统**: Python logging
- **重试机制**: 自研重试装饰器

## 系统架构

```
StockSchool/
├── src/
│   ├── ai/            # AI模型训练与预测
│   │   ├── training_pipeline.py  # 模型训练流水线
│   │   └── prediction.py         # 每日预测脚本
│   ├── api/           # Web API接口
│   ├── compute/       # 计算引擎
│   │   ├── factor_engine.py      # 因子计算引擎
│   │   ├── indicators.py         # 技术指标和基本面指标
│   │   ├── processing.py         # 因子预处理
│   │   ├── quality.py            # 数据质量控制
│   │   └── tasks.py              # Celery任务定义
│   ├── data/          # 数据同步
│   │   ├── tushare_sync.py       # Tushare数据同步
│   │   └── akshare_sync.py       # AkShare数据同步
│   ├── database/      # 数据库相关
│   ├── features/      # 特征商店
│   ├── monitoring/    # 监控告警
│   ├── strategy/      # 策略评估与AI模型
│   │   ├── evaluation.py         # 策略评估
│   │   ├── model_explainer.py    # 模型解释器
│   │   └── ai_model.py           # AI模型训练与预测
│   ├── tests/         # 单元测试
│   └── utils/         # 工具函数
├── scripts/           # 工具脚本
│   ├── full_test_v1_1_6.py      # 全流程实测编排器
│   └── clear_database.py        # 数据库清理工具
├── run_daily_workflow.py         # 日度工作流调度脚本
├── config.yml         # 配置文件
├── database_schema.sql # 数据库结构
├── requirements.txt   # 依赖包
├── log.md            # 开发日志
└── README.md         # 项目说明
```

## 安装和配置

### 1. 环境要求

- Python 3.8+
- pip

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置文件

复制并修改配置文件：

```bash
cp config.yaml.example config.yaml
```

主要配置项说明：

```yaml
# Tushare配置
tushare:
  token: "your_tushare_token"  # 替换为你的Tushare token

# 数据库配置
database:
  path: "data/stock_data.db"

# API配置
api_params:
  host: "0.0.0.0"
  port: 8000
  reload: true
  log_level: "info"

# 其他参数配置...
```

### 4. 数据库初始化

首次运行前需要初始化数据库和同步基础数据：

```python
from src.data.tushare_sync import TushareSynchronizer
from src.utils.config_loader import Config

config = Config()
syncer = TushareSynchronizer(config)

# 同步基础数据
syncer.sync_stock_basic()
syncer.sync_trade_calendar()
```

## 使用方法

### 0. 运维和调试控制台 (v1.1.6新增)

系统提供了完整的运维和调试控制台，集成了日常运行、监控和故障排查功能：

```bash
# 启动运维控制台
python run.py
# 选择 "3. 运维和调试控制台"
```

**运维功能包括**：
- **飞行前检查** - 环境健康检查（Docker、数据库、API连通性）
- **Celery Worker启动** - 启动和监控分布式任务队列
- **日常工作流执行** - 触发完整的数据同步到AI预测流程
- **数据质量检验** - 实时数据质量监控和验证
- **数据修复和回填** - 自动检测和修复缺失数据
- **紧急情况诊断** - 故障排查和系统恢复

**测试验证**：
```bash
# 运行运维功能测试
python test/test_run_operations.py
```

### 1. 数据同步

#### 同步基础数据
```bash
# 同步股票基本信息
python -m src.data.tushare_sync --action sync_stock_basic

# 同步交易日历
python -m src.data.tushare_sync --action sync_trade_calendar

# 同步日线数据
python -m src.data.tushare_sync --action sync_daily --start-date 2024-01-01
```

#### 数据修复和回填 (v1.1.6新增)
```bash
# 运行数据修复脚本，检查并修复缺失的历史数据
python fix_data_sync.py

# 检查数据库状态
python check_database_status.py

# 验证数据完整性
python check_data.py
```



#### 同步情绪数据
```bash
# 同步市场情绪数据
python -m src.data.akshare_sync --data-type sentiment

# 同步资金流向数据
python -m src.data.akshare_sync --data-type money_flow
```

### 2. 因子计算

#### 计算技术因子
```bash
# 计算所有股票的技术因子
python -m src.compute.factor_engine --action calculate

# 计算指定股票的因子
python -m src.compute.factor_engine --action calculate --ts-code 000001.SZ
```

#### 因子预处理
```bash
# 执行因子预处理流程
python -m src.compute.processing --action process

# 计算因子相关性
python -m src.compute.processing --action correlation
```

### 3. AI模型训练与预测

#### 模型训练
```bash
# 训练股票预测模型
python -m src.ai.training_pipeline --model-type xgboost --target-days 5

# 使用自定义参数训练
python -m src.ai.training_pipeline --model-type random_forest --cv-folds 5 --test-size 0.2
```

#### 每日预测
```bash
# 运行每日预测
python -m src.ai.prediction

# 使用指定模型预测
python -m src.ai.prediction --model-path models/custom_model.pkl

# 预测指定日期
python -m src.ai.prediction --trade-date 2024-12-20
```

### 4. 全流程自动化调度 (v1.1.6)

#### 启动Celery Worker
```bash
# 启动Celery worker
celery -A src.compute.tasks worker --loglevel=info

# 启动Celery beat (定时任务调度)
celery -A src.compute.tasks beat --loglevel=info
```

#### 运行全流程测试
```bash
# 清空数据库（可选）
python scripts/clear_database.py

# 运行完整的日度工作流
python run_daily_workflow.py

# 运行简化测试工作流
python run_daily_workflow.py --simple
```

#### 全流程实测编排器
```bash
# 使用专门的实测编排器
python scripts/full_test_v1_1_6.py
```

#### 工作流阶段说明
1. **数据同步阶段**: 并行同步测试股票池的历史数据
2. **因子计算阶段**: 并行计算各股票的技术和基本面因子
3. **AI模型训练阶段**: 基于历史数据训练LightGBM预测模型
4. **AI预测阶段**: 使用训练好的模型进行批量预测
5. **数据质量检查阶段**: 全面的数据质量验证和报告

#### 测试配置
测试配置在`config.yml`的`full_test_config`节中定义：
```yaml
full_test_config:
  start_date: '2024-01-01'
  end_date: '2024-01-15'
  stock_pool: 
    - '000001.SZ'  # 平安银行
    - '000002.SZ'  # 万科A
    - '600000.SH'  # 浦发银行
    - '600036.SH'  # 招商银行
    - '000858.SZ'  # 五粮液
```

### 5. 启动API服务

```bash
python -m src.api.main
```

或者使用uvicorn：

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. API文档

启动服务后，访问以下地址查看API文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3. 主要API接口

#### 健康检查
```
GET /health
```

#### 股票基础信息
```
GET /api/stocks/basic?limit=100
```

#### 股票日线数据
```
GET /api/stocks/daily?ts_code=000001.SZ&start_date=20240101&end_date=20241231
```

#### 因子值查询
```
GET /api/factors/values?ts_code=000001.SZ&factor_name=rsi&start_date=20240101
```

#### 数据同步
```
POST /api/sync/stocks
POST /api/sync/calendar
POST /api/sync/daily
```

#### 因子计算
```
POST /api/compute/factors
```

#### 数据质量检查
```
GET /api/quality/check
```

#### 策略评估
```
POST /api/strategy/evaluate
```

#### 性能监控
```
GET /api/monitoring/metrics
GET /api/monitoring/alerts
```

## 核心模块说明

### 数据同步模块 (src/data/)

- **TushareSynchronizer**: 负责从Tushare获取和同步股票数据
- 支持增量更新和错误重试
- 可配置批处理大小和同步频率

### 计算引擎 (src/compute/)

- **FactorEngine**: 因子计算引擎，支持技术因子和基本面因子计算
- **Indicators**: 技术指标和基本面指标计算库，包含20+种技术指标和9种基本面指标
- **FactorProcessor**: 因子预处理器，提供去极值、标准化、中性化等功能
- **QualityController**: 数据质量控制，包括异常值检测和缺失值处理

### AI模块 (src/ai/)

- **ModelTrainingPipeline**: 模型训练流水线，支持多种机器学习模型
- **StockPredictor**: 股票预测器，实现每日预测和结果保存功能
- 支持模型类型：线性回归、随机森林、XGBoost、LightGBM、神经网络

### 特征商店 (src/features/)

- **FeatureStore**: 特征定义、存储和检索系统
- 支持特征统计信息计算和缓存机制

### 策略模块 (src/strategy/)

- **StrategyEvaluator**: 策略评估器，提供多种绩效指标
- **ModelExplainer**: 模型解释器，支持SHAP分析

### 监控模块 (src/monitoring/)

- **AlertEngine**: 告警引擎，支持多种通知渠道
- **PerformanceMonitor**: 性能监控器，实时收集系统指标

## 配置参数说明

系统支持高度可配置化，所有关键参数都可以通过`config.yaml`文件进行调整。主要配置分类：

- **api_params**: API服务相关参数
- **data_sync_params**: 数据同步参数
- **compute_params**: 计算引擎参数
- **feature_params**: 特征工程参数
- **strategy_params**: 策略评估参数
- **monitoring_params**: 监控告警参数
- **quality_params**: 质量控制参数
- **utils_params**: 工具函数参数

详细配置说明请参考`log.md`文件。

## 最近更新

- **v1.1.6 全流程实测阶段**：
  - **AI模型模块 (`ai_model.py`)**：实现了AI模型的训练、预测和管理。
  - **Celery任务扩展 (`tasks.py`)**：扩展了AI相关的Celery任务，包括 `train_ai_model`、`run_prediction` 和 `batch_prediction`。
  - **配置更新 (`config.yml`)**：新增 `full_test_config` 配置节，用于定义全流程测试的参数。
  - **全流程调度系统**：构建了主工作流调度脚本 (`run_daily_workflow.py`) 和全流程实测编排器 (`scripts/full_test_v1_1_6.py`)。
  - **AI模型能力**：支持多模型训练、批量预测和模型持久化。
  - **任务调度架构**：利用Celery的 `group`、`chain` 和 `chord` 功能进行任务编排。
  - **测试配置**：允许灵活设置测试日期范围和股票池。
  - **工作流阶段**：划分了数据同步、因子计算、AI模型训练、AI预测和数据质量检查等阶段。
  - **质量保证体系**：通过数据质量检查确保任务流执行一致性、输入数据完整性、中间产品覆盖率和最终成品时效性。
  - **相关工具和脚本**：创建了 `scripts/clear_database.py` 用于清空数据库。
  - **技术特性**：模块化设计、异步任务处理、可扩展性、容错机制和性能监控。
  - **部署准备**：支持Docker部署和Celery分布式任务队列。
  - **调试与验证**：
    - **`clear_database.py` 脚本运行问题**：
      - 初始错误：`NameError: name 'ConfigLoader' is not defined`
      - 修复：将 `scripts/clear_database.py` 中 `DatabaseCleaner` 类初始化方法中的 `Config()` 修改为 `Config()`。
      - 修复：`scripts/clear_database.py` 中数据库删除操作的执行方式问题，通过导入 `sqlalchemy.text` 并使用 `conn.execute(text(f"DELETE FROM {table_name}"))` 解决了 `Not an executable object` 错误。
    - **`run_daily_workflow.py` 脚本运行问题**：
      - 初始错误：`NameError: name 'ConfigLoader' is not defined`
      - 修复：将 `run_daily_workflow.py` 中 `ConfigLoader` 的导入和实例化错误，将其从 `ConfigLoader` 修改为 `Config`。
      - 修复：为 `run_daily_workflow.py` 添加 `.env` 文件加载机制，确保脚本能正确加载环境变量。

    - **调试结果**：
      - Celery worker 成功启动并连接到 Redis。
      - `clear_database.py` 脚本成功运行，数据库清空功能正常。
      - `run_daily_workflow.py` 脚本成功启动简化工作流。
  - **下一步计划**：进行全面的集成测试和性能优化。


### v1.1.5 真实数据测试完成 (2025年07月28日)

完成了基于真实数据的端到端集成测试，验证了系统的稳定性和准确性：

#### 测试环境配置
- ✅ **Docker测试环境**: 配置独立的PostgreSQL测试数据库
- ✅ **测试数据管理**: 实现测试数据的自动生成和质量控制
- ✅ **环境隔离**: 确保测试环境与生产环境完全隔离

#### 端到端集成测试
- ✅ **数据库连接测试**: 验证数据库连接和基础操作
- ✅ **完整流水线测试**: 从数据获取到因子计算的完整流程验证
- ✅ **因子计算精度验证**: 使用黄金数据验证RSI等技术指标的计算精度
- ✅ **数据质量验证**: 确保价格数据的逻辑一致性（开盘价在最高最低价之间等）
- ✅ **计算一致性测试**: 验证MACD、布林带等复合指标的计算稳定性

#### 技术指标修复
- ✅ **RSI计算优化**: 修复RSI计算逻辑，确保计算精度达到预期
- ✅ **参数类型统一**: 统一技术指标函数的输入参数类型（DataFrame vs Series）
- ✅ **返回值标准化**: 标准化复合指标（MACD、布林带）的返回值结构

#### 测试架构改进
- ✅ **模块化测试**: 将集成测试拆分为独立的测试模块
- ✅ **测试数据生成**: 实现符合真实市场规律的测试数据生成算法
- ✅ **断言逻辑优化**: 改进测试断言，适应真实市场数据的波动特性

**测试覆盖率**: 6个核心测试用例全部通过，覆盖数据库操作、因子计算、数据质量等关键功能模块。

### v1.1 核心功能升级 (2025年7月)

本次重大更新完成了StockSchool项目v1.1阶段的核心功能开发，实现了完整的量化投研流水线：

#### 数据补全与入库
- ✅ **数据库架构升级**: 新增`sentiment_data`、`factor_library`、`prediction_results`等核心表
- ✅ **多数据源集成**: 完善Tushare数据同步，新增AkShare情绪数据同步
- ✅ **配置参数化**: 所有数据同步参数支持配置文件管理

#### 核心因子计算
- ✅ **基本面因子**: 新增9种基本面指标计算（ROE、ROA、成长性、杠杆等）
- ✅ **因子预处理**: 实现去极值、标准化、中性化等预处理功能
- ✅ **统一因子库**: 技术因子和基本面因子统一存储管理

#### AI模型训练与预测
- ✅ **训练流水线**: 支持5种机器学习模型的完整训练流程
- ✅ **每日预测**: 自动化预测脚本，支持模型加载和结果保存
- ✅ **模型管理**: 模型版本控制和性能评估

#### 技术架构优化
- ✅ **模块化设计**: 清晰的代码结构和职责分离
- ✅ **错误处理**: 完善的异常捕获和重试机制
- ✅ **日志系统**: 详细的操作日志和性能监控
- ✅ **配置管理**: 统一的配置文件和参数管理

**核心特性**:
1. 端到端的量化投研流程
2. 多因子模型构建能力
3. 自动化预测和决策支持
4. 高度可配置和可扩展
5. 生产级的稳定性和性能

详细开发记录请查看 <mcfile name="log.md" path="d:\Users\xuxuz\Desktop\StockSchool\log.md"></mcfile>。

### 硬编码参数配置化 (2025年07月28日)

将项目中的硬编码参数全面替换为从配置文件动态加载，提高系统可配置性和灵活性。

## 开发指南

### 添加新的技术指标

1. 在`src/compute/technical.py`中添加指标计算函数
2. 在`src/compute/factor_engine.py`中注册新指标
3. 更新配置文件中的相关参数

### 添加新的评估指标

1. 在`src/strategy/evaluation.py`中添加指标计算方法
2. 更新API接口返回的指标列表

### 添加新的告警规则

1. 在`src/monitoring/alerts.py`中定义告警规则
2. 配置通知渠道和触发条件

## 测试

运行单元测试：

```bash
python -m pytest src/tests/ -v
```

运行特定测试：

```bash
python -m pytest src/tests/test_database.py -v
```

## 部署

### Docker部署

```bash
# 构建镜像
docker build -t stockschool .

# 运行容器
docker run -p 8000:8000 -v $(pwd)/config.yaml:/app/config.yaml stockschool
```

### 生产环境部署

1. 使用Gunicorn作为WSGI服务器
2. 配置Nginx作为反向代理
3. 使用Redis作为Celery消息队列
4. 配置日志轮转和监控告警

## 性能优化

- 合理设置批处理大小和并发数
- 使用数据库索引优化查询性能
- 启用特征缓存机制
- 定期清理历史数据

## 故障排除

### 常见问题

1. **Tushare API限制**: 检查token有效性和调用频率
2. **数据库锁定**: 检查并发访问和事务处理
3. **内存不足**: 调整批处理大小和缓存配置
4. **网络超时**: 增加重试次数和超时时间

### 日志查看

系统日志位于`logs/`目录下，按模块和日期分类存储。

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交代码变更
4. 编写测试用例
5. 提交Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

---

**注意**: 使用本系统前请确保已获得相应的数据使用许可，并遵守相关法律法规。投资有风险，策略仅供参考。