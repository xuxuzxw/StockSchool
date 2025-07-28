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

StockSchool 提供了一个统一的命令行控制台 `run.py`，用于管理和操作系统的各项功能。通过交互式菜单，您可以轻松启动API服务、执行数据同步、以及进行系统运维和调试。

### `run.py` 功能详解

`run.py` 是 StockSchool 项目的主入口点，提供了一个命令行界面，用于启动API服务器、运行数据同步、以及进行运维和调试。以下是其主要功能模块的详细介绍：

#### 1. 主菜单功能

-   **启动API服务器 (`start_api_server`)**：
    -   **作用**：启动 StockSchool 的后端 API 服务，提供数据查询、因子获取、模型预测等接口。
    -   **使用场景**：当您需要通过前端界面或外部应用访问 StockSchool 的数据和功能时，必须先启动此服务。

-   **运行数据同步 (`run_data_sync`)**：
    -   **作用**：执行数据同步流程，确保系统中的数据是最新的。
    -   **使用场景**：需要更新股票数据、财务数据等时使用。
    -   **数据同步子菜单 (`data_sync_menu`)**：
        -   **同步每日数据 (`sync_daily_data`)**：
            *   **作用**：从数据源（如 Tushare）同步最新的每日股票行情数据、财务数据等。
            *   **使用场景**：每日收盘后或在需要获取最新市场数据时执行。
        -   **同步单只股票数据 (`sync_stock_data`)**：
            *   **作用**：针对指定的单只股票，同步其历史数据或最新数据。
            *   **使用场景**：当某只股票数据缺失、需要更新特定股票数据或进行调试时使用。
        -   **回填历史数据 (`backfill_history_data`)**：
            *   **作用**：用于补全指定股票或所有股票的缺失历史数据。
            *   **使用场景**：系统首次部署、数据源切换或发现历史数据不完整时使用。

-   **运维和调试 (`operations_menu`)**：
    -   **作用**：进入一个子菜单，提供更高级的系统维护和故障诊断功能。
    -   **使用场景**：当系统出现问题、需要进行数据修复、性能分析或手动触发特定任务时使用。

#### 2. 运维和调试控制台菜单功能

-   **飞行前检查 (`pre_flight_check`)**：
    -   **作用**：在系统启动或关键操作前，执行一系列预检查，确保环境配置、数据库连接、依赖项等都处于正常状态。
    -   **使用场景**：部署新环境、更新系统版本或在系统行为异常时进行初步诊断。

-   **启动Celery Worker (`start_celery_worker`)**：
    -   **作用**：启动 Celery 后台任务处理器，负责执行数据同步、因子计算、数据质量检查等耗时任务。
    -   **使用场景**：为了确保系统数据持续更新和自动化任务的正常运行，Celery Worker 通常需要保持常驻。
    -   **常驻功能**：
        -   **每日数据同步**：通过 `sync_daily_data` 任务，确保股票、指数等每日行情数据及时更新。
        -   **每日因子计算**：通过 `calculate_daily_factors` 任务，对最新数据进行处理，生成用于模型训练和策略分析的各种技术和基本面因子。
        -   **每周数据质量检查**：通过 `stockschool.weekly_quality_check` 任务，定期检查数据的完整性、准确性和一致性，确保数据质量。
        -   **每月全量因子重算**：通过 `stockschool.monthly_full_factor_recalculation` 任务，定期对所有历史数据进行因子重算，以应对数据修正或算法更新。
        -   **按需触发任务**：例如 `sync_stock_data`（同步单只股票数据）和 `calculate_stock_factors`（计算单只股票因子），用于数据补全或特定股票的分析。

-   **运行日常工作流 (`run_daily_workflow`)**：
    -   **作用**：执行每日例行的数据处理和分析流程，包括数据同步、因子计算、模型预测等。
    -   **使用场景**：每日收盘后或特定时间点，用于自动化更新系统数据和分析结果。

-   **数据质量检验 (`data_quality_check`)**：
    -   **作用**：对数据库中的数据进行深度检查，识别并报告潜在的数据质量问题，如缺失值、异常值、不一致性等。
    -   **使用场景**：定期维护或在发现数据异常时进行详细排查。

-   **数据修复和回填 (`fix_data_sync`)**：
    -   **作用**：针对数据质量问题或历史数据缺失，提供修复和回填机制，确保数据的完整性和准确性。
    -   **使用场景**：当发现某段时间的数据有误或缺失时，用于纠正和补充。

-   **紧急情况诊断 (`emergency_diagnosis`)**：
    -   **作用**：提供一系列工具和诊断流程，用于在系统发生严重故障或性能问题时，快速定位问题根源。
    -   **使用场景**：系统崩溃、响应缓慢或数据处理停滞等紧急情况。

通过这些功能，`run.py` 使得 StockSchool 系统的管理和维护变得更加高效和便捷。

### 0. 主控制台 `run.py`

运行 `run.py` 将启动主控制台，提供以下核心功能：

```bash
python run.py
```

**主菜单功能详解**：

*   **1. 启动API服务器**：
    *   **作用**：启动 StockSchool 的 FastAPI 后端服务，提供Web界面和API接口。
    *   **何时使用**：需要访问系统Web界面、进行数据查询或与其他应用集成时。

*   **2. 运行数据同步**：
    *   **作用**：从 Tushare 等数据源获取最新的金融数据（如股票基本信息、交易日历、日线行情等），并存储到数据库中。会提示您选择具体的同步类型。
    *   **何时使用**：更新系统数据，确保数据最新，或首次部署时填充数据。

*   **3. 运维和调试控制台**：
    *   **作用**：进入一个子菜单，包含系统运维和故障诊断的各种工具。
    *   **何时使用**：检查系统健康、处理后台任务、数据质量检查、修复数据或诊断故障时。

*   **4. 退出**：
    *   **作用**：安全地退出 `run.py` 程序。
    *   **何时使用**：完成所有操作，需要关闭控制台时。

### 0.1 运维和调试控制台 (v1.1.6新增)

系统提供了完整的运维和调试控制台，集成了日常运行、监控和故障排查功能。在主菜单选择“3. 运维和调试控制台”后，您将进入此菜单：

```bash
# 启动运维控制台
python run.py
# 选择 "3. 运维和调试控制台"
```

**运维功能包括**：

*   **1. 飞行前检查 (Pre-Flight Check)**：
    *   **作用**：执行环境健康检查，包括 Docker 服务状态、环境变量配置、数据库连接以及 Tushare API 连通性。
    *   **何时使用**：首次部署、系统升级或遇到问题时，作为第一步快速诊断系统环境。

*   **2. 启动Celery Worker**：
    *   **作用**：启动 Celery 后台任务处理器，负责执行耗时较长的异步任务（如复杂数据计算、模型训练）。
    *   **何时使用**：需要系统执行后台任务，或确保所有组件正常运行时。

*   **3. 运行日常工作流**：
    *   **作用**：执行预定义的日常数据处理流水线，通常包括数据同步、因子计算、模型预测等自动化步骤。
    *   **何时使用**：每天或定期运行，以自动化数据更新和分析流程。

*   **4. 数据质量检验**：
    *   **作用**：检查数据库中数据的完整性、一致性和准确性。
    *   **何时使用**：定期进行，或在数据同步后验证数据质量。

*   **5. 数据修复和回填**：
    *   **作用**：处理数据缺失或错误的情况，可以回填历史数据或修复损坏的数据。
    *   **何时使用**：数据质量检查发现问题，或需要补充特定时间段的数据时。

*   **6. 紧急情况诊断**：
    *   **作用**：提供一个子菜单，用于针对特定问题（如数据同步失败、数据库连接问题、API服务异常、系统资源检查）进行快速诊断。
    *   **何时使用**：系统出现故障，需要快速定位问题根源时。

*   **7. 返回主菜单**：
    *   **作用**：返回到 `run.py` 的主菜单。
    *   **何时使用**：完成运维操作后，回到主界面进行其他操作。

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