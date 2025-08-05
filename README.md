# StockSchool 量化投研系统

## 📚 完整文档

### 📖 用户文档
- [📖 用户手册](./docs/user_manual.md) - 详细的系统使用指南
- [🔧 部署指南](./docs/deployment_guide.md) - 生产环境部署和维护指南
- [📊 API文档](./docs/api_documentation.md) - 完整的API接口文档

## 项目简介

StockSchool是一个基于Python的量化投研系统，集成了数据获取、因子计算、策略评估、性能监控和告警等功能。本系统采用模块化设计，支持高度可配置化，适用于量化投资研究和策略开发。

## 主要特性

### 数据获取与同步
- **多数据源支持**: 集成Tushare和AkShare数据接口
- **全面数据覆盖**: 股票基本信息、日线行情、财务数据、市场情绪数据
- **增量同步**: 支持增量更新和断点续传
- **数据质量控制**: 自动数据清洗、异常值检测和缺失值处理
- **财务数据批量插入优化**: 将财务数据同步从逐条插入优化为批量插入，显著提高数据同步性能，减少数据库交互次数，降低网络延迟，提高吞吐量，并通过数据库约束和事务机制确保数据一致性。
  - **性能基准测试**: 集成 `timeit` 装饰器，量化批量插入带来的效率提升。
  - **自动降级**: 实现批量操作失败时自动回退到逐条插入模式的逻辑，确保数据同步的鲁棒性。
  - **事务支持**: 批量操作支持事务，确保数据一致性。

### 因子计算引擎
- **技术因子**: 30+种技术指标，包括动量、趋势、波动率、成交量因子
- **基本面因子**: 盈利能力、成长性、杠杆、流动性等财务指标
- **情绪面因子**: 资金流向、关注度、情绪强度等市场情绪指标
  - **完整测试覆盖**: 情绪因子引擎已通过36个单元测试，包括资金流向、关注度、情绪强度和事件类因子
  - **抽象方法实现**: 所有因子计算器都正确实现了抽象基类的必需方法
  - **兼容性修复**: 修复了pandas pct_change方法的弃用警告，确保代码向前兼容
- **因子预处理**: 去极值、标准化、中性化等预处理功能
- **高性能计算**: 支持批量处理和并行计算
- **统一架构**: 基于抽象基类的可扩展因子计算框架
- **任务调度**: 支持优先级队列和多线程并行计算的调度器
- **因子宽表**: 优化的时序数据存储结构，支持高效查询

### AI模型训练与预测
- **多模型支持**: 线性回归、随机森林、XGBoost、LightGBM、神经网络
- **训练流水线**: 完整的模型训练、验证、保存流程
- **每日预测**: 自动化的每日股票预测脚本
- **模型解释**: 支持SHAP值计算和特征重要性分析
## 模型解释性功能
- **特征重要性分析**：支持默认/排列/SHAP三种方法
- **单样本解释**：提供SHAP值分解
- **特征交互分析**：分析特征间交互作用
- **可视化展示**：包括特征重要性图、SHAP总结图、预测解释图
- **性能基准测试**：提供完整的性能验证框架
- **批量预测**: 支持历史时间段的批量预测和回测

### 统一配置管理系统
- **多环境支持**: 支持开发(development)、测试(testing)、生产(production)等多环境配置
- **分层配置架构**: 
  - `config/base.yml` - 基础配置
  - `config/factor_config.yaml` - 因子计算专用配置
  - `config/monitoring.yaml` - 监控告警配置
  - `config/development.yml` - 开发环境配置
  - `config/production.yml` - 生产环境配置
  - `config/testing.yml` - 测试环境配置
- **配置热更新**: 支持运行时配置更新，无需重启服务
- **配置验证**: 内置配置验证机制，确保配置正确性
- **变更历史**: 记录配置变更历史，支持回滚操作
- **向后兼容**: 完全兼容旧版配置接口，平滑迁移
- **降级机制**: 配置系统故障时自动降级到简单模式
- **迁移工具**: 提供自动迁移工具，批量更新项目中的配置引用

### 4. 全流程自动化调度
- **Celery异步任务**: 利用Celery实现数据同步、因子计算、模型训练和预测等任务的异步处理和调度，提高系统响应速度和并发能力。
- **任务编排**: 支持复杂的任务链和任务组，确保数据处理流程的顺序性和完整性。
- **错误处理与重试**: 内置健壮的错误处理机制和任务重试策略，提高系统稳定性。

### 5. 持续集成/持续部署 (CI/CD)
- **自动化测试**: 集成 `scripts/full_test_v1_1_6.py` 全流程测试脚本到 CI/CD 流程中，确保每次代码提交都能自动进行数据同步、因子计算、AI模型训练和预测的完整性验证。
- **GitHub Actions**: 利用 GitHub Actions 实现自动化构建、测试和部署，提高开发效率和代码质量。
- **持续稳定性验证**: 确保系统在持续迭代过程中保持稳定性和可靠性。

### 6. 单元测试覆盖
- **监控模块测试**: `src/tests/test_alerts.py` 覆盖告警系统核心功能，测试覆盖率96%
- **策略模块测试**: `src/tests/test_ai_model.py` 覆盖AI模型训练和预测功能，测试覆盖率64%
- **全面测试**: 总计67个测试用例，确保系统各模块功能正确性

### 知识图谱系统
- **核心功能**: 代码知识图谱，用于可视化和分析代码结构、类与方法关系
- **实体类型**: 包含类(Class)、方法(Method)和因子(Factor)等实体
- **关系网络**: 支持contains(包含)、uses(使用)等关系类型
- **数据质量**: 定期审查与整理，确保实体命名规范统一，关系网络准确反映代码结构
- **最新改进**: 
  - 完成全面的数据质量审查与整理，统一实体类型，规范命名格式
  - 修复关系网络，准确反映代码结构
- 为 `ModelTrainingPipeline` 类及其所有相关方法（`__init__`, `prepare_training_data`, `_calculate_target_returns`, `_clean_training_data`, `train_model`, `_get_default_model_params`, `_calculate_metrics`, `_get_feature_importance`, `save_model`, `load_model`, `run_training_pipeline`, `_select_best_model`）创建了规范的实体和关系，并添加了详细的观察值，确保了代码功能、特性和实现细节的全面记录。
- 为 `StockPredictor` 类及其所有相关方法（包括 `__init__`、`load_model`、`get_latest_factors`、`prepare_prediction_data`、`make_predictions`、`save_predictions`、`get_stock_info`、`generate_prediction_report` 和 `run_daily_prediction`）以及 `main` 函数创建了规范的实体和关系，并添加了详细的功能描述观察值，以全面记录其功能、特性和实现细节。
  - 清理了潜在的瞬时数据和冗余内容，提高了图谱的可用性和维护性

### 系统特性
- **特征工程**: 完整的特征商店系统，支持特征定义、存储和检索
- **策略评估**: 15+种策略评估指标，包括收益率、风险指标、绩效比率等
- **监控告警**: 实时性能监控和多渠道告警系统
- **Web API**: 基于FastAPI的RESTful API接口
- **高度可配置**: 所有关键参数支持配置文件动态加载
- **代码质量管理**: 渐进式标准化策略，自动化语法检查，知识图谱代码分析


## 🚀 第二阶段优化 (v2.0.0)

### 性能优化
- **并行计算引擎**: 多进程并行因子计算，支持CPU/GPU加速
- **智能缓存系统**: Redis+内存多级缓存，LRU淘汰策略
- **负载均衡**: Nginx反向代理，支持多实例部署
- **批处理优化**: 智能批量大小计算，内存自适应管理

### 监控增强
- **实时监控**: Prometheus+Grafana监控栈
- **数据质量监控**: 实时数据质量检查和告警
- **性能仪表板**: 可视化系统性能指标
- **智能告警**: 多渠道告警通知系统

### 架构优化
- **设计模式**: 依赖注入、观察者模式、工厂模式
- **模块化重构**: 单一职责原则，降低耦合度
- **容错机制**: 重试策略、熔断降级、健康检查
- **配置管理**: 集中式配置管理，支持热更新

### 部署优化
- **容器化部署**: Docker+Docker Compose完整方案
- **负载均衡**: 多实例负载均衡配置
- **自动扩展**: 基于CPU/内存的自动扩展
- **健康检查**: 多层次健康检查和自愈机制

### 快速开始 (第二阶段)
```bash
# 使用Docker快速部署
docker-compose -f docker-compose.stage2.yml up -d

# 运行性能测试
python scripts/performance_test_runner.py

# 查看监控仪表板
open http://localhost:3000
```

### 配置文件
- `stage2_optimization_config.yml` - 优化配置
- `docker-compose.stage2.yml` - 容器编排
- `Dockerfile.stage2` - 生产镜像
- `requirements-stage2.txt` - 优化依赖

## 技术栈

- **后端框架**: FastAPI
- **数据库**: PostgreSQL 12+ (主数据库) / SQLite (降级方案)
- **数据源**: Tushare Pro API, AkShare
- **机器学习**: scikit-learn, XGBoost, LightGBM, SHAP
- **数据处理**: pandas, numpy
- **技术指标**: talib, 自研指标库
- **配置管理**: PyYAML + 热更新机制
- **任务调度**: Celery分布式任务队列 + schedule库
- **日志系统**: Python logging
- **重试机制**: 自研重试装饰器
- **代码质量**: 自动化语法检查 + 知识图谱分析

## 硬件加速要求
- **GPU支持**：需安装CUDA 11.7+驱动
- **内存建议**：至少32GB RAM（推荐48GB）
- **CPU要求**：支持多线程处理（推荐12核以上）
- **依赖版本**：
  - PyTorch 2.0.1+cu117
  - SHAP 0.43.0
  - mmap2 0.1.0
  - pynvml (可选，用于GPU监控)

## GPU工具模块
StockSchool提供完整的GPU工具模块，支持：
- **GPU可用性检测**：自动检测CUDA可用性并选择最优计算设备
- **动态批量大小计算**：根据可用内存自动调整批量大小
- **GPU内存监控**：实时监控GPU内存使用情况
- **自动降级策略**：内存不足时自动降级到CPU模式
- **内存不足处理**：智能处理OOM情况并自动重试

### 核心功能

#### 1. GPU管理器 (GPUManager)
```python
from src.utils.gpu_utils import GPUManager

# 创建GPU管理器实例
gpu_manager = GPUManager()

# 检查GPU可用性
if gpu_manager.is_gpu_available():
    print("GPU可用")
else:
    print("使用CPU模式")

# 获取设备信息
device = gpu_manager.get_device()
print(f"当前设备: {device}")

# 获取详细GPU信息
gpu_info = gpu_manager.get_gpu_info()
print(f"GPU信息: {gpu_info}")
```

#### 2. 动态批量大小计算
```python
# 获取最优批量大小
batch_size = gpu_manager.get_optimal_batch_size()
print(f"推荐批量大小: {batch_size}")

# 根据数据大小调整批量大小
batch_size = gpu_manager.get_optimal_batch_size(data_size=10000)
```

#### 3. 内存监控与检查
```python
# 检查内存是否足够
sufficient = gpu_manager.check_memory_sufficient(required_memory=1000)  # 1000MB
print(f"内存是否足够: {sufficient}")

# 处理内存不足情况
if not sufficient:
    should_retry = gpu_manager.handle_oom(retry_count=0, max_retries=3)
```

#### 4. 自动降级策略
```python
# 当GPU不可用或内存不足时自动降级到CPU
cpu_device = gpu_manager.fallback_to_cpu()
print(f"已降级到CPU: {cpu_device}")
```

### 便捷函数
```python
from src.utils.gpu_utils import (
    get_device, is_gpu_available, get_gpu_info,
    get_batch_size, check_memory_sufficient,
    handle_oom, fallback_to_cpu
)

# 便捷函数使用
device = get_device()
available = is_gpu_available()
info = get_gpu_info()
batch_size = get_batch_size()
sufficient = check_memory_sufficient(1000)
should_retry = handle_oom(0, 3)
cpu_device = fallback_to_cpu()
```

### 性能特点
- **毫秒级响应**：设备切换和信息获取平均耗时小于1ms
- **智能内存管理**：自动监控内存使用并调整批量大小
- **优雅降级**：无缝切换CPU/GPU模式，保证程序连续运行
- **异常处理**：完善的OOM处理和重试机制

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

### 3. 环境变量配置

StockSchool支持通过环境变量进行配置，优先级高于配置文件：

```bash
# 必需环境变量
export TUSHARE_TOKEN="your_tushare_token"  # Tushare API token

# 可选环境变量
export DATABASE_URL="postgresql://user:password@localhost:5432/stockschool"  # 数据库连接
export REDIS_URL="redis://localhost:6379/0"  # Redis连接
export CELERY_BROKER_URL="redis://localhost:6379/0"  # Celery消息队列
export LOG_LEVEL="INFO"  # 日志级别
```

Windows系统：
```cmd
set TUSHARE_TOKEN=your_tushare_token
set DATABASE_URL=postgresql://user:password@localhost:5432/stockschool
```

### 4. 配置文件

StockSchool使用多个配置文件进行系统配置：

**主要配置文件：**
- `config.yml` - 主配置文件，包含数据源、数据库、AI模型等配置
- `config/data_sync.yaml` - 数据同步专用配置
- `config/monitoring.yaml` - 监控和告警配置

**配置文件结构示例：**

```yaml
# 数据源配置
data_sync_params:
  tushare:
    enabled: true
    token: ${TUSHARE_TOKEN}  # 使用环境变量
    api_limit: 200
    retry_times: 3
    retry_delay: 1
  
  akshare:
    enabled: true
    api_limit: 100
    retry_times: 3
    retry_delay: 1

# 数据库配置
database:
  url: postgresql://user:password@localhost:5432/stockschool

# AI模型配置
ai_model:
  enabled: true
  models: ['random_forest', 'xgboost', 'lightgbm']
  
# 监控配置
monitoring:
  enabled: true
  prometheus_url: http://localhost:9090
```

### 4. 数据库初始化

首次运行前需要初始化数据库和同步基础数据：

```bash
# 通过命令行参数执行数据库初始化
python run.py --init-db

# 或者通过交互式菜单执行
python run.py
# 然后选择 4. 数据库初始化
```

手动初始化方式：

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

# 同步申万行业分类数据
python -m src.data.tushare_sync --mode industry --level L1  # 同步一级行业
python -m src.data.tushare_sync --mode industry --level L2  # 同步二级行业
python -m src.data.tushare_sync --mode industry --level L3  # 同步三级行业
python -m src.data.tushare_sync --mode industry_full        # 同步所有级别行业

# 增量更新特定股票的行业数据
python -m src.data.tushare_sync --mode industry_update --stock-file stock_list.txt
```

#### 申万行业分类同步说明

系统支持申万行业分类数据的完整同步功能，包括以下特性：

1. **多级行业支持**：支持申万L1/L2/L3三级行业分类数据同步
2. **独立同步**：可独立同步指定级别的行业数据
3. **完整同步**：可一次性同步所有级别的行业数据
4. **增量更新**：支持对特定股票列表进行行业数据增量更新
5. **批量处理**：采用批量API调用和数据库操作，提高同步效率
6. **错误处理**：完善的异常处理机制，确保同步过程的稳定性

使用示例：
```bash
# 同步申万一级行业数据
python -m src.data.tushare_sync --mode industry --level L1

# 同步申万二级行业数据
python -m src.data.tushare_sync --mode industry --level L2

# 同步申万三级行业数据
python -m src.data.tushare_sync --mode industry --level L3

# 同步所有级别的申万行业数据
python -m src.data.tushare_sync --mode industry_full

# 更新特定股票的行业数据（股票代码列表保存在文件中）
python -m src.data.tushare_sync --mode industry_update --stock-file stocks.txt
```

股票代码文件格式（stocks.txt）示例：
```
000001.SZ
000002.SZ
600000.SH
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

- **v1.1.7 代码质量验证阶段** (2025年1月):
  - **语法验证**: 完成184个Python文件的语法检查，修复2个语法错误
  - **依赖管理**: 安装缺失的schedule库，确保任务调度功能完整
  - **代码查重**: 通过知识图谱识别重复代码，优化ModelTrainingPipeline相关类结构
  - **渐进式标准化**: 采用"先实践后标准"策略，基于实际开发经验制定规范
  - **测试优化**: 清理测试文件中的调试信息，保留必要的验证输出

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

#### 知识图谱实体和关系创建
- **src/compute/factor_engine.py 模块的知识图谱实体和关系创建**：
  - 为 <mcfile name="factor_engine.py" path="src/compute/factor_engine.py"></mcfile> 文件创建了类型为 `File` 的知识图谱实体。
  - 为 <mcsymbol name="FactorEngine" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="18" type="class"></mcsymbol> 类创建了类型为 `Class` 的知识图谱实体，并添加了“因子计算引擎，负责从数据库获取数据，计算因子，并存储结果”的功能描述。
  - 在 <mcfile name="factor_engine.py" path="src/compute/factor_engine.py"></mcfile> 文件与 <mcsymbol name="FactorEngine" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="18" type="class"></mcsymbol> 类之间创建了 `contains` 关系。
  - 为 <mcsymbol name="__init__" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="24" type="function"></mcsymbol> 方法创建了类型为 `Method` 的知识图谱实体，并添加了“初始化因子引擎，设置数据库连接和因子计算器”的功能描述。
  - 在 <mcsymbol name="FactorEngine" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="18" type="class"></mcsymbol> 类与 <mcsymbol name="__init__" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="24" type="function"></mcsymbol> 方法之间创建了 `contains` 关系。
  - 为 <mcsymbol name="get_stock_data" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="35" type="function"></mcsymbol> 方法创建了类型为 `Method` 的知识图谱实体，并添加了“从数据库获取指定股票在特定日期范围内的数据”的功能描述。
  - 在 <mcsymbol name="FactorEngine" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="18" type="class"></mcsymbol> 类与 <mcsymbol name="get_stock_data" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="35" type="function"></mcsymbol> 方法之间创建了 `contains` 关系。
  - 为 <mcsymbol name="get_all_stocks" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="55" type="function"></mcsymbol> 方法创建了类型为 `Method` 的知识图谱实体，并添加了“获取所有已上市股票的代码列表”的功能描述。
  - 在 <mcsymbol name="FactorEngine" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="18" type="class"></mcsymbol> 类与 <mcsymbol name="get_all_stocks" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="55" type="function"></mcsymbol> 方法之间创建了 `contains` 关系。
  - 为 <mcsymbol name="create_factor_tables" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="66" type="function"></mcsymbol> 方法创建了类型为 `Method` 的知识图谱实体，并添加了“在数据库中创建用于存储股票因子和技术因子的表”的功能描述。
  - 在 <mcsymbol name="FactorEngine" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="18" type="class"></mcsymbol> 类与 <mcsymbol name="create_factor_tables" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="66" type="function"></mcsymbol> 方法之间创建了 `contains` 关系。
  - 为 <mcsymbol name="calculate_stock_factors" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="120" type="function"></mcsymbol> 方法创建了类型为 `Method` 的知识图谱实体，并添加了“计算单只股票的技术因子和基本面因子”的功能描述。
  - 在 <mcsymbol name="FactorEngine" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="18" type="class"></mcsymbol> 类与 <mcsymbol name="calculate_stock_factors" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="120" type="function"></mcsymbol> 方法之间创建了 `contains` 关系。
  - 为 <mcsymbol name="calculate_all_factors" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="217" type="function"></mcsymbol> 方法创建了类型为 `Method` 的知识图谱实体，并添加了“计算所有股票的技术因子”的功能描述。
  - 在 <mcsymbol name="FactorEngine" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="18" type="class"></mcsymbol> 类与 <mcsymbol name="calculate_all_factors" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="217" type="function"></mcsymbol> 方法之间创建了 `contains` 关系。
  - 为 <mcsymbol name="_calculate_fundamental_factors" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="280" type="function"></mcsymbol> 方法创建了类型为 `Method` 的知识图谱实体，并添加了“计算基本面因子”的功能描述。
  - 在 <mcsymbol name="FactorEngine" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="18" type="class"></mcsymbol> 类与 <mcsymbol name="_calculate_fundamental_factors" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="280" type="function"></mcsymbol> 方法之间创建了 `contains` 关系。
  - 为 <mcsymbol name="_get_factor_category" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="400" type="function"></mcsymbol> 方法创建了类型为 `Method` 的知识图谱实体，并添加了“获取因子分类”的功能描述。
  - 在 <mcsymbol name="FactorEngine" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="18" type="class"></mcsymbol> 类与 <mcsymbol name="_get_factor_category" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="400" type="function"></mcsymbol> 方法之间创建了 `contains` 关系。
  - 为 <mcsymbol name="calculate_all_factors_with_fundamental" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="419" type="function"></mcsymbol> 方法创建了类型为 `Method` 的知识图谱实体，并添加了“计算所有因子（包括技术因子和基本面因子）”的功能描述。
  - 在 <mcsymbol name="FactorEngine" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="18" type="class"></mcsymbol> 类与 <mcsymbol name="calculate_all_factors_with_fundamental" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="419" type="function"></mcsymbol> 方法之间创建了 `contains` 关系。
  - 为 <mcsymbol name="get_factor_data" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="445" type="function"></mcsymbol> 方法创建了类型为 `Method` 的知识图谱实体，并添加了“获取因子数据”的功能描述。
  - 在 <mcsymbol name="FactorEngine" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="18" type="class"></mcsymbol> 类与 <mcsymbol name="get_factor_data" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="445" type="function"></mcsymbol> 方法之间创建了 `contains` 关系。
  - 为 <mcsymbol name="get_factor_statistics" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="480" type="function"></mcsymbol> 方法创建了类型为 `Method` 的知识图谱实体，并添加了“获取因子统计信息”的功能描述。
  - 在 <mcsymbol name="FactorEngine" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="18" type="class"></mcsymbol> 类与 <mcsymbol name="get_factor_statistics" filename="factor_engine.py" path="src/compute/factor_engine.py" startline="480" type="function"></mcsymbol> 方法之间创建了 `contains` 关系。

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
