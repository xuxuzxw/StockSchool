# StockSchool 项目开发日志

## v1.1 阶段开发记录

#### 数据库架构升级
- ✅ **数据库Schema更新** - 在`database_schema.sql`中添加了`sentiment_data`表，用于存储市场情绪、资金流向等数据
- ✅ **配置文件更新** - 在`config.yml`中添加了情绪数据、资金流向数据和北向资金数据的同步配置参数

#### 数据补全与入库
- ✅ **基础数据同步** - `tushare_sync.py`已实现基础股票数据同步
- ✅ **情绪数据同步** - `akshare_sync.py`已实现情绪数据同步功能
- 🔄 **财务数据补全** - 需要完善`tushare_sync.py`添加财务数据和指标数据同步

#### 核心因子计算
- ✅ **技术因子计算** - `factor_engine.py`已实现技术因子计算功能
- ✅ **基本面指标计算** - 在`indicators.py`中添加了`FundamentalIndicators`类和`FundamentalFactorCalculator`类
- ✅ **基本面因子集成** - 升级`factor_engine.py`，添加了`_calculate_fundamental_factors`方法和`calculate_all_factors_with_fundamental`方法
- ✅ **因子预处理** - 创建了`processing.py`，实现因子标准化、去极值、中性化等预处理功能

#### AI模型训练与预测
- ✅ **训练流水线** - 创建了`training_pipeline.py`，实现完整的模型训练流水线
- ✅ **预测脚本** - 创建了`prediction.py`，实现每日预测功能







---

- **v1.1.6 全流程实测阶段完成**：
  - AI模型模块 (`ai_model.py`) 已创建，实现了AI模型的训练、预测和管理。
  - `tasks.py` 中已扩展了AI相关的Celery任务 (`train_ai_model`, `run_prediction`, `batch_prediction`)。
  - `config.yml` 已更新，包含了 `full_test_config` 配置节，用于定义全流程测试的参数。
  - 全流程调度系统已构建，包括主工作流调度脚本 (`run_daily_workflow.py`) 和全流程实测编排器 (`scripts/full_test_v1_1_6.py`)。
  - AI模型能力已实现，支持多模型训练、批量预测和模型持久化。
  - 任务调度架构已设计，利用Celery的 `group`、`chain` 和 `chord` 功能进行任务编排。
  - 测试配置已定义，允许灵活设置测试日期范围和股票池。
  - 工作流阶段已划分，包括数据同步、因子计算、AI模型训练、AI预测和数据质量检查。
  - 质量保证体系已建立，通过数据质量检查确保任务流执行一致性、输入数据完整性、中间产品覆盖率和最终成品时效性。
  - 相关工具和脚本（如 `scripts/clear_database.py` 用于清空数据库）已创建。
  - 技术特性包括模块化设计、异步任务处理、可扩展性、容错机制和性能监控。
  - 部署准备已完成，支持Docker部署和Celery分布式任务队列。
  - 下一步计划是进行全面的集成测试和性能优化。

- **v1.1.6 全流程实测调试记录**：
  - **Celery Worker 启动问题**：
    - 初始错误：`ModuleNotFoundError: No module named 'utils'`
    - 修复：修改 `src/compute/tasks.py` 中的导入语句，将相对导入改为绝对导入（例如：`from src.utils.db import get_db_engine`）。
    - 初始错误：`ImportError: cannot import name 'TushareDataSync' from 'src.data.tushare_sync'`
    - 修复：将 `src/compute/tasks.py` 中 `TushareDataSync` 的导入修改为 `TushareSynchronizer`，因为 `src/data/tushare_sync.py` 中定义的类名为 `TushareSynchronizer`。
    - 初始错误：`consumer: Cannot connect to redis://localhost:6379/0: Authentication required..`
    - 修复：在 `src/compute/tasks.py` 中硬编码 Redis 密码进行测试（`redis://:redis123@localhost:6379/0`），验证连接成功后，改回使用环境变量 `REDIS_URL`。
  - **`clear_database.py` 脚本运行问题**：
    - 初始错误：`ImportError: cannot import name 'ConfigLoader' from 'src.utils.config_loader'`
    - 修复：将 `scripts/clear_database.py` 中的 `ConfigLoader` 导入修改为 `Config`，因为 `src/utils/config_loader.py` 中定义的类名为 `Config`。
    - 初始错误：`NameError: name 'ConfigLoader' is not defined`
    - 修复：将 `scripts/clear_database.py` 中 `DatabaseCleaner` 类初始化方法中的 `ConfigLoader()` 修改为 `Config()`。
    - 修复：`scripts/clear_database.py` 中数据库删除操作的执行方式问题，通过导入 `sqlalchemy.text` 并使用 `conn.execute(text(f"DELETE FROM {table_name}"))` 解决了 `Not an executable object` 错误。
  - **`run_daily_workflow.py` 脚本运行问题**：
    - 初始错误：`NameError: name 'ConfigLoader' is not defined`
    - 修复：将 `run_daily_workflow.py` 中所有 `ConfigLoader` 的引用修改为 `Config`。
    - 改进：在 `run_daily_workflow.py` 中添加 `load_dotenv()`，确保脚本能正确加载 `.env` 文件中的环境变量。
  - **调试结果**：
    - Celery worker 成功启动并连接到 Redis。
    - `clear_database.py` 脚本成功运行，数据库清空功能正常。
    - `run_daily_workflow.py` 脚本成功启动简化工作流.

## v1.1.5 真实数据测试阶段

### 测试环境配置
- ✅ **独立测试数据库** - 配置了独立的PostgreSQL测试数据库服务（端口15433）
- ✅ **Docker容器化** - 使用TimescaleDB镜像，支持时间序列数据优化
- ✅ **数据库隔离** - 测试数据库与生产数据库完全隔离，确保测试安全性

### 端到端集成测试
- ✅ **数据库连接测试** - 验证测试数据库连接和基本操作
- ✅ **数据库隔离测试** - 确保测试数据不会影响生产环境
- ✅ **完整流水线测试** - 从数据同步到因子计算的端到端测试
- ✅ **黄金数据校验** - 使用预设数据验证RSI等技术指标计算精度
- ✅ **因子计算一致性** - 验证相同输入产生相同输出的一致性
- ✅ **数据质量验证** - 检查股票数据的完整性和合理性

### 技术指标修复
- ✅ **RSI计算修复** - 修正了RSI计算函数的导入路径和调用方式
- ✅ **MACD计算优化** - 统一了MACD函数返回DataFrame格式
- ✅ **布林带计算验证** - 优化了布林带合理性验证逻辑

### 测试数据管理
- ✅ **测试数据生成器** - 创建了`TestDatabaseManager`类管理测试数据
- ✅ **数据质量保证** - 确保生成的测试数据符合金融数据规范
- ✅ **自动化清理** - 实现测试环境的自动设置和清理

### 架构决策记录
- ✅ **ADR文档** - 创建了`docs/adr/001-choice-of-ml-model.md`记录LightGBM选择决策
- ✅ **技术选型文档** - 详细记录了机器学习模型选择的技术考量

### 性能分析
- ✅ **性能剖析工具** - 修复了性能剖析工具的模块导入问题
- ✅ **性能报告生成** - 生成了详细的性能分析报告
- ✅ **瓶颈识别** - 识别了系统中的潜在性能瓶颈

### 测试结果
```
============================= test session starts =============================
src/tests/test_integration_pipeline.py::test_database_connection PASSED  [ 16%]
src/tests/test_integration_pipeline.py::test_database_isolation PASSED   [ 33%]
src/tests/test_integration_pipeline.py::test_full_pipeline_for_sample_data PASSED [ 50%]
src/tests/test_integration_pipeline.py::test_factor_golden_data_validation PASSED [ 66%]
src/tests/test_integration_pipeline.py::test_factor_calculation_consistency PASSED [ 83%]
src/tests/test_integration_pipeline.py::test_data_quality_validation PASSED [100%]

============================== 6 passed in 1.70s ==============================
```

### 技术改进
- **模块导入优化** - 修复了多个模块的导入路径问题
- **函数接口统一** - 统一了技术指标计算函数的接口规范
- **错误处理增强** - 改进了测试中的错误处理和异常捕获
- **数据验证逻辑** - 优化了金融数据的合理性验证逻辑

---

---

## v1.1.6 全流程实测阶段





## v1.1.6 数据同步问题调试记录 (2025-07-28)

### 数据库状态分析 (2025-07-28 04:46)
通过`check_database_status.py`脚本检查发现：

**数据库现状（修复后）：**
- `stock_daily`表：大幅增加，包含最近30个交易日完整数据
- 数据时间范围：2025-04-30 到 2025-07-25 (30个交易日)
- `trade_calendar`表：2,036条记录，1,349个交易日 (2020-01-01 到 2025-07-28)
- `stock_basic`表：5,417条记录，活跃股票5,417只
- **关键发现：通过数据修复脚本成功回填了历史数据，000001.SZ从29条增加到59条记录**

### 问题根源分析
1. **历史数据缺失的根本原因**
   - `get_last_trade_date()`函数逻辑：如果表为空返回一年前日期，如果有数据返回最大日期
   - 当前返回20250725，导致只从最近日期开始同步
   - 缺少历史数据回填机制

2. **当日数据缺失原因**
   - 2025-07-28是交易日但数据库中无此日期数据
   - 可能原因：Tushare API当日数据延迟，或同步逻辑问题

3. **数据同步策略问题**
   - `update_daily_data`限制最多同步30天
   - 没有历史数据初始化机制
   - 缺少数据完整性检查

### 问题排查过程

#### 1. 活跃股票数量问题
- **问题现象**: `check_data.py`显示活跃股票数量为0
- **根本原因**: `sync_stock_basic`函数中`fields`参数缺少`list_status`字段
- **解决方案**: 
  - 在`pro.stock_basic`的`fields`参数中添加`list_status`
  - 在`stock_basic`表创建语句中添加`list_status`字段
  - 重新同步数据后，活跃股票数量恢复为5417

#### 2. 日志记录问题
- **问题现象**: `sync_stock_basic`函数执行失败，错误`name 'logger' is not defined`
- **解决方案**: 在`src/data/tushare_sync.py`文件顶部添加`from loguru import logger`

#### 3. 历史数据不足问题
- **问题现象**: `000001.SZ`只有29条数据，时间范围为2025-06-17到2025-07-25
- **根本原因**: 
  - `get_last_trade_date()`返回最新日期，导致只做增量同步
  - 缺少历史数据初始化机制
  - `update_daily_data`函数限制最多同步30天

#### 4. 当日数据缺失问题
- **当前日期**: 2025-07-28 (周一，确认为交易日)
- **数据库最新**: 2025-07-25 (周五)
- **缺失日期**: 2025-07-28当日数据
- **可能原因**: Tushare API数据延迟或同步逻辑问题

### 已解决问题
- ✅ 活跃股票数量为0 → 修复`list_status`字段同步
- ✅ 日志记录缺失 → 添加`logger`导入
- ✅ 数据库连接问题 → 确认使用端口15432正常连接
- ✅ **历史数据缺失** → 通过`fix_data_sync.py`脚本成功回填最近30个交易日数据
- ✅ **数据修复机制** → 创建专门的数据修复脚本，支持批量历史数据同步

### 数据修复成果
- **修复脚本**: `fix_data_sync.py`
- **回填范围**: 2025-04-30 到 2025-06-16 (30个交易日)
- **数据增量**: 000001.SZ从29条增加到59条记录
- **同步效果**: 每个交易日约5,400条股票记录

### 待解决问题
- ❌ **当日数据缺失**：2025-07-28交易日Tushare暂无数据（正常情况，等待数据发布）
- ❌ **更长历史数据**：如需要更早期数据，可扩展脚本同步范围

### 下一步计划
1. 监控当日数据发布情况
2. 根据需要扩展历史数据范围
3. 将数据修复脚本集成到定期维护流程
4. 优化数据同步的错误处理和重试机制

## v1.1.6 调试工具整合 (2025-07-28)

### 调试工具整合
- ✅ **调试工具迁移** - 将所有调试脚本移动到`test/`目录
  - `check_database_status.py` - 数据库状态检查
  - `check_data.py` - 数据验证
  - `fix_data_sync.py` - 数据修复和回填
- ✅ **临时文件清理** - 删除开发过程中的临时文件
- ✅ **数据库端口检查** - 发现并准备修复15433端口相关代码

### 发现的问题
- ⚠️ **测试数据库端口** - 发现以下文件中使用15433端口:
  - `src/utils/test_db.py` - 测试数据库连接配置
  - `src/tests/test_integration_pipeline.py` - 集成测试中的端口验证
  - `docker-compose.yml` - test_db服务端口映射
  - `log.md` - 文档中的端口说明

### 已完成事项
- ✅ **调试功能集成** - 将1.16调试.md中的运维指令完整集成到run.py，并通过“飞行前检查”验证修复成功。
  - 飞行前检查 (Pre-Flight Check) - 环境健康检查
  - Celery Worker启动和监控
  - 日常工作流执行
  - 数据质量检验
  - 数据修复和回填
  - 紧急情况诊断
- ✅ **运维控制台** - 创建了完整的运维和调试控制台界面
- ✅ **功能测试** - 创建并通过了run.py运维功能的完整测试

### 待处理事项
- 🔄 **端口配置统一** - 用户将删除15433数据库，需要更新相关代码
  - `src/utils/test_db.py` - 测试数据库连接配置
  - `src/tests/test_integration_pipeline.py` - 集成测试中的端口验证
  - `docker-compose.yml` - test_db服务端口映射
  - 相关文档更新

---

## 硬编码参数配置化修改日志

## 修改概述
本次修改将项目中的硬编码参数替换为从配置文件动态加载，提高了系统的可配置性和灵活性。

## 修改文件列表

### 1. src/data/tushare_sync.py
**修改内容：**
- `batch_size` 默认值从硬编码 `1000` 改为从配置 `data_sync_params.batch_size` 加载
- `sleep_time` 默认值从硬编码 `1` 改为从配置 `data_sync_params.sleep_time` 加载
- `max_retries` 默认值从硬编码 `3` 改为从配置 `data_sync_params.max_retries` 加载
- `chunk_size` 默认值从硬编码 `10000` 改为从配置 `data_sync_params.chunk_size` 加载

### 2. src/compute/factor_engine.py
**修改内容：**
- `batch_size` 默认值从硬编码 `1000` 改为从配置 `compute_params.batch_size` 加载
- `max_workers` 默认值从硬编码 `4` 改为从配置 `compute_params.max_workers` 加载
- `chunk_size` 默认值从硬编码 `5000` 改为从配置 `compute_params.chunk_size` 加载

### 3. src/compute/quality.py
**修改内容：**
- `sample_size` 默认值从硬编码 `1000` 改为从配置 `quality_params.sample_size` 加载
- `threshold` 默认值从硬编码 `0.95` 改为从配置 `quality_params.threshold` 加载
- `max_gap_days` 默认值从硬编码 `7` 改为从配置 `quality_params.max_gap_days` 加载
- `outlier_threshold` 默认值从硬编码 `3` 改为从配置 `quality_params.outlier_threshold` 加载
- `min_data_points` 默认值从硬编码 `30` 改为从配置 `quality_params.min_data_points` 加载

### 4. src/features/feature_store.py
**修改内容：**
- `batch_size` 默认值从硬编码 `1000` 改为从配置 `feature_params.batch_size` 加载
- `cache_size` 默认值从硬编码 `10000` 改为从配置 `feature_params.cache_size` 加载
- `max_workers` 默认值从硬编码 `4` 改为从配置 `feature_params.max_workers` 加载
- `chunk_size` 默认值从硬编码 `5000` 改为从配置 `feature_params.chunk_size` 加载
- `top_n` 默认值从硬编码 `20` 改为从配置 `feature_params.top_n` 加载

### 5. src/monitoring/alerts.py
**修改内容：**
- `max_history` 默认值从硬编码 `1000` 改为从配置 `monitoring_params.max_history` 加载
- `timeout` 默认值从硬编码 `30` 改为从配置 `monitoring_params.webhook_timeout` 加载
- `alert_check_limit` 默认值从硬编码 `1000` 改为从配置 `monitoring_params.alert_check_limit` 加载
- `stats_limit` 默认值从硬编码 `1000` 改为从配置 `monitoring_params.stats_limit` 加载

### 6. src/monitoring/performance.py
**修改内容：**
- `collection_interval` 默认值从硬编码 `60` 改为从配置 `monitoring_params.collection_interval` 加载
- `metrics_max_history` 默认值从硬编码 `1000` 改为从配置 `monitoring_params.metrics_max_history` 加载
- `metric_window_minutes` 默认值从硬编码 `60` 改为从配置 `monitoring_params.metric_window_minutes` 加载
- `separator_length` 默认值从硬编码 `100` 改为从配置 `monitoring_params.separator_length` 加载
- `slow_query_threshold` 默认值从硬编码 `1.0` 改为从配置 `monitoring_params.slow_query_threshold` 加载
- `save_interval` 默认值从硬编码 `300` 改为从配置 `monitoring_params.save_interval` 加载
- `thread_join_timeout` 默认值从硬编码 `5` 改为从配置 `monitoring_params.thread_join_timeout` 加载
- `slow_query_limit` 默认值从硬编码 `20` 改为从配置 `monitoring_params.slow_query_limit` 加载
- `slow_query_threshold_check` 默认值从硬编码 `10` 改为从配置 `monitoring_params.slow_query_threshold_check` 加载
- `data_retention_days` 默认值从硬编码 `30` 改为从配置 `monitoring_params.data_retention_days` 加载

### 7. src/strategy/evaluation.py
**修改内容：**
- `rolling_window` 默认值从硬编码 `60` 改为从配置 `strategy_params.rolling_window` 加载

### 8. src/strategy/model_explainer.py
**修改内容：**
- `shap_sample_size` 默认值从硬编码 `1000` 改为从配置 `strategy_params.shap_sample_size` 加载
- `max_features_display` 默认值从硬编码 `15` 改为从配置 `strategy_params.max_features_display` 加载
- `top_features_count` 默认值从硬编码 `10` 改为从配置 `strategy_params.top_features_count` 加载

### 9. src/utils/retry.py
**修改内容：**
- `max_retries` 默认值从硬编码 `3` 改为从配置 `utils_params.max_retries` 加载
- `retry_sleep_base` 默认值从硬编码 `2` 改为从配置 `utils_params.retry_sleep_base` 加载

### 10. src/api/main.py
**修改内容：**
- `host` 默认值从硬编码 `"0.0.0.0"` 改为从配置 `api_params.host` 加载
- `port` 默认值从硬编码 `8000` 改为从配置 `api_params.port` 加载
- `reload` 默认值从硬编码 `True` 改为从配置 `api_params.reload` 加载
- `log_level` 默认值从硬编码 `"info"` 改为从配置 `api_params.log_level` 加载

## 配置参数说明

### 需要在config.yaml中添加的新配置项：

```yaml
# API参数
api_params:
  host: "0.0.0.0"
  port: 8000
  reload: true
  log_level: "info"
  default_limit: 100
  min_limit: 1
  max_limit: 1000

# 工具参数
utils_params:
  max_retries: 3
  retry_sleep_base: 2

# 监控参数
monitoring_params:
  max_history: 1000
  webhook_timeout: 30
  alert_check_limit: 1000
  stats_limit: 1000
  collection_interval: 60
  metrics_max_history: 1000
  metric_window_minutes: 60
  separator_length: 100
  slow_query_threshold: 1.0
  save_interval: 300
  thread_join_timeout: 5
  slow_query_limit: 20
  slow_query_threshold_check: 10
  data_retention_days: 30

# 策略参数扩展
strategy_params:
  rolling_window: 60
  shap_sample_size: 1000
  max_features_display: 15
  top_features_count: 10

# 特征参数
feature_params:
  batch_size: 1000
  cache_size: 10000
  max_workers: 4
  chunk_size: 5000
  top_n: 20

# 质量检查参数
quality_params:
  sample_size: 1000
  threshold: 0.95
  max_gap_days: 7
  outlier_threshold: 3
  min_data_points: 30

# 计算参数
compute_params:
  batch_size: 1000
  max_workers: 4
  chunk_size: 5000

# 数据同步参数扩展
data_sync_params:
  batch_size: 1000
  sleep_time: 1
  max_retries: 3
  chunk_size: 10000
```

## 修改效果

1. **提高可配置性**：所有关键参数都可以通过配置文件进行调整，无需修改代码
2. **便于环境适配**：不同环境可以使用不同的配置参数
3. **提升维护性**：参数集中管理，便于统一调整和维护
4. **增强灵活性**：可以根据实际需求动态调整系统行为

## 注意事项

1. 所有修改都保持了向后兼容性，如果配置文件中没有相应参数，会使用原来的默认值
2. 建议在部署前检查配置文件是否包含所有必要的参数
3. 部分参数（如批处理大小、线程数等）的调整可能会影响系统性能，建议根据实际环境进行测试和调优


## Celery任务进度显示优化

1. **修改文件**: `src/compute/tasks.py`
   - 在 `sync_daily_data` 任务中添加进度更新，通过 `self.update_state` 方法报告当前正在同步的数据类型（如"同步基础数据"和"同步交易数据"）
   - 在 `calculate_daily_factors` 任务中添加进度更新，报告当前正在计算的股票数量和总进度

2. **修改文件**: `run_daily_workflow.py`
   - 重构 `run_full_workflow` 函数，将任务链拆分为五个独立阶段（数据同步、因子计算、AI训练、批量预测、质量检查）
   - 为每个阶段添加独立的进度监控逻辑
   - 数据同步和因子计算任务组现在会显示每个子任务的详细进度
   - 添加超时机制（2小时）

3. **新增功能**:
   - 数据同步任务显示当前步骤和进度百分比
   - 因子计算任务显示已处理股票数量、总数量和当前股票代码
   - 所有任务阶段都有明确的开始和完成状态提示

4. **修复问题**:
   - 修复 `AsyncResult` 初始化时缺少 `app` 实例的问题
   - 添加缺失的 `import time` 语句

这些修改使得在运行完整工作流时，控制台能够实时显示更详细的进度信息，帮助用户更好地了解任务执行状态。