# StockSchool 完整程序构建指南

## 项目概述

StockSchool是一个基于Python 3.11的量化投研系统，采用"学校"隐喻构建智能股票分析和推荐平台。系统将股票视为"学生"，各种量化因子作为"老师"进行评分，AI模型作为"校长"进行综合决策。

### 核心理念
- **数据层**: 多维度数据获取和存储（Tushare为主，AkShare为辅）
- **计算层**: 三大"教研组"因子计算（技术面、基本面、情绪面）
- **决策层**: AI"校长"智能决策和模型解释

## 技术架构总览

### 系统架构图
```
StockSchool/
├── 数据获取层 (src/data/)
│   ├── Tushare数据源
│   ├── AkShare数据源
│   └── 数据质量监控
├── 数据存储层 (PostgreSQL + TimescaleDB)
│   ├── 基础信息表
│   ├── 时序行情数据
│   ├── 财务数据
│   └── 因子数据
├── 计算引擎层 (src/compute/)
│   ├── 技术因子引擎
│   ├── 基本面因子引擎
│   ├── 情绪因子引擎
│   └── 因子标准化和评分
├── AI决策层 (src/ai/)
│   ├── 模型训练流水线
│   ├── 预测引擎
│   └── 模型解释器
├── API服务层 (src/api/)
│   ├── RESTful API
│   ├── 认证授权
│   └── 特征商店API
├── 监控告警层 (src/monitoring/)
│   ├── 性能监控
│   ├── 数据质量监控
│   └── 告警通知
└── 任务调度层 (Celery + Redis)
    ├── 数据同步任务
    ├── 因子计算任务
    └── AI训练预测任务
```

## 核心技术栈

### 后端框架
- **Python**: 3.11 (必需版本)
- **Web框架**: FastAPI 0.100.0
- **异步处理**: Celery 5.3.1 + Redis 4.6.0
- **数据库**: PostgreSQL 12+ + TimescaleDB扩展

### 数据处理
- **数据获取**: Tushare 1.2.88, AkShare 1.10.3
- **数据处理**: pandas 2.0.3, numpy 1.24.3
- **数据库连接**: psycopg2-binary 2.9.7

### 机器学习
- **核心库**: scikit-learn 1.3.0
- **深度学习**: PyTorch 1.13.1+cu117 {推荐GPU版本}
- **模型解释**: SHAP 0.43.0 {支持CUDA加速}
- **可视化**: matplotlib 3.7.2, plotly 5.15.0

### 系统监控
- **GPU监控**: pynvml 11.5.0
- **系统监控**: psutil 5.9.5
- **日志系统**: loguru 0.7.0

## 环境要求

### 硬件要求
- **CPU**: 推荐12核以上，支持多线程处理
- **内存**: 至少32GB RAM（推荐48GB）
- **GPU**: 可选，需CUDA 11.7+驱动支持
- **存储**: 至少100GB可用空间

### 软件环境
- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.11 (严格要求)
- **数据库**: PostgreSQL 12+ with TimescaleDB
- **容器**: Docker + Docker Compose (推荐)

## 项目结构详解

### 源代码结构
```
src/
├── __init__.py                 # 包初始化
├── main.py                     # 主入口文件
├── check_stocks.py            # 股票检查工具
├── ai/                        # AI模型模块
│   ├── training_pipeline.py   # 模型训练流水线
│   ├── prediction.py          # 预测引擎
│   └── model_explainer.py     # 模型解释器
├── api/                       # API接口模块
│   ├── main.py               # API主入口
│   ├── factor_api.py         # 因子相关API
│   ├── auth.py               # 认证授权
│   └── feature_store_api.py  # 特征商店API
├── compute/                   # 计算引擎模块
│   ├── factor_engine.py      # 因子计算引擎
│   ├── technical_factor_engine.py    # 技术因子引擎
│   ├── fundamental_factor_engine.py  # 基本面因子引擎
│   ├── sentiment_factor_engine.py    # 情绪因子引擎
│   ├── factor_standardizer.py       # 因子标准化器
│   ├── factor_effectiveness_analyzer.py # 因子有效性分析
│   ├── parallel_factor_calculator.py    # 并行计算器
│   ├── incremental_calculator.py        # 增量计算器
│   ├── factor_cache.py              # 因子缓存
│   ├── data_compression_archiver.py # 数据压缩归档
│   ├── indicators.py         # 技术指标库
│   ├── processing.py         # 因子预处理
│   ├── quality.py           # 数据质量控制
│   └── tasks.py             # Celery任务定义
├── config/                   # 配置管理模块
│   ├── manager.py           # 配置管理器
│   ├── unified_config.py    # 统一配置管理器
│   └── migration_guide.py   # 配置迁移工具
├── data/                    # 数据同步模块
│   ├── sources/             # 数据源实现
│   ├── sync_manager.py      # 同步管理器
│   ├── incremental_update.py # 增量更新
│   ├── industry_classification.py # 行业分类
│   └── data_quality_monitor.py    # 数据质量监控
├── database/                # 数据库相关
├── features/                # 特征商店模块
│   ├── factor_feature_store.py    # 因子特征商店
│   └── feature_store_adapter.py   # 特征商店适配器
├── monitoring/              # 监控告警模块
│   ├── alerts.py           # 告警引擎
│   ├── performance.py      # 性能监控
│   └── notifications.py    # 通知渠道
├── strategy/               # 策略评估模块
│   ├── evaluation.py       # 策略评估
│   └── ai_model.py         # AI模型管理
├── tests/                  # 测试模块
├── ui/                     # 用户界面模块
├── utils/                  # 工具函数模块
│   ├── config_loader.py    # 配置加载器
│   ├── db.py              # 数据库工具
│   ├── gpu_utils.py       # GPU工具
│   └── retry.py           # 重试装饰器
└── websocket/             # WebSocket模块
```

### 配置文件结构
```
config/
├── base.yml              # 基础配置
├── factor_config.yaml    # 因子计算配置
├── monitoring.yaml       # 监控告警配置
├── development.yml       # 开发环境配置
├── production.yml        # 生产环境配置
└── testing.yml          # 测试环境配置
```

### 脚本工具
```
scripts/
├── deployment/           # 部署脚本
├── monitoring/          # 监控脚本
├── clear_database.py    # 数据库清理
├── full_test_v1_1_6.py  # 全流程测试
├── migrate_config_system.py # 配置迁移
└── performance_test_runner.py # 性能测试
```

## 数据库设计

### 核心表结构

#### 基础信息表
- `stock_basic`: 股票基础信息
- `trade_calendar`: 交易日历
- `sw_industry_classify`: 申万行业分类

#### 时序数据表 (TimescaleDB超表)
- `stock_daily`: 日线行情数据
- `daily_basic`: 每日基本指标
- `adj_factor`: 复权因子

#### 财务数据表
- `income`: 利润表
- `balance_sheet`: 资产负债表
- `cash_flow`: 现金流量表
- `financial_indicator`: 财务指标

#### 因子数据表 (宽表设计)
- `technical_factors`: 技术面因子宽表
- `fundamental_factors`: 基本面因子宽表
- `sentiment_factors`: 情绪面因子宽表
- `factor_metadata`: 因子元数据表
- `factor_scores`: 因子评分表

#### 特征商店表
- `factor_definitions`: 因子定义
- `factor_versions`: 因子版本管理
- `data_lineage`: 数据血缘
- `quality_metrics`: 质量指标

### 索引策略
```sql
-- 必需的复合索引
CREATE INDEX idx_stock_daily_composite ON stock_daily (ts_code, trade_date);
CREATE INDEX idx_daily_basic_composite ON daily_basic (ts_code, trade_date);
CREATE INDEX idx_industry_composite ON sw_industry_classify (ts_code, industry_level);

-- TimescaleDB时序优化
SELECT create_hypertable('stock_daily', 'trade_date', if_not_exists => TRUE);
SELECT create_hypertable('daily_basic', 'trade_date', if_not_exists => TRUE);
```

## 安装部署指南

### 1. 环境准备

#### 使用Docker部署 (推荐)
```bash
# 克隆项目
git clone <repository-url>
cd StockSchool

# 配置环境变量
cp .env.prod.example .env
# 编辑.env文件，设置必需的环境变量

# 启动服务
docker-compose -f docker-compose.prod.yml up -d
```

#### 手动安装
```bash
# 1. 安装Python依赖
pip install -r requirements.txt

# 2. 安装PostgreSQL和TimescaleDB
# Ubuntu/Debian:
sudo apt-get install postgresql-12 timescaledb-postgresql-12

# 3. 创建数据库
sudo -u postgres createdb stockschool
sudo -u postgres psql stockschool -c "CREATE EXTENSION timescaledb;"

# 4. 初始化数据库结构
psql -U postgres -d stockschool -f database_schema.sql
```

### 2. 环境变量配置

#### 必需环境变量
```bash
# Tushare API令牌 (必需)
export TUSHARE_TOKEN="your_tushare_token"

# 数据库连接 (可选，有默认值)
export DATABASE_URL="postgresql://user:password@localhost:5432/stockschool"

# Redis连接 (可选)
export REDIS_URL="redis://localhost:6379/0"
export CELERY_BROKER_URL="redis://localhost:6379/0"

# 日志级别 (可选)
export LOG_LEVEL="INFO"
```

#### Windows系统
```cmd
set TUSHARE_TOKEN=your_tushare_token
set DATABASE_URL=postgresql://user:password@localhost:5432/stockschool
```

### 3. 数据库初始化

#### 方式一：命令行参数
```bash
python run.py --init-db
```

#### 方式二：交互式菜单
```bash
python run.py
# 选择 "4. 数据库初始化"
```

#### 方式三：手动执行 (备用)
```bash
psql -U stockschool -d postgres -f database_schema.sql
```

## 核心功能模块详解

### 1. 数据获取与同步模块

#### 数据源配置
```yaml
# config/base.yml
data_sources:
  tushare:
    enabled: true
    token: ${TUSHARE_TOKEN}
    api_limit: 200
    retry_times: 3
    retry_delay: 1
  akshare:
    enabled: true
    api_limit: 100
    retry_times: 3
    retry_delay: 1
```

#### 主要功能
- **多数据源支持**: Tushare(主) + AkShare(辅)
- **增量同步**: 基于交易日历的智能增量更新
- **数据质量控制**: 3σ原则异常检测，行业均值填充
- **申万行业分类**: 支持L1/L2/L3三级行业数据同步

#### 使用方式
```bash
# 同步基础数据
python -m src.data.sync_manager --action sync_stock_basic

# 同步日线数据
python -m src.data.sync_manager --action sync_daily --start-date 2024-01-01

# 同步申万行业数据
python -m src.data.sync_manager --mode industry --level L1
```

### 2. 因子计算引擎

#### 三大因子引擎架构
```python
# 技术面因子引擎
from src.compute.technical_factor_engine import TechnicalFactorEngine

# 基本面因子引擎  
from src.compute.fundamental_factor_engine import FundamentalFactorEngine

# 情绪面因子引擎
from src.compute.sentiment_factor_engine import SentimentFactorEngine
```

#### 因子类型覆盖

**技术面因子 (30+种)**
- 动量类: RSI(6日/14日)、威廉指标、动量指标、ROC
- 趋势类: SMA(5/10/20/60日)、EMA(12/26日)、MACD
- 波动率类: 历史波动率、ATR、布林带
- 成交量类: 成交量均线、量比、VPT、MFI

**基本面因子 (20+种)**
- 估值因子: PE、PB、PS、EV/EBITDA
- 盈利能力: ROE、ROA、毛利率、净利率
- 成长性: 营收增长率、净利润增长率
- 杠杆因子: 资产负债率、权益乘数

**情绪面因子 (20+种)**
- 资金流向: 5日/20日资金流向、净流入率
- 关注度: 换手率、成交量、价格波动关注度
- 情绪强度: 基于动量和波动率的情绪评估
- 事件因子: 异常成交量、涨跌停、跳空信号

#### 因子计算配置
```yaml
# config/factor_config.yaml
factor_params:
  rsi:
    window: 14
  macd:
    fast_period: 12
    slow_period: 26
    signal_period: 9
  bollinger:
    window: 20
    num_std: 2
```

### 3. 因子标准化和评分系统

#### 标准化方法
- **Z-score标准化**: 均值0，标准差1
- **分位数标准化**: 转换为均匀分布
- **排名标准化**: 基于排名的相对位置
- **鲁棒标准化**: 基于中位数和MAD，抗异常值

#### 行业内评分
```python
from src.compute.factor_standardizer import IndustryFactorScorer

scorer = IndustryFactorScorer()
# 按行业分组标准化，消除行业间差异
industry_scores = scorer.score_by_industry(factor_data, industry_data)
```

#### 因子合成
- **等权重合成**: 简单平均
- **加权合成**: 自定义权重
- **IC加权**: 基于信息系数的智能权重
- **PCA合成**: 主成分分析降维

### 4. 因子有效性检验

#### 检验指标体系
```python
from src.compute.factor_effectiveness_analyzer import FactorEffectivenessAnalyzer

analyzer = FactorEffectivenessAnalyzer()

# IC分析 (信息系数)
ic_results = analyzer.calculate_ic_analysis(factor_data, return_data)

# IR分析 (信息比率)  
ir_results = analyzer.calculate_ir_analysis(factor_data, return_data)

# 分层回测
layered_results = analyzer.layered_backtest(factor_data, return_data, layers=5)

# 因子衰减分析
decay_results = analyzer.factor_decay_analysis(factor_data, return_data, max_periods=20)
```

#### 综合评级系统
- **A级 (80-100分)**: 优秀因子，可直接使用
- **B级 (70-79分)**: 良好因子，推荐使用  
- **C级 (60-69分)**: 一般因子，谨慎使用
- **D级 (50-59分)**: 较差因子，需要改进
- **F级 (0-49分)**: 无效因子，不建议使用

### 5. AI模型训练与预测

#### 支持的模型类型
- 线性回归 (Linear Regression)
- 随机森林 (Random Forest)  
- XGBoost
- LightGBM (推荐)
- 神经网络 (Neural Network)

#### 训练流水线
```python
from src.ai.training_pipeline import ModelTrainingPipeline

pipeline = ModelTrainingPipeline()
training_result = pipeline.run_training_pipeline(
    factor_names=['rsi_14', 'macd', 'pe_ratio'],
    start_date='2020-01-01',
    end_date='2023-12-31',
    model_type='lightgbm'
)
```

#### 模型解释
```python
from src.strategy.model_explainer import ModelExplainer

explainer = ModelExplainer()
# SHAP值分析
shap_values = explainer.explain_prediction(model, X_test)
# 特征重要性
importance = explainer.get_feature_importance(model, method='shap')
```

### 6. 性能优化系统

#### 并行计算
```python
from src.compute.parallel_factor_calculator import ParallelFactorCalculator

calculator = ParallelFactorCalculator()
# 自动检测系统资源，智能分配进程数
results = calculator.calculate_factors_parallel(stock_list, factor_list)
```

#### 缓存系统
- **双层缓存**: Redis + 内存缓存
- **LRU策略**: 最近最少使用淘汰
- **数据压缩**: 自动压缩存储
- **缓存统计**: 命中率监控

#### 增量计算
```python
from src.compute.incremental_calculator import IncrementalFactorCalculator

calc = IncrementalFactorCalculator()
# 只计算缺失的日期，避免重复计算
calc.calculate_incremental_factors(start_date='2024-01-01')
```

### 7. API接口系统

#### RESTful API设计
```python
# 获取因子数据
GET /api/v1/factors?ts_codes=000001.SZ&factors=rsi_14&date=2024-01-01

# 触发因子计算
POST /api/v1/factors/calculate
{
    "ts_codes": ["000001.SZ", "000002.SZ"],
    "factor_types": ["technical", "fundamental"],
    "date_range": ["2024-01-01", "2024-01-31"]
}

# 获取模型预测
POST /api/v1/models/predict
{
    "model_id": "lightgbm_v1",
    "ts_codes": ["000001.SZ"],
    "date": "2024-01-01"
}
```

#### 认证授权
- **JWT令牌认证**: 无状态认证机制
- **基于角色的权限控制**: RBAC权限模型
- **令牌黑名单**: 安全退出机制

### 8. 特征商店系统

#### 版本管理
```python
from src.features.factor_feature_store import FactorFeatureStore

store = FactorFeatureStore()
# 自动版本控制
version_id = store.register_factor(
    name='rsi_14_v2',
    algorithm_hash='abc123',
    metadata={'window': 14, 'method': 'wilder'}
)
```

#### 数据血缘跟踪
- 源表追踪: 记录数据来源表
- 依赖因子: 追踪因子间依赖关系
- 转换逻辑: 记录计算过程
- 血缘可视化: 图形化展示数据流

### 9. 监控告警系统

#### 监控指标
```python
from src.monitoring.performance import PerformanceMonitor

monitor = PerformanceMonitor()
# 系统资源监控
metrics = monitor.collect_system_metrics()
# 数据质量监控  
quality_metrics = monitor.collect_data_quality_metrics()
```

#### 告警规则
- **CPU使用率**: 阈值80%
- **内存使用率**: 阈值85%  
- **错误率**: 阈值5%
- **响应时间**: 阈值1000ms
- **数据异常**: 3σ原则检测

## 使用方法详解

### 主控制台 `run.py`

#### 启动主控制台
```bash
python run.py
```

#### 主要功能菜单
1. **启动API服务器**: 启动FastAPI后端服务
2. **运行数据同步**: 数据获取和同步管理
   - 同步每日数据
   - 同步单只股票数据  
   - 回填历史数据
3. **运维和调试控制台**: 系统运维和故障诊断
   - 飞行前检查
   - 启动Celery Worker
   - 运行日常工作流
   - 数据质量检验
   - 数据修复和回填
   - 紧急情况诊断

### 数据同步操作

#### 基础数据同步
```bash
# 同步股票基本信息
python -m src.data.sync_manager --action sync_stock_basic

# 同步交易日历
python -m src.data.sync_manager --action sync_trade_calendar

# 同步日线数据
python -m src.data.sync_manager --action sync_daily --start-date 2024-01-01
```

#### 申万行业分类同步
```bash
# 同步一级行业
python -m src.data.sync_manager --mode industry --level L1

# 同步所有级别行业
python -m src.data.sync_manager --mode industry_full

# 更新特定股票行业数据
python -m src.data.sync_manager --mode industry_update --stock-file stocks.txt
```

### 因子计算操作

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

### AI模型训练与预测

#### 模型训练
```bash
# 训练股票预测模型
python -m src.ai.training_pipeline --model-type lightgbm --target-days 5

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

### 全流程自动化调度

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

# 运行全流程实测编排器
python scripts/full_test_v1_1_6.py
```

### API服务使用

#### 启动API服务
```bash
# 方式一：使用run.py
python run.py
# 选择 "1. 启动API服务器"

# 方式二：直接启动
python -m src.api.main

# 方式三：使用uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### API文档访问
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 配置管理详解

### 统一配置系统

#### 配置文件层次
```
config/
├── base.yml              # 基础配置（所有环境共享）
├── development.yml       # 开发环境配置
├── production.yml        # 生产环境配置
├── testing.yml          # 测试环境配置
├── factor_config.yaml    # 因子计算专用配置
└── monitoring.yaml       # 监控告警配置
```

#### 环境变量优先级
1. 环境变量 (最高优先级)
2. 环境特定配置文件 (如production.yml)
3. 基础配置文件 (base.yml)
4. 默认值 (最低优先级)

#### 配置热更新
```python
from src.config.unified_config import UnifiedConfig

config = UnifiedConfig()
# 运行时重新加载配置
config.reload()
```

### 主要配置参数

#### 因子计算参数
```yaml
factor_params:
  min_data_days: 60        # 最小数据天数要求
  rsi:
    window: 14             # RSI窗口期
  macd:
    fast_period: 12        # MACD快线周期
    slow_period: 26        # MACD慢线周期
    signal_period: 9       # MACD信号线周期
```

#### 模型训练参数
```yaml
training_params:
  model_name: 'LightGBM'
  prediction_window: 5     # 预测未来5天收益率
  test_size: 0.2          # 测试集比例
  lgbm_params:
    objective: 'regression_l1'
    n_estimators: 1000
    learning_rate: 0.05
```

#### GPU加速配置
```yaml
feature_params:
  use_cuda: true           # 启用GPU加速
  max_gpu_memory: 20480    # 限制GPU显存使用（MB）
  fallback_to_cpu: true    # GPU失败时自动回退到CPU
  cuda_device: 0           # CUDA设备ID
```

## 部署运维指南

### Docker容器化部署

#### 生产环境部署
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  postgres:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database_schema.sql:/docker-entrypoint-initdb.d/01-schema.sql
    
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - TUSHARE_TOKEN=${TUSHARE_TOKEN}
    depends_on:
      - postgres
      - redis
    ports:
      - "8000:8000"
```

#### 部署命令
```bash
# 构建并启动服务
docker-compose -f docker-compose.prod.yml up -d

# 查看服务状态
docker-compose -f docker-compose.prod.yml ps

# 查看日志
docker-compose -f docker-compose.prod.yml logs -f api
```

### 监控和日志

#### 系统监控
```python
# 启动监控服务
python scripts/start_monitoring.py

# 查看性能指标
python -c "
from src.monitoring.performance import PerformanceMonitor
monitor = PerformanceMonitor()
print(monitor.get_system_status())
"
```

#### 日志管理
```python
# 配置日志系统
from src.utils.logging_config import setup_logging
setup_logging()

# 日志文件位置
logs/
├── app.log              # 应用日志
├── error.log            # 错误日志
├── celery.log           # Celery任务日志
└── api.log              # API访问日志
```

### 备份和恢复

#### 数据库备份
```bash
# 全量备份
docker exec stockschool_postgres pg_dump -U stockschool -d stockschool | gzip > backup_$(date +%Y%m%d).sql.gz

# 恢复数据库
gunzip -c backup_20241201.sql.gz | docker exec -i stockschool_postgres psql -U stockschool -d stockschool
```

#### 配置备份
```bash
# 备份配置文件
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/ .env

# 备份模型文件
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
```

## 故障排查指南

### 常见问题诊断

#### 数据库连接问题
```bash
# 检查数据库连接
python -c "
from src.utils.db import get_db_engine
engine = get_db_engine()
with engine.connect() as conn:
    result = conn.execute('SELECT version();')
    print(result.fetchone())
"
```

#### Celery任务问题
```bash
# 检查Celery worker状态
celery -A src.compute.tasks inspect active

# 检查任务队列
celery -A src.compute.tasks inspect reserved

# 清空任务队列
celery -A src.compute.tasks purge
```

#### GPU相关问题
```python
# 检查GPU可用性
from src.utils.gpu_utils import is_gpu_available, get_gpu_info
print(f"GPU可用: {is_gpu_available()}")
print(f"GPU信息: {get_gpu_info()}")
```

### 性能优化建议

#### 数据库优化
```sql
-- 检查慢查询
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- 检查索引使用情况
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

#### 内存优化
```python
# 监控内存使用
import psutil
process = psutil.Process()
memory_info = process.memory_info()
print(f"内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")
```

## 重复内容标识

### ~~已删除的旧架构文件~~
- ~~`src/data/tushare_sync.py`~~ - 已被新的数据源架构替代
- ~~`src/data/akshare_sync.py`~~ - 已被新的数据源架构替代  
- ~~旧版配置系统~~ - 已迁移到统一配置管理系统

### README.md中的重复内容
- ~~第二阶段优化部分重复~~ - 建议合并重复的"第二阶段优化"章节
- ~~技术栈描述重复~~ - MySQL/SQLite描述已过时，实际使用PostgreSQL

### {推荐优化内容}

#### {配置系统优化}
```python
# 推荐使用新的统一配置系统
from src.config.unified_config import UnifiedConfig
config = UnifiedConfig()

# 而不是旧的配置加载器
# from src.utils.config_loader import Config  # 已弃用
```

#### {数据库连接优化}
```python
# 推荐使用连接池
from src.utils.db import get_db_engine
engine = get_db_engine()  # 自动管理连接池

# 而不是直接创建连接
# import psycopg2  # 不推荐直接使用
```

#### {GPU加速优化}
```python
# 推荐启用GPU加速（如果可用）
feature_params:
  use_cuda: true
  max_gpu_memory: 20480
  fallback_to_cpu: true  # 自动降级保证兼容性
```

## 开发规范

### 代码规范
- 遵循PEP8代码风格
- 使用类型提示(Type Hints)
- 函数和类必须包含docstring
- 变量命名使用snake_case，类名使用PascalCase

### 测试规范
- 单元测试覆盖率 ≥ 80%
- 关键业务逻辑覆盖率 ≥ 95%
- 新增代码覆盖率 ≥ 90%

### 提交规范
- 使用语义化提交信息
- 每次提交包含相关测试
- 重大变更需要更新文档

## 版本历史

### v1.1.7 (2025-01-04) - 统一配置管理系统
- ✅ 建立统一配置管理器`UnifiedConfig`
- ✅ 自动迁移30个文件的配置导入语句
- ✅ 支持多环境配置、热更新、降级机制

### v1.1.6 (2024-12) - 全流程实测阶段  
- ✅ AI模型训练与预测完整流程
- ✅ Celery任务编排和调度系统
- ✅ 全流程测试和质量保证体系

### v1.1.5 (2024-07) - 真实数据测试完成
- ✅ Docker测试环境和数据质量验证
- ✅ 因子计算精度验证和技术指标修复
- ✅ 端到端集成测试覆盖

### v1.1 (2024-07) - 核心功能升级
- ✅ 完整的量化投研流水线
- ✅ 多因子模型构建能力  
- ✅ 自动化预测和决策支持

## 总结

StockSchool是一个功能完整、架构清晰的量化投研系统，具备以下特点：

### 技术优势
1. **模块化架构**: 清晰的职责分离，易于维护和扩展
2. **高性能计算**: 并行计算、缓存优化、GPU加速
3. **完整的数据流**: 从数据获取到AI决策的完整链路
4. **企业级特性**: 监控告警、容错机制、版本管理

### 业务价值  
1. **多维度分析**: 技术面、基本面、情绪面三维因子体系
2. **智能决策**: AI模型自动学习和预测
3. **可解释性**: SHAP分析提供模型解释
4. **个性化**: 支持用户自定义策略和参数

### 部署优势
1. **容器化**: Docker支持，便于部署和扩展
2. **配置化**: 高度可配置，适应不同环境需求  
3. **监控完善**: 全方位监控和告警机制
4. **文档完整**: 详细的使用和部署文档

该系统为量化投资研究和策略开发提供了完整的技术基础设施，具备生产级的稳定性和性能。