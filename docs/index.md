# StockSchool 文档中心

## 📚 文档目录

### 🎯 快速入门
- [🚀 快速开始](../README.md#快速开始) - 5分钟快速上手指南
- [📋 安装指南](../README.md#安装和配置) - 详细的安装和配置说明
- [🎮 使用教程](../README.md#使用方法) - 系统基本使用方法

### 📖 用户手册
- [📘 用户手册](./user_manual.md) - 完整的系统使用指南
  - [系统概述](./user_manual.md#系统概述)
  - [安装指南](./user_manual.md#安装指南)
  - [快速开始](./user_manual.md#快速开始)
  - [核心功能](./user_manual.md#核心功能)
  - [API使用](./user_manual.md#api使用)
  - [模型解释](./user_manual.md#模型解释)
  - [性能优化](./user_manual.md#性能优化)
  - [故障排除](./user_manual.md#故障排除)
  - [最佳实践](./user_manual.md#最佳实践)

### 🛠️ 部署指南
- [🔧 部署指南](./deployment_guide.md) - 生产环境部署和维护
  - [部署环境准备](./deployment_guide.md#部署环境准备)
  - [本地开发部署](./deployment_guide.md#本地开发部署)
  - [生产环境部署](./deployment_guide.md#生产环境部署)
  - [Docker部署](./deployment_guide.md#docker部署)
  - [Kubernetes部署](./deployment_guide.md#kubernetes部署)
  - [负载均衡配置](./deployment_guide.md#负载均衡配置)
  - [监控告警配置](./deployment_guide.md#监控告警配置)
  - [备份恢复策略](./deployment_guide.md#备份恢复策略)
  - [安全配置](./deployment_guide.md#安全配置)
  - [性能调优](./deployment_guide.md#性能调优)

### 📡 API文档
- [📊 API文档](./api_documentation.md) - 完整的API接口文档
  - [基础信息](./api_documentation.md#基础信息)
  - [API端点](./api_documentation.md#api端点)
  - [错误处理](./api_documentation.md#错误处理)
  - [认证](./api_documentation.md#认证)
  - [限流](./api_documentation.md#限流)
  - [WebSocket支持](./api_documentation.md#websocket支持)
  - [版本历史](./api_documentation.md#版本历史)
  - [客户端SDK](./api_documentation.md#客户端sdk)
  - [性能优化建议](./api_documentation.md#性能优化建议)

### 🧠 核心模块

#### 数据管理
- [💾 数据同步](../README.md#1-数据同步) - 数据获取和同步机制
- [🔍 数据质量控制](../src/compute/quality.py) - 数据质量检查和控制
- [🔄 数据修复](../scripts/fix_data_sync.py) - 数据修复和回填工具

#### 因子计算
- [🧮 技术因子](../src/compute/indicators.py) - 20+种技术指标计算
- [📊 基本面因子](../src/compute/indicators.py) - 财务指标计算
- [⚙️ 因子预处理](../src/compute/processing.py) - 去极值、标准化、中性化
- [🏭 因子引擎](../src/compute/factor_engine.py) - 因子计算引擎

#### AI模型
- [🤖 模型训练](../src/ai/training_pipeline.py) - 机器学习模型训练流水线
- [🔮 模型预测](../src/ai/prediction.py) - 每日股票预测
- [📈 模型评估](../src/strategy/evaluation.py) - 策略评估和绩效分析
- [🧠 模型解释](../src/strategy/model_explainer.py) - SHAP和置换重要性解释

#### 模型解释模块
- [🔍 SHAP解释器](../src/strategy/shap_explainer.py) - SHAP模型解释
- [🔄 置换解释器](../src/strategy/permutation_explainer.py) - 置换重要性解释
- [🎨 可视化模块](../src/strategy/visualization.py) - 模型解释可视化

#### GPU工具
- [⚡ GPU管理器](../src/utils/gpu_utils.py) - GPU资源管理和优化
- [🧠 GPU加速](../README.md#gpu工具模块) - GPU加速配置和使用

### 🔧 开发工具

#### 自动化调度
- [⏰ Celery任务](../src/compute/tasks.py) - 异步任务处理
- [🔁 工作流调度](../run_daily_workflow.py) - 日度工作流调度
- [🧪 全流程测试](../scripts/full_test_v1_1_6.py) - 完整测试编排器

#### 监控告警
- [📊 性能监控](../src/monitoring/performance.py) - 系统性能监控
- [🚨 告警引擎](../src/monitoring/alerts.py) - 实时告警系统
- [📈 监控仪表板](../src/monitoring/explainer_monitor.py) - 模型解释监控

#### 测试验证
- [🧪 单元测试](../src/tests/) - 完整的测试套件
- [📊 测试覆盖率](../logs/check_test_coverage.py) - 测试覆盖率检查
- [⚡ 性能基准](../logs/run_performance_benchmark.py) - 性能基准测试
- [🛡️ 异常处理](../logs/test_exception_handling.py) - 异常情况处理测试

### 📊 系统架构

#### 目录结构
```
StockSchool/
├── src/                    # 源代码目录
│   ├── ai/                # AI模型训练与预测
│   ├── api/               # Web API接口
│   ├── compute/           # 计算引擎
│   ├── data/              # 数据同步
│   ├── database/          # 数据库相关
│   ├── features/          # 特征商店
│   ├── monitoring/        # 监控告警
│   ├── strategy/          # 策略评估与AI模型
│   ├── tests/             # 单元测试
│   └── utils/             # 工具函数
├── scripts/               # 工具脚本
├── docs/                  # 文档目录
├── logs/                  # 日志目录
├── models/                # 模型文件
├── cache/                 # 缓存文件
└── data/                  # 数据文件
```

#### 核心组件
- [🌐 API服务](../src/api/) - FastAPI RESTful接口
- [⚙️ 计算引擎](../src/compute/) - 因子计算和数据处理
- [🧠 AI模块](../src/ai/) - 机器学习模型训练和预测
- [📊 策略模块](../src/strategy/) - 策略评估和模型解释
- [👁️ 监控模块](../src/monitoring/) - 性能监控和告警系统
- [🛠️ 工具函数](../src/utils/) - 通用工具和辅助函数

### 🎯 最佳实践

#### 性能优化
- [⚡ GPU加速最佳实践](../README.md#性能特点)
- [キャッシング 缓存策略](../src/features/feature_store.py)
- [🔄 批量处理优化](../src/utils/gpu_utils.py)
- [🧠 内存管理](../src/utils/gpu_utils.py)

#### 安全配置
- [🔒 API安全](../src/api/main.py)
- [🛡️ 数据安全](../src/database/)
- [🔐 认证授权](../src/utils/auth.py)
- [防火墙配置](./deployment_guide.md#网络安全)

#### 监控告警
- [📈 系统监控](../src/monitoring/performance.py)
- [🚨 告警规则](../src/monitoring/alerts.py)
- [📊 性能指标](../src/monitoring/explainer_monitor.py)
- [🔔 通知渠道](../src/monitoring/alerts.py)

### 🛠️ 运维管理

#### 日常维护
- [🧹 日志轮转](../src/monitoring/logger.py)
- [💾 数据备份](./deployment_guide.md#备份恢复策略)
- [🔄 系统升级](./deployment_guide.md#升级维护)
- [🔍 健康检查](../src/api/main.py)

#### 故障处理
- [🚑 紧急诊断](../run.py)
- [🔄 自动恢复](../src/utils/gpu_utils.py)
- [📊 性能调优](./deployment_guide.md#性能调优)
- [🛡️ 容错机制](../src/utils/retry.py)

#### 性能调优
- [⚡ GPU优化](../src/utils/gpu_utils.py)
- [🧠 内存优化](../src/utils/gpu_utils.py)
- [🔄 并行处理](../src/compute/tasks.py)
- [キャッシング 缓存优化](../src/features/feature_store.py)

### 📈 开发指南

#### 扩展开发
- [➕ 添加新因子](../src/compute/indicators.py)
- [🤖 添加新模型](../src/ai/training_pipeline.py)
- [📈 添加新指标](../src/strategy/evaluation.py)
- [🔍 添加新解释方法](../src/strategy/model_explainer.py)

#### 模块集成
- [🔗 API集成](../src/api/)
- [🔄 任务调度](../src/compute/tasks.py)
- [📊 数据流](../src/compute/factor_engine.py)
- [🧠 模型管道](../src/ai/training_pipeline.py)

#### 测试开发
- [🧪 单元测试](../src/tests/)
- [📊 集成测试](../tests/)
- [⚡ 性能测试](../logs/run_performance_benchmark.py)
- [🛡️ 压力测试](../logs/test_exception_handling.py)

### 📞 支持资源

#### 技术支持
- [📧 邮箱支持](mailto:support@stockschool.com)
- [🌐 在线文档](https://docs.stockschool.com)
- [💬 社区论坛](https://community.stockschool.com)
- [🐛 问题反馈](https://github.com/yourusername/stockschool/issues)

#### 学习资源
- [🎓 教程视频](https://youtube.com/stockschool)
- [📚 技术博客](https://blog.stockschool.com)
- [🔬 案例研究](https://case.stockschool.com)
- [📊 最佳实践](./user_manual.md#最佳实践)

#### 贡献指南
- [📋 贡献流程](../README.md#贡献指南)
- [📝 代码规范](../CONTRIBUTING.md)
- [🧪 测试指南](../src/tests/)
- [📄 文档贡献](./index.md)

---

## 🚀 快速导航

### 常用链接
- [🏠 主页](../README.md)
- [📖 用户手册](./user_manual.md)
- [🔧 部署指南](./deployment_guide.md)
- [📊 API文档](./api_documentation.md)
- [🧪 测试套件](../src/tests/)
- [📈 开发日志](../log.md)
- [📋 配置文件](../config.yml)
- [🗄️ 数据库结构](../database_schema.sql)

### 快速搜索
使用 `Ctrl+F` 或 `Cmd+F` 快速搜索关键词：
- `API` - API相关文档
- `GPU` - GPU加速和优化
- `模型` - AI模型训练和预测
- `因子` - 因子计算和处理
- `部署` - 部署和运维指南
- `监控` - 性能监控和告警
- `测试` - 测试和验证文档

---

*文档版本: v1.1.7*
*最后更新: 2025-07-31*
*文档中心: https://docs.stockschool.com*
