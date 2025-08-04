# StockSchool AI策略系统 - 第三阶段

## 概述

第三阶段是StockSchool AI策略系统的高级功能模块，提供了完整的AI模型监控、系统优化、文档生成、测试框架和部署管理功能。

## 核心功能模块

### 1. 模型监控 (Model Monitor)
- **功能**: AI模型性能监控、数据漂移检测、自动告警
- **特性**:
  - 实时性能监控
  - 数据漂移检测
  - 自动重训练触发
  - 多渠道告警通知
  - 监控仪表板

### 2. 系统优化 (System Optimizer)
- **功能**: 系统性能优化、资源管理、自动调优
- **特性**:
  - 系统指标收集
  - 性能瓶颈识别
  - 自动缓存清理
  - 数据库优化
  - 日志归档管理

### 3. 文档生成 (Document Generator)
- **功能**: 自动化文档生成、API文档、系统文档
- **特性**:
  - API文档自动生成
  - 系统架构文档
  - 用户操作手册
  - 部署指南
  - 多格式输出支持

### 4. 测试框架 (Test Framework)
- **功能**: 全面的自动化测试框架
- **特性**:
  - 单元测试
  - 集成测试
  - 性能测试
  - UI测试
  - 并行测试执行
  - 详细测试报告

### 5. 部署管理 (Deployment Manager)
- **功能**: 智能部署管理、多环境支持、自动回滚
- **特性**:
  - 多环境部署
  - Docker/Kubernetes支持
  - 健康检查
  - 自动回滚
  - 部署历史追踪

## 快速开始

### 环境要求

- Python 3.8+
- PostgreSQL 12+
- Redis 6+
- Docker (可选)
- Kubernetes (可选)

### 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt

# 安装额外的第三阶段依赖
pip install -r requirements_stage3.txt
```

### 配置环境变量

```bash
# 数据库配置
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=stockschool
export DB_USER=postgres
export DB_PASSWORD=your_password

# Redis配置
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_PASSWORD=your_redis_password

# JWT密钥
export JWT_SECRET_KEY=your_jwt_secret_key

# 部署环境
export DEPLOY_ENV=development

# 日志级别
export LOG_LEVEL=INFO
```

### 启动系统

#### 1. 初始化模式
```bash
# 仅初始化数据库和配置
python start_stage3.py --mode init
```

#### 2. 完整启动模式
```bash
# 启动所有服务（包括API服务器）
python start_stage3.py --mode start

# 指定端口启动
python start_stage3.py --mode start --port 8080

# 不启动API服务器
python start_stage3.py --mode start --no-api
```

#### 3. 健康检查模式
```bash
# 运行系统健康检查
python start_stage3.py --mode health
```

#### 4. 信息显示模式
```bash
# 显示系统信息
python start_stage3.py --mode info
```

#### 5. 交互模式
```bash
# 进入交互式命令行
python start_stage3.py --mode interactive
```

## API接口

第三阶段提供了完整的RESTful API接口，默认运行在 `http://localhost:8000`

### 模型监控API

```bash
# 设置模型监控
POST /api/v3/model-monitor/setup

# 执行模型检查
POST /api/v3/model-monitor/check

# 获取监控仪表板
GET /api/v3/model-monitor/dashboard

# 获取模型告警
GET /api/v3/model-monitor/alerts
```

### 系统优化API

```bash
# 创建优化任务
POST /api/v3/system-optimizer/optimize

# 获取系统健康状态
GET /api/v3/system-optimizer/health

# 获取系统指标
GET /api/v3/system-optimizer/metrics
```

### 文档生成API

```bash
# 生成文档
POST /api/v3/doc-generator/generate

# 获取已生成文档列表
GET /api/v3/doc-generator/documents
```

### 测试框架API

```bash
# 执行测试
POST /api/v3/test-framework/run

# 获取测试结果
GET /api/v3/test-framework/results

# 获取测试报告
GET /api/v3/test-framework/report
```

### 部署管理API

```bash
# 启动部署
POST /api/v3/deployment-manager/deploy

# 回滚部署
POST /api/v3/deployment-manager/rollback

# 获取部署状态
GET /api/v3/deployment-manager/status

# 执行健康检查
POST /api/v3/deployment-manager/health-check

# 获取部署报告
GET /api/v3/deployment-manager/report
```

## 配置说明

### 主配置文件

配置文件位于 `config/stage3_config.py`，包含所有模块的配置参数：

- `ModelMonitorConfig`: 模型监控配置
- `SystemOptimizerConfig`: 系统优化配置
- `DocumentGeneratorConfig`: 文档生成配置
- `TestFrameworkConfig`: 测试框架配置
- `DeploymentManagerConfig`: 部署管理配置
- `DatabaseConfig`: 数据库配置
- `RedisConfig`: Redis配置
- `LoggingConfig`: 日志配置
- `SecurityConfig`: 安全配置

### 环境配置

系统支持三种环境配置：

- `development`: 开发环境
- `staging`: 测试环境
- `production`: 生产环境

通过环境变量 `ENVIRONMENT` 指定当前环境。

## 使用示例

### 1. 设置模型监控

```python
from src.ai.strategy.model_monitor import ModelMonitor

# 创建监控实例
monitor = ModelMonitor()

# 设置监控配置
config = {
    'model_name': 'lstm_predictor',
    'model_version': '1.0.0',
    'monitoring_metrics': ['accuracy', 'precision', 'recall'],
    'drift_detection_enabled': True,
    'alert_thresholds': {
        'accuracy_drop': 0.05,
        'drift_score': 0.1
    }
}

# 启动监控
await monitor.setup_monitoring(config)
```

### 2. 执行系统优化

```python
from src.ai.strategy.system_optimizer import SystemOptimizer

# 创建优化器实例
optimizer = SystemOptimizer()

# 创建优化任务
task_config = {
    'task_type': 'comprehensive',
    'target_metrics': ['cpu_usage', 'memory_usage', 'response_time'],
    'optimization_level': 'aggressive'
}

# 执行优化
result = await optimizer.create_optimization_task(task_config)
```

### 3. 生成文档

```python
from src.ai.strategy.doc_generator import DocumentGenerator

# 创建文档生成器实例
generator = DocumentGenerator()

# 生成API文档
api_docs = await generator.generate_api_documentation()

# 生成系统文档
system_docs = await generator.generate_system_documentation()

# 生成用户手册
user_manual = await generator.generate_user_manual()
```

### 4. 执行测试

```python
from src.ai.strategy.test_framework import TestFramework

# 创建测试框架实例
test_framework = TestFramework()

# 运行测试套件
results = await test_framework.run_test_suite(
    suite_name='api_tests',
    parallel=True
)

# 生成测试报告
report = await test_framework.generate_test_report(
    results=results,
    format='html'
)
```

### 5. 部署应用

```python
from src.ai.strategy.deployment_manager import DeploymentManager

# 创建部署管理器实例
deployment_manager = DeploymentManager()

# 部署应用
deployment_config = {
    'application_name': 'stockschool-api',
    'version': '3.0.0',
    'environment': 'staging',
    'deployment_type': 'docker'
}

result = await deployment_manager.deploy_application(deployment_config)
```

## 监控和维护

### 系统健康检查

```bash
# 检查系统整体健康状态
curl http://localhost:8000/api/v3/health

# 获取详细系统信息
curl http://localhost:8000/api/v3/system-info
```

### 日志查看

```bash
# 查看系统日志
tail -f logs/stage3.log

# 查看特定模块日志
grep "ModelMonitor" logs/stage3.log
```

### 性能监控

系统提供了内置的性能监控功能，可以通过以下方式查看：

1. **监控仪表板**: `http://localhost:8000/api/v3/model-monitor/dashboard`
2. **系统指标**: `http://localhost:8000/api/v3/system-optimizer/metrics`
3. **健康报告**: `http://localhost:8000/api/v3/system-optimizer/health`

## 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查数据库服务是否运行
   - 验证连接参数是否正确
   - 确认数据库用户权限

2. **Redis连接失败**
   - 检查Redis服务是否运行
   - 验证Redis配置参数
   - 检查网络连接

3. **模块初始化失败**
   - 查看详细错误日志
   - 检查依赖是否安装完整
   - 验证配置文件格式

4. **API服务无法访问**
   - 检查端口是否被占用
   - 验证防火墙设置
   - 确认服务启动状态

### 调试模式

```bash
# 启用调试模式
export LOG_LEVEL=DEBUG
python start_stage3.py --mode start
```

### 重置系统

```bash
# 重新初始化数据库
python start_stage3.py --mode init

# 清理缓存
redis-cli FLUSHALL
```

## 扩展开发

### 添加新的监控指标

1. 在 `ModelMonitor` 类中添加新的指标收集方法
2. 更新数据库表结构
3. 修改监控配置
4. 添加相应的API接口

### 自定义优化策略

1. 继承 `SystemOptimizer` 类
2. 实现自定义优化逻辑
3. 注册新的优化任务类型
4. 更新配置文件

### 扩展测试框架

1. 添加新的测试类型
2. 实现测试执行器
3. 扩展测试报告格式
4. 集成外部测试工具

## 版本历史

### v3.0.0 (当前版本)
- 初始发布
- 包含所有核心功能模块
- 完整的API接口
- 基础监控和优化功能

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 支持

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件
- 查看文档

---

**注意**: 第三阶段是一个高级功能模块，建议在熟悉前两个阶段的基础上使用。