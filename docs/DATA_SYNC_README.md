# StockSchool 数据同步增强功能

<!--
## 概述

StockSchool 数据同步增强功能是一个完整的多数据源同步解决方案，实现了智能化、自动化的股票数据获取和管理。该功能包含四个核心模块：

1. **Akshare数据源集成** - 获取情绪面数据（新闻情绪、用户关注度、人气榜）
2. **申万行业分类管理** - 管理三级行业分类体系和股票行业归属
3. **智能增量更新引擎** - 自动检测缺失数据并智能调度同步任务
4. **统一数据同步管理** - 协调多数据源同步，提供统一管理界面
-->

<!--
## 核心特性

### 🎯 智能化
- 自动检测缺失数据，基于交易日历智能识别需要同步的日期
- 智能任务调度，根据数据类型和时间自动确定优先级
- 自适应重试机制，根据错误类型采用不同的重试策略

### 🔄 可靠性
- 完整的错误处理和重试机制
- 同步状态持久化，支持断点续传
- 数据质量监控，3σ原则异常检测
- UPSERT模式避免重复数据

### ⚡ 高效性
- 并发任务执行，支持多线程同步
- API调用频率限制，避免触发限制
- 增量更新机制，只同步缺失数据
- 数据压缩和分区优化存储

### 📊 可观测性
- 详细的同步状态监控
- 数据质量评分和健康检查
- 性能统计和趋势分析
- 实时告警和异常通知
-->

<!--
## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export TUSHARE_TOKEN="your_tushare_token"
export POSTGRES_PASSWORD="your_db_password"

# 启动数据库
docker-compose up -d postgres redis
```

### 2. 数据库初始化

```bash
# 执行数据库初始化
python run.py --init-db

# 应用数据库更新
psql -U stockschool -d stockschool -f database_updates/akshare_tables.sql
```

### 3. 运行演示

```bash
# 运行功能演示
python demo_data_sync.py
```

### 4. 基本使用

```bash
# 查看同步状态
<del>python src/data/sync_manager.py --mode status</del> ~~[已更新为data_sync_scheduler.py]~~
python src/data/data_sync_scheduler.py --mode status

# 执行快速同步（最新数据）
<del>python src/data/sync_manager.py --mode quick</del> ~~[已更新为data_sync_scheduler.py]~~
python src/data/data_sync_scheduler.py --mode quick

# 执行完整同步
<del>python src/data/sync_manager.py --mode full --include-incremental</del> ~~[已更新为data_sync_scheduler.py]~~
python src/data/data_sync_scheduler.py --mode full --include-incremental

# 清理旧数据
<del>python src/data/sync_manager.py --mode cleanup --cleanup-days 90</del> ~~[已更新为data_sync_scheduler.py]~~
python src/data/data_sync_scheduler.py --mode cleanup --cleanup-days 90
```
-->

<!--
## 详细使用指南

### Akshare数据同步

```bash
# 同步新闻情绪数据
python src/data/akshare_sync.py --mode news --start-date 2024-01-01 --end-date 2024-01-07

# 同步用户关注度数据
python src/data/akshare_sync.py --mode attention --start-date 2024-01-01 --end-date 2024-01-07

# 同步人气榜数据
python src/data/akshare_sync.py --mode popularity --start-date 2024-01-01 --end-date 2024-01-07

# 同步所有Akshare数据
python src/data/akshare_sync.py --mode all --start-date 2024-01-01 --end-date 2024-01-07
```

### 申万行业分类同步

```bash
# 更新申万行业分类数据
python src/data/sw_classification_sync.py --mode update

# 同步股票行业归属
python src/data/sw_classification_sync.py --mode stock_mapping

# 查看行业分类统计
python src/data/sw_classification_sync.py --mode stats
```
-->

<!--
## 配置选项

### 主配置文件 (config/data_sync.yaml)

```yaml
# 数据同步主配置
main:
  # 是否启用数据同步
  enabled: true
  # 并发任务数量
  max_concurrent_tasks: 5
  # 日志级别
  log_level: INFO
  # 告警通知邮箱
  alert_email: support@stockschool.com

# Akshare配置
akshare:
  # 是否启用Akshare数据源
  enabled: true
  # API调用间隔 (秒)
  api_interval: 2
  # 最大重试次数
  max_retries: 3
  # 重试间隔 (秒)
  retry_interval: 5
  # 超时时间 (秒)
  timeout: 30
  # 数据类型配置
  data_types:
    news: true
    attention: true
    popularity: true

# 申万行业配置
sw_classification:
  enabled: true
  # 同步频率 (天)黄
  sync_frequency_days: 7

# 数据库配置
database:
  # PostgreSQL连接信息
  postgres:
    host: localhost
    port: 5432
    user: stockschool
    password: ${POSTGRES_PASSWORD}
    database: stockschool
  # Redis连接信息
  redis:
    host: localhost
    port: 6379
    db: 0
```
-->

<!--
## 数据质量控制

### 数据验证规则

1. **完整性检查** - 确保所有必填字段都有值
2. **格式检查** - 验证数据格式是否符合预期（日期、数值等）
3. **范围检查** - 确保数值在合理范围内
4. **一致性检查** - 验证不同数据源之间的数据一致性
5. **异常检测** - 使用3σ原则检测异常值

### 数据质量报告

```bash
# 生成数据质量报告
python src/data/quality_report.py --output data_quality_report.html
```
-->

<!--
## 常见问题

1. **Q: 数据同步失败如何处理?**
   A: 首先查看日志文件 (logs/data_sync.log) 中的错误信息，根据错误类型采取相应措施。常见问题包括API密钥无效、网络连接问题、数据库权限不足等。

2. **Q: 如何提高数据同步速度?**
   A: 可以尝试增加并发任务数量、优化网络连接、使用缓存机制减少重复计算等方法。
-->

3. **Q: 数据同步过程中如何避免API调用限制?**
   A: 系统已内置API调用频率限制机制，会自动控制调用频率。也可以在配置文件中调整api_interval参数。

4. **Q: 如何定期执行数据同步?**
   A: 可以使用crontab或Windows任务计划程序定期执行同步命令，例如每天凌晨2点执行完整同步。

5. **Q: 数据同步会占用大量存储空间吗?**
   A: 系统采用增量更新机制，只会存储新数据和变更数据，同时提供数据清理功能，可以定期清理旧数据。

## 性能优化建议

1. **使用SSD存储** - 提高数据库读写性能
2. **增加内存** - 提高Redis缓存命中率
3. **优化网络连接** - 确保与数据源API的网络连接稳定
4. **合理设置并发数** - 根据服务器性能调整并发任务数量
5. **定期清理旧数据** - 减少数据库存储压力

## 故障排除

### 常见错误及解决方法

| 错误类型 | 可能原因 | 解决方法 |
|---------|---------|---------|
| API调用失败 | 网络问题、API密钥无效 | 检查网络连接、验证API密钥 |
| 数据库连接失败 | 数据库未启动、连接参数错误 | 检查数据库状态、验证连接参数 |
| 数据格式错误 | 数据源返回格式变更 | 更新数据解析代码 |
| 同步任务卡死 | 资源耗尽、死锁 | 重启同步服务、优化资源配置 |

### 联系支持

如果遇到无法解决的问题，请发送邮件至 [support@stockschool.com](mailto:support@stockschool.com)，并附上详细的错误日志。

### 申万行业分类管理

```bash
# 同步行业分类数据
python src/data/industry_classification.py --mode classification --level all

# 同步股票行业归属映射
python src/data/industry_classification.py --mode mapping

# 完整同步
python src/data/industry_classification.py --mode full

# 验证数据完整性
python src/data/industry_classification.py --mode validate

# 查询股票行业归属
python src/data/industry_classification.py --mode query --ts-code 000001.SZ --date 2024-01-01
```

### 智能增量更新

```bash
# 检查缺失数据
python src/data/incremental_update.py --mode check --data-sources tushare akshare --data-types daily daily_basic

# 调度增量同步任务
python src/data/incremental_update.py --mode schedule --days-back 7

# 执行同步任务
python src/data/incremental_update.py --mode execute --max-concurrent 4

# 查看状态摘要
python src/data/incremental_update.py --mode status
```

## API使用示例

### Python API

```python
<del>from src.data.sync_manager import DataSyncManager</del> ~~[已更新为DataSyncScheduler]~~
from src.data.data_sync_scheduler import DataSyncScheduler
from src.data.akshare_sync import AkshareSynchronizer
from src.data.industry_classification import IndustryClassificationManager
from src.data.incremental_update import IncrementalUpdateManager

# 统一同步管理
<del>manager = DataSyncManager()</del> ~~[已更新为DataSyncScheduler]~~
manager = DataSyncScheduler()

# 执行完整同步
result = manager.full_sync(
    data_sources=['tushare', 'akshare', 'industry'],
    include_incremental=True
)

# 执行快速同步
result = manager.quick_sync(data_types=['daily', 'daily_basic'])

# 获取同步状态
status = manager.get_sync_status()

# Akshare数据同步
akshare_sync = AkshareSynchronizer()

# 同步新闻情绪数据
result = akshare_sync.sync_news_sentiment(
    start_date='2024-01-01',
    end_date='2024-01-07'
)

# 行业分类管理
industry_manager = IndustryClassificationManager()

# 查询股票行业归属
industry_info = industry_manager.get_stock_industry_history(
    ts_code='000001.SZ',
    date='2024-01-01'
)

# 增量更新管理
incremental_manager = IncrementalUpdateManager()

# 获取缺失日期
missing_dates = incremental_manager.get_missing_dates(
    data_type='daily',
    data_source='tushare'
)

# 调度同步任务
created_tasks = incremental_manager.schedule_incremental_sync(
    data_sources=['tushare', 'akshare'],
    data_types=['daily', 'news_sentiment'],
    days_back=7
)
```

## 配置说明

### 数据源配置 (config.yml)

```yaml
data_sources:
  tushare:
    enabled: true
    api_limit: 200  # 每分钟调用限制
    retry_times: 3
    retry_delay: 1
    
  akshare:
    enabled: true
    api_limit: 100
    retry_times: 3
    retry_delay: 2

sync_strategy:
  incremental:
    enabled: true
    check_interval: 3600  # 1小时检查一次
    max_workers: 4
    batch_size: 100
    
  full:
    enabled: true
    schedule: "0 2 * * 0"  # 每周凌晨2点全量同步

data_quality:
  outlier_detection:
    enabled: true
    threshold: 3  # 3σ原则
    
  missing_value_handling:
    method: "forward_fill_industry_mean"
    
  anomaly_alert:
    enabled: true
    webhook_url: "${ALERT_WEBHOOK_URL}"
```

### 环境变量

```bash
# 必需的环境变量
TUSHARE_TOKEN=your_tushare_token_here
POSTGRES_PASSWORD=your_secure_password
REDIS_PASSWORD=your_redis_password

# 可选的环境变量
DATABASE_URL=postgresql://stockschool:password@localhost:5432/stockschool
REDIS_URL=redis://:password@localhost:6379/0
ALERT_WEBHOOK_URL=https://your-webhook-url
LOG_LEVEL=INFO
ENVIRONMENT=production
```

## 数据库表结构

### 新增表

1. **news_sentiment** - 新闻情绪数据
2. **user_attention** - 用户关注度数据
3. **popularity_ranking** - 人气榜数据
4. **industry_classification** - 申万行业分类
5. **stock_industry_mapping** - 股票行业归属映射
6. **sw_industry_history** - 行业变更历史
7. **sync_status** - 同步状态管理
8. **data_quality_metrics** - 数据质量监控

### 索引优化

- 按 `stock_code` + `end_date` 建立复合索引
- 使用 TimescaleDB 的 `create_hypertable` 进行时间分区
- 启用数据压缩优化存储性能

## 监控和告警

### 数据质量监控

- **完整性检查**: 检测缺失数据和覆盖率
- **准确性检查**: 3σ原则异常值检测
- **时效性检查**: 数据更新延迟监控
- **一致性检查**: 跨数据源数据一致性验证

### 同步状态监控

- **实时状态**: 当前运行的同步任务
- **历史记录**: 同步任务的历史执行记录
- **性能统计**: 同步速度、成功率等指标
- **错误分析**: 错误分类和趋势分析

### 告警机制

- **数据异常告警**: 数据质量下降时发送告警
- **同步失败告警**: 同步任务失败时通知
- **性能告警**: 性能指标异常时提醒
- **系统告警**: 系统资源异常时警报

## 性能优化

### 数据库优化

```sql
-- TimescaleDB 优化配置
SELECT set_chunk_time_interval('stock_daily', INTERVAL '1 month');
SELECT add_compression_policy('stock_daily', INTERVAL '3 months');
SELECT add_retention_policy('stock_daily', INTERVAL '5 years');

-- 复合索引
CREATE INDEX idx_stock_daily_composite ON stock_daily (ts_code, trade_date);
CREATE INDEX idx_news_sentiment_composite ON news_sentiment (ts_code, news_date);
```

### 应用优化

- **并发控制**: 使用线程池控制并发数量
- **内存管理**: 及时释放大型DataFrame
- **缓存机制**: Redis缓存频繁查询的数据
- **批量操作**: 使用批量插入提高写入性能

## 故障排查

### 常见问题

1. **API调用限制**
   ```bash
   # 检查API调用频率配置
   grep -r "api_limit" config.yml
   
   # 调整重试延迟
   export RETRY_DELAY=3
   ```

2. **数据库连接问题**
   ```bash
   # 检查数据库连接
   python -c "from src.utils.db import get_db_engine; print(get_db_engine())"
   
   # 检查TimescaleDB扩展
   psql -U stockschool -d stockschool -c "SELECT * FROM pg_extension WHERE extname='timescaledb';"
   ```

3. **同步任务卡住**
   ```bash
   # 查看运行中的任务
   python src/data/incremental_update.py --mode status
   
   # 重置任务队列
   python -c "from src.data.incremental_update import IncrementalUpdateManager; m=IncrementalUpdateManager(); m.task_queue.clear()"
   ```

### 日志分析

```bash
# 查看同步日志
tail -f logs/app.log | grep -E "(sync|error|failed)"

# 查看特定数据源日志
tail -f logs/app.log | grep "akshare"

# 查看性能日志
tail -f logs/app.log | grep -E "(duration|performance)"
```

## 测试

### 运行测试

```bash
# 运行所有测试
pytest src/tests/ -v

# 运行特定模块测试
pytest src/tests/test_akshare_sync.py -v
pytest src/tests/test_industry_classification.py -v
pytest src/tests/test_incremental_update.py -v

# 运行性能测试
pytest src/tests/ -m performance -v

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

### 测试数据

测试使用模拟数据，不会调用真实的API接口。测试覆盖：

- 数据同步逻辑
- 错误处理机制
- 数据标准化
- 状态管理
- 性能基准

## 部署建议

### 生产环境部署

1. **使用Docker部署**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

2. **配置定时任务**
   ```bash
   # 每日增量同步
   0 9 * * * /usr/bin/python /app/src/data/sync_manager.py --mode quick
   
   # 每周完整同步
   0 2 * * 0 /usr/bin/python /app/src/data/sync_manager.py --mode full
   
   # 每月数据清理
   0 3 1 * * /usr/bin/python /app/src/data/sync_manager.py --mode cleanup
   ```

3. **监控配置**
   - 配置Prometheus指标收集
   - 设置Grafana监控面板
   - 配置告警规则和通知渠道

### 扩展性考虑

- **水平扩展**: 支持多实例部署，通过Redis协调任务
- **数据分片**: 按股票代码或时间范围分片处理
- **缓存优化**: 使用Redis缓存热点数据
- **异步处理**: 使用Celery处理长时间运行的任务

## 贡献指南

### 开发环境设置

```bash
# 克隆项目
git clone https://github.com/your-org/stockschool.git
cd stockschool

# 安装开发依赖
pip install -r requirements-dev.txt

# 安装pre-commit钩子
pre-commit install

# 运行测试
pytest
```

### 代码规范

- 遵循PEP8代码风格
- 使用类型提示
- 编写完整的docstring
- 单元测试覆盖率 > 80%

### 提交规范

```bash
# 功能开发
git commit -m "feat: 添加新闻情绪数据同步功能"

# 问题修复
git commit -m "fix: 修复API调用频率限制问题"

# 文档更新
git commit -m "docs: 更新数据同步使用指南"
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 支持

如有问题或建议，请：

1. 查看 [FAQ](FAQ.md)
2. 搜索 [Issues](https://github.com/your-org/stockschool/issues)
3. 创建新的 Issue
4. 联系开发团队

---

**StockSchool Team**  
*让量化投资更智能*