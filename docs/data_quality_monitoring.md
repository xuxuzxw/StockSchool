# 数据质量监控系统

## 概述

数据质量监控系统是StockSchool监控框架的重要组成部分，专门用于监控和评估数据的质量状况。该系统支持多维度的数据质量检查，包括完整性、准确性、时效性、一致性和唯一性等。

## 功能特性

### 1. 多维度质量检查
- **完整性（Completeness）**: 检查数据的空值情况
- **准确性（Accuracy）**: 检查数据的范围和异常值
- **时效性（Timeliness）**: 检查数据的新鲜度
- **一致性（Consistency）**: 检查数据的重复情况
- **唯一性（Uniqueness）**: 检查数据的唯一性约束

### 2. 灵活的规则配置
- 支持自定义质量规则
- 可配置的阈值和检查条件
- 支持多种检查类型

### 3. 自动化监控
- 定期执行质量检查
- 自动生成质量报告
- 智能告警机制

### 4. 历史数据追踪
- 质量趋势分析
- 历史数据保留
- 质量变化监控

## 快速开始

### 1. 基本使用

```python
from src.monitoring.data_quality import DataQualityMonitor
from src.monitoring.config import MonitoringConfig

# 初始化配置
config = MonitoringConfig()

# 创建数据质量监控器
monitor = DataQualityMonitor(config)

# 添加质量规则
rule = {
    'id': 'stock_price_null_check',
    'name': '股价空值检查',
    'description': '检查股价数据的完整性',
    'table_name': 'stock_prices',
    'column_name': 'price',
    'dimension': 'completeness',
    'check_type': 'null_check',
    'threshold': 0.05,  # 允许5%的空值
    'enabled': True
}

monitor.add_rule(rule)

# 执行质量检查
results = await monitor.check_data_quality('stock_prices')
print(f"质量检查结果: {results}")

# 生成质量报告
report = await monitor.generate_quality_report('stock_prices')
print(f"质量报告: {report}")
```

### 2. 使用调度器

```python
from src.monitoring.data_quality_scheduler import DataQualityScheduler

# 创建调度器
scheduler = DataQualityScheduler(monitor, config)

# 添加监控表
scheduler.add_table('stock_prices')
scheduler.add_table('financial_reports')

# 启动调度器
scheduler.start()

# 获取调度器状态
status = scheduler.get_status()
print(f"调度器状态: {status}")
```

### 3. 集成到监控系统

```python
from src.monitoring import MonitoringSystem

# 创建监控系统
monitoring = MonitoringSystem()

# 启动监控系统（包含数据质量监控）
await monitoring.start()

# 获取数据质量报告
report = await monitoring.get_data_quality_report('stock_prices')

# 手动执行质量检查
results = await monitoring.check_data_quality('stock_prices')

# 获取质量历史
history = await monitoring.get_data_quality_history('stock_prices', days=7)
```

## 配置说明

### 1. 基本配置

在 `config/monitoring_example.yaml` 中配置数据质量监控：

```yaml
data_quality_monitoring:
  enabled: true
  check_interval: 300  # 检查间隔（秒）
  database_path: "data/data_quality.db"
  retention_days: 30
  batch_size: 1000
```

### 2. 质量规则配置

```yaml
monitored_tables:
  - name: "stock_prices"
    rules:
      - dimension: "completeness"
        check_type: "null_check"
        column: "price"
        threshold: 0.05
      - dimension: "accuracy"
        check_type: "range_check"
        column: "price"
        min_value: 0
        max_value: 10000
```

### 3. 阈值配置

```yaml
thresholds:
  data_quality_overall: 80
  data_quality_completeness: 85
  data_quality_accuracy: 90
  data_quality_timeliness: 75
  data_quality_consistency: 85
  data_quality_uniqueness: 95
```

## 质量检查类型

### 1. 空值检查（null_check）
检查指定列的空值比例是否超过阈值。

```python
rule = {
    'dimension': 'completeness',
    'check_type': 'null_check',
    'column_name': 'price',
    'threshold': 0.05  # 允许5%的空值
}
```

### 2. 范围检查（range_check）
检查数值是否在指定范围内。

```python
rule = {
    'dimension': 'accuracy',
    'check_type': 'range_check',
    'column_name': 'price',
    'min_value': 0,
    'max_value': 10000
}
```

### 3. 异常值检查（outlier_check）
使用统计方法检测异常值。

```python
rule = {
    'dimension': 'accuracy',
    'check_type': 'outlier_check',
    'column_name': 'price',
    'method': 'iqr',  # 或 'zscore'
    'threshold': 3.0
}
```

### 4. 新鲜度检查（freshness_check）
检查数据的时效性。

```python
rule = {
    'dimension': 'timeliness',
    'check_type': 'freshness_check',
    'column_name': 'created_at',
    'max_age_hours': 24
}
```

### 5. 重复检查（duplicate_check）
检查数据的重复情况。

```python
rule = {
    'dimension': 'consistency',
    'check_type': 'duplicate_check',
    'columns': ['company_id', 'report_date'],
    'threshold': 0.02  # 允许2%的重复
}
```

## 质量报告

质量报告包含以下信息：

```python
{
    'table_name': 'stock_prices',
    'timestamp': '2024-01-15T10:30:00',
    'overall_score': 85.5,
    'dimension_scores': {
        'completeness': 90.0,
        'accuracy': 88.0,
        'timeliness': 82.0,
        'consistency': 85.0,
        'uniqueness': 95.0
    },
    'check_results': [
        {
            'rule_id': 'price_null_check',
            'passed': True,
            'score': 95.0,
            'details': '空值比例: 2.1%'
        }
    ],
    'recommendations': [
        '建议检查价格数据的异常值',
        '考虑增加数据验证规则'
    ]
}
```

## 告警机制

当数据质量低于阈值时，系统会自动创建告警：

- **严重告警**: 总体质量分数 < 60
- **警告告警**: 总体质量分数 < 80
- **信息告警**: 单个维度分数 < 阈值

## 最佳实践

### 1. 规则设计
- 根据业务需求设置合理的阈值
- 优先监控关键业务数据
- 定期审查和调整规则

### 2. 性能优化
- 合理设置检查间隔
- 使用批处理减少数据库负载
- 定期清理历史数据

### 3. 监控策略
- 建立质量基线
- 监控质量趋势
- 及时响应质量告警

### 4. 团队协作
- 明确数据质量责任
- 建立质量改进流程
- 定期进行质量评审

## 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查数据库配置
   - 确认数据库服务状态
   - 验证连接权限

2. **质量检查失败**
   - 检查表和列是否存在
   - 验证规则配置
   - 查看错误日志

3. **性能问题**
   - 调整批处理大小
   - 优化检查间隔
   - 添加数据库索引

### 日志分析

查看监控日志获取详细信息：

```bash
tail -f logs/monitoring.log | grep "data_quality"
```

## API参考

详细的API文档请参考代码注释和类型提示。主要类和方法：

- `DataQualityMonitor`: 核心监控类
- `DataQualityScheduler`: 调度器类
- `QualityRule`: 质量规则数据结构
- `QualityResult`: 检查结果数据结构
- `QualityReport`: 质量报告数据结构

## 扩展开发

### 添加新的检查类型

1. 在 `DataQualityMonitor` 中添加新的检查方法
2. 更新 `QualityCheckType` 枚举
3. 在配置中添加相应的参数
4. 编写单元测试

### 集成外部数据源

1. 扩展数据库连接配置
2. 添加数据源适配器
3. 更新质量检查逻辑
4. 测试兼容性

通过以上文档，您可以全面了解和使用数据质量监控系统。如有问题，请查看代码注释或联系开发团队。