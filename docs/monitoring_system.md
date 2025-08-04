# StockSchool 监控系统

## 概述

StockSchool 监控系统是一个全面的实时监控解决方案，用于监控系统健康状态、性能指标和告警管理。该系统提供了Web界面、API接口和WebSocket实时推送功能。

## 功能特性

### 🔍 系统监控
- **系统资源监控**: CPU、内存、磁盘使用率
- **数据库监控**: 连接状态、响应时间、连接数
- **Redis监控**: 连接状态、响应时间、内存使用
- **API监控**: 响应时间、错误率、请求统计
- **Celery监控**: 队列长度、工作者状态

### 🚨 告警系统
- **灵活的告警规则**: 支持多种指标和操作符
- **多级告警**: INFO、WARNING、CRITICAL
- **告警持续时间**: 避免瞬时波动触发告警
- **告警历史**: 完整的告警记录和统计

### 📊 性能监控
- **请求统计**: 总请求数、平均响应时间、错误率
- **端点分析**: 各API端点的详细性能数据
- **实时指标**: 基于时间窗口的性能聚合

### 🔔 通知系统
- **多渠道通知**: 邮件、Webhook、钉钉
- **通知限流**: 防止通知轰炸
- **通知模板**: 可自定义的通知格式
- **通知统计**: 发送成功率和失败统计

### 🌐 实时推送
- **WebSocket连接**: 实时数据推送
- **订阅机制**: 客户端可选择订阅的数据类型
- **连接管理**: 自动重连和连接清理

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │   API Gateway   │    │  WebSocket Hub  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────────────────────────────────────┐
         │              Monitoring System                  │
         └─────────────────────────────────────────────────┘
                                 │
    ┌────────────┬────────────────┼────────────────┬────────────┐
    │            │                │                │            │
┌───▼───┐  ┌────▼────┐  ┌────────▼────────┐  ┌───▼───┐  ┌────▼────┐
│Health │  │Performance│  │   Alert Engine  │  │Notify │  │ Config  │
│Checker│  │ Monitor   │  │                 │  │Manager│  │ Manager │
└───────┘  └─────────┘  └─────────────────┘  └───────┘  └─────────┘
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置监控系统

编辑 `config/monitoring.yaml` 文件：

```yaml
# 基本配置
system:
  check_interval: 30
  health_check_timeout: 10

# 数据库监控
database:
  enabled: true
  check_interval: 60

# 告警配置
alerting:
  enabled: true
  check_interval: 30

# 通知配置
notifications:
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    username: "your-email@gmail.com"
    password: "your-password"
```

### 3. 启动监控系统

#### 方式一：集成到主应用

监控系统已集成到主FastAPI应用中，启动主应用即可：

```bash
python src/api/main.py
```

#### 方式二：独立运行

```bash
python scripts/start_monitoring.py
```

### 4. 访问监控界面

打开浏览器访问：`http://localhost:8000/templates/monitoring.html`

## API 接口

### 健康检查

```http
GET /api/v1/monitoring/health
```

响应示例：
```json
{
  "overall_status": "healthy",
  "system": {
    "status": "healthy",
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "disk_usage": 23.1
  },
  "database": {
    "status": "healthy",
    "response_time": 12.5,
    "connections": 5
  },
  "redis": {
    "status": "healthy",
    "response_time": 2.1,
    "memory_usage": 1048576
  }
}
```

### 性能指标

```http
GET /api/v1/monitoring/performance
```

### 告警管理

```http
# 获取活跃告警
GET /api/v1/monitoring/alerts

# 获取告警统计
GET /api/v1/monitoring/alerts/stats

# 确认告警
POST /api/v1/monitoring/alerts/{alert_id}/acknowledge

# 解决告警
POST /api/v1/monitoring/alerts/{alert_id}/resolve
```

### 告警规则管理

```http
# 获取告警规则
GET /api/v1/monitoring/rules

# 创建告警规则
POST /api/v1/monitoring/rules

# 更新告警规则
PUT /api/v1/monitoring/rules/{rule_id}

# 删除告警规则
DELETE /api/v1/monitoring/rules/{rule_id}
```

## WebSocket 接口

### 连接

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/monitoring');
```

### 订阅数据

```javascript
// 订阅健康状态更新
ws.send(JSON.stringify({
  type: 'subscribe',
  topics: ['health', 'alerts', 'performance']
}));

// 取消订阅
ws.send(JSON.stringify({
  type: 'unsubscribe',
  topics: ['performance']
}));

// 获取当前数据
ws.send(JSON.stringify({
  type: 'get_current_data'
}));

// 心跳检测
ws.send(JSON.stringify({
  type: 'ping'
}));
```

### 接收数据

```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'health_update':
      updateHealthDisplay(data.data);
      break;
    case 'alert_update':
      updateAlertsDisplay(data.data);
      break;
    case 'performance_update':
      updatePerformanceDisplay(data.data);
      break;
    case 'pong':
      console.log('心跳响应');
      break;
  }
};
```

## 配置说明

### 系统监控配置

```yaml
system:
  check_interval: 30          # 检查间隔（秒）
  health_check_timeout: 10    # 健康检查超时（秒）
```

### 告警规则配置

```yaml
alerting:
  default_rules:
    - name: "高CPU使用率"
      metric: "system.cpu_usage"
      operator: ">"
      threshold: 80
      severity: "warning"
      duration: 300           # 持续时间（秒）
      description: "系统CPU使用率超过80%"
```

支持的操作符：
- `>`: 大于
- `>=`: 大于等于
- `<`: 小于
- `<=`: 小于等于
- `==`: 等于
- `!=`: 不等于

支持的严重级别：
- `info`: 信息
- `warning`: 警告
- `critical`: 严重

### 通知配置

```yaml
notifications:
  # 邮件通知
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    username: "your-email@gmail.com"
    password: "your-app-password"
    from_email: "your-email@gmail.com"
    to_emails:
      - "admin@example.com"
    use_tls: true
    
  # Webhook通知
  webhook:
    enabled: true
    url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
    timeout: 10
    retry_count: 3
    
  # 钉钉通知
  dingtalk:
    enabled: true
    webhook_url: "https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN"
    secret: "YOUR_SECRET"
```

## 监控指标

### 系统指标
- `system.cpu_usage`: CPU使用率（%）
- `system.memory_usage`: 内存使用率（%）
- `system.disk_usage`: 磁盘使用率（%）
- `system.status`: 系统状态（healthy/warning/error）

### 数据库指标
- `database.status`: 数据库状态
- `database.response_time`: 响应时间（ms）
- `database.connections`: 连接数

### Redis指标
- `redis.status`: Redis状态
- `redis.response_time`: 响应时间（ms）
- `redis.memory_usage`: 内存使用量（bytes）

### API指标
- `api.total_requests`: 总请求数
- `api.avg_response_time`: 平均响应时间（ms）
- `api.error_rate`: 错误率（%）
- `api.success_rate`: 成功率（%）

## 故障排除

### 常见问题

1. **监控系统无法启动**
   - 检查配置文件是否正确
   - 确认数据库和Redis连接
   - 查看日志文件获取详细错误信息

2. **WebSocket连接失败**
   - 检查防火墙设置
   - 确认WebSocket端口是否开放
   - 检查代理服务器配置

3. **告警不触发**
   - 检查告警规则配置
   - 确认指标数据是否正常收集
   - 检查告警引擎是否启动

4. **通知发送失败**
   - 检查通知渠道配置
   - 确认网络连接
   - 查看通知统计信息

### 日志文件

- 主应用日志: `logs/app.log`
- 监控系统日志: `logs/monitoring.log`
- 独立运行日志: `logs/monitoring_standalone.log`

### 调试模式

启用调试模式获取更详细的日志：

```yaml
logging:
  level: "DEBUG"
```

## 性能优化

### 配置优化

1. **调整检查间隔**
   ```yaml
   system:
     check_interval: 60  # 增加间隔减少系统负载
   ```

2. **禁用不需要的监控**
   ```yaml
   redis:
     enabled: false  # 如果不使用Redis
   ```

3. **调整数据保留期**
   ```yaml
   cleanup:
     retention:
       performance_metrics: 3  # 减少保留天数
   ```

### 扩展性

- 支持多实例部署
- 可配置的数据存储后端
- 插件化的通知渠道
- 自定义指标收集器

## 开发指南

### 添加自定义指标

```python
from src.monitoring.performance import PerformanceMonitor

monitor = PerformanceMonitor()

# 记录自定义指标
monitor.record_custom_metric("custom.metric", 42.0)
```

### 添加自定义告警规则

```python
from src.monitoring.alerts import AlertRule, AlertSeverity

rule = AlertRule(
    name="自定义规则",
    metric="custom.metric",
    operator=">",
    threshold=100,
    severity=AlertSeverity.WARNING,
    duration=300,
    description="自定义指标超过阈值"
)

alert_engine.add_rule(rule)
```

### 添加自定义通知渠道

```python
from src.monitoring.notifications import BaseNotificationChannel

class CustomNotificationChannel(BaseNotificationChannel):
    async def send_notification(self, title: str, message: str, severity: str) -> bool:
        # 实现自定义通知逻辑
        return True
```

## 测试

运行测试套件：

```bash
# 运行所有测试
python tests/test_monitoring.py

# 运行特定测试
pytest tests/test_monitoring.py::TestPerformanceMonitor -v
```

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。