# StockSchool API 使用示例

## 概述

StockSchool API 提供了完整的因子计算、查询、管理和分析功能。本文档提供了详细的使用示例和最佳实践。

## 认证

### 1. 用户登录

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123"
  }'
```

响应示例：
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user_info": {
    "user_id": "admin_001",
    "username": "admin",
    "email": "admin@stockschool.com",
    "roles": ["admin"],
    "permissions": ["factor:read", "factor:write", ...]
  }
}
```

### 2. 获取当前用户信息

```bash
curl -X GET "http://localhost:8000/api/v1/auth/me" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## 因子数据查询

### 1. 查询单只股票的因子数据

```bash
curl -X GET "http://localhost:8000/api/v1/factors" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "ts_codes": ["000001.SZ"],
    "factor_names": ["sma_5", "sma_20", "rsi_14"],
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "standardized": false
  }'
```

### 2. 查询多只股票的技术面因子

```bash
curl -X GET "http://localhost:8000/api/v1/factors" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "ts_codes": ["000001.SZ", "000002.SZ", "600000.SH"],
    "factor_types": ["technical"],
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "standardized": true
  }'
```

### 3. 查询标准化因子数据

```bash
curl -X GET "http://localhost:8000/api/v1/factors" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "ts_codes": ["000001.SZ"],
    "factor_names": ["pe_ttm", "pb", "roe"],
    "start_date": "2024-01-01",
    "standardized": true
  }'
```

## 因子计算

### 1. 触发单只股票的因子计算

```bash
curl -X POST "http://localhost:8000/api/v1/factors/calculate" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "ts_codes": ["000001.SZ"],
    "factor_types": ["technical", "fundamental"],
    "calculation_date": "2024-01-31",
    "force_recalculate": false,
    "priority": "high"
  }'
```

响应示例：
```json
{
  "success": true,
  "data": {
    "task_id": "api_20240131_143022",
    "mode": "single_stock",
    "estimated_duration": "5-30分钟",
    "status_url": "/api/v1/factors/tasks/api_20240131_143022"
  },
  "message": "因子计算任务已提交",
  "timestamp": "2024-01-31T14:30:22.123456"
}
```

### 2. 触发批量股票计算

```bash
curl -X POST "http://localhost:8000/api/v1/factors/calculate" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "ts_codes": ["000001.SZ", "000002.SZ", "600000.SH"],
    "factor_types": ["technical"],
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "priority": "normal"
  }'
```

### 3. 查询计算任务状态

```bash
curl -X GET "http://localhost:8000/api/v1/factors/tasks/api_20240131_143022" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

响应示例：
```json
{
  "success": true,
  "data": {
    "task_id": "api_20240131_143022",
    "status": "running",
    "created_at": "2024-01-31T14:30:22",
    "started_at": "2024-01-31T14:30:25",
    "completed_at": null,
    "progress": 65.5,
    "message": "正在计算技术面因子..."
  },
  "message": "任务状态查询成功"
}
```

## 因子标准化

### 1. Z-score标准化

```bash
curl -X POST "http://localhost:8000/api/v1/factors/standardize" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "factor_names": ["sma_5", "sma_20", "rsi_14"],
    "method": "zscore",
    "calculation_date": "2024-01-31",
    "industry_neutral": false,
    "outlier_method": "clip"
  }'
```

### 2. 行业中性化标准化

```bash
curl -X POST "http://localhost:8000/api/v1/factors/standardize" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "factor_names": ["pe_ttm", "pb", "roe"],
    "method": "quantile",
    "calculation_date": "2024-01-31",
    "industry_neutral": true,
    "outlier_method": "winsorize"
  }'
```

## 因子有效性分析

### 1. 提交有效性分析任务

```bash
curl -X POST "http://localhost:8000/api/v1/factors/effectiveness" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "factor_names": ["sma_5", "rsi_14", "pe_ttm"],
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "return_periods": [1, 5, 20],
    "analysis_types": ["ic", "ir", "layered_backtest"]
  }'
```

### 2. 查询分析结果

```bash
curl -X GET "http://localhost:8000/api/v1/factors/analysis/analysis_20240131_143500" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## 因子元数据

### 1. 获取所有因子元数据

```bash
curl -X GET "http://localhost:8000/api/v1/factors/metadata" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### 2. 获取特定类型因子元数据

```bash
curl -X GET "http://localhost:8000/api/v1/factors/metadata?factor_type=technical" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## 因子管理（管理员功能）

### 1. 获取因子定义列表

```bash
curl -X GET "http://localhost:8000/api/v1/factor-management/factors" \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN"
```

### 2. 创建新的因子定义

```bash
curl -X POST "http://localhost:8000/api/v1/factor-management/factors" \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "factor_name": "custom_indicator",
    "factor_type": "technical",
    "category": "custom",
    "description": "自定义技术指标",
    "formula": "CUSTOM_CALC(close, volume)",
    "parameters": {"window": 10, "threshold": 0.5},
    "data_requirements": ["stock_daily"],
    "update_frequency": "daily",
    "is_active": true
  }'
```

### 3. 更新因子定义

```bash
curl -X PUT "http://localhost:8000/api/v1/factor-management/factors/custom_indicator" \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "factor_name": "custom_indicator",
    "factor_type": "technical",
    "category": "custom",
    "description": "更新后的自定义技术指标",
    "formula": "CUSTOM_CALC_V2(close, volume, high, low)",
    "parameters": {"window": 15, "threshold": 0.6},
    "data_requirements": ["stock_daily"],
    "update_frequency": "daily",
    "is_active": true
  }'
```

## 任务调度管理

### 1. 获取调度任务列表

```bash
curl -X GET "http://localhost:8000/api/v1/factor-management/schedules" \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN"
```

### 2. 创建调度任务

```bash
curl -X POST "http://localhost:8000/api/v1/factor-management/schedules" \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "daily_factor_calculation",
    "task_type": "factor_calculation",
    "schedule_time": "18:00",
    "priority": "high",
    "dependencies": ["data_sync"],
    "parameters": {
      "factor_types": ["technical", "fundamental"],
      "calculation_mode": "incremental"
    },
    "is_enabled": true
  }'
```

### 3. 启用/禁用调度任务

```bash
# 启用任务
curl -X PUT "http://localhost:8000/api/v1/factor-management/schedules/task_001/enable" \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN"

# 禁用任务
curl -X PUT "http://localhost:8000/api/v1/factor-management/schedules/task_001/disable" \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN"
```

## 监控和告警

### 1. 获取系统监控状态

```bash
curl -X GET "http://localhost:8000/api/v1/factor-management/monitoring/status" \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN"
```

### 2. 获取性能指标

```bash
curl -X GET "http://localhost:8000/api/v1/factor-management/monitoring/metrics?start_time=2024-01-31T00:00:00&end_time=2024-01-31T23:59:59" \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN"
```

### 3. 获取告警信息

```bash
curl -X GET "http://localhost:8000/api/v1/factor-management/monitoring/alerts?alert_level=error&is_resolved=false" \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN"
```

## 缓存管理

### 1. 获取缓存状态

```bash
curl -X GET "http://localhost:8000/api/v1/factor-management/cache/status" \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN"
```

### 2. 清空缓存

```bash
# 清空所有缓存
curl -X POST "http://localhost:8000/api/v1/factor-management/cache/clear" \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN"

# 清空特定类型缓存
curl -X POST "http://localhost:8000/api/v1/factor-management/cache/clear?cache_type=factor_data" \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN"
```

### 3. 缓存预热

```bash
curl -X POST "http://localhost:8000/api/v1/factor-management/cache/warm-up" \
  -H "Authorization: Bearer ADMIN_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "factor_names": ["sma_5", "sma_20", "rsi_14"],
    "date_range": 30
  }'
```

## Python客户端示例

### 1. 基础使用

```python
import requests
import json
from datetime import date, datetime

class StockSchoolAPI:
    def __init__(self, base_url="http://localhost:8000", token=None):
        self.base_url = base_url
        self.token = token
        self.session = requests.Session()
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
    
    def login(self, username, password):
        """用户登录"""
        response = self.session.post(
            f"{self.base_url}/api/v1/auth/login",
            json={"username": username, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            self.token = data["access_token"]
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            return data
        else:
            raise Exception(f"登录失败: {response.text}")
    
    def get_factors(self, ts_codes, factor_names=None, factor_types=None, 
                   start_date=None, end_date=None, standardized=False):
        """获取因子数据"""
        payload = {
            "ts_codes": ts_codes,
            "standardized": standardized
        }
        
        if factor_names:
            payload["factor_names"] = factor_names
        if factor_types:
            payload["factor_types"] = factor_types
        if start_date:
            payload["start_date"] = start_date.isoformat() if isinstance(start_date, date) else start_date
        if end_date:
            payload["end_date"] = end_date.isoformat() if isinstance(end_date, date) else end_date
        
        response = self.session.get(
            f"{self.base_url}/api/v1/factors",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"获取因子数据失败: {response.text}")
    
    def calculate_factors(self, ts_codes=None, factor_names=None, factor_types=None,
                         calculation_date=None, start_date=None, end_date=None,
                         force_recalculate=False, priority="normal"):
        """触发因子计算"""
        payload = {
            "force_recalculate": force_recalculate,
            "priority": priority
        }
        
        if ts_codes:
            payload["ts_codes"] = ts_codes
        if factor_names:
            payload["factor_names"] = factor_names
        if factor_types:
            payload["factor_types"] = factor_types
        if calculation_date:
            payload["calculation_date"] = calculation_date.isoformat() if isinstance(calculation_date, date) else calculation_date
        if start_date:
            payload["start_date"] = start_date.isoformat() if isinstance(start_date, date) else start_date
        if end_date:
            payload["end_date"] = end_date.isoformat() if isinstance(end_date, date) else end_date
        
        response = self.session.post(
            f"{self.base_url}/api/v1/factors/calculate",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"触发因子计算失败: {response.text}")
    
    def get_task_status(self, task_id):
        """查询任务状态"""
        response = self.session.get(f"{self.base_url}/api/v1/factors/tasks/{task_id}")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"查询任务状态失败: {response.text}")

# 使用示例
if __name__ == "__main__":
    # 创建API客户端
    api = StockSchoolAPI()
    
    # 登录
    login_result = api.login("admin", "admin123")
    print("登录成功:", login_result["user_info"]["username"])
    
    # 获取因子数据
    factors = api.get_factors(
        ts_codes=["000001.SZ", "000002.SZ"],
        factor_names=["sma_5", "sma_20", "rsi_14"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31)
    )
    print(f"获取到 {factors['data']['total_count']} 条因子数据")
    
    # 触发因子计算
    calc_result = api.calculate_factors(
        ts_codes=["000001.SZ"],
        factor_types=["technical"],
        calculation_date=date(2024, 1, 31),
        priority="high"
    )
    task_id = calc_result["data"]["task_id"]
    print(f"计算任务已提交: {task_id}")
    
    # 查询任务状态
    import time
    while True:
        status = api.get_task_status(task_id)
        task_status = status["data"]["status"]
        progress = status["data"]["progress"]
        
        print(f"任务状态: {task_status}, 进度: {progress}%")
        
        if task_status in ["completed", "failed"]:
            break
        
        time.sleep(10)  # 等待10秒后再次查询
```

### 2. 批量数据处理

```python
import pandas as pd
from datetime import date, timedelta

def batch_get_factors(api, stock_list, factor_list, start_date, end_date, batch_size=50):
    """批量获取因子数据"""
    all_data = []
    
    # 按批次处理股票列表
    for i in range(0, len(stock_list), batch_size):
        batch_stocks = stock_list[i:i+batch_size]
        
        try:
            result = api.get_factors(
                ts_codes=batch_stocks,
                factor_names=factor_list,
                start_date=start_date,
                end_date=end_date
            )
            
            if result["success"]:
                all_data.extend(result["data"]["factors"])
                print(f"已处理 {i+len(batch_stocks)}/{len(stock_list)} 只股票")
            
        except Exception as e:
            print(f"批次 {i//batch_size + 1} 处理失败: {e}")
            continue
    
    # 转换为DataFrame
    if all_data:
        df_list = []
        for item in all_data:
            row_data = {
                'ts_code': item['ts_code'],
                'factor_date': item['factor_date']
            }
            row_data.update(item['factor_values'])
            df_list.append(row_data)
        
        return pd.DataFrame(df_list)
    else:
        return pd.DataFrame()

# 使用示例
api = StockSchoolAPI()
api.login("admin", "admin123")

# 获取股票列表（这里使用示例数据）
stock_list = ["000001.SZ", "000002.SZ", "600000.SH", "600036.SH", "000858.SZ"]
factor_list = ["sma_5", "sma_20", "rsi_14", "pe_ttm", "pb"]

# 批量获取最近30天的因子数据
end_date = date.today()
start_date = end_date - timedelta(days=30)

df = batch_get_factors(api, stock_list, factor_list, start_date, end_date)
print(f"获取到 {len(df)} 条记录")
print(df.head())
```

## 错误处理

### 常见错误码

- `401`: 未授权 - 需要登录或token已过期
- `403`: 权限不足 - 需要相应的权限
- `404`: 资源不存在 - 请求的资源未找到
- `422`: 参数验证失败 - 请求参数格式错误
- `500`: 服务器内部错误 - 联系管理员

### 错误响应格式

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "请求参数验证失败",
    "details": [
      {
        "loc": ["ts_codes"],
        "msg": "股票代码列表不能为空",
        "type": "value_error"
      }
    ]
  },
  "timestamp": "2024-01-31T14:30:22.123456"
}
```

## 最佳实践

### 1. 认证和安全
- 定期更新访问令牌
- 不要在客户端代码中硬编码密码
- 使用HTTPS进行生产环境通信

### 2. 性能优化
- 使用批量请求减少API调用次数
- 合理设置查询日期范围
- 利用缓存机制提高查询效率

### 3. 错误处理
- 实现重试机制处理临时性错误
- 记录详细的错误日志便于调试
- 对用户友好的错误提示

### 4. 数据管理
- 定期清理过期的计算任务
- 监控系统资源使用情况
- 合理配置缓存策略

## 支持和反馈

如有问题或建议，请联系：
- 邮箱: support@stockschool.com
- 文档: https://docs.stockschool.com
- GitHub: https://github.com/stockschool/api