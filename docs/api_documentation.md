# StockSchool API 文档

## 概述

StockSchool API 提供了完整的量化投研服务接口，包括数据获取、因子计算、模型解释、预测分析等功能。API基于FastAPI构建，支持RESTful风格的HTTP请求。

## 基础信息

- **基础URL**: `http://localhost:8000/api/v1`
- **协议**: HTTP/HTTPS
- **数据格式**: JSON
- **认证**: API密钥（通过请求头或查询参数）

## API端点

### 1. 健康检查

#### GET /health
检查API服务健康状态

**请求示例**:
```bash
curl -X GET "http://localhost:8000/health"
```

**响应**:
```json
{
  "status": "healthy",
  "timestamp": "2025-07-31T12:00:00Z",
  "version": "1.1.7"
}
```

### 2. 股票数据接口

#### GET /stocks/basic
获取股票基础信息

**参数**:
- `limit` (integer, optional): 返回记录数，默认100
- `offset` (integer, optional): 偏移量，默认0

**请求示例**:
```bash
curl -X GET "http://localhost:8000/api/v1/stocks/basic?limit=50"
```

**响应**:
```json
{
  "data": [
    {
      "ts_code": "000001.SZ",
      "symbol": "000001",
      "name": "平安银行",
      "area": "深圳",
      "industry": "银行",
      "market": "主板"
    }
  ],
  "total": 1000,
  "limit": 50,
  "offset": 0
}
```

#### GET /stocks/daily
获取股票日线数据

**参数**:
- `ts_code` (string, required): 股票代码
- `start_date` (string, optional): 开始日期 (YYYYMMDD)
- `end_date` (string, optional): 结束日期 (YYYYMMDD)
- `limit` (integer, optional): 返回记录数

**请求示例**:
```bash
curl -X GET "http://localhost:8000/api/v1/stocks/daily?ts_code=000001.SZ&start_date=20240101&end_date=20241231"
```

**响应**:
```json
{
  "data": [
    {
      "ts_code": "000001.SZ",
      "trade_date": "20240101",
      "open": 15.2,
      "high": 15.8,
      "low": 15.1,
      "close": 15.6,
      "volume": 1000000,
      "amount": 15600000
    }
  ],
  "total": 250
}
```

### 3. 因子数据接口

#### GET /factors/values
获取因子值

**参数**:
- `ts_code` (string, optional): 股票代码
- `factor_name` (string, required): 因子名称
- `start_date` (string, optional): 开始日期
- `end_date` (string, optional): 结束日期

**请求示例**:
```bash
curl -X GET "http://localhost:8000/api/v1/factors/values?ts_code=000001.SZ&factor_name=rsi&start_date=20240101"
```

**响应**:
```json
{
  "data": [
    {
      "ts_code": "000001.SZ",
      "trade_date": "20240101",
      "factor_name": "rsi",
      "factor_value": 65.2
    }
  ]
}
```

#### POST /factors/calculate
计算因子

**请求体**:
```json
{
  "ts_code": "000001.SZ",
  "factor_names": ["rsi", "macd", "bollinger"],
  "start_date": "20240101",
  "end_date": "20241231"
}
```

**响应**:
```json
{
  "status": "success",
  "message": "因子计算完成",
  "data": {
    "calculated_factors": 3,
    "processing_time": 2.5
  }
}
```

### 4. 模型解释接口

#### POST /explain/feature-importance
计算特征重要性

**请求体**:
```json
{
  "model_path": "/models/stock_model.pkl",
  "data": {
    "features": ["open", "high", "low", "close", "volume"],
    "X": [[15.2, 15.8, 15.1, 15.6, 1000000]]
  },
  "method": "shap",
  "background_samples": 100
}
```

**响应**:
```json
{
  "status": "success",
  "data": {
    "feature_importance": [
      {
        "feature": "close",
        "importance": 0.35,
        "importance_std": 0.05
      },
      {
        "feature": "volume",
        "importance": 0.28,
        "importance_std": 0.03
      }
    ],
    "method": "shap",
    "processing_time": 1.2
  }
}
```

#### POST /explain/prediction
解释单个预测

**请求体**:
```json
{
  "model_path": "/models/stock_model.pkl",
  "sample": {
    "features": ["open", "high", "low", "close", "volume"],
    "values": [15.2, 15.8, 15.1, 15.6, 1000000]
  },
  "explanation_type": "shap"
}
```

**响应**:
```json
{
  "status": "success",
  "data": {
    "prediction": 16.2,
    "base_value": 15.5,
    "shap_values": [0.1, 0.2, -0.05, 0.3, 0.15],
    "feature_names": ["open", "high", "low", "close", "volume"],
    "feature_values": [15.2, 15.8, 15.1, 15.6, 1000000]
  }
}
```

#### POST /explain/batch
批量解释预测

**请求体**:
```json
{
  "model_path": "/models/stock_model.pkl",
  "samples": [
    {
      "features": ["open", "high", "low", "close", "volume"],
      "values": [15.2, 15.8, 15.1, 15.6, 1000000]
    },
    {
      "features": ["open", "high", "low", "close", "volume"],
      "values": [16.1, 16.5, 16.0, 16.3, 800000]
    }
  ],
  "batch_size": 100
}
```

**响应**:
```json
{
  "status": "success",
  "data": [
    {
      "sample_index": 0,
      "prediction": 16.2,
      "shap_values": [0.1, 0.2, -0.05, 0.3, 0.15]
    },
    {
      "sample_index": 1,
      "prediction": 16.8,
      "shap_values": [0.15, 0.25, -0.1, 0.35, 0.2]
    }
  ],
  "total_samples": 2,
  "processing_time": 2.5
}
```

### 5. 预测接口

#### POST /predict/single
单个股票预测

**请求体**:
```json
{
  "ts_code": "000001.SZ",
  "model_name": "LightGBM",
  "prediction_days": 5,
  "features": ["rsi", "macd", "volume_ratio"]
}
```

**响应**:
```json
{
  "status": "success",
  "data": {
    "ts_code": "000001.SZ",
    "predictions": [
      {
        "date": "20240102",
        "predicted_price": 16.2,
        "confidence": 0.85
      },
      {
        "date": "20240103",
        "predicted_price": 16.5,
        "confidence": 0.82
      }
    ]
  }
}
```

#### POST /predict/batch
批量股票预测

**请求体**:
```json
{
  "ts_codes": ["000001.SZ", "000002.SZ", "600000.SH"],
  "model_name": "XGBoost",
  "prediction_days": 3
}
```

**响应**:
```json
{
  "status": "success",
  "data": {
    "predictions": [
      {
        "ts_code": "000001.SZ",
        "results": [
          {
            "date": "20240102",
            "predicted_price": 16.2,
            "confidence": 0.85
          }
        ]
      }
    ]
  }
}
```

### 6. 监控接口

#### GET /monitoring/metrics
获取系统性能指标

**请求示例**:
```bash
curl -X GET "http://localhost:8000/api/v1/monitoring/metrics"
```

**响应**:
```json
{
  "cpu_usage": 45.2,
  "memory_usage": 60.5,
  "disk_usage": 75.3,
  "gpu_usage": 28.1,
  "active_connections": 15,
  "request_rate": 120
}
```

#### GET /monitoring/alerts
获取告警信息

**参数**:
- `level` (string, optional): 告警级别 (INFO, WARNING, ERROR, CRITICAL)
- `limit` (integer, optional): 返回记录数

**响应**:
```json
{
  "alerts": [
    {
      "level": "WARNING",
      "title": "高CPU使用率",
      "message": "CPU使用率达到85%",
      "timestamp": "2025-07-31T12:00:00Z",
      "source": "system_monitor"
    }
  ]
}
```

### 7. 数据同步接口

#### POST /sync/stocks
同步股票数据

**请求体**:
```json
{
  "ts_codes": ["000001.SZ", "000002.SZ"],
  "start_date": "20240101",
  "end_date": "20241231"
}
```

**响应**:
```json
{
  "status": "success",
  "message": "数据同步完成",
  "synced_stocks": 2,
  "processing_time": 15.2
}
```

#### POST /sync/calendar
同步交易日历

**请求体**:
```json
{
  "start_year": 2024,
  "end_year": 2025
}
```

**响应**:
```json
{
  "status": "success",
  "message": "交易日历同步完成",
  "synced_days": 500,
  "processing_time": 2.1
}
```

## 错误处理

### 错误响应格式
```json
{
  "error": {
    "code": 400,
    "message": "请求参数错误",
    "details": "缺少必需参数: ts_code"
  }
}
```

### 常见错误码
- `400`: 请求参数错误
- `401`: 未授权
- `404`: 资源未找到
- `500`: 服务器内部错误
- `503`: 服务不可用

## 认证

### API密钥
在请求头中添加API密钥：
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" "http://localhost:8000/api/v1/stocks/basic"
```

或作为查询参数：
```bash
curl "http://localhost:8000/api/v1/stocks/basic?api_key=YOUR_API_KEY"
```

## 限流

API接口有请求频率限制：
- **普通用户**: 100次/分钟
- **付费用户**: 1000次/分钟
- **企业用户**: 10000次/分钟

超过限制将返回429状态码。

## WebSocket支持

部分实时数据接口支持WebSocket连接：

### 连接地址
`ws://localhost:8000/ws/realtime`

### 订阅消息
```json
{
  "action": "subscribe",
  "channel": "stock_updates",
  "symbols": ["000001.SZ", "000002.SZ"]
}
```

### 推送消息
```json
{
  "channel": "stock_updates",
  "data": {
    "ts_code": "000001.SZ",
    "price": 15.6,
    "volume": 1000,
    "timestamp": "2025-07-31T12:00:00Z"
  }
}
```

## 版本历史

### v1.1.7 (2025-07-31)
- 新增模型解释API接口
- 优化GPU加速支持
- 增强错误处理机制

### v1.1.6 (2025-07-30)
- 完善预测API功能
- 添加批量处理支持
- 改进性能监控

### v1.1.5 (2025-07-29)
- 基础API接口完成
- 数据同步功能实现
- 因子计算接口上线

## 客户端SDK

### Python SDK
```python
from stockschool import StockSchoolClient

client = StockSchoolClient(api_key="YOUR_API_KEY")

# 获取股票数据
stocks = client.get_stock_basic(limit=100)

# 计算因子
factors = client.calculate_factors("000001.SZ", ["rsi", "macd"])

# 模型解释
explanation = client.explain_prediction("/models/model.pkl", sample_data)
```

### JavaScript SDK
```javascript
import { StockSchoolClient } from 'stockschool-sdk';

const client = new StockSchoolClient({ apiKey: 'YOUR_API_KEY' });

// 获取股票数据
const stocks = await client.getStockBasic({ limit: 100 });

// 预测股票价格
const prediction = await client.predictStock('000001.SZ', { days: 5 });
```

## 性能优化建议

1. **批量请求**: 尽可能使用批量接口减少网络开销
2. **缓存利用**: 合理使用缓存避免重复计算
3. **连接复用**: 复用HTTP连接提高效率
4. **异步处理**: 对于耗时操作使用异步接口

## 联系方式

如有API使用问题，请联系：
- 邮箱: api@stockschool.com
- 文档: https://docs.stockschool.com
- 支持: https://support.stockschool.com
