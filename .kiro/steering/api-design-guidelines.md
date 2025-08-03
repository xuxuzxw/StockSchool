# StockSchool API设计指南

## API设计原则

### RESTful设计
- 使用标准HTTP方法：GET（查询）、POST（创建）、PUT（更新）、DELETE（删除）
- 资源导向的URL设计：`/api/v1/stocks/{ts_code}/factors`
- 使用HTTP状态码表示操作结果
- 支持内容协商（JSON优先）

### 版本管理
```python
# URL版本控制
/api/v1/stocks
/api/v2/stocks

# 请求头版本控制
Accept: application/vnd.stockschool.v1+json
```

### 统一响应格式
```python
# 成功响应
{
    "success": true,
    "data": {...},
    "message": "操作成功",
    "timestamp": "2024-01-01T00:00:00Z"
}

# 错误响应
{
    "success": false,
    "error": {
        "code": "INVALID_PARAMETER",
        "message": "参数验证失败",
        "details": {...}
    },
    "timestamp": "2024-01-01T00:00:00Z"
}
```

## 核心API接口规范

### 股票数据API
```python
# 获取股票基本信息
GET /api/v1/stocks
GET /api/v1/stocks/{ts_code}

# 获取股票行情数据
GET /api/v1/stocks/{ts_code}/daily?start_date=2024-01-01&end_date=2024-12-31

# 获取股票财务数据
GET /api/v1/stocks/{ts_code}/financial?report_type=annual&year=2024
```

### 因子数据API
```python
# 获取因子数据
GET /api/v1/factors?ts_codes=000001.SZ,000002.SZ&factors=rsi_14,macd&date=2024-01-01

# 获取因子统计信息
GET /api/v1/factors/statistics?factor_name=rsi_14&start_date=2024-01-01

# 触发因子计算
POST /api/v1/factors/calculate
{
    "ts_codes": ["000001.SZ", "000002.SZ"],
    "factor_types": ["technical", "fundamental"],
    "date_range": ["2024-01-01", "2024-01-31"]
}
```

### AI模型API
```python
# 获取模型预测
POST /api/v1/models/predict
{
    "model_id": "lightgbm_v1",
    "ts_codes": ["000001.SZ", "000002.SZ"],
    "date": "2024-01-01"
}

# 获取模型解释
GET /api/v1/models/{model_id}/explain?ts_code=000001.SZ&date=2024-01-01

# 启动模型训练
POST /api/v1/models/train
{
    "model_type": "lightgbm",
    "factors": ["rsi_14", "macd", "pe_ratio"],
    "training_period": ["2020-01-01", "2023-12-31"]
}
```

## 参数验证规范

### 请求参数验证
```python
from pydantic import BaseModel, validator
from typing import List, Optional
from datetime import date

class FactorQueryRequest(BaseModel):
    ts_codes: List[str]
    factors: List[str]
    start_date: date
    end_date: Optional[date] = None
    
    @validator('ts_codes')
    def validate_ts_codes(cls, v):
        if len(v) > 100:
            raise ValueError('最多支持100只股票')
        return v
    
    @validator('factors')
    def validate_factors(cls, v):
        allowed_factors = ['rsi_14', 'macd', 'pe_ratio']  # 从配置获取
        invalid_factors = set(v) - set(allowed_factors)
        if invalid_factors:
            raise ValueError(f'不支持的因子: {invalid_factors}')
        return v
```

### 响应数据序列化
```python
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime

class APIResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: str = ""
    timestamp: datetime = datetime.now()
    
class ErrorResponse(BaseModel):
    success: bool = False
    error: Dict[str, Any]
    timestamp: datetime = datetime.now()
```

## 错误处理规范

### 错误码定义
```python
class ErrorCode:
    # 客户端错误 4xx
    INVALID_PARAMETER = "INVALID_PARAMETER"
    MISSING_PARAMETER = "MISSING_PARAMETER"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    
    # 服务端错误 5xx
    INTERNAL_ERROR = "INTERNAL_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"
    CALCULATION_ERROR = "CALCULATION_ERROR"
```

### 异常处理中间件
```python
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

async def error_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": "HTTP_ERROR",
                    "message": exc.detail
                },
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # 记录未处理异常
    logger.error(f"未处理异常: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "服务器内部错误"
            },
            "timestamp": datetime.now().isoformat()
        }
    )
```

## 性能优化规范

### 分页查询
```python
class PaginationParams(BaseModel):
    page: int = 1
    page_size: int = 20
    
    @validator('page_size')
    def validate_page_size(cls, v):
        if v > 1000:
            raise ValueError('每页最多1000条记录')
        return v

# 分页响应
{
    "success": true,
    "data": {
        "items": [...],
        "pagination": {
            "page": 1,
            "page_size": 20,
            "total": 1000,
            "total_pages": 50
        }
    }
}
```

### 缓存策略
```python
from functools import wraps
import redis

def cache_response(expire_seconds: int = 300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"api:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # 尝试从缓存获取
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # 执行函数并缓存结果
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expire_seconds, json.dumps(result))
            
            return result
        return wrapper
    return decorator
```

### 限流控制
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/v1/stocks")
@limiter.limit("100/minute")
async def get_stocks(request: Request):
    # API实现
    pass
```

## 文档规范

### OpenAPI文档
```python
from fastapi import FastAPI

app = FastAPI(
    title="StockSchool API",
    description="量化投资系统API接口",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/api/v1/stocks/{ts_code}", 
         summary="获取股票信息",
         description="根据股票代码获取股票基本信息",
         response_description="股票基本信息")
async def get_stock(ts_code: str):
    """
    获取指定股票的基本信息
    
    - **ts_code**: 股票代码，如 000001.SZ
    
    返回股票的基本信息，包括名称、行业、上市日期等
    """
    pass
```

### API使用示例
```python
# 在文档中提供完整的使用示例
"""
## 使用示例

### 获取股票因子数据

```bash
curl -X GET "http://localhost:8000/api/v1/factors" \
  -H "Content-Type: application/json" \
  -d '{
    "ts_codes": ["000001.SZ", "000002.SZ"],
    "factors": ["rsi_14", "macd"],
    "start_date": "2024-01-01",
    "end_date": "2024-01-31"
  }'
```

### Python客户端示例

```python
import requests

response = requests.get(
    "http://localhost:8000/api/v1/factors",
    json={
        "ts_codes": ["000001.SZ"],
        "factors": ["rsi_14"],
        "date": "2024-01-01"
    }
)

data = response.json()
if data["success"]:
    factors = data["data"]
    print(factors)
```
"""
```

## 安全规范

### 认证授权
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # 验证token逻辑
    if not is_valid_token(token):
        raise HTTPException(status_code=401, detail="无效的访问令牌")
    return get_user_from_token(token)

@app.get("/api/v1/protected")
async def protected_endpoint(user = Depends(verify_token)):
    return {"message": f"Hello {user.username}"}
```

### 输入验证
```python
# 防止SQL注入
from sqlalchemy import text

def safe_query(ts_code: str):
    # 使用参数化查询
    query = text("SELECT * FROM stocks WHERE ts_code = :ts_code")
    return db.execute(query, {"ts_code": ts_code})

# 防止XSS
import html

def sanitize_input(user_input: str) -> str:
    return html.escape(user_input)
```

## 测试规范

### API测试
```python
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    return TestClient(app)

def test_get_stock_info(client):
    response = client.get("/api/v1/stocks/000001.SZ")
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert "ts_code" in data["data"]
    assert data["data"]["ts_code"] == "000001.SZ"

def test_invalid_stock_code(client):
    response = client.get("/api/v1/stocks/INVALID")
    assert response.status_code == 404
    
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "NOT_FOUND"
```

### 性能测试
```python
import asyncio
import aiohttp
import time

async def performance_test():
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        tasks = []
        for i in range(100):  # 并发100个请求
            task = session.get("http://localhost:8000/api/v1/stocks")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"100个并发请求耗时: {duration:.2f}秒")
        print(f"平均响应时间: {duration/100:.3f}秒")
```