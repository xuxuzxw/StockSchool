# StockSchool 编码规范

## Python代码规范

### 基本规范
- 遵循PEP8代码风格
- 使用类型提示(Type Hints)
- 函数和类必须包含docstring
- 变量命名使用snake_case，类名使用PascalCase

### 导入规范
```python
# 标准库导入
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# 第三方库导入
import pandas as pd
import numpy as np
from sqlalchemy import text

# 项目内部导入
from src.utils.config_loader import config
from src.utils.db import get_db_engine
```

### 异常处理规范
```python
# 使用项目统一的重试装饰器
from src.utils.retry import idempotent_retry

@idempotent_retry(max_retries=3)
def sync_data():
    try:
        # 业务逻辑
        pass
    except Exception as e:
        logger.error(f"数据同步失败: {e}")
        raise
```

### 日志规范
```python
from loguru import logger

# 使用结构化日志
logger.info(f"开始同步股票 {ts_code} 的数据")
logger.error(f"股票 {ts_code} 数据同步失败: {error_msg}")
```

## 数据库操作规范

### 连接管理
```python
# 使用统一的数据库连接
from src.utils.db import get_db_engine

engine = get_db_engine()
with engine.connect() as conn:
    result = conn.execute(text("SELECT * FROM stock_basic"))
```

### SQL查询规范
- 使用参数化查询防止SQL注入
- 复杂查询使用sqlalchemy的text()
- 批量操作使用事务管理

```python
# 正确的参数化查询
query = text("""
    SELECT * FROM stock_daily 
    WHERE ts_code = :ts_code 
    AND trade_date >= :start_date
""")
result = conn.execute(query, {
    'ts_code': ts_code,
    'start_date': start_date
})
```

## 因子计算规范

### 函数签名规范
```python
def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    计算RSI指标
    
    Args:
        data: 包含close列的价格数据
        window: 计算窗口期，默认14
        
    Returns:
        RSI值序列
        
    Raises:
        ValueError: 当数据不足时抛出异常
    """
```

### 数据验证规范
```python
# 输入数据验证
if data.empty:
    raise ValueError("输入数据不能为空")
    
if len(data) < window:
    logger.warning(f"数据长度 {len(data)} 小于窗口期 {window}")
    return pd.Series(dtype=float)

# 必要列检查
required_columns = ['close', 'high', 'low']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"缺少必要列: {missing_columns}")
```

## 配置管理规范

### 配置文件使用
```python
from src.utils.config_loader import config

# 获取配置值，提供默认值
window = config.get('factor_params.rsi.window', 14)
batch_size = config.get('data_sync_params.batch_size', 1000)
```

### 环境变量使用
```python
import os

# 敏感信息从环境变量获取
token = os.getenv("TUSHARE_TOKEN")
if not token:
    raise ValueError("TUSHARE_TOKEN环境变量未设置")
```

## 测试规范

### 单元测试
```python
import pytest
import pandas as pd
from src.compute.technical import calculate_rsi

def test_rsi_calculation():
    # 准备测试数据
    data = pd.DataFrame({
        'close': [10, 12, 11, 13, 14, 15, 14, 16, 17, 18]
    })
    
    # 执行测试
    rsi = calculate_rsi(data, window=14)
    
    # 断言验证
    assert not rsi.empty, "RSI计算结果不应为空"
    assert 0 <= rsi.iloc[-1] <= 100, "RSI值应在0-100之间"
```

### 集成测试
```python
@pytest.fixture(scope="module")
def test_database():
    """测试数据库fixture"""
    from src.utils.test_db import get_test_db_manager
    
    db_manager = get_test_db_manager()
    db_manager.setup_test_environment()
    
    yield db_manager
    
    db_manager.cleanup_test_environment()
```

## 性能优化规范

### 数据处理优化
```python
# 使用向量化操作
data['returns'] = data['close'].pct_change()

# 避免循环，使用pandas内置函数
data['sma_20'] = data['close'].rolling(window=20).mean()

# 批量数据库操作
df.to_sql('table_name', engine, if_exists='append', method='multi')
```

### 内存管理
```python
# 及时释放大型DataFrame
del large_dataframe

# 使用生成器处理大数据集
def process_stocks_batch(stock_list, batch_size=100):
    for i in range(0, len(stock_list), batch_size):
        yield stock_list[i:i+batch_size]
```

## 文档规范

### API文档
- 所有公共函数必须包含完整的docstring
- 使用Google风格的docstring格式
- 包含参数类型、返回值类型和异常说明

### 代码注释
```python
# 计算RSI指标的核心逻辑
# RSI = 100 - (100 / (1 + RS))
# 其中 RS = 平均上涨幅度 / 平均下跌幅度
gains = price_changes.where(price_changes > 0, 0)
losses = -price_changes.where(price_changes < 0, 0)
```