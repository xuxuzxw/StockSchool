# 因子计算引擎技术文档

## 概述

StockSchool因子计算引擎是一个高性能、可扩展的量化因子计算系统，支持技术面、基本面和情绪面三大类因子的计算。本文档详细介绍了系统的技术架构、因子定义、计算方法和使用指南。

## 系统架构

### 整体架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │  Factor Engine  │    │ Feature Store   │
│   (FastAPI)     │────│   (Core)        │────│  (Storage)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Authentication │    │ Compute Engines │    │   TimescaleDB   │
│   & Authorization│    │ (Tech/Fund/Sent)│    │   (Database)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Cache Layer   │    │  Task Queue     │    │   Monitoring    │
│   (Redis)       │    │  (Celery)       │    │   & Logging     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 核心组件

#### 1. 因子引擎 (FactorEngine)
- **职责**: 统一的因子计算入口，协调各个专业引擎
- **特性**: 支持批量计算、并行处理、错误恢复
- **位置**: `src/compute/factor_engine.py`

#### 2. 专业计算引擎
- **技术面引擎** (`TechnicalFactorEngine`): 计算技术指标
- **基本面引擎** (`FundamentalFactorEngine`): 计算财务指标
- **情绪面引擎** (`SentimentFactorEngine`): 计算市场情绪指标

#### 3. 特征商店 (FeatureStore)
- **职责**: 因子数据的版本化存储和管理
- **特性**: 数据血缘追踪、版本控制、质量监控
- **位置**: `src/features/factor_feature_store.py`

#### 4. API层
- **因子API**: 因子计算和查询接口
- **管理API**: 因子定义和调度管理
- **特征商店API**: 特征数据管理接口

## 因子分类和定义

### 技术面因子

#### 1. 移动平均类
- **SMA (Simple Moving Average)**: 简单移动平均
  - 公式: `SMA(n) = (P1 + P2 + ... + Pn) / n`
  - 参数: window (窗口期，默认5、10、20、60)
  - 用途: 趋势识别、支撑阻力位

- **EMA (Exponential Moving Average)**: 指数移动平均
  - 公式: `EMA(t) = α × P(t) + (1-α) × EMA(t-1)`
  - 参数: window (窗口期), alpha (平滑系数)
  - 用途: 更敏感的趋势跟踪

#### 2. 动量指标类
- **RSI (Relative Strength Index)**: 相对强弱指数
  - 公式: `RSI = 100 - (100 / (1 + RS))`，其中 `RS = 平均涨幅 / 平均跌幅`
  - 参数: window (默认14)
  - 取值范围: 0-100
  - 用途: 超买超卖判断

- **MACD (Moving Average Convergence Divergence)**: 指数平滑移动平均线
  - 公式: 
    - `DIF = EMA(12) - EMA(26)`
    - `DEA = EMA(DIF, 9)`
    - `MACD = 2 × (DIF - DEA)`
  - 用途: 趋势转换信号

#### 3. 波动率指标
- **Bollinger Bands**: 布林带
  - 公式:
    - `中轨 = SMA(n)`
    - `上轨 = SMA(n) + k × STD(n)`
    - `下轨 = SMA(n) - k × STD(n)`
  - 参数: window (默认20), k (默认2)
  - 用途: 价格通道、波动率分析

- **ATR (Average True Range)**: 平均真实波幅
  - 公式: `ATR = SMA(TR, n)`，其中 `TR = max(H-L, |H-C1|, |L-C1|)`
  - 参数: window (默认14)
  - 用途: 波动率测量、止损设置

### 基本面因子

#### 1. 估值指标
- **PE (Price-to-Earnings Ratio)**: 市盈率
  - 公式: `PE = 股价 / 每股收益`
  - 数据来源: 财务报表、股价数据
  - 用途: 估值水平判断

- **PB (Price-to-Book Ratio)**: 市净率
  - 公式: `PB = 股价 / 每股净资产`
  - 用途: 资产价值评估

- **PS (Price-to-Sales Ratio)**: 市销率
  - 公式: `PS = 市值 / 营业收入`
  - 用途: 成长性评估

#### 2. 盈利能力指标
- **ROE (Return on Equity)**: 净资产收益率
  - 公式: `ROE = 净利润 / 平均净资产`
  - 用途: 盈利能力评估

- **ROA (Return on Assets)**: 总资产收益率
  - 公式: `ROA = 净利润 / 平均总资产`
  - 用途: 资产使用效率

- **Gross Margin**: 毛利率
  - 公式: `毛利率 = (营业收入 - 营业成本) / 营业收入`
  - 用途: 盈利质量分析

#### 3. 财务质量指标
- **Debt-to-Equity**: 资产负债率
  - 公式: `资产负债率 = 总负债 / 总资产`
  - 用途: 财务风险评估

- **Current Ratio**: 流动比率
  - 公式: `流动比率 = 流动资产 / 流动负债`
  - 用途: 短期偿债能力

### 情绪面因子

#### 1. 资金流向指标
- **Money Flow**: 资金流向
  - 公式: `MF = 典型价格 × 成交量`，其中 `典型价格 = (H + L + C) / 3`
  - 用途: 资金进出判断

- **Volume Ratio**: 量比
  - 公式: `量比 = 当日成交量 / 历史平均成交量`
  - 用途: 成交活跃度分析

#### 2. 市场关注度
- **Turnover Rate**: 换手率
  - 公式: `换手率 = 成交量 / 流通股本`
  - 用途: 流动性和关注度分析

- **Price Change Rate**: 价格变化率
  - 公式: `涨跌幅 = (当前价格 - 前收盘价) / 前收盘价`
  - 用途: 市场情绪强度

## 计算方法和算法

### 数据预处理

#### 1. 数据清洗
```python
def clean_stock_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    股票数据清洗
    
    处理步骤:
    1. 去除重复数据
    2. 处理缺失值
    3. 异常值检测和处理
    4. 数据类型转换
    """
    # 去重
    data = data.drop_duplicates(subset=['ts_code', 'trade_date'])
    
    # 缺失值处理
    data['close'] = data['close'].fillna(method='ffill')
    data['volume'] = data['volume'].fillna(0)
    
    # 异常值处理 (3σ原则)
    for col in ['open', 'high', 'low', 'close']:
        mean_val = data[col].mean()
        std_val = data[col].std()
        data[col] = data[col].clip(
            lower=mean_val - 3*std_val,
            upper=mean_val + 3*std_val
        )
    
    return data
```

#### 2. 数据标准化
```python
def standardize_factor(factor_data: pd.Series, method: str = 'zscore') -> pd.Series:
    """
    因子标准化
    
    支持方法:
    - zscore: Z-score标准化
    - minmax: 最小-最大标准化
    - rank: 排名标准化
    """
    if method == 'zscore':
        return (factor_data - factor_data.mean()) / factor_data.std()
    elif method == 'minmax':
        return (factor_data - factor_data.min()) / (factor_data.max() - factor_data.min())
    elif method == 'rank':
        return factor_data.rank(pct=True)
    else:
        raise ValueError(f"不支持的标准化方法: {method}")
```

### 并行计算策略

#### 1. 股票级并行
```python
def parallel_stock_calculation(ts_codes: List[str], calc_func, **kwargs):
    """
    股票级并行计算
    
    策略: 将股票列表分批，每批并行计算
    """
    from concurrent.futures import ThreadPoolExecutor
    
    batch_size = 50  # 每批50只股票
    results = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        for i in range(0, len(ts_codes), batch_size):
            batch = ts_codes[i:i+batch_size]
            future = executor.submit(calc_func, batch, **kwargs)
            futures.append(future)
        
        for future in futures:
            result = future.result()
            results.append(result)
    
    return pd.concat(results, ignore_index=True)
```

#### 2. 因子级并行
```python
def parallel_factor_calculation(factor_configs: List[dict], data: pd.DataFrame):
    """
    因子级并行计算
    
    策略: 不同因子并行计算，最后合并结果
    """
    from concurrent.futures import ProcessPoolExecutor
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {
            executor.submit(calculate_single_factor, config, data): config['name']
            for config in factor_configs
        }
        
        results = {}
        for future in futures:
            factor_name = futures[future]
            try:
                results[factor_name] = future.result()
            except Exception as e:
                logger.error(f"因子{factor_name}计算失败: {e}")
                results[factor_name] = pd.Series(dtype=float)
    
    return results
```

### 缓存策略

#### 1. 多级缓存
```python
class FactorCache:
    """
    因子计算缓存系统
    
    缓存层级:
    1. 内存缓存 (最快，容量小)
    2. Redis缓存 (快速，容量中等)
    3. 数据库缓存 (慢速，容量大)
    """
    
    def __init__(self):
        self.memory_cache = {}
        self.redis_client = redis.Redis()
    
    def get_cached_factor(self, cache_key: str):
        # 1. 检查内存缓存
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # 2. 检查Redis缓存
        redis_data = self.redis_client.get(cache_key)
        if redis_data:
            data = pickle.loads(redis_data)
            self.memory_cache[cache_key] = data  # 回填内存缓存
            return data
        
        # 3. 检查数据库缓存
        db_data = self.get_from_database(cache_key)
        if db_data is not None:
            self.set_cache(cache_key, db_data)
            return db_data
        
        return None
    
    def set_cache(self, cache_key: str, data, expire_seconds: int = 3600):
        # 设置多级缓存
        self.memory_cache[cache_key] = data
        self.redis_client.setex(cache_key, expire_seconds, pickle.dumps(data))
```

## 性能优化

### 1. 数据库优化

#### 索引策略
```sql
-- 股票日线数据索引
CREATE INDEX idx_stock_daily_ts_code_date ON stock_daily (ts_code, trade_date);
CREATE INDEX idx_stock_daily_date ON stock_daily (trade_date);

-- 因子数据索引
CREATE INDEX idx_factor_data_composite ON factor_data (ts_code, factor_name, factor_date);
CREATE INDEX idx_factor_data_date ON factor_data (factor_date);

-- 时序数据分区 (TimescaleDB)
SELECT create_hypertable('factor_data', 'factor_date', chunk_time_interval => INTERVAL '1 month');
```

#### 查询优化
```python
def optimized_data_query(ts_codes: List[str], start_date: str, end_date: str):
    """
    优化的数据查询
    
    优化策略:
    1. 使用参数化查询
    2. 批量查询减少数据库连接
    3. 只查询必要字段
    4. 使用适当的索引
    """
    query = """
    SELECT ts_code, trade_date, open, high, low, close, volume
    FROM stock_daily 
    WHERE ts_code = ANY(%s) 
    AND trade_date BETWEEN %s AND %s
    ORDER BY ts_code, trade_date
    """
    
    with engine.connect() as conn:
        result = conn.execute(query, (ts_codes, start_date, end_date))
        return pd.DataFrame(result.fetchall(), columns=result.keys())
```

### 2. 内存优化

#### 数据类型优化
```python
def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    优化DataFrame内存使用
    
    策略:
    1. 使用合适的数据类型
    2. 分类数据使用category类型
    3. 整数使用最小可能类型
    """
    # 股票代码使用category
    if 'ts_code' in df.columns:
        df['ts_code'] = df['ts_code'].astype('category')
    
    # 价格数据使用float32
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if col in df.columns:
            df[col] = df[col].astype('float32')
    
    # 成交量使用合适的整数类型
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], downcast='integer')
    
    return df
```

#### 批处理策略
```python
def batch_process_stocks(ts_codes: List[str], batch_size: int = 100):
    """
    批处理股票数据
    
    优势:
    1. 控制内存使用
    2. 提高数据库查询效率
    3. 支持大规模数据处理
    """
    for i in range(0, len(ts_codes), batch_size):
        batch = ts_codes[i:i+batch_size]
        
        # 处理当前批次
        batch_data = load_stock_data(batch)
        batch_factors = calculate_factors(batch_data)
        save_factors(batch_factors)
        
        # 清理内存
        del batch_data, batch_factors
        gc.collect()
```

### 3. 算法优化

#### 向量化计算
```python
def vectorized_rsi_calculation(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    向量化RSI计算
    
    优势:
    1. 避免Python循环
    2. 利用NumPy优化
    3. 显著提升计算速度
    """
    delta = prices.diff()
    
    # 使用向量化操作
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # 使用pandas的rolling函数
    avg_gains = gains.rolling(window=window, min_periods=1).mean()
    avg_losses = losses.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi
```

## 错误处理和监控

### 1. 异常处理策略

#### 分层异常处理
```python
class FactorCalculationError(Exception):
    """因子计算基础异常"""
    pass

class DataQualityError(FactorCalculationError):
    """数据质量异常"""
    pass

class CalculationTimeoutError(FactorCalculationError):
    """计算超时异常"""
    pass

def robust_factor_calculation(calc_func, *args, **kwargs):
    """
    健壮的因子计算包装器
    
    特性:
    1. 自动重试
    2. 异常分类处理
    3. 降级策略
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            return calc_func(*args, **kwargs)
            
        except DataQualityError as e:
            logger.warning(f"数据质量问题: {e}")
            # 数据质量问题不重试，返回空结果
            return pd.Series(dtype=float)
            
        except CalculationTimeoutError as e:
            logger.warning(f"计算超时: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                return pd.Series(dtype=float)
            time.sleep(2 ** retry_count)  # 指数退避
            
        except Exception as e:
            logger.error(f"未知错误: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                raise
            time.sleep(1)
```

### 2. 监控和告警

#### 性能监控
```python
class FactorCalculationMonitor:
    """因子计算监控器"""
    
    def __init__(self):
        self.metrics = {
            'calculation_count': 0,
            'success_count': 0,
            'error_count': 0,
            'total_time': 0,
            'avg_time': 0
        }
    
    def record_calculation(self, factor_name: str, calculation_time: float, success: bool):
        """记录计算指标"""
        self.metrics['calculation_count'] += 1
        self.metrics['total_time'] += calculation_time
        
        if success:
            self.metrics['success_count'] += 1
        else:
            self.metrics['error_count'] += 1
        
        self.metrics['avg_time'] = self.metrics['total_time'] / self.metrics['calculation_count']
        
        # 性能告警
        if calculation_time > 30:  # 30秒阈值
            self.send_alert(f"因子{factor_name}计算时间过长: {calculation_time:.2f}秒")
        
        # 错误率告警
        error_rate = self.metrics['error_count'] / self.metrics['calculation_count']
        if error_rate > 0.1:  # 10%错误率阈值
            self.send_alert(f"因子计算错误率过高: {error_rate:.2%}")
    
    def send_alert(self, message: str):
        """发送告警"""
        logger.error(f"ALERT: {message}")
        # 这里可以集成钉钉、邮件等告警方式
```

## 部署和运维

### 1. 容器化部署

#### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY src/ ./src/
COPY config.yml .

# 创建非root用户
RUN useradd -m -u 1000 stockschool && chown -R stockschool:stockschool /app
USER stockschool

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  factor-engine:
    build: .
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/stockschool
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    ports:
      - "8000:8000"
    
  postgres:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_DB: stockschool
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### 2. 监控和日志

#### 日志配置
```python
import logging
from logging.handlers import RotatingFileHandler
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)

# 配置日志
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = RotatingFileHandler(
        'logs/factor_engine.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
```

## 最佳实践

### 1. 因子开发流程

1. **需求分析**: 明确因子的金融含义和预期效果
2. **数据准备**: 确保数据质量和完整性
3. **算法实现**: 使用向量化计算，注意边界条件
4. **单元测试**: 验证计算正确性和边界情况
5. **性能测试**: 确保计算效率满足要求
6. **回测验证**: 验证因子的有效性
7. **生产部署**: 监控运行状态和性能指标

### 2. 代码规范

#### 函数设计
```python
def calculate_factor(
    data: pd.DataFrame,
    window: int = 20,
    **kwargs
) -> pd.Series:
    """
    因子计算函数模板
    
    Args:
        data: 股票数据，必须包含['close']列
        window: 计算窗口期
        **kwargs: 其他参数
    
    Returns:
        因子值序列
    
    Raises:
        ValueError: 数据不足或参数无效
        DataQualityError: 数据质量问题
    """
    # 1. 参数验证
    if data.empty:
        raise ValueError("输入数据不能为空")
    
    if len(data) < window:
        raise ValueError(f"数据长度{len(data)}小于窗口期{window}")
    
    if 'close' not in data.columns:
        raise ValueError("数据必须包含'close'列")
    
    # 2. 数据质量检查
    if data['close'].isna().sum() > len(data) * 0.1:
        raise DataQualityError("收盘价缺失值过多")
    
    # 3. 因子计算
    try:
        factor_values = data['close'].rolling(window=window).mean()
        return factor_values
    except Exception as e:
        logger.error(f"因子计算失败: {e}")
        raise
```

### 3. 性能优化建议

1. **数据预加载**: 批量加载数据，减少数据库查询次数
2. **并行计算**: 合理使用多线程/多进程，注意GIL限制
3. **内存管理**: 及时释放大对象，使用生成器处理大数据集
4. **缓存策略**: 合理设置缓存过期时间，避免脏数据
5. **算法优化**: 优先使用向量化操作，避免Python循环

### 4. 故障排除

#### 常见问题和解决方案

1. **计算结果为空**
   - 检查输入数据是否为空
   - 验证数据时间范围是否正确
   - 确认因子参数是否合理

2. **计算速度慢**
   - 检查数据库索引是否正确
   - 优化SQL查询语句
   - 考虑增加缓存或并行计算

3. **内存使用过高**
   - 减少批处理大小
   - 及时释放不需要的数据
   - 使用更高效的数据类型

4. **数据不一致**
   - 检查数据源的一致性
   - 验证计算逻辑的正确性
   - 确认缓存是否过期

## 总结

StockSchool因子计算引擎通过模块化设计、并行计算、多级缓存等技术手段，实现了高性能、高可靠性的量化因子计算。系统支持三大类因子的计算，具备完善的错误处理、监控告警和运维支持能力。

通过遵循本文档的技术规范和最佳实践，可以确保因子计算的准确性、效率和稳定性，为量化投资决策提供可靠的数据支持。