# StockSchool 测试标准

## 测试策略概述

### 测试金字塔
```
    /\
   /  \     E2E Tests (10%)
  /____\    
 /      \   Integration Tests (20%)
/________\  Unit Tests (70%)
```

- **单元测试**: 测试单个函数/方法的功能
- **集成测试**: 测试模块间的交互
- **端到端测试**: 测试完整的业务流程

### 测试覆盖率要求
- 单元测试覆盖率 ≥ 80%
- 关键业务逻辑覆盖率 ≥ 95%
- 新增代码覆盖率 ≥ 90%

## 单元测试规范

### 测试文件组织
```
src/
├── compute/
│   ├── factor_engine.py
│   └── technical.py
tests/
├── unit/
│   ├── compute/
│   │   ├── test_factor_engine.py
│   │   └── test_technical.py
│   └── conftest.py
├── integration/
└── e2e/
```

### 测试命名规范
```python
# 测试类命名: Test + 被测试类名
class TestFactorEngine:
    pass

# 测试方法命名: test_方法名_场景_期望结果
def test_calculate_rsi_with_valid_data_returns_correct_values(self):
    pass

def test_calculate_rsi_with_insufficient_data_raises_exception(self):
    pass

def test_calculate_rsi_with_empty_data_returns_empty_series(self):
    pass
```

### 测试结构 (AAA模式)
```python
def test_calculate_rsi_with_valid_data_returns_correct_values(self):
    # Arrange - 准备测试数据
    data = pd.DataFrame({
        'close': [10, 11, 12, 11, 13, 14, 13, 15, 16, 15, 17, 18, 17, 19, 20]
    })
    expected_rsi_range = (0, 100)
    
    # Act - 执行被测试的方法
    result = calculate_rsi(data, window=14)
    
    # Assert - 验证结果
    assert not result.empty, "RSI结果不应为空"
    assert result.iloc[-1] >= expected_rsi_range[0], "RSI值应大于等于0"
    assert result.iloc[-1] <= expected_rsi_range[1], "RSI值应小于等于100"
    assert not pd.isna(result.iloc[-1]), "最新RSI值不应为NaN"
```

### 测试数据管理
```python
# conftest.py - 共享测试数据和fixture
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_stock_data():
    """生成样本股票数据"""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)  # 确保可重现性
    
    prices = 100 + np.random.randn(100).cumsum()
    volumes = np.random.randint(1000, 10000, 100)
    
    return pd.DataFrame({
        'trade_date': dates,
        'ts_code': '000001.SZ',
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': volumes
    })

@pytest.fixture
def sample_financial_data():
    """生成样本财务数据"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ'],
        'end_date': ['2024-03-31', '2024-03-31'],
        'revenue': [1000000, 800000],
        'net_profit': [100000, 80000],
        'total_assets': [5000000, 4000000],
        'total_equity': [2000000, 1600000]
    })
```

### 异常测试
```python
def test_calculate_rsi_with_insufficient_data_raises_exception(self):
    """测试数据不足时的异常处理"""
    # 只有5条数据，但RSI需要14条
    insufficient_data = pd.DataFrame({
        'close': [10, 11, 12, 11, 13]
    })
    
    with pytest.raises(ValueError, match="数据长度不足"):
        calculate_rsi(insufficient_data, window=14)

def test_calculate_rsi_with_invalid_column_raises_exception(self):
    """测试缺少必要列时的异常处理"""
    invalid_data = pd.DataFrame({
        'price': [10, 11, 12, 11, 13, 14, 13, 15, 16, 15, 17, 18, 17, 19, 20]
    })
    
    with pytest.raises(KeyError, match="缺少必要列"):
        calculate_rsi(invalid_data, window=14)
```

### 参数化测试
```python
@pytest.mark.parametrize("window,expected_length", [
    (5, 95),   # 100条数据，5日窗口，前4个为NaN
    (10, 90),  # 100条数据，10日窗口，前9个为NaN
    (20, 80),  # 100条数据，20日窗口，前19个为NaN
])
def test_calculate_rsi_with_different_windows(sample_stock_data, window, expected_length):
    """测试不同窗口期的RSI计算"""
    result = calculate_rsi(sample_stock_data, window=window)
    
    # 验证非NaN值的数量
    valid_count = result.notna().sum()
    assert valid_count == expected_length
```

## 集成测试规范

### 数据库集成测试
```python
import pytest
from src.utils.test_db import get_test_db_manager

@pytest.fixture(scope="module")
def test_database():
    """测试数据库fixture"""
    db_manager = get_test_db_manager()
    
    # 设置测试环境
    db_manager.setup_test_environment()
    
    yield db_manager
    
    # 清理测试环境
    db_manager.cleanup_test_environment()

def test_factor_engine_integration(test_database, sample_stock_data):
    """测试因子引擎的集成功能"""
    # 插入测试数据
    test_database.insert_stock_data(sample_stock_data)
    
    # 创建因子引擎
    engine = FactorEngine()
    engine.engine = test_database.get_engine()
    
    # 执行因子计算
    result = engine.calculate_stock_factors('000001.SZ')
    
    # 验证结果
    assert result is True
    
    # 验证数据库中的结果
    factors = test_database.query_factors('000001.SZ')
    assert not factors.empty
    assert 'rsi_14' in factors.columns
```

### API集成测试
```python
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture
def api_client():
    return TestClient(app)

def test_factor_calculation_api_integration(api_client, test_database):
    """测试因子计算API的集成功能"""
    # 准备测试数据
    test_database.insert_stock_data(sample_stock_data)
    
    # 调用API
    response = api_client.post("/api/v1/factors/calculate", json={
        "ts_codes": ["000001.SZ"],
        "factor_types": ["technical"],
        "date": "2024-01-31"
    })
    
    # 验证响应
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    
    # 验证计算结果
    factors_response = api_client.get("/api/v1/factors", params={
        "ts_codes": "000001.SZ",
        "factors": "rsi_14",
        "date": "2024-01-31"
    })
    
    assert factors_response.status_code == 200
    factors_data = factors_response.json()
    assert factors_data["success"] is True
    assert len(factors_data["data"]) > 0
```

## 端到端测试规范

### 业务流程测试
```python
def test_complete_factor_calculation_workflow(test_database):
    """测试完整的因子计算工作流"""
    # 1. 数据同步
    synchronizer = TushareSynchronizer()
    synchronizer.engine = test_database.get_engine()
    
    sync_result = synchronizer.sync_stock_basic()
    assert sync_result is True
    
    # 2. 因子计算
    engine = FactorEngine()
    engine.engine = test_database.get_engine()
    
    calc_result = engine.calculate_all_factors(
        start_date='2024-01-01',
        end_date='2024-01-31',
        stock_list=['000001.SZ']
    )
    assert calc_result['success_count'] > 0
    
    # 3. AI模型训练
    trainer = AIModelTrainer()
    trainer.engine = test_database.get_engine()
    
    training_result = trainer.train_model(
        factor_names=['rsi_14', 'macd'],
        start_date='2024-01-01',
        end_date='2024-01-31'
    )
    assert training_result['test_metrics']['r2'] > 0
    
    # 4. 预测服务
    predictor = AIModelPredictor()
    predictor.load_model(training_result['model_path'])
    
    predictions = predictor.predict(['000001.SZ'], '2024-02-01')
    assert '000001.SZ' in predictions
    assert isinstance(predictions['000001.SZ'], float)
```

## 性能测试规范

### 基准测试
```python
import time
import pytest

def test_factor_calculation_performance(sample_stock_data):
    """测试因子计算性能"""
    # 扩展数据集到1000条记录
    large_dataset = pd.concat([sample_stock_data] * 10, ignore_index=True)
    
    start_time = time.time()
    result = calculate_rsi(large_dataset, window=14)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    # 验证性能要求：1000条数据应在1秒内完成
    assert execution_time < 1.0, f"计算时间过长: {execution_time:.2f}秒"
    assert not result.empty, "计算结果不应为空"

@pytest.mark.performance
def test_batch_factor_calculation_performance():
    """测试批量因子计算性能"""
    stock_list = [f"00000{i}.SZ" for i in range(1, 101)]  # 100只股票
    
    engine = FactorEngine()
    
    start_time = time.time()
    result = engine.calculate_all_factors(
        start_date='2024-01-01',
        end_date='2024-01-31',
        stock_list=stock_list
    )
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    # 验证性能要求：100只股票1个月数据应在60秒内完成
    assert execution_time < 60.0, f"批量计算时间过长: {execution_time:.2f}秒"
    assert result['success_count'] > 0, "应有成功计算的股票"
```

### 内存使用测试
```python
import psutil
import os

def test_memory_usage_during_calculation():
    """测试计算过程中的内存使用"""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 执行大量计算
    large_dataset = pd.DataFrame({
        'close': np.random.randn(10000).cumsum() + 100
    })
    
    result = calculate_rsi(large_dataset, window=14)
    
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = peak_memory - initial_memory
    
    # 验证内存使用合理（不超过100MB增长）
    assert memory_increase < 100, f"内存使用过多: {memory_increase:.2f}MB"
    assert not result.empty, "计算结果不应为空"
```

## 测试数据管理

### 黄金数据测试
```python
def test_rsi_calculation_with_golden_data():
    """使用黄金数据验证RSI计算准确性"""
    # 预先计算好的标准数据
    golden_input = pd.DataFrame({
        'close': [
            10.00, 10.50, 10.20, 10.80, 10.60, 10.90, 11.00,
            10.80, 11.20, 11.50, 11.30, 11.80, 11.60, 12.00
        ]
    })
    
    # 预期的RSI值（手工计算或权威工具计算）
    expected_rsi = 73.81
    
    result = calculate_rsi(golden_input, window=14)
    actual_rsi = result.iloc[-1]
    
    # 允许1%的误差
    assert abs(actual_rsi - expected_rsi) < 1.0, \
        f"RSI计算不准确，期望: {expected_rsi}, 实际: {actual_rsi:.2f}"
```

### 测试数据生成
```python
def generate_test_stock_data(
    ts_code: str = "000001.SZ",
    start_date: str = "2024-01-01",
    periods: int = 100,
    initial_price: float = 100.0,
    volatility: float = 0.02
) -> pd.DataFrame:
    """生成测试用股票数据"""
    dates = pd.date_range(start_date, periods=periods, freq='D')
    
    # 使用几何布朗运动生成价格
    np.random.seed(42)
    returns = np.random.normal(0, volatility, periods)
    prices = initial_price * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'ts_code': ts_code,
        'trade_date': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, periods)
    })
```

## 测试执行和报告

### 测试命令
```bash
# 运行所有测试
pytest

# 运行特定类型的测试
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# 运行特定模块的测试
pytest tests/unit/compute/test_technical.py

# 运行带覆盖率报告的测试
pytest --cov=src --cov-report=html

# 运行性能测试
pytest -m performance

# 并行运行测试
pytest -n auto
```

### 持续集成配置
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: timescale/timescaledb:latest-pg15
        env:
          POSTGRES_PASSWORD: test123
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-xdist
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml --cov-fail-under=80
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

### 测试报告
```python
# pytest.ini
[tool:pytest]
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=src
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml
    --junit-xml=test-results.xml

markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    slow: Slow running tests

testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```