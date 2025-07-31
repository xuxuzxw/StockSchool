# StockSchool 用户手册

## 目录
1. [系统概述](#系统概述)
2. [安装指南](#安装指南)
3. [快速开始](#快速开始)
4. [核心功能](#核心功能)
5. [API使用](#api使用)
6. [模型解释](#模型解释)
7. [性能优化](#性能优化)
8. [故障排除](#故障排除)
9. [最佳实践](#最佳实践)

## 系统概述

StockSchool是一个专业的量化投研平台，旨在为投资者和研究人员提供完整的股票分析、因子计算、模型训练和预测服务。系统采用模块化设计，支持GPU加速，提供丰富的API接口和可视化功能。

### 主要特性
- **多数据源支持**: 集成Tushare、AkShare等数据源
- **因子计算引擎**: 50+种技术因子和基本面因子
- **AI模型训练**: 支持多种机器学习算法
- **模型解释**: SHAP、置换重要性等解释方法
- **可视化分析**: 丰富的图表和交互式界面
- **GPU加速**: CUDA支持的高性能计算
- **实时监控**: 系统性能和告警监控

## 安装指南

### 系统要求
- **操作系统**: Windows 10/11, Linux, macOS
- **Python版本**: 3.8+
- **内存**: 至少16GB RAM（推荐32GB+）
- **GPU**: CUDA兼容显卡（可选，推荐）
- **存储**: 至少50GB可用空间

### 安装步骤

#### 1. 克隆项目
```bash
git clone https://github.com/yourusername/stockschool.git
cd stockschool
```

#### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows
```

#### 3. 安装依赖
```bash
pip install -r requirements.txt
```

#### 4. 配置环境变量
创建 `.env` 文件：
```bash
TUSHARE_TOKEN=your_tushare_token
DATABASE_URL=sqlite:///stock_data.db
API_KEY=your_api_key
```

#### 5. 初始化数据库
```bash
python run.py --init-db
```

#### 6. 验证安装
```bash
python -m pytest tests/ -v
```

## 快速开始

### 1. 启动系统
```bash
# 启动API服务
python run.py
# 选择 "1. 启动API服务器"

# 或直接启动
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 2. 数据同步
```bash
# 同步基础数据
python run.py
# 选择 "2. 运行数据同步"
# 然后选择具体的数据同步选项
```

### 3. 因子计算
```bash
# 计算技术因子
python -m src.compute.factor_engine --action calculate --ts-code 000001.SZ
```

### 4. 模型训练
```bash
# 训练预测模型
python -m src.ai.training_pipeline --model-type lightgbm --target-days 5
```

### 5. 模型解释
```bash
# 启动模型解释API
python run_explainer_api.py
```

## 核心功能

### 数据管理

#### 数据同步
StockSchool支持多种数据源的自动同步：

```python
from src.data.tushare_sync import TushareSynchronizer
from src.utils.config_loader import Config

config = Config()
syncer = TushareSynchronizer(config)

# 同步股票基本信息
syncer.sync_stock_basic()

# 同步日线数据
syncer.sync_daily(start_date='20240101', end_date='20241231')

# 同步财务数据
syncer.sync_financial_data()
```

#### 数据质量控制
```python
from src.compute.quality import QualityController

qc = QualityController()
issues = qc.check_data_quality('stock_daily', date='20240101')
```

### 因子计算

#### 技术因子
```python
from src.compute.indicators import TechnicalIndicators

ti = TechnicalIndicators()
df = ti.calculate_rsi(data, window=14)
df = ti.calculate_macd(data)
df = ti.calculate_bollinger_bands(data)
```

#### 基本面因子
```python
from src.compute.indicators import FundamentalIndicators

fi = FundamentalIndicators()
df = fi.calculate_roe(financial_data)
df = fi.calculate_pe_ratio(market_data, financial_data)
```

### AI模型训练

#### 训练流程
```python
from src.ai.training_pipeline import ModelTrainingPipeline

pipeline = ModelTrainingPipeline()
model, metrics = pipeline.train_model(
    model_type='lightgbm',
    target_days=5,
    test_size=0.2
)
```

#### 模型预测
```python
from src.ai.prediction import StockPredictor

predictor = StockPredictor()
predictions = predictor.predict_stocks(
    ts_codes=['000001.SZ', '000002.SZ'],
    model_path='/models/stock_model.pkl'
)
```

## API使用

### Python客户端
```python
import requests
import json

class StockSchoolClient:
    def __init__(self, base_url='http://localhost:8000', api_key=None):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {'Content-Type': 'application/json'}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def get_stock_basic(self, limit=100, offset=0):
        """获取股票基础信息"""
        url = f"{self.base_url}/api/v1/stocks/basic"
        params = {'limit': limit, 'offset': offset}
        if self.api_key:
            params['api_key'] = self.api_key
        
        response = requests.get(url, headers=self.headers, params=params)
        return response.json()
    
    def get_stock_daily(self, ts_code, start_date=None, end_date=None):
        """获取股票日线数据"""
        url = f"{self.base_url}/api/v1/stocks/daily"
        params = {'ts_code': ts_code}
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        if self.api_key:
            params['api_key'] = self.api_key
        
        response = requests.get(url, headers=self.headers, params=params)
        return response.json()
    
    def calculate_factors(self, ts_code, factor_names, start_date=None, end_date=None):
        """计算因子"""
        url = f"{self.base_url}/api/v1/factors/calculate"
        data = {
            'ts_code': ts_code,
            'factor_names': factor_names
        }
        if start_date:
            data['start_date'] = start_date
        if end_date:
            data['end_date'] = end_date
        
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()
    
    def explain_prediction(self, model_path, sample_data, method='shap'):
        """解释预测"""
        url = f"{self.base_url}/api/v1/explain/prediction"
        data = {
            'model_path': model_path,
            'sample': sample_data,
            'explanation_type': method
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()

# 使用示例
client = StockSchoolClient(api_key='your_api_key')

# 获取股票列表
stocks = client.get_stock_basic(limit=50)

# 获取具体股票数据
daily_data = client.get_stock_daily('000001.SZ', '20240101', '20241231')

# 计算因子
factors = client.calculate_factors('000001.SZ', ['rsi', 'macd', 'bollinger'])

# 模型解释
sample = {
    'features': ['open', 'high', 'low', 'close', 'volume'],
    'values': [15.2, 15.8, 15.1, 15.6, 1000000]
}
explanation = client.explain_prediction('/models/model.pkl', sample)
```

### 命令行工具
```bash
# 健康检查
curl http://localhost:8000/health

# 获取股票基础信息
curl "http://localhost:8000/api/v1/stocks/basic?limit=10"

# 获取股票日线数据
curl "http://localhost:8000/api/v1/stocks/daily?ts_code=000001.SZ&start_date=20240101"

# 计算因子
curl -X POST "http://localhost:8000/api/v1/factors/calculate" \
     -H "Content-Type: application/json" \
     -d '{"ts_code": "000001.SZ", "factor_names": ["rsi", "macd"]}'

# 模型解释
curl -X POST "http://localhost:8000/api/v1/explain/feature-importance" \
     -H "Content-Type: application/json" \
     -d '{"model_path": "/models/model.pkl", "method": "shap"}'
```

## 模型解释

### SHAP解释器
```python
from src.strategy.shap_explainer import SHAPExplainer
import pandas as pd

# 创建解释器
explainer = SHAPExplainer(model, feature_names)

# 计算特征重要性
importance_df = explainer.calculate_feature_importance(X_test, y_test)

# 解释单个预测
explanation = explainer.explain_prediction(X_test, sample_idx=0)

# 批量解释
explanations = explainer.batch_explain(X_test.head(100))
```

### 置换解释器
```python
from src.strategy.permutation_explainer import PermutationExplainer

# 创建解释器
explainer = PermutationExplainer(model, feature_names)

# 计算置换重要性
importance_df = explainer.calculate_feature_importance(X_test, y_test, n_repeats=10)

# 计算模型性能
performance = explainer.get_model_performance(X_test, y_test)

# 计算特征交互
interactions = explainer.calculate_interaction_strength(X_test, y_test)
```

### 可视化
```python
from src.strategy.visualization import ModelVisualizer

# 创建可视化器
visualizer = ModelVisualizer(model, feature_names)

# 绘制特征重要性图
fig = visualizer.plot_feature_importance(importance_df)

# 创建交互式图表
interactive_fig = visualizer.create_interactive_feature_importance(importance_df)

# 绘制预测解释图
explanation_fig = visualizer.plot_prediction_explanation(explanation)

# 保存图表
visualizer.save_plot(fig, 'feature_importance', format='png')
```

## 性能优化

### GPU加速配置
```yaml
# config.yml
feature_params:
  use_cuda: true          # 启用GPU加速
  shap_batch_size: 500    # SHAP计算批量大小
  max_cache_size: 40960   # 缓存限制（MB）
  chunk_size: 10000       # 分块处理大小
  max_gpu_memory: 20480   # 限制GPU显存使用（MB）
  gpu_oom_retry: 3        # OOM重试次数
  fallback_to_cpu: true   # GPU失败时自动回退到CPU
```

### 批量处理优化
```python
from src.utils.gpu_utils import get_batch_size

# 获取最优批量大小
batch_size = get_batch_size(data_size=len(X))

# 分批处理大数据集
for i in range(0, len(large_dataset), batch_size):
    batch = large_dataset[i:i+batch_size]
    process_batch(batch)
```

### 内存管理
```python
from src.utils.gpu_utils import handle_oom, fallback_to_cpu

try:
    # 执行内存密集型操作
    result = memory_intensive_operation()
except torch.cuda.OutOfMemoryError:
    # 处理内存不足
    if handle_oom(retry_count=0, max_retries=3):
        # 重试
        result = memory_intensive_operation()
    else:
        # 降级到CPU
        cpu_device = fallback_to_cpu()
        result = memory_intensive_operation(device=cpu_device)
```

## 故障排除

### 常见问题

#### 1. GPU相关问题
```bash
# 检查CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"

# 检查GPU状态
nvidia-smi

# 重新安装PyTorch CUDA版本
pip uninstall torch torchvision torchaudio
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

#### 2. 数据同步问题
```bash
# 检查Tushare token
echo $TUSHARE_TOKEN

# 测试API连接
python -c "import tushare as ts; ts.set_token('your_token'); pro = ts.pro_api(); print(pro.stock_basic())"

# 清理数据库
python scripts/clear_database.py
```

#### 3. 模型加载问题
```bash
# 检查模型文件
ls -la models/

# 验证模型完整性
python -c "import joblib; model = joblib.load('models/stock_model.pkl'); print(type(model))"
```

#### 4. 内存不足问题
```bash
# 监控系统资源
htop  # Linux
# 或
taskmgr  # Windows

# 调整配置参数
# 在config.yml中减少batch_size和cache_size
```

### 日志查看
```bash
# 查看主日志
tail -f logs/stockschool_$(date +%Y-%m-%d).log

# 查看错误日志
tail -f logs/error_$(date +%Y-%m-%d).log

# 查看API日志
tail -f logs/api_$(date +%Y-%m-%d).log
```

### 性能监控
```python
from src.monitoring.performance import PerformanceMonitor

monitor = PerformanceMonitor()

# 获取系统指标
metrics = monitor.get_system_metrics()

# 监控GPU使用
gpu_info = monitor.get_gpu_metrics()

# 性能分析
profiler = monitor.start_profiling()
# 执行操作
profiler.stop()
report = profiler.get_report()
```

## 最佳实践

### 1. 数据管理最佳实践

#### 定期数据同步
```bash
# 设置定时任务
# crontab -e
0 20 * * 1-5 cd /path/to/stockschool && python run.py --sync-daily
```

#### 数据备份
```bash
# 定期备份数据库
sqlite3 stock_data.db ".backup backup_$(date +%Y%m%d).db"
```

#### 数据质量检查
```python
# 定期运行数据质量检查
from src.compute.quality import QualityController

qc = QualityController()
qc.run_daily_check()
```

### 2. 模型管理最佳实践

#### 模型版本控制
```python
import joblib
from datetime import datetime

# 保存带时间戳的模型
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = f'models/stock_model_{timestamp}.pkl'
joblib.dump(model, model_path)

# 记录模型元数据
metadata = {
    'timestamp': timestamp,
    'model_type': 'LightGBM',
    'features': feature_names,
    'performance': metrics,
    'training_data_period': '2020-2024'
}
```

#### 模型验证
```python
# 交叉验证
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# 回测验证
def backtest_model(model, data, start_date, end_date):
    # 实现回测逻辑
    pass
```

### 3. 性能优化最佳实践

#### 缓存策略
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_factor_calculation(data_hash, parameters):
    # 计算因子
    pass

# 使用缓存
data_hash = hashlib.md5(data.to_json().encode()).hexdigest()
result = cached_factor_calculation(data_hash, parameters)
```

#### 并行处理
```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# 多线程处理
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_stock, ts_code) for ts_code in stock_list]
    results = [future.result() for future in futures]

# 多进程处理
if __name__ == '__main__':
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(process_stock, stock_list)
```

### 4. 监控告警最佳实践

#### 设置告警规则
```python
from src.monitoring.alerts import AlertEngine

alert_engine = AlertEngine()

# 设置数据质量告警
alert_engine.add_rule(
    name='data_quality_alert',
    condition=lambda data: data['missing_rate'] > 0.05,
    level='HIGH',
    title='数据质量异常',
    message='缺失数据率超过5%'
)

# 设置性能告警
alert_engine.add_rule(
    name='performance_alert',
    condition=lambda data: data['processing_time'] > 300,
    level='MEDIUM',
    title='处理时间过长',
    message='数据处理时间超过5分钟'
)
```

#### 通知配置
```yaml
# config.yml
monitoring_params:
  alerts:
    email:
      smtp_server: smtp.gmail.com
      smtp_port: 587
      username: your_email@gmail.com
      password: your_password
    webhook:
      url: https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

### 5. 安全最佳实践

#### API安全
```python
# 使用API密钥认证
from fastapi import Depends, HTTPException
from src.utils.auth import verify_api_key

@app.get("/api/v1/stocks/basic")
async def get_stock_basic(api_key: str = Depends(verify_api_key)):
    # 业务逻辑
    pass

# 限流配置
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/v1/stocks/basic")
@limiter.limit("100/minute")
async def get_stock_basic(request: Request):
    # 业务逻辑
    pass
```

#### 数据安全
```python
# 敏感信息加密
from cryptography.fernet import Fernet

# 生成密钥
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# 加密敏感数据
encrypted_token = cipher_suite.encrypt(b'your_sensitive_token')

# 解密
decrypted_token = cipher_suite.decrypt(encrypted_token)
```

## 附录

### 配置文件详解
```yaml
# config.yml 详细说明
api_params:
  host: "0.0.0.0"           # API监听地址
  port: 8000               # API端口
  workers: 1               # 工作进程数
  reload: true             # 开发模式自动重载
  log_level: "info"        # 日志级别

data_sync_params:
  batch_size: 1000         # 数据同步批量大小
  retry_times: 3           # 重试次数
  sleep_interval: 0.3      # API调用间隔

feature_params:
  lookback_period: 14      # 回看周期
  use_cuda: true           # 启用GPU加速
  shap_batch_size: 500     # SHAP批量大小
  max_cache_size: 40960    # 最大缓存大小(MB)

monitoring_params:
  collection_interval: 60  # 监控收集间隔(秒)
  cpu_threshold: 80.0      # CPU使用率阈值
  memory_threshold: 85.0   # 内存使用率阈值
```

### 常用命令参考
```bash
# 系统管理
python run.py                    # 启动主控制台
python run_explainer_api.py      # 启动模型解释API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000  # 直接启动API

# 数据操作
python -m src.data.tushare_sync --action sync_stock_basic  # 同步股票基础数据
python -m src.data.tushare_sync --action sync_daily        # 同步日线数据
python scripts/clear_database.py                           # 清空数据库

# 计算任务
python -m src.compute.factor_engine --action calculate      # 计算因子
python -m src.ai.training_pipeline --model-type lightgbm   # 训练模型
python -m src.ai.prediction                                # 运行预测

# 测试验证
python -m pytest tests/ -v                                 # 运行所有测试
python -m pytest tests/test_api.py -v                      # 运行API测试
python logs/check_test_coverage.py                         # 检查测试覆盖率
python logs/run_performance_benchmark.py                   # 运行性能基准测试

# 监控维护
tail -f logs/stockschool_$(date +%Y-%m-%d).log             # 查看实时日志
nvidia-smi                                                 # 查看GPU状态
htop                                                       # 查看系统资源
```

### 性能基准参考
- **因子计算**: 1000只股票 < 30秒
- **模型训练**: 5年数据 < 5分钟  
- **单个预测解释**: < 1秒
- **批量预测解释**: 1000个样本 < 30秒
- **GPU加速**: 性能提升2-10倍（取决于任务类型）

### 联系支持
- **文档**: https://docs.stockschool.com
- **GitHub**: https://github.com/yourusername/stockschool
- **邮箱**: support@stockschool.com
- **社区**: https://community.stockschool.com

---
*文档版本: v1.1.7*
*最后更新: 2025-07-31*
