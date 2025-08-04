# 配置系统迁移报告

扫描目录: d:\Users\xuxuz\Desktop\StockSchool\src
发现需要迁移的文件: 30个

## 需要迁移的文件:

### src\ai\training_pipeline.py
匹配的模式:
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\api\auth.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\api\cache.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\api\config.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\api\explainer_api.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\api\factor_management_api.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\api\main.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\compute\feature_store_adapter.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\compute\indicators.py
匹配的模式:
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\compute\processing.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\compute\tasks.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\data\data_quality_monitor.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\data\data_sync_scheduler.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\data\timescale_optimizer.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\data\tushare_sync.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\features\factor_feature_store.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\features\feature_store.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\monitoring\alerts.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\monitoring\api.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\monitoring\collectors.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\monitoring\logger.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\monitoring\notifications.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\monitoring\performance.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\strategy\ai_model.py
匹配的模式:
  - from\s+.*\.utils\.config_loader\s+import\s+Config

### src\strategy\model_explainer.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\strategy\shap_explainer.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\tests\test_config.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config
  - from\s+src\.utils\.config_loader\s+import\s+Config
  - from\s+.*\.utils\.config_loader\s+import\s+Config

### src\tests\test_gpu_utils.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\ui\explainer_dashboard.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

### src\utils\gpu_utils.py
匹配的模式:
  - from\s+src\.utils\.config_loader\s+import\s+config
  - from\s+.*\.utils\.config_loader\s+import\s+config

## 迁移建议:

1. 将所有旧的导入语句替换为:
   ```python
   from src.config.unified_config import config
   ```

2. 确保配置调用方式保持不变:
   ```python
   value = config.get('key.path', default_value)
   ```

3. 新的统一配置系统提供以下额外功能:
   - 多环境支持
   - 配置热更新
   - 配置验证
   - 变更历史和回滚
   - 更好的错误处理