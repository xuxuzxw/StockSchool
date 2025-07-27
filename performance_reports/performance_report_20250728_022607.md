# StockSchool 性能分析报告
生成时间: 2025-07-28 02:26:07

## factor_engine 性能分析

- 总调用次数: 472
- 总执行时间: 0.0011 秒

### 累计耗时最长的前5个函数

| 排名 | 调用次数 | 累计时间(秒) | 单次时间(秒) | 函数 |
|------|----------|--------------|--------------|------|
| 1 | 1 | 0.0010 | 0.001000 | D:\Users\xuxuz\Desktop\StockSchool\src\utils\profiler.py:78(test_factor_calculation) |
| 2 | 1 | 0.0010 | 0.001000 | D:\Users\xuxuz\Desktop\StockSchool\src\compute\factor_engine.py:28(__init__) |
| 3 | 1 | 0.0010 | 0.001000 | D:\Users\xuxuz\Desktop\StockSchool\src\utils\db.py:31(get_db_engine) |
| 4 | 1 | 0.0010 | 0.001000 | <string>:1(create_engine) |
| 5 | 2/1 | 0.0010 | 0.001000 | C:\Users\xuxuz\AppData\Local\Programs\Python\Python311\Lib\site-packages\sqlalchemy\util\deprecations.py:249(warned) |

### 性能瓶颈分析

- 未发现明显的性能瓶颈

---

## training_pipeline 性能分析

- 总调用次数: 1,016
- 总执行时间: 0.0027 秒

### 累计耗时最长的前5个函数

| 排名 | 调用次数 | 累计时间(秒) | 单次时间(秒) | 函数 |
|------|----------|--------------|--------------|------|
| 1 | 1 | 0.0030 | 0.003000 | D:\Users\xuxuz\Desktop\StockSchool\src\utils\profiler.py:117(test_training_pipeline) |
| 2 | 1 | 0.0030 | 0.003000 | D:\Users\xuxuz\Desktop\StockSchool\src\ai\training_pipeline.py:44(__init__) |
| 3 | 2 | 0.0020 | 0.001000 | D:\Users\xuxuz\Desktop\StockSchool\src\utils\db.py:31(get_db_engine) |
| 4 | 2 | 0.0010 | 0.001000 | <string>:1(create_engine) |
| 5 | 4/2 | 0.0010 | 0.001000 | C:\Users\xuxuz\AppData\Local\Programs\Python\Python311\Lib\site-packages\sqlalchemy\util\deprecations.py:249(warned) |

### 性能瓶颈分析

- 未发现明显的性能瓶颈

---

## 优化建议

- 使用数据库索引优化查询性能
- 考虑使用缓存机制减少重复计算
- 批量处理数据以减少数据库连接开销
- 使用向量化操作替代循环计算
- 考虑使用多进程或多线程并行处理
- 优化数据结构和算法复杂度
- 减少不必要的数据复制和转换
- 使用更高效的数据格式（如Parquet）