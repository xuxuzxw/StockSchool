# StockSchool 配置文件管理指南

## 概述

StockSchool 使用环境变量文件来管理不同环境的配置。本指南将帮助您理解和管理这些配置文件。

## 配置文件结构

### 主要配置文件

| 文件名 | 用途 | 说明 |
|--------|------|------|
| `.env` | 开发环境配置 | 默认的开发环境配置文件 |
| `.env.acceptance` | 验收测试环境配置 | 用于验收测试的配置 |
| `.env.prod.example` | 生产环境配置模板 | 生产环境配置示例，需要复制并修改 |
| `config/env_template.env` | 配置模板 | 标准配置模板，包含所有可用配置项 |

### 配置文件优先级

1. 环境变量 (最高优先级)
2. `.env.{ENVIRONMENT}` 文件
3. `.env` 文件
4. 默认值 (最低优先级)

## 配置分类

### 1. 环境标识配置

```bash
ENVIRONMENT=development          # 环境类型
APP_NAME=StockSchool            # 应用名称
APP_VERSION=1.0.0               # 应用版本
DEBUG=true                      # 调试模式
LOG_LEVEL=INFO                  # 日志级别
TZ=Asia/Shanghai                # 时区设置
```

### 2. 数据库配置

```bash
# PostgreSQL + TimescaleDB
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=stockschool
POSTGRES_USER=stockschool
POSTGRES_PASSWORD=your_password_here
DATABASE_URL=postgresql://user:pass@host:port/db

# 连接池配置
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600
```

### 3. Redis配置

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
REDIS_DB=0
REDIS_URL=redis://:password@host:port/db

# 连接池配置
REDIS_POOL_SIZE=20
REDIS_POOL_TIMEOUT=10
REDIS_MAX_CONNECTIONS=50
```

### 4. 数据源API配置

```bash
# Tushare API
TUSHARE_TOKEN=your_tushare_token_here
TUSHARE_BASE_URL=http://api.tushare.pro
TUSHARE_TIMEOUT=30
TUSHARE_MAX_RETRIES=3

# AkShare API
AKSHARE_TOKEN=your_akshare_token_here
AKSHARE_ENABLED=true
AKSHARE_TIMEOUT=20
```

### 5. AI大模型API配置

```bash
# OpenAI API
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# Gemini API
GEMINI_API_KEY=your_gemini_api_key

# 通用配置
AI_API_TIMEOUT=30
AI_API_MAX_RETRIES=3
AI_API_TEMPERATURE=0.1
```

### 6. 安全配置

```bash
SECRET_KEY=your_secret_key_here           # 至少32字符
JWT_SECRET_KEY=your_jwt_secret_key        # JWT密钥
JWT_ALGORITHM=HS256                       # JWT算法
JWT_EXPIRATION_HOURS=24                   # JWT过期时间

# API安全
API_RATE_LIMIT_PER_MINUTE=100
API_RATE_LIMIT_PER_HOUR=1000
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ORIGINS=http://localhost:8000
```

## 配置管理工具

我们提供了一个配置管理工具 `scripts/config_manager.py` 来帮助您管理配置文件。

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用方法

#### 1. 验证配置文件

```bash
# 验证默认配置文件
python scripts/config_manager.py validate

# 验证指定配置文件
python scripts/config_manager.py validate --file .env.acceptance --env acceptance

# 验证生产环境配置
python scripts/config_manager.py validate --file .env.prod --env production
```

#### 2. 生成配置文件

```bash
# 生成开发环境配置
python scripts/config_manager.py generate --env development

# 生成生产环境配置
python scripts/config_manager.py generate --env production --output .env.prod

# 生成验收测试配置
python scripts/config_manager.py generate --env acceptance
```

#### 3. 比较配置文件

```bash
# 比较两个配置文件
python scripts/config_manager.py compare --file .env --file2 .env.acceptance

# 比较开发和生产配置
python scripts/config_manager.py compare --file .env --file2 .env.prod
```

#### 4. 列出所有配置文件

```bash
python scripts/config_manager.py list
```

#### 5. 安全检查

```bash
# 检查配置文件安全性
python scripts/config_manager.py security --file .env

# 检查生产环境配置安全性
python scripts/config_manager.py security --file .env.prod
```

## 环境特定配置

### 开发环境 (development)

- 启用调试模式
- 详细日志输出
- 较小的资源限制
- 允许热重载

```bash
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
RELOAD=true
```

### 验收测试环境 (acceptance)

- 启用测试模式
- 调试级别日志
- 测试数据路径配置
- 性能基准设置

```bash
ENVIRONMENT=acceptance
TEST_MODE=true
LOG_LEVEL=DEBUG
TEST_DATA_PATH=./test_data
```

### 生产环境 (production)

- 禁用调试模式
- 警告级别日志
- 强化安全配置
- 监控和告警配置

```bash
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
MONITORING_ENABLED=true
```

## 安全最佳实践

### 1. 密码和密钥管理

- 使用强密码（至少8个字符）
- SECRET_KEY 和 JWT_SECRET_KEY 至少32个字符
- 不要在代码中硬编码敏感信息
- 定期轮换密钥和密码

### 2. 生产环境安全

- 禁用调试模式 (`DEBUG=false`)
- 使用警告或错误日志级别
- 配置适当的CORS源
- 启用HTTPS
- 设置合理的API限流

### 3. 敏感信息保护

以下配置项包含敏感信息，需要特别保护：

- `POSTGRES_PASSWORD`
- `REDIS_PASSWORD`
- `SECRET_KEY`
- `JWT_SECRET_KEY`
- `TUSHARE_TOKEN`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `SMTP_PASSWORD`

### 4. 文件权限

```bash
# 设置配置文件权限
chmod 600 .env*
```

## 故障排查

### 常见问题

#### 1. 数据库连接失败

检查以下配置：
- `POSTGRES_HOST`、`POSTGRES_PORT` 是否正确
- `POSTGRES_USER`、`POSTGRES_PASSWORD` 是否有效
- `DATABASE_URL` 格式是否正确

#### 2. Redis连接失败

检查以下配置：
- `REDIS_HOST`、`REDIS_PORT` 是否正确
- `REDIS_PASSWORD` 是否匹配
- `REDIS_URL` 格式是否正确

#### 3. API调用失败

检查以下配置：
- `TUSHARE_TOKEN` 是否有效
- API超时设置是否合理
- 网络连接是否正常

#### 4. 端口冲突

使用配置管理工具检查端口冲突：

```bash
python scripts/config_manager.py validate --file .env
```

### 调试技巧

#### 1. 启用详细日志

```bash
LOG_LEVEL=DEBUG
LOG_TO_CONSOLE=true
```

#### 2. 检查环境变量

```bash
# 在Python中检查环境变量
import os
print(f"POSTGRES_HOST: {os.getenv('POSTGRES_HOST')}")
print(f"REDIS_HOST: {os.getenv('REDIS_HOST')}")
```

#### 3. 验证配置加载

```bash
# 使用配置管理工具验证
python scripts/config_manager.py validate --file .env --env development
```

## 配置文件模板

### 快速开始模板

```bash
# 复制模板文件
cp config/env_template.env .env

# 编辑配置文件
nano .env

# 验证配置
python scripts/config_manager.py validate
```

### 生产环境部署模板

```bash
# 生成生产环境配置
python scripts/config_manager.py generate --env production --output .env.prod

# 编辑生产环境配置
nano .env.prod

# 安全检查
python scripts/config_manager.py security --file .env.prod

# 验证配置
python scripts/config_manager.py validate --file .env.prod --env production
```

## 更新和维护

### 配置文件更新流程

1. 备份现有配置文件
2. 更新配置模板
3. 使用配置管理工具生成新配置
4. 合并自定义配置
5. 验证新配置
6. 测试应用启动

### 版本控制

- 将 `.env.example` 文件加入版本控制
- 不要将包含敏感信息的 `.env` 文件加入版本控制
- 在 `.gitignore` 中排除敏感配置文件

```gitignore
# 环境配置文件
.env
.env.local
.env.production
.env.*.local
```

## 支持和帮助

如果您在配置过程中遇到问题，可以：

1. 查看本文档的故障排查部分
2. 使用配置管理工具进行诊断
3. 检查应用日志文件
4. 联系技术支持团队

---

**注意**: 请确保在生产环境中使用强密码和安全的配置设置。定期审查和更新配置文件以保持系统安全。