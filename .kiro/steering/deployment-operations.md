# StockSchool 部署运维指南

## 部署架构

### 生产环境架构
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Web Gateway   │    │   API Gateway   │
│    (Nginx)      │────│   (Nginx)       │────│   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Static Files  │    │   Compute       │
│   Dashboard     │    │   (Frontend)    │    │   Services      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   TimescaleDB   │    │   Redis Cache   │    │   Task Queue    │
│   (Primary)     │────│   (Session)     │────│   (Celery)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
┌─────────────────┐
│   TimescaleDB   │
│   (Replica)     │
└─────────────────┘
```

### 容器化部署

#### Docker Compose 生产配置
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # 数据库服务
  postgres:
    image: timescale/timescaledb:latest-pg15
    container_name: stockschool_postgres
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      TIMESCALEDB_TELEMETRY: off
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database_schema.sql:/docker-entrypoint-initdb.d/01-schema.sql
      - ./backups:/backups
    ports:
      - "5432:5432"
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis缓存
  redis:
    image: redis:7-alpine
    container_name: stockschool_redis
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

  # API服务
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: stockschool_api
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - TUSHARE_TOKEN=${TUSHARE_TOKEN}
    depends_on:
      - postgres
      - redis
    ports:
      - "8000:8000"
    restart: unless-stopped
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2G
          cpus: '1'

  # 计算服务
  compute:
    build:
      context: .
      dockerfile: Dockerfile.compute
    container_name: stockschool_compute
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'

  # 前端服务
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    container_name: stockschool_frontend
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://api:8000
    depends_on:
      - api
    restart: unless-stopped

  # Nginx反向代理
  nginx:
    image: nginx:alpine
    container_name: stockschool_nginx
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - api
      - frontend
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

#### Dockerfile 配置

```dockerfile
# Dockerfile.api
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
COPY run.py .

# 创建非root用户
RUN useradd -m -u 1000 stockschool && chown -R stockschool:stockschool /app
USER stockschool

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 环境配置管理

### 环境变量配置
```bash
# .env.prod
# 数据库配置
POSTGRES_DB=stockschool
POSTGRES_USER=stockschool
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgresql://stockschool:your_secure_password@postgres:5432/stockschool

# Redis配置
REDIS_PASSWORD=your_redis_password
REDIS_URL=redis://:your_redis_password@redis:6379/0

# API配置
TUSHARE_TOKEN=your_tushare_token
SECRET_KEY=your_secret_key
ENVIRONMENT=production

# 监控配置
ALERT_WEBHOOK_URL=https://your-webhook-url
LOG_LEVEL=INFO

# SSL配置
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem
```

### 配置文件管理
```yaml
# config/production.yml
database:
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600

api:
  workers: 4
  max_requests: 1000
  max_requests_jitter: 100
  timeout: 30

cache:
  default_timeout: 300
  max_entries: 10000

logging:
  level: INFO
  format: json
  handlers:
    - console
    - file
    - syslog

monitoring:
  metrics_enabled: true
  tracing_enabled: true
  health_check_interval: 30
```

## 部署流程

### 自动化部署脚本
```bash
#!/bin/bash
# deploy.sh

set -e

echo "开始部署StockSchool系统..."

# 1. 环境检查
echo "检查部署环境..."
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "错误: Docker Compose未安装"
    exit 1
fi

# 2. 拉取最新代码
echo "拉取最新代码..."
git pull origin main

# 3. 构建镜像
echo "构建Docker镜像..."
docker-compose -f docker-compose.prod.yml build --no-cache

# 4. 数据库迁移
echo "执行数据库迁移..."
docker-compose -f docker-compose.prod.yml run --rm api python -m alembic upgrade head

# 5. 启动服务
echo "启动服务..."
docker-compose -f docker-compose.prod.yml up -d

# 6. 健康检查
echo "执行健康检查..."
sleep 30
if curl -f http://localhost/health; then
    echo "✅ 部署成功"
else
    echo "❌ 部署失败，请检查日志"
    docker-compose -f docker-compose.prod.yml logs
    exit 1
fi

echo "部署完成！"
```

### 滚动更新脚本
```bash
#!/bin/bash
# rolling_update.sh

set -e

echo "开始滚动更新..."

# 1. 构建新镜像
docker-compose -f docker-compose.prod.yml build api

# 2. 逐个更新API服务实例
for i in {1..2}; do
    echo "更新API服务实例 $i..."
    
    # 停止一个实例
    docker-compose -f docker-compose.prod.yml stop api_$i
    
    # 启动新实例
    docker-compose -f docker-compose.prod.yml up -d api_$i
    
    # 健康检查
    sleep 10
    if ! curl -f http://localhost/health; then
        echo "实例 $i 健康检查失败，回滚..."
        docker-compose -f docker-compose.prod.yml restart api_$i
        exit 1
    fi
    
    echo "实例 $i 更新成功"
done

echo "滚动更新完成！"
```

## 监控和日志

### 日志配置
```python
# src/utils/logging_config.py
import logging
import logging.handlers
import json
from datetime import datetime

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

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # 文件处理器
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/app.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    
    # Syslog处理器（生产环境）
    if os.getenv('ENVIRONMENT') == 'production':
        syslog_handler = logging.handlers.SysLogHandler(
            address=('localhost', 514)
        )
        syslog_handler.setFormatter(JSONFormatter())
        logger.addHandler(syslog_handler)
```

### 监控配置
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'stockschool-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'stockschool-postgres'
    static_configs:
      - targets: ['postgres:5432']
    
  - job_name: 'stockschool-redis'
    static_configs:
      - targets: ['redis:6379']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

## 备份和恢复

### 数据库备份脚本
```bash
#!/bin/bash
# backup.sh

set -e

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="stockschool"

echo "开始数据库备份..."

# 创建备份目录
mkdir -p $BACKUP_DIR

# 全量备份
docker exec stockschool_postgres pg_dump -U stockschool -d $DB_NAME | gzip > $BACKUP_DIR/full_backup_$DATE.sql.gz

# 增量备份（WAL文件）
docker exec stockschool_postgres pg_basebackup -U stockschool -D $BACKUP_DIR/incremental_$DATE -Ft -z -P

# 清理旧备份（保留7天）
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete
find $BACKUP_DIR -name "incremental_*" -mtime +7 -exec rm -rf {} \;

echo "备份完成: $BACKUP_DIR/full_backup_$DATE.sql.gz"
```

### 数据恢复脚本
```bash
#!/bin/bash
# restore.sh

set -e

if [ $# -ne 1 ]; then
    echo "用法: $0 <backup_file>"
    exit 1
fi

BACKUP_FILE=$1

echo "开始数据库恢复..."

# 停止应用服务
docker-compose -f docker-compose.prod.yml stop api compute frontend

# 恢复数据库
if [[ $BACKUP_FILE == *.gz ]]; then
    gunzip -c $BACKUP_FILE | docker exec -i stockschool_postgres psql -U stockschool -d stockschool
else
    docker exec -i stockschool_postgres psql -U stockschool -d stockschool < $BACKUP_FILE
fi

# 重启服务
docker-compose -f docker-compose.prod.yml start api compute frontend

echo "数据库恢复完成"
```

## 性能优化

### 数据库优化
```sql
-- postgresql.conf 优化配置
shared_buffers = 1GB                    # 25% of RAM
effective_cache_size = 3GB               # 75% of RAM
work_mem = 64MB                          # For complex queries
maintenance_work_mem = 256MB             # For maintenance operations
checkpoint_completion_target = 0.9       # Spread checkpoints
wal_buffers = 16MB                       # WAL buffer size
default_statistics_target = 100          # Statistics target

-- TimescaleDB优化
SELECT set_chunk_time_interval('stock_daily', INTERVAL '1 month');
SELECT add_compression_policy('stock_daily', INTERVAL '3 months');
SELECT add_retention_policy('stock_daily', INTERVAL '5 years');
```

### 应用优化
```python
# 连接池配置
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Redis连接池
import redis
from redis.connection import ConnectionPool

redis_pool = ConnectionPool(
    host='redis',
    port=6379,
    password=REDIS_PASSWORD,
    max_connections=50,
    retry_on_timeout=True
)

redis_client = redis.Redis(connection_pool=redis_pool)
```

## 安全配置

### SSL/TLS配置
```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;
    
    # 其他安全头
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    location / {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 防火墙配置
```bash
# ufw防火墙配置
ufw default deny incoming
ufw default allow outgoing

# 允许SSH
ufw allow ssh

# 允许HTTP/HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# 允许内部服务通信
ufw allow from 172.20.0.0/16 to any port 5432  # PostgreSQL
ufw allow from 172.20.0.0/16 to any port 6379  # Redis

ufw enable
```

## 故障排查

### 常见问题诊断
```bash
# 检查服务状态
docker-compose -f docker-compose.prod.yml ps

# 查看服务日志
docker-compose -f docker-compose.prod.yml logs -f api

# 检查数据库连接
docker exec stockschool_postgres psql -U stockschool -d stockschool -c "SELECT version();"

# 检查Redis连接
docker exec stockschool_redis redis-cli -a $REDIS_PASSWORD ping

# 检查磁盘空间
df -h

# 检查内存使用
free -h

# 检查网络连接
netstat -tlnp | grep :8000
```

### 性能诊断
```sql
-- 查看慢查询
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- 查看数据库连接
SELECT count(*) as connections,
       state,
       application_name
FROM pg_stat_activity
GROUP BY state, application_name;

-- 查看表大小
SELECT schemaname,tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;
```