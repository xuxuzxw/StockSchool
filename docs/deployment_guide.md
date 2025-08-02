# StockSchool 部署指南

## 目录
1. [部署环境准备](#部署环境准备)
2. [本地开发部署](#本地开发部署)
3. [生产环境部署](#生产环境部署)
4. [Docker部署](#docker部署)
5. [Kubernetes部署](#kubernetes部署)
6. [负载均衡配置](#负载均衡配置)
7. [监控告警配置](#监控告警配置)
8. [备份恢复策略](#备份恢复策略)
9. [安全配置](#安全配置)
10. [性能调优](#性能调优)

## 部署环境准备

### 硬件要求

#### 开发环境
- **CPU**: 4核以上
- **内存**: 16GB RAM
- **存储**: 50GB SSD
- **GPU**: 可选（CUDA兼容显卡）

#### 生产环境
- **CPU**: 8核以上
- **内存**: 32GB RAM（推荐64GB+）
- **存储**: 1TB SSD（推荐NVMe）
- **GPU**: 1-2块高性能显卡（如RTX 3080/4090）

### 软件依赖

#### 操作系统
- **Linux**: Ubuntu 20.04/22.04 LTS, CentOS 8+
- **Windows**: Windows Server 2019/2022
- **macOS**: macOS 12+（仅开发环境）

#### 核心组件
```bash
# Python 3.8+
python --version

# 数据库
# PostgreSQL 13+ 或 MySQL 8.0+
psql --version

# Redis 6.0+（用于Celery）
redis-server --version

# Node.js 16+（用于前端）
node --version

# Docker 20+（可选）
docker --version
```

### 环境变量配置

创建 `.env` 文件：
```bash
# 基础配置
ENV=production
DEBUG=False
SECRET_KEY=your_secret_key_here
ALLOWED_HOSTS=localhost,127.0.0.1,your_domain.com

# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/stockschool
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis配置（Celery）
REDIS_URL=redis://localhost:6379/0

# Tushare配置
TUSHARE_TOKEN=your_tushare_token_here

# API配置
API_KEY=your_api_key_here
API_RATE_LIMIT=1000/minute

# GPU配置
CUDA_VISIBLE_DEVICES=0,1
TORCH_CUDA_ARCH_LIST=7.5,8.0,8.6

# 日志配置
LOG_LEVEL=INFO
LOG_FILE_MAX_SIZE=100MB
LOG_FILE_BACKUP_COUNT=10

# 缓存配置
CACHE_BACKEND=redis
CACHE_LOCATION=redis://localhost:6379/1
CACHE_TIMEOUT=3600

# 邮件配置（告警）
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your_email@gmail.com
EMAIL_HOST_PASSWORD=your_app_password
EMAIL_USE_TLS=True

# Slack/Webhook配置
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

## 本地开发部署

### 1. 克隆项目
```bash
git clone https://github.com/yourusername/stockschool.git
cd stockschool
```

### 2. 创建虚拟环境
```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. 安装依赖
```bash
# 安装核心依赖
pip install -r requirements.txt

# 安装开发依赖（可选）
pip install -r requirements-dev.txt

# 安装GPU支持（如果需要）
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### 4. 数据库初始化
```bash
# 创建数据库
createdb stockschool

# 初始化数据库结构
python run.py --init-db

# 或手动执行SQL脚本
psql -U username -d stockschool -f database_schema.sql
```

### 5. 配置文件设置
```bash
# 复制配置模板
cp config.yaml.example config.yaml

# 编辑配置文件
vim config.yaml
```

### 6. 启动开发服务器
```bash
# 启动API服务
python run.py
# 选择 "1. 启动API服务器"

# 或直接启动
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 启动Celery Worker（后台任务）
celery -A src.compute.tasks worker --loglevel=info

# 启动Celery Beat（定时任务）
celery -A src.compute.tasks beat --loglevel=info
```

### 7. 验证部署
```bash
# 检查健康状态
curl http://localhost:8000/health

# 运行测试
python -m pytest tests/ -v

# 检查API文档
# 访问 http://localhost:8000/docs
```

## 生产环境部署

### 1. 系统配置

#### 用户和权限
```bash
# 创建专用用户
sudo useradd -r -s /bin/false stockschool
sudo mkdir -p /opt/stockschool
sudo chown stockschool:stockschool /opt/stockschool

# 设置目录权限
sudo chmod 755 /opt/stockschool
```

#### 系统优化
```bash
# 调整文件描述符限制
echo "stockschool soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "stockschool hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# 调整内核参数
echo "vm.swappiness=1" | sudo tee -a /etc/sysctl.conf
echo "net.core.somaxconn=65535" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### 2. 数据库部署

#### PostgreSQL配置
```bash
# 安装PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# 创建数据库和用户
sudo -u postgres psql
CREATE DATABASE stockschool;
CREATE USER stockschool_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE stockschool TO stockschool_user;
\q

# 优化PostgreSQL配置
sudo vim /etc/postgresql/13/main/postgresql.conf
```

关键配置项：
```conf
# 内存配置
shared_buffers = 2GB
effective_cache_size = 6GB
work_mem = 16MB
maintenance_work_mem = 512MB

# 连接配置
max_connections = 200
superuser_reserved_connections = 3

# WAL配置
wal_buffers = 16MB
checkpoint_completion_target = 0.9
wal_writer_delay = 10ms

# 查询优化
random_page_cost = 1.1
effective_io_concurrency = 200
```

### 3. Redis部署
```bash
# 安装Redis
sudo apt install redis-server

# 配置Redis
sudo vim /etc/redis/redis.conf
```

关键配置项：
```conf
# 内存配置
maxmemory 2gb
maxmemory-policy allkeys-lru

# 持久化
save 900 1
save 300 10
save 60 10000

# 安全
requirepass your_redis_password
bind 127.0.0.1 ::1

# 性能
tcp-keepalive 300
timeout 0
```

### 4. 应用部署

#### Gunicorn配置
创建 `gunicorn.conf.py`：
```python
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 300
keepalive = 2
preload_app = True
user = "stockschool"
group = "stockschool"
tmp_upload_dir = None
errorlog = "/var/log/stockschool/gunicorn_error.log"
accesslog = "/var/log/stockschool/gunicorn_access.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
```

#### Systemd服务配置
创建 `/etc/systemd/system/stockschool.service`：
```ini
[Unit]
Description=StockSchool API Service
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=stockschool
Group=stockschool
WorkingDirectory=/opt/stockschool
EnvironmentFile=/opt/stockschool/.env
ExecStart=/opt/stockschool/venv/bin/gunicorn -c gunicorn.conf.py src.api.main:app
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

创建Celery Worker服务 `/etc/systemd/system/stockschool-celery.service`：
```ini
[Unit]
Description=StockSchool Celery Worker
After=network.target redis.service

[Service]
Type=simple
User=stockschool
Group=stockschool
WorkingDirectory=/opt/stockschool
EnvironmentFile=/opt/stockschool/.env
ExecStart=/opt/stockschool/venv/bin/celery -A src.compute.tasks worker --loglevel=info --hostname=worker1@%h
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

创建Celery Beat服务 `/etc/systemd/system/stockschool-celery-beat.service`：
```ini
[Unit]
Description=StockSchool Celery Beat
After=network.target redis.service

[Service]
Type=simple
User=stockschool
Group=stockschool
WorkingDirectory=/opt/stockschool
EnvironmentFile=/opt/stockschool/.env
ExecStart=/opt/stockschool/venv/bin/celery -A src.compute.tasks beat --loglevel=info --schedule-file=/var/run/stockschool/celerybeat-schedule
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### 启动服务
```bash
# 重新加载systemd配置
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start stockschool
sudo systemctl start stockschool-celery
sudo systemctl start stockschool-celery-beat

# 设置开机自启
sudo systemctl enable stockschool
sudo systemctl enable stockschool-celery
sudo systemctl enable stockschool-celery-beat

# 检查服务状态
sudo systemctl status stockschool
sudo systemctl status stockschool-celery
sudo systemctl status stockschool-celery-beat
```

### 5. Nginx反向代理

安装和配置Nginx：
```bash
# 安装Nginx
sudo apt install nginx

# 创建配置文件 /etc/nginx/sites-available/stockschool
sudo vim /etc/nginx/sites-available/stockschool
```

```nginx
upstream stockschool_api {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your_domain.com;
    
    # SSL配置（推荐）
    # listen 443 ssl;
    # ssl_certificate /path/to/your/certificate.crt;
    # ssl_certificate_key /path/to/your/private.key;
    
    # 安全头
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # 日志
    access_log /var/log/nginx/stockschool_access.log;
    error_log /var/log/nginx/stockschool_error.log;
    
    # API路由
    location /api/ {
        proxy_pass http://stockschool_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # 健康检查
    location /health {
        proxy_pass http://stockschool_api;
        proxy_set_header Host $host;
    }
    
    # 静态文件（如果有前端）
    location /static/ {
        alias /opt/stockschool/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # 限制请求频率
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/s;
    limit_req zone=api burst=200 nodelay;
}
```

启用配置：
```bash
# 创建软链接
sudo ln -s /etc/nginx/sites-available/stockschool /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重启Nginx
sudo systemctl restart nginx
```

## Docker部署

### 1. Docker Compose配置

创建 `docker-compose.yml`：
```yaml
version: '3.8'

services:
  # 数据库
  db:
    image: postgres:13
    container_name: stockschool_db
    environment:
      POSTGRES_DB: stockschool
      POSTGRES_USER: stockschool_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database_schema.sql:/docker-entrypoint-initdb.d/01-schema.sql
    ports:
      - "5432:5432"
    networks:
      - stockschool_network
    restart: unless-stopped

  # Redis
  redis:
    image: redis:6-alpine
    container_name: stockschool_redis
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - stockschool_network
    restart: unless-stopped

  # API服务
  api:
    build: .
    container_name: stockschool_api
    depends_on:
      - db
      - redis
    environment:
      - DATABASE_URL=postgresql://stockschool_user:${DB_PASSWORD}@db:5432/stockschool
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - TUSHARE_TOKEN=${TUSHARE_TOKEN}
      - API_KEY=${API_KEY}
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
      - ./cache:/app/cache
    networks:
      - stockschool_network
    restart: unless-stopped

  # Celery Worker
  worker:
    build: .
    container_name: stockschool_worker
    depends_on:
      - db
      - redis
    environment:
      - DATABASE_URL=postgresql://stockschool_user:${DB_PASSWORD}@db:5432/stockschool
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - TUSHARE_TOKEN=${TUSHARE_TOKEN}
    command: celery -A src.compute.tasks worker --loglevel=info
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
      - ./cache:/app/cache
    networks:
      - stockschool_network
    restart: unless-stopped

  # Celery Beat
  beat:
    build: .
    container_name: stockschool_beat
    depends_on:
      - db
      - redis
    environment:
      - DATABASE_URL=postgresql://stockschool_user:${DB_PASSWORD}@db:5432/stockschool
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
    command: celery -A src.compute.tasks beat --loglevel=info
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    networks:
      - stockschool_network
    restart: unless-stopped

  # Nginx
  nginx:
    image: nginx:alpine
    container_name: stockschool_nginx
    depends_on:
      - api
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    networks:
      - stockschool_network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:

networks:
  stockschool_network:
    driver: bridge
```

### 2. Dockerfile

创建 `Dockerfile`：
```dockerfile
# 使用官方Python运行时作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装GPU支持（如果需要）
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p logs models cache data

# 创建非root用户
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["gunicorn", "-c", "gunicorn.conf.py", "src.api.main:app"]
```

### 3. 环境变量文件

创建 `.env` 文件：
```bash
# .env
DB_PASSWORD=your_secure_db_password
REDIS_PASSWORD=your_secure_redis_password
TUSHARE_TOKEN=your_tushare_token
API_KEY=your_api_key
```

### 4. 启动Docker环境

```bash
# 构建镜像
docker-compose build

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 初始化数据库
docker-compose exec api python run.py --init-db

# 停止服务
docker-compose down
```

## Kubernetes部署

### 1. 命名空间配置

创建 `namespace.yaml`：
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: stockschool
```

### 2. 配置管理

创建 `configmap.yaml`：
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: stockschool-config
  namespace: stockschool
data:
  database-url: "postgresql://stockschool_user:$(DB_PASSWORD)@stockschool-db:5432/stockschool"
  redis-url: "redis://:$(REDIS_PASSWORD)@stockschool-redis:6379/0"
  log-level: "INFO"
  api-host: "0.0.0.0"
  api-port: "8000"
```

### 3. 密钥管理

创建 `secrets.yaml`：
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: stockschool-secrets
  namespace: stockschool
type: Opaque
data:
  db-password: base64_encoded_password
  redis-password: base64_encoded_password
  tushare-token: base64_encoded_token
  api-key: base64_encoded_api_key
```

### 4. 数据库部署

创建 `postgres-deployment.yaml`：
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stockschool-db
  namespace: stockschool
spec:
  replicas: 1
  selector:
    matchLabels:
      app: stockschool-db
  template:
    metadata:
      labels:
        app: stockschool-db
    spec:
      containers:
      - name: postgres
        image: postgres:13
        env:
        - name: POSTGRES_DB
          value: "stockschool"
        - name: POSTGRES_USER
          value: "stockschool_user"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: stockschool-secrets
              key: db-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: stockschool-db
  namespace: stockschool
spec:
  selector:
    app: stockschool-db
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

### 5. Redis部署

创建 `redis-deployment.yaml`：
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stockschool-redis
  namespace: stockschool
spec:
  replicas: 1
  selector:
    matchLabels:
      app: stockschool-redis
  template:
    metadata:
      labels:
        app: stockschool-redis
    spec:
      containers:
      - name: redis
        image: redis:6-alpine
        command: ["redis-server", "--requirepass", "$(REDIS_PASSWORD)"]
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: stockschool-secrets
              key: redis-password
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-storage
          mountPath: /data
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: stockschool-redis
  namespace: stockschool
spec:
  selector:
    app: stockschool-redis
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP
```

### 6. API服务部署

创建 `api-deployment.yaml`：
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stockschool-api
  namespace: stockschool
spec:
  replicas: 3
  selector:
    matchLabels:
      app: stockschool-api
  template:
    metadata:
      labels:
        app: stockschool-api
    spec:
      containers:
      - name: api
        image: your_registry/stockschool:latest
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: stockschool-config
              key: database-url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: stockschool-config
              key: redis-url
        - name: TUSHARE_TOKEN
          valueFrom:
            secretKeyRef:
              name: stockschool-secrets
              key: tushare-token
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: stockschool-secrets
              key: api-key
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: stockschool-config
              key: log-level
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: models
          mountPath: /app/models
        - name: cache
          mountPath: /app/cache
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
      volumes:
      - name: logs
        persistentVolumeClaim:
          claimName: logs-pvc
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: cache
        persistentVolumeClaim:
          claimName: cache-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: stockschool-api
  namespace: stockschool
spec:
  selector:
    app: stockschool-api
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: stockschool-ingress
  namespace: stockschool
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: api.stockschool.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: stockschool-api
            port:
              number: 8000
```

### 7. Celery Worker部署

创建 `worker-deployment.yaml`：
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stockschool-worker
  namespace: stockschool
spec:
  replicas: 2
  selector:
    matchLabels:
      app: stockschool-worker
  template:
    metadata:
      labels:
        app: stockschool-worker
    spec:
      containers:
      - name: worker
        image: your_registry/stockschool:latest
        command: ["celery", "-A", "src.compute.tasks", "worker", "--loglevel=info"]
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: stockschool-config
              key: database-url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: stockschool-config
              key: redis-url
        - name: TUSHARE_TOKEN
          valueFrom:
            secretKeyRef:
              name: stockschool-secrets
              key: tushare-token
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: models
          mountPath: /app/models
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
      volumes:
      - name: logs
        persistentVolumeClaim:
          claimName: logs-pvc
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

### 8. 持久化存储

创建 `pvcs.yaml`：
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: stockschool
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: stockschool
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: logs-pvc
  namespace: stockschool
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
  namespace: stockschool
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: cache-pvc
  namespace: stockschool
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
```

### 9. 部署到Kubernetes

```bash
# 创建命名空间
kubectl apply -f namespace.yaml

# 创建配置和密钥
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml

# 创建持久化存储
kubectl apply -f pvcs.yaml

# 部署数据库和Redis
kubectl apply -f postgres-deployment.yaml
kubectl apply -f redis-deployment.yaml

# 部署应用服务
kubectl apply -f api-deployment.yaml
kubectl apply -f worker-deployment.yaml

# 检查部署状态
kubectl get pods -n stockschool
kubectl get services -n stockschool
kubectl get ingress -n stockschool

# 查看日志
kubectl logs -f deployment/stockschool-api -n stockschool
kubectl logs -f deployment/stockschool-worker -n stockschool
```

## 负载均衡配置

### 1. Nginx负载均衡

创建 `nginx-lb.conf`：
```nginx
upstream stockschool_api {
    least_conn;
    server stockschool-api-1:8000 weight=3 max_fails=3 fail_timeout=30s;
    server stockschool-api-2:8000 weight=3 max_fails=3 fail_timeout=30s;
    server stockschool-api-3:8000 weight=2 max_fails=3 fail_timeout=30s;
}

upstream stockschool_worker {
    ip_hash;
    server stockschool-worker-1:5672;
    server stockschool-worker-2:5672;
}

server {
    listen 80;
    server_name api.stockschool.com;
    
    # SSL配置
    listen 443 ssl http2;
    ssl_certificate /etc/nginx/ssl/stockschool.crt;
    ssl_certificate_key /etc/nginx/ssl/stockschool.key;
    
    # 安全头
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    # 限流
    limit_req_zone $binary_remote_addr zone=api:10m rate=1000r/m;
    limit_req zone=api burst=2000 nodelay;
    
    location / {
        proxy_pass http://stockschool_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超时设置
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # 缓冲设置
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }
    
    # 健康检查
    location /health {
        access_log off;
        proxy_pass http://stockschool_api;
        proxy_set_header Host $host;
    }
}
```

### 2. HAProxy配置

创建 `haproxy.cfg`：
```haproxy
global
    log /dev/log local0
    log /dev/log local1 notice
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin expose-fd listeners
    stats timeout 30s
    user haproxy
    group haproxy
    daemon

    # SSL配置
    ca-base /etc/ssl/certs
    crt-base /etc/ssl/private

defaults
    log global
    mode http
    option httplog
    option dontlognull
    timeout connect 5000
    timeout client 50000
    timeout server 50000
    errorfile 400 /etc/haproxy/errors/400.http
    errorfile 403 /etc/haproxy/errors/403.http
    errorfile 408 /etc/haproxy/errors/408.http
    errorfile 500 /etc/haproxy/errors/500.http
    errorfile 502 /etc/haproxy/errors/502.http
    errorfile 503 /etc/haproxy/errors/503.http
    errorfile 504 /etc/haproxy/errors/504.http

frontend stockschool_frontend
    bind *:80
    bind *:443 ssl crt /etc/haproxy/certs/stockschool.pem
    mode http
    option httplog
    option forwardfor
    default_backend stockschool_backend

    # 健康检查
    monitor-uri /health

    # 限流
    stick-table type ip size 1m expire 5m store gpc0,http_req_rate(10s),http_err_rate(10s)
    tcp-request content track-sc0 src
    tcp-request content reject if { sc0_get_gpc0 gt 0 }

backend stockschool_backend
    mode http
    balance leastconn
    option httpchk GET /health
    http-check expect status 200
    default-server inter 3000 rise 2 fall 3
    
    server api1 stockschool-api-1:8000 check weight 3
    server api2 stockschool-api-2:8000 check weight 3
    server api3 stockschool-api-3:8000 check weight 2

    # 慢启动
    server api4 stockschool-api-4:8000 check weight 1 slowstart 60000

# 统计页面
listen stats
    bind :9000
    mode http
    stats enable
    stats hide-version
    stats realm Haproxy\ Statistics
    stats uri /stats
    stats auth admin:your_stats_password
```

## 监控告警配置

### 1. Prometheus配置

创建 `prometheus.yml`：
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

scrape_configs:
  - job_name: 'stockschool-api'
    static_configs:
      - targets: ['stockschool-api:8000']
    metrics_path: '/metrics'
    
  - job_name: 'stockschool-worker'
    static_configs:
      - targets: ['stockschool-worker:5555']
      
  - job_name: 'postgresql'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
      
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

### 2. Grafana仪表板

创建StockSchool监控仪表板：
```json
{
  "dashboard": {
    "id": null,
    "title": "StockSchool监控面板",
    "tags": ["stockschool", "quant"],
    "timezone": "browser",
    "schemaVersion": 16,
    "version": 0,
    "panels": [
      {
        "id": 1,
        "type": "graph",
        "title": "API请求率",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{handler}}"
          }
        ]
      },
      {
        "id": 2,
        "type": "singlestat",
        "title": "系统健康状态",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "up{job='stockschool-api'}",
            "format": "time_series"
          }
        ],
        "valueMaps": [
          {
            "value": "1",
            "text": "健康"
          },
          {
            "value": "0",
            "text": "异常"
          }
        ]
      },
      {
        "id": 3,
        "type": "graph",
        "title": "数据库连接数",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "pg_stat_database_numbackends",
            "legendFormat": "连接数"
          }
        ]
      },
      {
        "id": 4,
        "type": "graph",
        "title": "Redis内存使用",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "redis_memory_used_bytes",
            "legendFormat": "已使用内存"
          }
        ]
      },
      {
        "id": 5,
        "type": "graph",
        "title": "Celery任务队列",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "celery_tasks_total",
            "legendFormat": "总任务数"
          }
        ]
      }
    ]
  }
}
```

### 3. 告警规则

创建 `alert_rules.yml`：
```yaml
groups:
- name: stockschool-alerts
  rules:
  - alert: HighCPUUsage
    expr: rate(node_cpu_seconds_total{mode!="idle"}[5m]) > 0.8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "CPU使用率过高"
      description: "CPU使用率超过80%持续5分钟"

  - alert: HighMemoryUsage
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "内存使用率过高"
      description: "内存使用率超过85%持续5分钟"

  - alert: DatabaseDown
    expr: up{job="postgresql"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "数据库服务不可用"
      description: "PostgreSQL数据库服务宕机"

  - alert: APIDown
    expr: up{job="stockschool-api"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "API服务不可用"
      description: "StockSchool API服务宕机"

  - alert: HighRequestLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "API响应时间过长"
      description: "API 95%请求响应时间超过5秒"

  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "错误率过高"
      description: "API错误率超过5%持续5分钟"
```

## 备份恢复策略

### 1. 数据库备份

#### PostgreSQL备份脚本
创建 `backup_postgres.sh`：
```bash
#!/bin/bash

# 配置
BACKUP_DIR="/opt/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="stockschool_${DATE}.sql"

# 创建备份目录
mkdir -p ${BACKUP_DIR}/${DATE}

# 执行备份
pg_dump -U stockschool_user -h localhost -d stockschool > ${BACKUP_DIR}/${DATE}/${BACKUP_NAME}

# 压缩备份文件
gzip ${BACKUP_DIR}/${DATE}/${BACKUP_NAME}

# 删除7天前的备份
find ${BACKUP_DIR} -type d -mtime +7 -exec rm -rf {} \;

# 记录日志
echo "$(date): PostgreSQL备份完成 - ${BACKUP_NAME}.gz" >> /var/log/stockschool/backup.log
```

#### 定时备份任务
```bash
# 添加到crontab
0 2 * * * /opt/stockschool/scripts/backup_postgres.sh
```

### 2. 模型备份

#### 模型备份脚本
创建 `backup_models.sh`：
```bash
#!/bin/bash

BACKUP_DIR="/opt/backups/models"
DATE=$(date +%Y%m%d)
MODEL_DIR="/opt/stockschool/models"

# 创建备份目录
mkdir -p ${BACKUP_DIR}/${DATE}

# 备份模型文件
rsync -av --delete ${MODEL_DIR}/ ${BACKUP_DIR}/${DATE}/

# 创建备份元数据
cat > ${BACKUP_DIR}/${DATE}/metadata.json << EOF
{
  "backup_date": "$(date -Iseconds)",
  "model_count": $(ls ${MODEL_DIR}/*.pkl | wc -l),
  "total_size": "$(du -sh ${MODEL_DIR} | cut -f1)",
  "backup_path": "${BACKUP_DIR}/${DATE}"
}
EOF

# 删除30天前的备份
find ${BACKUP_DIR} -type d -mtime +30 -exec rm -rf {} \;

echo "$(date): 模型备份完成" >> /var/log/stockschool/backup.log
```

### 3. 恢复流程

#### 数据库恢复
```bash
# 停止应用服务
sudo systemctl stop stockschool
sudo systemctl stop stockschool-celery

# 恢复数据库
gunzip < /opt/backups/postgres/20250731_020000/stockschool_20250731_020000.sql.gz | psql -U stockschool_user -d stockschool

# 启动服务
sudo systemctl start stockschool-celery
sudo systemctl start stockschool
```

#### 模型恢复
```bash
# 停止相关服务
sudo systemctl stop stockschool-worker

# 恢复模型
rsync -av /opt/backups/models/20250731/ /opt/stockschool/models/

# 启动服务
sudo systemctl start stockschool-worker
```

## 安全配置

### 1. 网络安全

#### 防火墙配置
```bash
# UFW配置
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 允许必要端口
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 5432/tcp from 192.168.1.0/24  # 仅允许内网访问数据库
sudo ufw allow 6379/tcp from 192.168.1.0/24  # 仅允许内网访问Redis

# 限制SSH访问
sudo ufw limit ssh
```

#### Fail2ban配置
```bash
# 安装Fail2ban
sudo apt install fail2ban

# 创建配置文件 /etc/fail2ban/jail.local
sudo vim /etc/fail2ban/jail.local
```

```ini
[DEFAULT]
bantime = 1h
findtime = 10m
maxretry = 5

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 1d

[nginx-http-auth]
enabled = true
port = http,https
filter = nginx-http-auth
logpath = /var/log/nginx/error.log
maxretry = 3
bantime = 1d

[nginx-botsearch]
enabled = true
port = http,https
filter = nginx-botsearch
logpath = /var/log/nginx/error.log
maxretry = 2
```

### 2. 应用安全

#### SSL/TLS配置
```nginx
# Nginx SSL配置
server {
    listen 443 ssl http2;
    server_name api.stockschool.com;
    
    # SSL证书
    ssl_certificate /etc/ssl/certs/stockschool.crt;
    ssl_certificate_key /etc/ssl/private/stockschool.key;
    
    # SSL安全配置
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # 其他安全头
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
}
```

#### API安全配置
```python
# FastAPI安全配置
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
import hashlib
import hmac

app = FastAPI()

# API密钥验证
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key_header: str = Depends(API_KEY_HEADER)):
    """验证API密钥"""
    if not api_key_header or not verify_key(api_key_header):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key_header

def verify_key(api_key: str) -> bool:
    """验证API密钥"""
    # 从配置中获取合法密钥列表
    valid_keys = get_valid_api_keys()
    return api_key in valid_keys

# 限流配置
limiter = Limiter(key_func=get_remote_address)

@app.get("/api/v1/stocks/basic", dependencies=[Depends(verify_api_key)])
@limiter.limit("1000/minute")
async def get_stock_basic(request: Request):
    # 业务逻辑
    pass

# CORS配置
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count"]
)
```

## 性能调优

### 1. 数据库优化

#### PostgreSQL优化配置
```conf
# postgresql.conf
# 内存配置
shared_buffers = 2GB                    # 物理内存的25%
effective_cache_size = 6GB             # 物理内存的75%
work_mem = 16MB                        # 每个查询操作的内存
maintenance_work_mem = 512MB           # 维护操作内存

# 连接配置
max_connections = 200
superuser_reserved_connections = 3

# WAL配置
wal_buffers = 16MB
checkpoint_completion_target = 0.9
wal_writer_delay = 10ms
max_wal_size = 4GB
min_wal_size = 1GB

# 查询优化
random_page_cost = 1.1                 # SSD设置为1.1
effective_io_concurrency = 200         # SSD设置为200
```

#### 数据库索引优化
```sql
-- 股票日线数据索引
CREATE INDEX idx_stock_daily_ts_code_date ON stock_daily(ts_code, trade_date);
CREATE INDEX idx_stock_daily_date ON stock_daily(trade_date);

-- 因子数据索引
CREATE INDEX idx_factor_data_ts_code_date ON factor_data(ts_code, trade_date);
CREATE INDEX idx_factor_data_factor_name ON factor_data(factor_name);

-- 预测结果索引
CREATE INDEX idx_prediction_results_ts_code_date ON prediction_results(ts_code, prediction_date);
```

### 2. 应用性能优化

#### Gunicorn优化配置
```python
# gunicorn.conf.py
import multiprocessing

# 工作进程数 = (CPU核心数 * 2) + 1
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 2000

# 资源限制
max_requests = 1000
max_requests_jitter = 100
timeout = 300
keepalive = 2

# 性能优化
preload_app = True
worker_tmp_dir = "/dev/shm"  # 使用内存tmpfs

# 日志配置
accesslog = "/var/log/stockschool/gunicorn_access.log"
errorlog = "/var/log/stockschool/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s %(p)s'
```

#### Redis优化配置
```conf
# redis.conf
# 内存配置
maxmemory 4gb
maxmemory-policy allkeys-lru

# 持久化配置
save 900 1
save 300 10
save 60 10000

# 网络配置
tcp-keepalive 300
timeout 0
tcp-backlog 511

# 性能配置
lazyfree-lazy-eviction yes
lazyfree-lazy-expire yes
lazyfree-lazy-server-del yes
replica-lazy-flush yes
```

### 3. GPU性能优化

#### CUDA优化配置
```bash
# 环境变量优化
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_CUDA_ARCH_LIST="7.5,8.0,8.6"
export CUDA_LAUNCH_BLOCKING=0

# 内存分配优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

#### PyTorch优化
```python
import torch

# GPU优化设置
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# 混合精度训练（如果支持）
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = loss_fn(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 维护操作

### 1. 日常维护脚本

创建 `daily_maintenance.sh`：
```bash
#!/bin/bash

# 日志轮转
logrotate /etc/logrotate.d/stockschool

# 清理临时文件
find /tmp -name "stockschool_*" -mtime +1 -delete

# 检查磁盘空间
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 85 ]; then
    echo "警告: 磁盘使用率超过85%"
    # 发送告警
fi

# 检查服务状态
systemctl is-active stockschool || systemctl start stockschool
systemctl is-active stockschool-celery || systemctl start stockschool-celery

# 记录维护日志
echo "$(date): 日常维护完成" >> /var/log/stockschool/maintenance.log
```

### 2. 监控脚本

创建 `monitor_services.sh`：
```bash
#!/bin/bash

SERVICES=("stockschool" "stockschool-celery" "stockschool-celery-beat")

for service in "${SERVICES[@]}"; do
    if ! systemctl is-active --quiet $service; then
        echo "服务 $service 未运行，正在启动..."
        systemctl start $service
        # 发送告警通知
        send_alert "服务恢复" "服务 $service 已重新启动"
    fi
done
```

### 3. 性能监控脚本

创建 `performance_monitor.sh`：
```bash
#!/bin/bash

# CPU使用率
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)

# 内存使用率
MEM_USAGE=$(free | grep Mem | awk '{printf("%.2f"), $3/$2 * 100.0}')

# 磁盘使用率
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')

# 记录到日志
echo "$(date): CPU=${CPU_USAGE}%, MEM=${MEM_USAGE}%, DISK=${DISK_USAGE}%" >> /var/log/stockschool/performance.log

# 性能告警
if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    send_alert "高CPU使用率" "CPU使用率: ${CPU_USAGE}%"
fi
```

## 故障排除

### 1. 常见问题诊断

#### 服务启动失败
```bash
# 检查服务状态
systemctl status stockschool
systemctl status stockschool-celery

# 查看详细日志
journalctl -u stockschool -f
journalctl -u stockschool-celery -f

# 检查端口占用
netstat -tlnp | grep :8000
```

#### 数据库连接问题
```bash
# 检查数据库服务
systemctl status postgresql

# 测试数据库连接
psql -U stockschool_user -d stockschool -c "SELECT 1;"

# 检查连接池
SELECT count(*) FROM pg_stat_activity;
```

#### Redis连接问题
```bash
# 检查Redis服务
systemctl status redis

# 测试Redis连接
redis-cli -a your_redis_password ping

# 查看Redis统计
redis-cli -a your_redis_password info
```

### 2. 性能问题诊断

#### API响应慢
```bash
# 查看慢查询
tail -f /var/log/stockschool/gunicorn_access.log | grep -E '([5-9][0-9]{2}|[0-9]{4,})'

# 数据库慢查询日志
tail -f /var/log/postgresql/postgresql-13-main.log | grep duration

# 系统资源监控
htop
iotop
```

#### 内存泄漏
```bash
# 监控内存使用
watch -n 1 'ps aux --sort=-%mem | head -20'

# 检查Python内存
python -m pympler.summary

# 重启服务释放内存
systemctl restart stockschool
```

## 升级维护

### 1. 版本升级流程

#### 代码升级
```bash
# 停止服务
sudo systemctl stop stockschool
sudo systemctl stop stockschool-celery

# 备份当前版本
cp -r /opt/stockschool /opt/stockschool_backup_$(date +%Y%m%d)

# 拉取新代码
cd /opt/stockschool
git pull origin main

# 安装新依赖
pip install -r requirements.txt

# 运行数据库迁移
python manage.py migrate

# 重启服务
sudo systemctl start stockschool-celery
sudo systemctl start stockschool

# 验证升级
curl http://localhost:8000/health
```

#### 数据库升级
```bash
# 备份数据库
pg_dump -U stockschool_user -d stockschool > backup_$(date +%Y%m%d).sql

# 运行迁移脚本
python manage.py migrate

# 验证数据完整性
python manage.py check_data_integrity
```

### 2. 回滚流程

#### 代码回滚
```bash
# 停止服务
sudo systemctl stop stockschool
sudo systemctl stop stockschool-celery

# 回滚代码
cd /opt/stockschool
git reset --hard HEAD~1

# 重新安装依赖
pip install -r requirements.txt

# 重启服务
sudo systemctl start stockschool-celery
sudo systemctl start stockschool
```

#### 数据库回滚
```bash
# 停止应用
sudo systemctl stop stockschool

# 恢复数据库备份
psql -U stockschool_user -d stockschool < backup_20250731.sql

# 重启应用
sudo systemctl start stockschool
```

---

## 附录

### 端口使用说明
- **8000**: API服务端口
- **5432**: PostgreSQL数据库
- **6379**: Redis服务
- **5555**: Celery Flower监控
- **9000**: HAProxy统计页面
- **9090**: Prometheus监控
- **3000**: Grafana仪表板

### 目录结构
```
/opt/stockschool/
├── app/                 # 应用代码
├── logs/                # 日志文件
├── models/              # 模型文件
├── cache/               # 缓存文件
├── data/                # 数据文件
├── backups/             # 备份文件
├── scripts/             # 运维脚本
└── venv/                # Python虚拟环境
```

### 联系支持
- **文档**: https://docs.stockschool.com
- **GitHub**: https://github.com/yourusername/stockschool
- **邮箱**: support@stockschool.com
- **社区**: https://community.stockschool.com

---
*部署指南版本: v1.1.7*
*最后更新: 2025-07-31*
