# StockSchool 生产环境部署指南

## 概述

本文档详细描述了StockSchool量化投研系统在生产环境中的部署、配置和运维流程。

## 系统架构

### 服务组件

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Web Gateway   │    │   API Gateway   │
│    (Nginx)      │────│   (Nginx)       │────│   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Grafana       │    │   Monitoring    │    │   Data Sync     │
│   Dashboard     │    │   Service       │    │   Service       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   TimescaleDB   │    │   Redis Cache   │    │   Prometheus    │
│   (Primary)     │────│   (Master)      │────│   (Metrics)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   TimescaleDB   │    │   Redis         │    │   AlertManager  │
│   (Replica)     │    │   Sentinel      │    │   (Alerts)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 技术栈

- **容器化**: Docker + Docker Compose
- **数据库**: PostgreSQL + TimescaleDB
- **缓存**: Redis + Redis Sentinel
- **监控**: Prometheus + Grafana + AlertManager
- **日志**: ELK Stack (Elasticsearch + Logstash + Kibana)
- **负载均衡**: Nginx
- **SSL/TLS**: Let's Encrypt 或自签名证书

## 部署前准备

### 系统要求

- **操作系统**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **CPU**: 最少4核，推荐8核
- **内存**: 最少16GB，推荐32GB
- **存储**: 最少100GB SSD，推荐500GB+
- **网络**: 稳定的互联网连接

### 软件依赖

```bash
# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 验证安装
docker --version
docker-compose --version
```

### 环境变量配置

创建 `.env` 文件：

```bash
# 数据库配置
POSTGRES_DB=stockschool
POSTGRES_USER=stockschool
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_REPLICA_PASSWORD=replica_password_here

# Redis配置
REDIS_PASSWORD=your_redis_password_here

# API配置
TUSHARE_TOKEN=your_tushare_token_here
SECRET_KEY=your_secret_key_here

# 监控配置
GRAFANA_PASSWORD=your_grafana_password_here
GRAFANA_SECRET_KEY=your_grafana_secret_key_here

# 告警配置
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=alerts@example.com
SMTP_PASSWORD=smtp_password_here
DEFAULT_ALERT_EMAIL=admin@example.com
CRITICAL_ALERT_EMAIL=critical@example.com
WARNING_ALERT_EMAIL=warning@example.com
DATASYNC_ALERT_EMAIL=datasync@example.com
DATABASE_ALERT_EMAIL=database@example.com

# Webhook配置
ALERT_WEBHOOK_URL=https://hooks.slack.com/your/webhook/url
CRITICAL_WEBHOOK_URL=https://hooks.slack.com/critical/webhook/url
WEBHOOK_TOKEN=your_webhook_token_here

# 备份配置
BACKUP_WEBHOOK_URL=https://hooks.slack.com/backup/webhook/url
BACKUP_EMAIL=backup@example.com
```

## 部署流程

### 1. 快速部署

```bash
# 克隆项目
git clone https://github.com/your-org/stockschool.git
cd stockschool

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，设置必要的环境变量

# 执行部署
chmod +x scripts/deployment/deploy.sh
./scripts/deployment/deploy.sh deploy
```

### 2. 分步部署

#### 步骤1: 准备环境

```bash
# 创建必要目录
mkdir -p logs backups data/{postgres,redis,grafana,prometheus}

# 生成SSL证书（生产环境请使用正式证书）
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout monitoring/nginx/ssl/key.pem \
    -out monitoring/nginx/ssl/cert.pem \
    -subj "/C=CN/ST=Beijing/L=Beijing/O=StockSchool/CN=your-domain.com"
```

#### 步骤2: 启动基础设施

```bash
# 启动数据库和缓存
docker-compose -f docker-compose.prod.yml up -d postgres postgres_replica redis redis_sentinel

# 等待服务就绪
sleep 30

# 验证数据库连接
docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U $POSTGRES_USER -d $POSTGRES_DB
```

#### 步骤3: 启动应用服务

```bash
# 构建应用镜像
docker-compose -f docker-compose.prod.yml build

# 启动应用服务
docker-compose -f docker-compose.prod.yml up -d api data_sync monitoring
```

#### 步骤4: 启动监控服务

```bash
# 启动监控栈
docker-compose -f docker-compose.prod.yml up -d prometheus grafana alertmanager
docker-compose -f docker-compose.prod.yml up -d node_exporter postgres_exporter redis_exporter

# 启动日志栈
docker-compose -f docker-compose.prod.yml up -d elasticsearch logstash kibana

# 启动负载均衡器
docker-compose -f docker-compose.prod.yml up -d nginx
```

### 3. 验证部署

```bash
# 检查服务状态
docker-compose -f docker-compose.prod.yml ps

# 健康检查
curl -f https://localhost/api/health
curl -f https://localhost/monitoring/health
curl -f https://localhost/grafana/api/health

# 查看日志
docker-compose -f docker-compose.prod.yml logs -f api
```

## 配置管理

### SSL证书配置

#### 使用Let's Encrypt

```bash
# 安装certbot
sudo apt-get install certbot

# 获取证书
sudo certbot certonly --standalone -d your-domain.com

# 复制证书到nginx目录
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem monitoring/nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem monitoring/nginx/ssl/key.pem

# 设置自动续期
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

### 数据库配置优化

```sql
-- postgresql.conf 优化配置
shared_buffers = 2GB                     -- 25% of RAM
effective_cache_size = 6GB               -- 75% of RAM
work_mem = 64MB                          -- For complex queries
maintenance_work_mem = 512MB             -- For maintenance operations
checkpoint_completion_target = 0.9       -- Spread checkpoints
wal_buffers = 16MB                       -- WAL buffer size
default_statistics_target = 100          -- Statistics target
random_page_cost = 1.1                   -- For SSD storage

-- TimescaleDB优化
SELECT set_chunk_time_interval('stock_daily', INTERVAL '1 month');
SELECT add_compression_policy('stock_daily', INTERVAL '3 months');
SELECT add_retention_policy('stock_daily', INTERVAL '5 years');
```

### Redis配置优化

```conf
# redis.conf 关键配置
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
```

## 监控配置

### Prometheus配置

监控指标包括：
- 系统资源使用率（CPU、内存、磁盘、网络）
- 应用性能指标（API响应时间、错误率、吞吐量）
- 数据库性能指标（连接数、查询时间、锁等待）
- 业务指标（数据同步状态、因子计算进度）

### Grafana仪表板

预配置的仪表板：
- **系统概览**: 整体系统状态和关键指标
- **API性能**: API请求量、响应时间、错误率
- **数据库监控**: 数据库连接、查询性能、存储使用
- **数据同步监控**: 同步状态、数据质量、错误统计
- **基础设施监控**: 服务器资源使用情况

### 告警规则

关键告警：
- 服务不可用
- API错误率过高
- 数据库连接失败
- 数据同步失败
- 系统资源不足

## 备份策略

### 自动备份

```bash
# 设置定时备份
echo "0 2 * * * /path/to/stockschool/scripts/deployment/backup.sh backup" | crontab -

# 手动备份
./scripts/deployment/backup.sh backup
```

### 备份内容

- PostgreSQL数据库完整备份
- Redis数据快照
- 配置文件备份
- Grafana仪表板备份
- 应用日志备份

### 恢复流程

```bash
# 列出可用备份
./scripts/deployment/backup.sh list

# 恢复指定日期的备份
./scripts/deployment/backup.sh restore 20240101_020000
```

## 运维操作

### 日常维护

```bash
# 查看服务状态
./scripts/deployment/deploy.sh status

# 查看日志
./scripts/deployment/deploy.sh logs [service_name]

# 重启服务
docker-compose -f docker-compose.prod.yml restart [service_name]

# 更新服务
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

### 扩容操作

```bash
# API服务扩容
docker-compose -f docker-compose.prod.yml up -d --scale api=3

# 数据同步服务扩容
docker-compose -f docker-compose.prod.yml up -d --scale data_sync=2
```

### 性能优化

1. **数据库优化**
   - 定期执行VACUUM和ANALYZE
   - 监控慢查询并优化
   - 调整连接池大小

2. **缓存优化**
   - 监控Redis内存使用
   - 优化缓存策略
   - 设置合适的过期时间

3. **应用优化**
   - 监控API响应时间
   - 优化数据同步频率
   - 调整并发参数

## 故障排查

### 常见问题

#### 1. 服务启动失败

```bash
# 检查容器状态
docker-compose -f docker-compose.prod.yml ps

# 查看容器日志
docker-compose -f docker-compose.prod.yml logs [service_name]

# 检查端口占用
netstat -tlnp | grep [port]

# 检查磁盘空间
df -h
```

#### 2. 数据库连接问题

```bash
# 检查数据库状态
docker-compose -f docker-compose.prod.yml exec postgres pg_isready

# 检查连接数
docker-compose -f docker-compose.prod.yml exec postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT count(*) FROM pg_stat_activity;"

# 检查慢查询
docker-compose -f docker-compose.prod.yml exec postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

#### 3. 数据同步问题

```bash
# 检查同步服务状态
curl -f https://localhost/monitoring/sync/status

# 查看同步日志
docker-compose -f docker-compose.prod.yml logs -f data_sync

# 手动触发同步
curl -X POST https://localhost/api/sync/trigger
```

#### 4. 监控告警问题

```bash
# 检查Prometheus目标状态
curl https://localhost/prometheus/api/v1/targets

# 检查AlertManager状态
curl https://localhost:9093/api/v1/status

# 测试告警规则
curl -X POST https://localhost/prometheus/-/reload
```

### 日志分析

```bash
# 查看错误日志
grep -i error logs/*.log

# 分析API访问日志
awk '{print $7}' /var/log/nginx/access.log | sort | uniq -c | sort -nr

# 监控系统资源
top
htop
iotop
```

## 安全配置

### 网络安全

```bash
# 配置防火墙
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
```

### 应用安全

- 使用强密码和密钥
- 定期更新依赖包
- 启用HTTPS
- 配置安全头
- 限制API访问频率

### 数据安全

- 数据库连接加密
- 敏感数据脱敏
- 定期安全审计
- 访问日志记录

## 性能基准

### 系统性能指标

- **API响应时间**: < 200ms (95%分位)
- **数据库查询时间**: < 100ms (平均)
- **数据同步延迟**: < 5分钟
- **系统可用性**: > 99.9%

### 容量规划

- **并发用户**: 1000+
- **API请求**: 10000+ QPS
- **数据存储**: 1TB+
- **日志保留**: 30天

## 联系支持

如遇到部署或运维问题，请联系：

- **技术支持**: support@stockschool.com
- **紧急联系**: emergency@stockschool.com
- **文档更新**: docs@stockschool.com

---

*本文档版本: v1.0*  
*最后更新: 2024-01-01*