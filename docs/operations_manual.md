# StockSchool 运维手册

## 概述

本手册为StockSchool量化投研系统的日常运维提供详细指导，包括系统监控、故障处理、性能优化、安全管理等方面的操作规程。

## 目录

1. [日常运维检查](#日常运维检查)
2. [监控和告警](#监控和告警)
3. [备份和恢复](#备份和恢复)
4. [性能优化](#性能优化)
5. [安全管理](#安全管理)
6. [故障处理](#故障处理)
7. [变更管理](#变更管理)
8. [应急响应](#应急响应)

## 日常运维检查

### 每日检查清单

#### 系统状态检查
```bash
# 1. 执行自动健康检查
./scripts/monitoring/health_check.sh

# 2. 检查服务状态
docker-compose -f docker-compose.prod.yml ps

# 3. 查看系统资源使用
./scripts/monitoring/performance_check.sh

# 4. 检查磁盘空间
df -h | awk '$5 > 80 {print "警告: " $1 " 磁盘使用率 " $5}'

# 5. 检查日志错误
grep -i "error\|exception\|critical" logs/*.log | tail -20
```

#### 业务指标检查
```bash
# 1. 检查数据同步状态
curl -s https://localhost/monitoring/sync/status | jq '.'

# 2. 检查API响应时间
curl -w "响应时间: %{time_total}s\n" -o /dev/null -s https://localhost/api/health

# 3. 检查数据库连接数
docker-compose -f docker-compose.prod.yml exec postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT count(*) as connections FROM pg_stat_activity;"

# 4. 检查Redis内存使用
docker-compose -f docker-compose.prod.yml exec redis redis-cli -a $REDIS_PASSWORD info memory | grep used_memory_human
```

### 每周检查清单

#### 系统维护
```bash
# 1. 清理Docker镜像和容器
docker system prune -f

# 2. 更新系统包
sudo apt update && sudo apt upgrade -y

# 3. 检查SSL证书有效期
openssl x509 -in monitoring/nginx/ssl/cert.pem -noout -dates

# 4. 分析慢查询日志
docker-compose -f docker-compose.prod.yml exec postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT query, mean_time, calls FROM pg_stat_statements WHERE mean_time > 1000 ORDER BY mean_time DESC LIMIT 10;"
```

#### 数据维护
```bash
# 1. 数据库统计信息更新
docker-compose -f docker-compose.prod.yml exec postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "ANALYZE;"

# 2. 清理过期日志
find logs/ -name "*.log" -mtime +30 -delete

# 3. 检查备份完整性
./scripts/deployment/backup.sh list
```

### 每月检查清单

#### 容量规划
```bash
# 1. 分析存储增长趋势
du -sh data/* | sort -hr

# 2. 检查数据库表大小
docker-compose -f docker-compose.prod.yml exec postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT schemaname,tablename,pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size FROM pg_tables ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC LIMIT 10;"

# 3. 分析API使用趋势
awk '{print $7}' /var/log/nginx/access.log | sort | uniq -c | sort -nr | head -20
```

#### 安全检查
```bash
# 1. 检查系统更新
sudo apt list --upgradable

# 2. 检查防火墙状态
sudo ufw status

# 3. 检查登录日志
sudo grep "Failed password" /var/log/auth.log | tail -10

# 4. 检查Docker安全配置
docker bench-security
```

## 监控和告警

### Grafana仪表板

#### 主要仪表板
1. **系统概览** - 整体系统状态和关键指标
2. **API性能** - API请求量、响应时间、错误率
3. **数据库监控** - 数据库连接、查询性能、存储使用
4. **数据同步监控** - 同步状态、数据质量、错误统计
5. **基础设施监控** - 服务器资源使用情况

#### 关键指标阈值
```yaml
# 系统指标
CPU使用率: > 80%
内存使用率: > 85%
磁盘使用率: > 85%
网络延迟: > 100ms

# 应用指标
API响应时间: > 2s (95%分位)
API错误率: > 5%
数据库连接数: > 80
Redis内存使用: > 80%

# 业务指标
数据同步延迟: > 2小时
数据质量评分: < 0.8
因子计算失败率: > 10%
```

### 告警配置

#### 告警级别
- **Critical**: 影响系统可用性的严重问题
- **Warning**: 需要关注但不影响核心功能的问题
- **Info**: 一般性信息通知

#### 告警通知渠道
1. **邮件通知** - 所有级别告警
2. **Slack通知** - Critical和Warning级别
3. **短信通知** - Critical级别（需配置）
4. **电话通知** - 系统完全不可用时

### 告警处理流程

#### Critical告警处理
1. **立即响应** (5分钟内)
   - 确认告警真实性
   - 评估影响范围
   - 启动应急响应流程

2. **问题定位** (15分钟内)
   - 查看相关日志
   - 检查系统资源
   - 确定根本原因

3. **问题解决** (30分钟内)
   - 实施临时解决方案
   - 恢复服务可用性
   - 记录处理过程

4. **后续跟进**
   - 实施永久解决方案
   - 更新文档和流程
   - 进行事后分析

## 备份和恢复

### 备份策略

#### 自动备份
```bash
# 设置每日自动备份
echo "0 2 * * * /path/to/stockschool/scripts/deployment/backup.sh backup" | crontab -

# 设置每周完整备份
echo "0 1 * * 0 /path/to/stockschool/scripts/deployment/backup.sh backup --full" | crontab -
```

#### 备份内容
1. **数据库备份** - PostgreSQL完整备份
2. **Redis备份** - Redis数据快照
3. **配置备份** - 系统配置文件
4. **日志备份** - 应用日志文件
5. **代码备份** - 应用代码和脚本

#### 备份验证
```bash
# 每周验证备份完整性
./scripts/deployment/backup.sh verify

# 每月进行恢复测试
./scripts/deployment/backup.sh restore-test
```

### 恢复流程

#### 数据库恢复
```bash
# 1. 停止应用服务
docker-compose -f docker-compose.prod.yml stop api data_sync

# 2. 备份当前数据
./scripts/deployment/backup.sh backup --emergency

# 3. 恢复数据库
./scripts/deployment/backup.sh restore 20240101_020000

# 4. 验证数据完整性
docker-compose -f docker-compose.prod.yml exec postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT count(*) FROM stock_basic;"

# 5. 重启应用服务
docker-compose -f docker-compose.prod.yml start api data_sync
```

#### 完整系统恢复
```bash
# 1. 停止所有服务
docker-compose -f docker-compose.prod.yml down

# 2. 恢复配置文件
tar -xzf backups/configs_backup_20240101_020000.tar.gz

# 3. 恢复数据
./scripts/deployment/backup.sh restore 20240101_020000

# 4. 重新部署系统
./scripts/deployment/deploy.sh deploy

# 5. 验证系统功能
./scripts/deployment/deploy.sh health
```

## 性能优化

### 数据库优化

#### 定期维护
```sql
-- 1. 更新统计信息
ANALYZE;

-- 2. 清理死元组
VACUUM;

-- 3. 重建索引
REINDEX DATABASE stockschool;

-- 4. 检查表膨胀
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
       pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as index_size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

#### 查询优化
```sql
-- 1. 查找慢查询
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
WHERE mean_time > 1000
ORDER BY mean_time DESC
LIMIT 10;

-- 2. 分析查询计划
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM your_query;

-- 3. 创建必要索引
CREATE INDEX CONCURRENTLY idx_stock_code_date ON stock_daily(ts_code, trade_date);
```

### 应用优化

#### 缓存策略
```python
# 1. Redis缓存配置
CACHE_CONFIG = {
    'default_timeout': 300,
    'key_prefix': 'stockschool:',
    'version': 1
}

# 2. 查询结果缓存
@cache.cached(timeout=300, key_prefix='stock_data')
def get_stock_data(ts_code, start_date, end_date):
    return query_database(ts_code, start_date, end_date)

# 3. 分页缓存
@cache.cached(timeout=600, key_prefix='stock_list')
def get_stock_list(page, page_size):
    return paginated_query(page, page_size)
```

#### 并发优化
```python
# 1. 异步处理
import asyncio
import aiohttp

async def fetch_multiple_stocks(stock_codes):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_stock_data(session, code) for code in stock_codes]
        return await asyncio.gather(*tasks)

# 2. 连接池优化
DATABASE_CONFIG = {
    'pool_size': 20,
    'max_overflow': 30,
    'pool_timeout': 30,
    'pool_recycle': 3600
}
```

### 系统优化

#### 内核参数调优
```bash
# /etc/sysctl.conf
vm.swappiness=10
vm.dirty_ratio=15
vm.dirty_background_ratio=5
net.core.somaxconn=65535
net.core.netdev_max_backlog=5000
net.ipv4.tcp_max_syn_backlog=65535
```

#### 文件系统优化
```bash
# 1. 调整文件描述符限制
echo '* soft nofile 65535' >> /etc/security/limits.conf
echo '* hard nofile 65535' >> /etc/security/limits.conf

# 2. 优化磁盘调度器
echo 'deadline' > /sys/block/sda/queue/scheduler

# 3. 禁用不必要的服务
systemctl disable bluetooth
systemctl disable cups
```

## 安全管理

### 访问控制

#### 用户权限管理
```bash
# 1. 创建运维用户
sudo useradd -m -s /bin/bash stockschool-ops
sudo usermod -aG docker stockschool-ops

# 2. 配置SSH密钥认证
sudo mkdir -p /home/stockschool-ops/.ssh
sudo cp authorized_keys /home/stockschool-ops/.ssh/
sudo chown -R stockschool-ops:stockschool-ops /home/stockschool-ops/.ssh
sudo chmod 700 /home/stockschool-ops/.ssh
sudo chmod 600 /home/stockschool-ops/.ssh/authorized_keys

# 3. 配置sudo权限
echo 'stockschool-ops ALL=(ALL) NOPASSWD: /usr/bin/docker, /usr/local/bin/docker-compose' | sudo tee /etc/sudoers.d/stockschool-ops
```

#### 网络安全
```bash
# 1. 配置防火墙
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# 2. 配置fail2ban
sudo apt install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### 数据安全

#### 数据库安全
```sql
-- 1. 创建只读用户
CREATE USER stockschool_readonly WITH PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE stockschool TO stockschool_readonly;
GRANT USAGE ON SCHEMA public TO stockschool_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO stockschool_readonly;

-- 2. 启用审计日志
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000;
SELECT pg_reload_conf();
```

#### 敏感数据处理
```python
# 1. 数据脱敏
def mask_sensitive_data(data):
    if 'phone' in data:
        data['phone'] = data['phone'][:3] + '****' + data['phone'][-4:]
    if 'email' in data:
        email_parts = data['email'].split('@')
        data['email'] = email_parts[0][:2] + '***@' + email_parts[1]
    return data

# 2. 加密存储
from cryptography.fernet import Fernet

def encrypt_data(data, key):
    f = Fernet(key)
    return f.encrypt(data.encode())

def decrypt_data(encrypted_data, key):
    f = Fernet(key)
    return f.decrypt(encrypted_data).decode()
```

### 安全审计

#### 日志审计
```bash
# 1. 系统登录审计
sudo grep "Failed password" /var/log/auth.log | tail -20

# 2. Docker操作审计
sudo grep "docker" /var/log/syslog | tail -20

# 3. 应用访问审计
grep "POST\|PUT\|DELETE" /var/log/nginx/access.log | tail -20
```

#### 安全扫描
```bash
# 1. 端口扫描
nmap -sS -O localhost

# 2. 漏洞扫描
sudo apt install lynis
sudo lynis audit system

# 3. Docker安全扫描
docker run --rm -it --net host --pid host --userns host --cap-add audit_control \
    -e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
    -v /etc:/etc:ro \
    -v /usr/bin/containerd:/usr/bin/containerd:ro \
    -v /usr/bin/runc:/usr/bin/runc:ro \
    -v /usr/lib/systemd:/usr/lib/systemd:ro \
    -v /var/lib:/var/lib:ro \
    -v /var/run/docker.sock:/var/run/docker.sock:ro \
    --label docker_bench_security \
    docker/docker-bench-security
```

## 故障处理

### 常见故障处理

#### 服务无法启动
```bash
# 1. 检查容器状态
docker-compose -f docker-compose.prod.yml ps

# 2. 查看容器日志
docker-compose -f docker-compose.prod.yml logs [service_name]

# 3. 检查端口占用
netstat -tlnp | grep [port]

# 4. 检查磁盘空间
df -h

# 5. 重启服务
docker-compose -f docker-compose.prod.yml restart [service_name]
```

#### 数据库连接问题
```bash
# 1. 检查数据库状态
docker-compose -f docker-compose.prod.yml exec postgres pg_isready

# 2. 检查连接数
docker-compose -f docker-compose.prod.yml exec postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT count(*) FROM pg_stat_activity;"

# 3. 终止空闲连接
docker-compose -f docker-compose.prod.yml exec postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND state_change < now() - interval '5 minutes';"

# 4. 重启数据库
docker-compose -f docker-compose.prod.yml restart postgres
```

#### 性能问题
```bash
# 1. 检查系统负载
top
htop

# 2. 检查磁盘I/O
iostat -x 1 5

# 3. 检查网络连接
netstat -an | wc -l

# 4. 分析慢查询
docker-compose -f docker-compose.prod.yml exec postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT query, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

### 故障升级流程

#### 故障级别定义
- **P0**: 系统完全不可用
- **P1**: 核心功能受影响
- **P2**: 部分功能受影响
- **P3**: 性能问题或非关键功能问题

#### 升级时间表
- **P0**: 立即升级
- **P1**: 30分钟内升级
- **P2**: 2小时内升级
- **P3**: 24小时内升级

## 变更管理

### 变更流程

#### 变更分类
1. **紧急变更** - 修复严重故障
2. **标准变更** - 预先批准的常规变更
3. **正常变更** - 需要评估和批准的变更

#### 变更步骤
1. **变更申请** - 填写变更申请表
2. **风险评估** - 评估变更风险和影响
3. **变更批准** - 获得相关方批准
4. **变更实施** - 按计划执行变更
5. **变更验证** - 验证变更效果
6. **变更关闭** - 记录变更结果

### 发布管理

#### 发布流程
```bash
# 1. 代码审查
git log --oneline origin/main..HEAD

# 2. 测试验证
./scripts/deployment/deploy.sh test

# 3. 备份当前版本
./scripts/deployment/backup.sh backup --pre-release

# 4. 滚动更新
./scripts/deployment/rolling_update.sh api

# 5. 验证发布
./scripts/deployment/deploy.sh health

# 6. 监控观察
# 观察监控指标30分钟
```

#### 回滚流程
```bash
# 1. 确认需要回滚
./scripts/deployment/deploy.sh health

# 2. 执行回滚
./scripts/deployment/rolling_update.sh --rollback api

# 3. 验证回滚
./scripts/deployment/deploy.sh health

# 4. 通知相关方
# 发送回滚通知
```

## 应急响应

### 应急响应团队

#### 角色定义
- **事件指挥官** - 负责整体协调
- **技术负责人** - 负责技术问题解决
- **沟通协调员** - 负责内外部沟通
- **记录员** - 负责记录处理过程

#### 联系方式
```
事件指挥官: +86-138-xxxx-xxxx
技术负责人: +86-139-xxxx-xxxx
沟通协调员: +86-137-xxxx-xxxx
```

### 应急预案

#### 数据库故障预案
1. **故障检测** - 监控告警或用户报告
2. **影响评估** - 确定影响范围和严重程度
3. **应急措施** - 切换到备用数据库
4. **故障修复** - 修复主数据库问题
5. **服务恢复** - 切换回主数据库
6. **事后分析** - 分析故障原因和改进措施

#### 网络故障预案
1. **故障确认** - 确认网络连接问题
2. **流量切换** - 切换到备用网络链路
3. **问题排查** - 联系网络服务提供商
4. **服务监控** - 监控服务可用性
5. **恢复验证** - 验证网络恢复正常

#### 安全事件预案
1. **事件确认** - 确认安全事件类型
2. **影响控制** - 隔离受影响系统
3. **证据保全** - 保存相关日志和证据
4. **漏洞修复** - 修复安全漏洞
5. **系统加固** - 加强安全防护措施
6. **事件报告** - 向相关部门报告

### 应急演练

#### 演练计划
- **月度演练** - 故障恢复演练
- **季度演练** - 完整灾难恢复演练
- **年度演练** - 综合应急响应演练

#### 演练内容
1. **数据库故障恢复**
2. **应用服务故障切换**
3. **网络中断应对**
4. **安全事件响应**
5. **完整系统恢复**

---

*本手册版本: v1.0*  
*最后更新: 2024-01-01*  
*下次审查: 2024-07-01*