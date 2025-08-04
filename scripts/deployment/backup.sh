#!/bin/bash
# StockSchool数据备份脚本

set -e

# 配置
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=7

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 创建备份目录
create_backup_dir() {
    mkdir -p "$BACKUP_DIR"
    log_info "备份目录: $BACKUP_DIR"
}

# 备份PostgreSQL数据库
backup_postgres() {
    log_info "开始备份PostgreSQL数据库..."
    
    local backup_file="$BACKUP_DIR/postgres_backup_$DATE.sql.gz"
    
    # 使用docker exec执行pg_dump
    docker-compose -f docker-compose.prod.yml exec -T postgres pg_dump \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        --verbose \
        --no-owner \
        --no-privileges \
        | gzip > "$backup_file"
    
    if [[ $? -eq 0 ]]; then
        log_success "PostgreSQL备份完成: $backup_file"
        
        # 记录备份大小
        local size=$(du -h "$backup_file" | cut -f1)
        log_info "备份文件大小: $size"
    else
        log_error "PostgreSQL备份失败"
        return 1
    fi
}

# 备份Redis数据
backup_redis() {
    log_info "开始备份Redis数据..."
    
    local backup_file="$BACKUP_DIR/redis_backup_$DATE.rdb"
    
    # 触发Redis保存
    docker-compose -f docker-compose.prod.yml exec -T redis redis-cli \
        -a "$REDIS_PASSWORD" \
        --no-auth-warning \
        BGSAVE
    
    # 等待保存完成
    sleep 5
    
    # 复制RDB文件
    docker cp stockschool_redis_prod:/data/dump.rdb "$backup_file"
    
    if [[ $? -eq 0 ]]; then
        log_success "Redis备份完成: $backup_file"
    else
        log_error "Redis备份失败"
        return 1
    fi
}

# 备份配置文件
backup_configs() {
    log_info "开始备份配置文件..."
    
    local config_backup="$BACKUP_DIR/configs_backup_$DATE.tar.gz"
    
    tar -czf "$config_backup" \
        config/ \
        monitoring/ \
        docker-compose.prod.yml \
        .env 2>/dev/null || true
    
    if [[ $? -eq 0 ]]; then
        log_success "配置文件备份完成: $config_backup"
    else
        log_error "配置文件备份失败"
        return 1
    fi
}

# 备份日志文件
backup_logs() {
    log_info "开始备份日志文件..."
    
    local logs_backup="$BACKUP_DIR/logs_backup_$DATE.tar.gz"
    
    if [[ -d "logs" ]] && [[ -n "$(ls -A logs/)" ]]; then
        tar -czf "$logs_backup" logs/
        
        if [[ $? -eq 0 ]]; then
            log_success "日志文件备份完成: $logs_backup"
        else
            log_error "日志文件备份失败"
            return 1
        fi
    else
        log_info "没有日志文件需要备份"
    fi
}

# 备份Grafana仪表板
backup_grafana() {
    log_info "开始备份Grafana仪表板..."
    
    local grafana_backup="$BACKUP_DIR/grafana_backup_$DATE.tar.gz"
    
    # 等待Grafana启动
    sleep 10
    
    # 备份Grafana数据目录
    docker run --rm \
        --volumes-from stockschool_grafana \
        -v "$BACKUP_DIR:/backup" \
        alpine:latest \
        tar -czf "/backup/grafana_backup_$DATE.tar.gz" -C /var/lib/grafana .
    
    if [[ $? -eq 0 ]]; then
        log_success "Grafana备份完成: $grafana_backup"
    else
        log_error "Grafana备份失败"
        return 1
    fi
}

# 清理旧备份
cleanup_old_backups() {
    log_info "清理${RETENTION_DAYS}天前的备份文件..."
    
    find "$BACKUP_DIR" -name "*backup_*" -type f -mtime +$RETENTION_DAYS -delete
    
    local remaining=$(find "$BACKUP_DIR" -name "*backup_*" -type f | wc -l)
    log_success "清理完成，剩余备份文件: $remaining 个"
}

# 验证备份完整性
verify_backups() {
    log_info "验证备份文件完整性..."
    
    local postgres_backup="$BACKUP_DIR/postgres_backup_$DATE.sql.gz"
    local redis_backup="$BACKUP_DIR/redis_backup_$DATE.rdb"
    local config_backup="$BACKUP_DIR/configs_backup_$DATE.tar.gz"
    
    # 验证PostgreSQL备份
    if [[ -f "$postgres_backup" ]]; then
        if gzip -t "$postgres_backup"; then
            log_success "PostgreSQL备份文件完整性验证通过"
        else
            log_error "PostgreSQL备份文件损坏"
            return 1
        fi
    fi
    
    # 验证配置备份
    if [[ -f "$config_backup" ]]; then
        if tar -tzf "$config_backup" > /dev/null; then
            log_success "配置备份文件完整性验证通过"
        else
            log_error "配置备份文件损坏"
            return 1
        fi
    fi
    
    log_success "备份文件完整性验证完成"
}

# 发送备份通知
send_notification() {
    local status=$1
    local message=$2
    
    if [[ -n "$BACKUP_WEBHOOK_URL" ]]; then
        curl -X POST "$BACKUP_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"text\":\"StockSchool备份通知: $message\"}" \
            > /dev/null 2>&1
    fi
    
    if [[ -n "$BACKUP_EMAIL" ]]; then
        echo "$message" | mail -s "StockSchool备份通知" "$BACKUP_EMAIL" 2>/dev/null || true
    fi
}

# 主备份函数
main_backup() {
    log_info "开始执行StockSchool系统备份..."
    
    local start_time=$(date +%s)
    
    create_backup_dir
    
    # 执行各项备份
    backup_postgres
    backup_redis
    backup_configs
    backup_logs
    backup_grafana
    
    # 验证备份
    verify_backups
    
    # 清理旧备份
    cleanup_old_backups
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "备份完成！耗时: ${duration}秒"
    
    # 发送成功通知
    send_notification "success" "备份成功完成，耗时${duration}秒"
}

# 恢复函数
restore_backup() {
    local backup_date=$1
    
    if [[ -z "$backup_date" ]]; then
        log_error "请指定备份日期，格式: YYYYMMDD_HHMMSS"
        exit 1
    fi
    
    log_info "开始恢复备份: $backup_date"
    
    # 停止服务
    log_info "停止应用服务..."
    docker-compose -f docker-compose.prod.yml stop api data_sync monitoring
    
    # 恢复PostgreSQL
    local postgres_backup="$BACKUP_DIR/postgres_backup_${backup_date}.sql.gz"
    if [[ -f "$postgres_backup" ]]; then
        log_info "恢复PostgreSQL数据库..."
        gunzip -c "$postgres_backup" | \
            docker-compose -f docker-compose.prod.yml exec -T postgres \
            psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"
        log_success "PostgreSQL恢复完成"
    else
        log_error "PostgreSQL备份文件不存在: $postgres_backup"
    fi
    
    # 恢复Redis
    local redis_backup="$BACKUP_DIR/redis_backup_${backup_date}.rdb"
    if [[ -f "$redis_backup" ]]; then
        log_info "恢复Redis数据..."
        docker-compose -f docker-compose.prod.yml stop redis
        docker cp "$redis_backup" stockschool_redis_prod:/data/dump.rdb
        docker-compose -f docker-compose.prod.yml start redis
        log_success "Redis恢复完成"
    else
        log_error "Redis备份文件不存在: $redis_backup"
    fi
    
    # 重启服务
    log_info "重启应用服务..."
    docker-compose -f docker-compose.prod.yml start api data_sync monitoring
    
    log_success "备份恢复完成"
}

# 列出可用备份
list_backups() {
    log_info "可用的备份文件:"
    echo
    
    find "$BACKUP_DIR" -name "*backup_*.sql.gz" -type f | sort | while read -r file; do
        local basename=$(basename "$file")
        local date_part=$(echo "$basename" | grep -o '[0-9]\{8\}_[0-9]\{6\}')
        local size=$(du -h "$file" | cut -f1)
        local mtime=$(stat -c %y "$file" | cut -d' ' -f1,2)
        
        echo "  $date_part - $size - $mtime"
    done
}

# 主函数
case "${1:-backup}" in
    "backup")
        main_backup
        ;;
    "restore")
        restore_backup "$2"
        ;;
    "list")
        list_backups
        ;;
    "cleanup")
        cleanup_old_backups
        ;;
    *)
        echo "用法: $0 {backup|restore|list|cleanup}"
        echo "  backup           - 执行完整备份"
        echo "  restore <date>   - 恢复指定日期的备份"
        echo "  list             - 列出可用备份"
        echo "  cleanup          - 清理旧备份"
        exit 1
        ;;
esac