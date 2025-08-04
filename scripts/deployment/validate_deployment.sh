#!/bin/bash
# StockSchool部署验证脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 计数器
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED_CHECKS++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED_CHECKS++))
}

check_result() {
    ((TOTAL_CHECKS++))
    if [ $1 -eq 0 ]; then
        log_success "$2"
    else
        log_error "$2"
    fi
}

# 检查Docker环境
check_docker_environment() {
    log_info "检查Docker环境..."
    
    # 检查Docker服务
    if systemctl is-active --quiet docker; then
        check_result 0 "Docker服务运行正常"
    else
        check_result 1 "Docker服务未运行"
    fi
    
    # 检查Docker Compose
    if command -v docker-compose &> /dev/null; then
        check_result 0 "Docker Compose已安装"
    else
        check_result 1 "Docker Compose未安装"
    fi
    
    # 检查Docker版本
    docker_version=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    if [[ $(echo "$docker_version 20.10.0" | tr " " "\n" | sort -V | head -n1) == "20.10.0" ]]; then
        check_result 0 "Docker版本符合要求 ($docker_version)"
    else
        check_result 1 "Docker版本过低 ($docker_version)"
    fi
}

# 检查环境变量
check_environment_variables() {
    log_info "检查环境变量..."
    
    required_vars=(
        "POSTGRES_DB"
        "POSTGRES_USER" 
        "POSTGRES_PASSWORD"
        "REDIS_PASSWORD"
        "TUSHARE_TOKEN"
        "SECRET_KEY"
        "GRAFANA_PASSWORD"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -n "${!var}" ]]; then
            check_result 0 "环境变量 $var 已设置"
        else
            check_result 1 "环境变量 $var 未设置"
        fi
    done
}

# 检查容器状态
check_container_status() {
    log_info "检查容器状态..."
    
    # 获取所有服务
    services=$(docker-compose -f docker-compose.prod.yml config --services)
    
    for service in $services; do
        if docker-compose -f docker-compose.prod.yml ps $service | grep -q "Up"; then
            check_result 0 "容器 $service 运行正常"
        else
            check_result 1 "容器 $service 未运行或异常"
        fi
    done
}

# 检查端口可用性
check_port_availability() {
    log_info "检查端口可用性..."
    
    ports=(
        "80:Nginx HTTP"
        "443:Nginx HTTPS"
        "5432:PostgreSQL"
        "6379:Redis"
        "9090:Prometheus"
        "3000:Grafana"
        "8000:API"
        "8002:Monitoring"
    )
    
    for port_info in "${ports[@]}"; do
        port=$(echo $port_info | cut -d: -f1)
        service=$(echo $port_info | cut -d: -f2)
        
        if netstat -tlnp | grep -q ":$port "; then
            check_result 0 "端口 $port ($service) 正在监听"
        else
            check_result 1 "端口 $port ($service) 未监听"
        fi
    done
}

# 检查服务健康状态
check_service_health() {
    log_info "检查服务健康状态..."
    
    # 检查API健康状态
    if curl -f -s --max-time 10 https://localhost/api/health > /dev/null 2>&1; then
        check_result 0 "API服务健康检查通过"
    else
        check_result 1 "API服务健康检查失败"
    fi
    
    # 检查监控服务健康状态
    if curl -f -s --max-time 10 https://localhost/monitoring/health > /dev/null 2>&1; then
        check_result 0 "监控服务健康检查通过"
    else
        check_result 1 "监控服务健康检查失败"
    fi
    
    # 检查Grafana健康状态
    if curl -f -s --max-time 10 https://localhost/grafana/api/health > /dev/null 2>&1; then
        check_result 0 "Grafana服务健康检查通过"
    else
        check_result 1 "Grafana服务健康检查失败"
    fi
    
    # 检查Prometheus健康状态
    if curl -f -s --max-time 10 https://localhost/prometheus/-/healthy > /dev/null 2>&1; then
        check_result 0 "Prometheus服务健康检查通过"
    else
        check_result 1 "Prometheus服务健康检查失败"
    fi
}

# 检查数据库连接
check_database_connection() {
    log_info "检查数据库连接..."
    
    # 检查PostgreSQL连接
    if docker-compose -f docker-compose.prod.yml exec -T postgres pg_isready -U $POSTGRES_USER -d $POSTGRES_DB > /dev/null 2>&1; then
        check_result 0 "PostgreSQL数据库连接正常"
    else
        check_result 1 "PostgreSQL数据库连接失败"
    fi
    
    # 检查Redis连接
    if docker-compose -f docker-compose.prod.yml exec -T redis redis-cli -a $REDIS_PASSWORD --no-auth-warning ping > /dev/null 2>&1; then
        check_result 0 "Redis缓存连接正常"
    else
        check_result 1 "Redis缓存连接失败"
    fi
}

# 检查SSL证书
check_ssl_certificate() {
    log_info "检查SSL证书..."
    
    if [[ -f "monitoring/nginx/ssl/cert.pem" ]] && [[ -f "monitoring/nginx/ssl/key.pem" ]]; then
        # 检查证书有效期
        if openssl x509 -in monitoring/nginx/ssl/cert.pem -noout -checkend 86400 > /dev/null 2>&1; then
            check_result 0 "SSL证书有效且未过期"
        else
            check_result 1 "SSL证书已过期或即将过期"
        fi
        
        # 检查证书和私钥匹配
        cert_hash=$(openssl x509 -in monitoring/nginx/ssl/cert.pem -noout -modulus | openssl md5)
        key_hash=$(openssl rsa -in monitoring/nginx/ssl/key.pem -noout -modulus | openssl md5)
        
        if [[ "$cert_hash" == "$key_hash" ]]; then
            check_result 0 "SSL证书和私钥匹配"
        else
            check_result 1 "SSL证书和私钥不匹配"
        fi
    else
        check_result 1 "SSL证书文件不存在"
    fi
}

# 检查监控配置
check_monitoring_configuration() {
    log_info "检查监控配置..."
    
    # 检查Prometheus配置
    if [[ -f "monitoring/prometheus/prometheus.yml" ]]; then
        check_result 0 "Prometheus配置文件存在"
    else
        check_result 1 "Prometheus配置文件不存在"
    fi
    
    # 检查Grafana数据源配置
    if [[ -f "monitoring/grafana/datasources/prometheus.yml" ]]; then
        check_result 0 "Grafana数据源配置存在"
    else
        check_result 1 "Grafana数据源配置不存在"
    fi
    
    # 检查告警规则
    if [[ -f "monitoring/prometheus/rules/stockschool_alerts.yml" ]]; then
        check_result 0 "Prometheus告警规则存在"
    else
        check_result 1 "Prometheus告警规则不存在"
    fi
    
    # 检查AlertManager配置
    if [[ -f "monitoring/alertmanager/alertmanager.yml" ]]; then
        check_result 0 "AlertManager配置存在"
    else
        check_result 1 "AlertManager配置不存在"
    fi
}

# 检查日志配置
check_logging_configuration() {
    log_info "检查日志配置..."
    
    # 检查日志目录
    if [[ -d "logs" ]]; then
        check_result 0 "日志目录存在"
    else
        check_result 1 "日志目录不存在"
    fi
    
    # 检查日志文件权限
    if [[ -w "logs" ]]; then
        check_result 0 "日志目录可写"
    else
        check_result 1 "日志目录不可写"
    fi
    
    # 检查Logstash配置
    if [[ -f "monitoring/logstash/pipeline/stockschool.conf" ]]; then
        check_result 0 "Logstash管道配置存在"
    else
        check_result 1 "Logstash管道配置不存在"
    fi
}

# 检查备份配置
check_backup_configuration() {
    log_info "检查备份配置..."
    
    # 检查备份目录
    if [[ -d "backups" ]]; then
        check_result 0 "备份目录存在"
    else
        check_result 1 "备份目录不存在"
    fi
    
    # 检查备份脚本
    if [[ -f "scripts/deployment/backup.sh" ]] && [[ -x "scripts/deployment/backup.sh" ]]; then
        check_result 0 "备份脚本存在且可执行"
    else
        check_result 1 "备份脚本不存在或不可执行"
    fi
    
    # 检查备份计划任务
    if crontab -l 2>/dev/null | grep -q "backup.sh"; then
        check_result 0 "备份计划任务已配置"
    else
        check_result 1 "备份计划任务未配置"
    fi
}

# 检查系统资源
check_system_resources() {
    log_info "检查系统资源..."
    
    # 检查磁盘空间
    disk_usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [[ $disk_usage -lt 80 ]]; then
        check_result 0 "磁盘空间充足 (${disk_usage}%)"
    else
        check_result 1 "磁盘空间不足 (${disk_usage}%)"
    fi
    
    # 检查内存使用
    memory_usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
    if [[ $memory_usage -lt 85 ]]; then
        check_result 0 "内存使用正常 (${memory_usage}%)"
    else
        check_result 1 "内存使用过高 (${memory_usage}%)"
    fi
    
    # 检查CPU负载
    cpu_load=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    cpu_cores=$(nproc)
    if (( $(echo "$cpu_load < $cpu_cores" | bc -l) )); then
        check_result 0 "CPU负载正常 ($cpu_load)"
    else
        check_result 1 "CPU负载过高 ($cpu_load)"
    fi
}

# 检查网络连通性
check_network_connectivity() {
    log_info "检查网络连通性..."
    
    # 检查外部API连通性
    if curl -f -s --max-time 10 https://api.tushare.pro > /dev/null 2>&1; then
        check_result 0 "Tushare API连通性正常"
    else
        check_result 1 "Tushare API连通性异常"
    fi
    
    # 检查DNS解析
    if nslookup google.com > /dev/null 2>&1; then
        check_result 0 "DNS解析正常"
    else
        check_result 1 "DNS解析异常"
    fi
}

# 检查安全配置
check_security_configuration() {
    log_info "检查安全配置..."
    
    # 检查防火墙状态
    if command -v ufw &> /dev/null; then
        if ufw status | grep -q "Status: active"; then
            check_result 0 "防火墙已启用"
        else
            check_result 1 "防火墙未启用"
        fi
    else
        check_result 1 "防火墙未安装"
    fi
    
    # 检查Docker安全配置
    if docker info | grep -q "Security Options"; then
        check_result 0 "Docker安全选项已配置"
    else
        check_result 1 "Docker安全选项未配置"
    fi
}

# 性能基准测试
run_performance_benchmark() {
    log_info "运行性能基准测试..."
    
    # API响应时间测试
    api_response_time=$(curl -w "%{time_total}" -o /dev/null -s https://localhost/api/health)
    if (( $(echo "$api_response_time < 2.0" | bc -l) )); then
        check_result 0 "API响应时间正常 (${api_response_time}s)"
    else
        check_result 1 "API响应时间过长 (${api_response_time}s)"
    fi
    
    # 数据库查询性能测试
    db_query_time=$(docker-compose -f docker-compose.prod.yml exec -T postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "\timing on" -c "SELECT count(*) FROM pg_stat_activity;" 2>&1 | grep "Time:" | awk '{print $2}' | sed 's/ms//')
    if [[ -n "$db_query_time" ]] && (( $(echo "$db_query_time < 100" | bc -l) )); then
        check_result 0 "数据库查询性能正常 (${db_query_time}ms)"
    else
        check_result 1 "数据库查询性能异常"
    fi
}

# 生成验证报告
generate_validation_report() {
    echo
    echo "=========================================="
    echo "           部署验证报告"
    echo "=========================================="
    echo "验证时间: $(date)"
    echo "总检查项: $TOTAL_CHECKS"
    echo "通过检查: $PASSED_CHECKS"
    echo "失败检查: $FAILED_CHECKS"
    echo
    
    if [[ $FAILED_CHECKS -eq 0 ]]; then
        log_success "🎉 所有检查项均通过，部署验证成功！"
        echo
        echo "系统访问地址："
        echo "  - 主页: https://localhost"
        echo "  - API文档: https://localhost/api/docs"
        echo "  - Grafana监控: https://localhost/grafana"
        echo "  - Prometheus: https://localhost/prometheus"
        echo
        return 0
    else
        log_error "❌ 部署验证失败，请检查失败的项目"
        echo
        echo "建议操作："
        echo "1. 查看详细日志: docker-compose -f docker-compose.prod.yml logs"
        echo "2. 检查系统资源: ./scripts/monitoring/performance_check.sh"
        echo "3. 重新部署: ./scripts/deployment/deploy.sh deploy"
        echo
        return 1
    fi
}

# 主函数
main() {
    echo "=========================================="
    echo "      StockSchool 部署验证开始"
    echo "=========================================="
    echo
    
    # 执行所有检查
    check_docker_environment
    check_environment_variables
    check_container_status
    check_port_availability
    check_service_health
    check_database_connection
    check_ssl_certificate
    check_monitoring_configuration
    check_logging_configuration
    check_backup_configuration
    check_system_resources
    check_network_connectivity
    check_security_configuration
    run_performance_benchmark
    
    # 生成报告
    generate_validation_report
}

# 解析命令行参数
case "${1:-validate}" in
    "validate")
        main
        ;;
    "quick")
        log_info "执行快速验证..."
        check_container_status
        check_service_health
        check_database_connection
        generate_validation_report
        ;;
    "security")
        log_info "执行安全检查..."
        check_ssl_certificate
        check_security_configuration
        generate_validation_report
        ;;
    "performance")
        log_info "执行性能测试..."
        run_performance_benchmark
        check_system_resources
        generate_validation_report
        ;;
    *)
        echo "用法: $0 {validate|quick|security|performance}"
        echo "  validate     - 完整验证 (默认)"
        echo "  quick        - 快速验证"
        echo "  security     - 安全检查"
        echo "  performance  - 性能测试"
        exit 1
        ;;
esac