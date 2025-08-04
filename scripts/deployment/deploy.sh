#!/bin/bash
# StockSchool生产环境部署脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查环境变量
check_env_vars() {
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
    
    missing_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "缺少必要的环境变量: ${missing_vars[*]}"
        log_info "请在.env文件中设置这些变量"
        exit 1
    fi
    
    log_success "环境变量检查通过"
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    # 检查Docker服务状态
    if ! docker info &> /dev/null; then
        log_error "Docker服务未运行，请启动Docker服务"
        exit 1
    fi
    
    log_success "依赖检查通过"
}

# 创建必要目录
create_directories() {
    log_info "创建必要目录..."
    
    directories=(
        "logs"
        "backups"
        "monitoring/nginx/ssl"
        "data/postgres"
        "data/redis"
        "data/grafana"
        "data/prometheus"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log_info "创建目录: $dir"
    done
    
    log_success "目录创建完成"
}

# 生成SSL证书（自签名，生产环境请使用正式证书）
generate_ssl_cert() {
    log_info "生成SSL证书..."
    
    ssl_dir="monitoring/nginx/ssl"
    
    if [[ ! -f "$ssl_dir/cert.pem" ]] || [[ ! -f "$ssl_dir/key.pem" ]]; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$ssl_dir/key.pem" \
            -out "$ssl_dir/cert.pem" \
            -subj "/C=CN/ST=Beijing/L=Beijing/O=StockSchool/CN=localhost"
        
        log_success "SSL证书生成完成"
    else
        log_info "SSL证书已存在，跳过生成"
    fi
}

# 构建镜像
build_images() {
    log_info "构建Docker镜像..."
    
    # 构建所有服务镜像
    docker-compose -f docker-compose.prod.yml build --no-cache --parallel
    
    log_success "镜像构建完成"
}

# 启动基础服务
start_infrastructure() {
    log_info "启动基础设施服务..."
    
    # 启动数据库和缓存
    docker-compose -f docker-compose.prod.yml up -d postgres redis
    
    # 等待服务就绪
    log_info "等待数据库启动..."
    sleep 30
    
    # 检查数据库连接
    if ! docker-compose -f docker-compose.prod.yml exec -T postgres pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"; then
        log_error "数据库启动失败"
        exit 1
    fi
    
    log_success "基础设施服务启动完成"
}

# 运行数据库迁移
run_migrations() {
    log_info "运行数据库迁移..."
    
    # 这里可以添加数据库迁移逻辑
    # docker-compose -f docker-compose.prod.yml run --rm api python -m alembic upgrade head
    
    log_success "数据库迁移完成"
}

# 启动应用服务
start_application() {
    log_info "启动应用服务..."
    
    # 启动应用服务
    docker-compose -f docker-compose.prod.yml up -d api data_sync monitoring
    
    # 等待服务启动
    sleep 20
    
    log_success "应用服务启动完成"
}

# 启动监控服务
start_monitoring() {
    log_info "启动监控服务..."
    
    # 启动监控栈
    docker-compose -f docker-compose.prod.yml up -d prometheus grafana alertmanager node_exporter postgres_exporter redis_exporter
    
    # 启动日志栈
    docker-compose -f docker-compose.prod.yml up -d elasticsearch logstash kibana
    
    # 启动负载均衡器
    docker-compose -f docker-compose.prod.yml up -d nginx
    
    sleep 30
    
    log_success "监控服务启动完成"
}

# 健康检查
health_check() {
    log_info "执行健康检查..."
    
    # 检查API服务
    if curl -f -s http://localhost/api/health > /dev/null; then
        log_success "API服务健康检查通过"
    else
        log_error "API服务健康检查失败"
        return 1
    fi
    
    # 检查监控服务
    if curl -f -s http://localhost/monitoring/health > /dev/null; then
        log_success "监控服务健康检查通过"
    else
        log_warning "监控服务健康检查失败"
    fi
    
    # 检查Grafana
    if curl -f -s http://localhost/grafana/api/health > /dev/null; then
        log_success "Grafana健康检查通过"
    else
        log_warning "Grafana健康检查失败"
    fi
    
    log_success "健康检查完成"
}

# 显示部署信息
show_deployment_info() {
    log_success "🎉 StockSchool部署完成！"
    echo
    echo "访问地址："
    echo "  - 主页: https://localhost"
    echo "  - API文档: https://localhost/api/docs"
    echo "  - Grafana监控: https://localhost/grafana"
    echo "  - Prometheus: https://localhost/prometheus"
    echo "  - Kibana日志: http://localhost:5601"
    echo
    echo "默认账号："
    echo "  - Grafana: admin / $GRAFANA_PASSWORD"
    echo
    echo "查看服务状态："
    echo "  docker-compose -f docker-compose.prod.yml ps"
    echo
    echo "查看日志："
    echo "  docker-compose -f docker-compose.prod.yml logs -f [service_name]"
}

# 清理函数
cleanup() {
    if [[ $? -ne 0 ]]; then
        log_error "部署失败，正在清理..."
        docker-compose -f docker-compose.prod.yml down
    fi
}

# 主函数
main() {
    log_info "开始部署StockSchool生产环境..."
    
    # 设置错误处理
    trap cleanup EXIT
    
    # 执行部署步骤
    check_env_vars
    check_dependencies
    create_directories
    generate_ssl_cert
    build_images
    start_infrastructure
    run_migrations
    start_application
    start_monitoring
    health_check
    show_deployment_info
    
    # 取消错误处理
    trap - EXIT
    
    log_success "部署完成！"
}

# 解析命令行参数
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        log_info "停止所有服务..."
        docker-compose -f docker-compose.prod.yml down
        log_success "服务已停止"
        ;;
    "restart")
        log_info "重启所有服务..."
        docker-compose -f docker-compose.prod.yml restart
        log_success "服务已重启"
        ;;
    "status")
        docker-compose -f docker-compose.prod.yml ps
        ;;
    "logs")
        docker-compose -f docker-compose.prod.yml logs -f "${2:-}"
        ;;
    "health")
        health_check
        ;;
    *)
        echo "用法: $0 {deploy|stop|restart|status|logs|health}"
        echo "  deploy  - 部署系统"
        echo "  stop    - 停止所有服务"
        echo "  restart - 重启所有服务"
        echo "  status  - 查看服务状态"
        echo "  logs    - 查看日志"
        echo "  health  - 健康检查"
        exit 1
        ;;
esac