#!/bin/bash
# StockSchool滚动更新脚本

set -e

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

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 配置
SERVICE_NAME=${1:-api}
COMPOSE_FILE="docker-compose.prod.yml"
HEALTH_CHECK_URL="https://localhost/api/health"
ROLLBACK_ON_FAILURE=${ROLLBACK_ON_FAILURE:-true}

# 检查服务是否存在
check_service_exists() {
    if ! docker-compose -f $COMPOSE_FILE config --services | grep -q "^${SERVICE_NAME}$"; then
        log_error "服务 $SERVICE_NAME 不存在"
        exit 1
    fi
}

# 健康检查
health_check() {
    local max_attempts=30
    local attempt=1
    
    log_info "执行健康检查..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s $HEALTH_CHECK_URL > /dev/null 2>&1; then
            log_success "健康检查通过"
            return 0
        fi
        
        log_info "健康检查失败，重试 $attempt/$max_attempts"
        sleep 10
        ((attempt++))
    done
    
    log_error "健康检查失败"
    return 1
}

# 获取当前运行的容器
get_running_containers() {
    docker-compose -f $COMPOSE_FILE ps -q $SERVICE_NAME
}

# 备份当前镜像
backup_current_image() {
    local current_image=$(docker-compose -f $COMPOSE_FILE images -q $SERVICE_NAME | head -1)
    if [ -n "$current_image" ]; then
        docker tag $current_image stockschool_${SERVICE_NAME}_backup:$(date +%Y%m%d_%H%M%S)
        log_success "当前镜像已备份"
    fi
}

# 构建新镜像
build_new_image() {
    log_info "构建新的 $SERVICE_NAME 镜像..."
    
    if docker-compose -f $COMPOSE_FILE build --no-cache $SERVICE_NAME; then
        log_success "镜像构建成功"
    else
        log_error "镜像构建失败"
        exit 1
    fi
}

# 滚动更新单个服务
rolling_update_service() {
    log_info "开始滚动更新服务: $SERVICE_NAME"
    
    # 获取当前副本数
    local current_containers=$(get_running_containers)
    local container_count=$(echo "$current_containers" | wc -l)
    
    if [ $container_count -eq 0 ]; then
        log_warning "没有运行中的容器，执行常规启动"
        docker-compose -f $COMPOSE_FILE up -d $SERVICE_NAME
        health_check
        return
    fi
    
    log_info "当前运行容器数: $container_count"
    
    # 逐个更新容器
    local container_index=1
    for container_id in $current_containers; do
        log_info "更新容器 $container_index/$container_count: $container_id"
        
        # 停止旧容器
        docker stop $container_id
        
        # 启动新容器
        docker-compose -f $COMPOSE_FILE up -d --no-deps $SERVICE_NAME
        
        # 等待新容器启动
        sleep 15
        
        # 健康检查
        if ! health_check; then
            log_error "新容器健康检查失败"
            
            if [ "$ROLLBACK_ON_FAILURE" = "true" ]; then
                log_warning "开始回滚..."
                rollback_service
                exit 1
            else
                log_error "更新失败，请手动处理"
                exit 1
            fi
        fi
        
        # 移除旧容器
        docker rm $container_id
        
        log_success "容器 $container_index 更新完成"
        ((container_index++))
        
        # 容器间更新间隔
        if [ $container_index -le $container_count ]; then
            sleep 10
        fi
    done
    
    log_success "服务 $SERVICE_NAME 滚动更新完成"
}

# 回滚服务
rollback_service() {
    log_warning "开始回滚服务: $SERVICE_NAME"
    
    # 查找最新的备份镜像
    local backup_image=$(docker images --format "table {{.Repository}}:{{.Tag}}" | grep "stockschool_${SERVICE_NAME}_backup" | head -1)
    
    if [ -z "$backup_image" ]; then
        log_error "没有找到备份镜像，无法回滚"
        return 1
    fi
    
    log_info "使用备份镜像回滚: $backup_image"
    
    # 停止当前服务
    docker-compose -f $COMPOSE_FILE stop $SERVICE_NAME
    
    # 标记备份镜像为当前镜像
    local current_image_name=$(docker-compose -f $COMPOSE_FILE config | grep "image:" | grep $SERVICE_NAME -A1 | tail -1 | awk '{print $2}')
    docker tag $backup_image $current_image_name
    
    # 启动服务
    docker-compose -f $COMPOSE_FILE up -d $SERVICE_NAME
    
    # 健康检查
    if health_check; then
        log_success "回滚成功"
    else
        log_error "回滚后健康检查失败"
        return 1
    fi
}

# 更新多个服务
update_multiple_services() {
    local services=("$@")
    
    for service in "${services[@]}"; do
        SERVICE_NAME=$service
        check_service_exists
        backup_current_image
        build_new_image
        rolling_update_service
    done
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项] [服务名]"
    echo
    echo "选项:"
    echo "  -h, --help              显示帮助信息"
    echo "  -r, --rollback          回滚指定服务"
    echo "  -a, --all               更新所有应用服务"
    echo "  --no-rollback           失败时不自动回滚"
    echo
    echo "服务名:"
    echo "  api                     API服务"
    echo "  data_sync               数据同步服务"
    echo "  monitoring              监控服务"
    echo
    echo "示例:"
    echo "  $0 api                  滚动更新API服务"
    echo "  $0 -r api               回滚API服务"
    echo "  $0 -a                   更新所有服务"
}

# 主函数
main() {
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -r|--rollback)
            if [ -z "$2" ]; then
                log_error "请指定要回滚的服务名"
                exit 1
            fi
            SERVICE_NAME=$2
            check_service_exists
            rollback_service
            ;;
        -a|--all)
            log_info "开始更新所有应用服务..."
            update_multiple_services "api" "data_sync" "monitoring"
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE=false
            if [ -z "$2" ]; then
                log_error "请指定要更新的服务名"
                exit 1
            fi
            SERVICE_NAME=$2
            check_service_exists
            backup_current_image
            build_new_image
            rolling_update_service
            ;;
        "")
            log_error "请指定服务名或选项"
            show_help
            exit 1
            ;;
        *)
            SERVICE_NAME=$1
            check_service_exists
            backup_current_image
            build_new_image
            rolling_update_service
            ;;
    esac
}

# 错误处理
trap 'log_error "脚本执行失败，请检查日志"' ERR

# 执行主函数
main "$@"