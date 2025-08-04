#!/bin/bash
# StockSchool监控系统设置脚本

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

# 创建Grafana仪表板
create_grafana_dashboards() {
    log_info "创建Grafana仪表板..."
    
    # 系统概览仪表板
    cat > monitoring/grafana/dashboards/json/system_overview.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "StockSchool系统概览",
    "tags": ["stockschool", "overview"],
    "style": "dark",
    "timezone": "Asia/Shanghai",
    "panels": [
      {
        "id": 1,
        "title": "服务状态",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=~\"stockschool.*\"}",
            "legendFormat": "{{job}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "green", "value": 1}
              ]
            },
            "mappings": [
              {"options": {"0": {"text": "Down"}}, "type": "value"},
              {"options": {"1": {"text": "Up"}}, "type": "value"}
            ]
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "API请求率 (QPS)",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"stockschool-api\"}[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ],
        "yAxes": [
          {"label": "请求/秒", "min": 0}
        ],
        "gridPos": {"h": 8, "w": 9, "x": 6, "y": 0}
      },
      {
        "id": 3,
        "title": "API响应时间",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"stockschool-api\"}[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket{job=\"stockschool-api\"}[5m]))",
            "legendFormat": "50th percentile"
          }
        ],
        "yAxes": [
          {"label": "秒", "min": 0}
        ],
        "gridPos": {"h": 8, "w": 9, "x": 15, "y": 0}
      },
      {
        "id": 4,
        "title": "数据库连接数",
        "type": "graph",
        "targets": [
          {
            "expr": "pg_stat_activity_count",
            "legendFormat": "活跃连接"
          },
          {
            "expr": "pg_settings_max_connections",
            "legendFormat": "最大连接数"
          }
        ],
        "yAxes": [
          {"label": "连接数", "min": 0}
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "id": 5,
        "title": "Redis内存使用",
        "type": "graph",
        "targets": [
          {
            "expr": "redis_memory_used_bytes",
            "legendFormat": "已使用内存"
          },
          {
            "expr": "redis_memory_max_bytes",
            "legendFormat": "最大内存"
          }
        ],
        "yAxes": [
          {"label": "字节", "min": 0}
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      },
      {
        "id": 6,
        "title": "系统资源使用",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg by(instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU使用率 %"
          },
          {
            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
            "legendFormat": "内存使用率 %"
          }
        ],
        "yAxes": [
          {"label": "百分比", "min": 0, "max": 100}
        ],
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16}
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
EOF

    # 数据同步监控仪表板
    cat > monitoring/grafana/dashboards/json/data_sync_monitoring.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "数据同步监控",
    "tags": ["stockschool", "datasync"],
    "style": "dark",
    "timezone": "Asia/Shanghai",
    "panels": [
      {
        "id": 1,
        "title": "同步状态",
        "type": "stat",
        "targets": [
          {
            "expr": "stockschool_sync_status",
            "legendFormat": "{{data_source}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 1},
                {"color": "green", "value": 2}
              ]
            },
            "mappings": [
              {"options": {"0": {"text": "失败"}}, "type": "value"},
              {"options": {"1": {"text": "进行中"}}, "type": "value"},
              {"options": {"2": {"text": "成功"}}, "type": "value"}
            ]
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "同步成功率",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(stockschool_sync_success_total[5m]) / (rate(stockschool_sync_success_total[5m]) + rate(stockschool_sync_failures_total[5m]))",
            "legendFormat": "成功率"
          }
        ],
        "yAxes": [
          {"label": "百分比", "min": 0, "max": 1}
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "数据质量评分",
        "type": "graph",
        "targets": [
          {
            "expr": "stockschool_data_quality_score",
            "legendFormat": "{{table}}"
          }
        ],
        "yAxes": [
          {"label": "评分", "min": 0, "max": 1}
        ],
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
      }
    ],
    "time": {
      "from": "now-6h",
      "to": "now"
    },
    "refresh": "1m"
  }
}
EOF

    log_success "Grafana仪表板创建完成"
}

# 配置Prometheus告警规则
setup_prometheus_alerts() {
    log_info "配置Prometheus告警规则..."
    
    # 业务告警规则
    cat > monitoring/prometheus/rules/business_alerts.yml << 'EOF'
groups:
  - name: stockschool.business
    rules:
      # 数据同步延迟告警
      - alert: DataSyncDelay
        expr: time() - stockschool_last_sync_timestamp > 7200
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "数据同步延迟"
          description: "数据同步延迟超过2小时，当前延迟: {{ $value }}秒"

      # 数据质量告警
      - alert: DataQualityDegraded
        expr: stockschool_data_quality_score < 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "数据质量下降"
          description: "数据质量评分低于0.8，当前评分: {{ $value }}"

      # API错误率告警
      - alert: HighAPIErrorRate
        expr: rate(http_requests_total{job="stockschool-api",status=~"5.."}[5m]) / rate(http_requests_total{job="stockschool-api"}[5m]) > 0.05
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "API错误率过高"
          description: "API 5xx错误率超过5%，当前值: {{ $value | humanizePercentage }}"

      # 因子计算失败告警
      - alert: FactorCalculationFailure
        expr: increase(stockschool_factor_calculation_failures_total[1h]) > 5
        for: 0m
        labels:
          severity: warning
        annotations:
          summary: "因子计算失败"
          description: "过去1小时内因子计算失败次数超过5次"

      # 用户活跃度异常告警
      - alert: UserActivityAnomaly
        expr: rate(http_requests_total{job="stockschool-api"}[1h]) < 0.1
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "用户活跃度异常"
          description: "API请求率异常低，可能存在问题"
EOF

    log_success "Prometheus告警规则配置完成"
}

# 配置AlertManager通知模板
setup_alertmanager_templates() {
    log_info "配置AlertManager通知模板..."
    
    mkdir -p monitoring/alertmanager/templates
    
    cat > monitoring/alertmanager/templates/email.tmpl << 'EOF'
{{ define "email.subject" }}
[{{ .Status | toUpper }}] StockSchool告警 - {{ .GroupLabels.alertname }}
{{ end }}

{{ define "email.html" }}
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>StockSchool告警通知</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
        .alert { margin: 10px 0; padding: 15px; border-radius: 5px; }
        .critical { background-color: #f8d7da; border: 1px solid #f5c6cb; }
        .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; }
        .resolved { background-color: #d4edda; border: 1px solid #c3e6cb; }
        .footer { margin-top: 20px; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <h2>🚨 StockSchool系统告警通知</h2>
        <p>告警时间: {{ .CommonAnnotations.timestamp | default "未知" }}</p>
    </div>
    
    {{ range .Alerts }}
    <div class="alert {{ if eq .Status "firing" }}{{ if eq .Labels.severity "critical" }}critical{{ else }}warning{{ end }}{{ else }}resolved{{ end }}">
        <h3>{{ .Annotations.summary }}</h3>
        <p><strong>描述:</strong> {{ .Annotations.description }}</p>
        <p><strong>级别:</strong> {{ .Labels.severity | title }}</p>
        <p><strong>服务:</strong> {{ .Labels.job | default "未知" }}</p>
        <p><strong>实例:</strong> {{ .Labels.instance | default "未知" }}</p>
        <p><strong>状态:</strong> {{ .Status | title }}</p>
        {{ if eq .Status "firing" }}
        <p><strong>开始时间:</strong> {{ .StartsAt.Format "2006-01-02 15:04:05" }}</p>
        {{ else }}
        <p><strong>恢复时间:</strong> {{ .EndsAt.Format "2006-01-02 15:04:05" }}</p>
        {{ end }}
    </div>
    {{ end }}
    
    <div class="footer">
        <p>此邮件由StockSchool监控系统自动发送</p>
        <p>如需帮助，请联系技术支持团队</p>
    </div>
</body>
</html>
{{ end }}
EOF

    log_success "AlertManager通知模板配置完成"
}

# 设置监控数据保留策略
setup_retention_policies() {
    log_info "设置监控数据保留策略..."
    
    # Prometheus数据保留策略已在docker-compose中配置
    # 这里可以添加其他保留策略配置
    
    log_success "监控数据保留策略设置完成"
}

# 创建监控脚本
create_monitoring_scripts() {
    log_info "创建监控脚本..."
    
    # 系统健康检查脚本
    cat > scripts/monitoring/health_check.sh << 'EOF'
#!/bin/bash
# 系统健康检查脚本

# 检查服务状态
echo "=== 服务状态检查 ==="
docker-compose -f docker-compose.prod.yml ps

# 检查关键端口
echo -e "\n=== 端口检查 ==="
for port in 80 443 5432 6379 9090 3000; do
    if netstat -tlnp | grep -q ":$port "; then
        echo "✅ 端口 $port 正常"
    else
        echo "❌ 端口 $port 异常"
    fi
done

# 检查磁盘空间
echo -e "\n=== 磁盘空间检查 ==="
df -h | grep -E "(/$|/var|/tmp)"

# 检查内存使用
echo -e "\n=== 内存使用检查 ==="
free -h

# 检查API健康状态
echo -e "\n=== API健康检查 ==="
if curl -f -s https://localhost/api/health > /dev/null; then
    echo "✅ API服务正常"
else
    echo "❌ API服务异常"
fi

# 检查数据库连接
echo -e "\n=== 数据库连接检查 ==="
if docker-compose -f docker-compose.prod.yml exec -T postgres pg_isready -U $POSTGRES_USER -d $POSTGRES_DB > /dev/null 2>&1; then
    echo "✅ 数据库连接正常"
else
    echo "❌ 数据库连接异常"
fi
EOF

    chmod +x scripts/monitoring/health_check.sh
    
    # 性能监控脚本
    cat > scripts/monitoring/performance_check.sh << 'EOF'
#!/bin/bash
# 性能监控脚本

echo "=== StockSchool性能监控报告 ==="
echo "生成时间: $(date)"
echo

# CPU使用率
echo "=== CPU使用率 ==="
top -bn1 | grep "Cpu(s)" | awk '{print "CPU使用率: " $2}' | sed 's/%us,//'

# 内存使用率
echo -e "\n=== 内存使用率 ==="
free | awk 'NR==2{printf "内存使用率: %.2f%%\n", $3*100/$2}'

# 磁盘I/O
echo -e "\n=== 磁盘I/O ==="
iostat -x 1 1 | tail -n +4

# 网络连接
echo -e "\n=== 网络连接统计 ==="
netstat -an | awk '/^tcp/ {++state[$NF]} END {for(key in state) print key, state[key]}'

# Docker容器资源使用
echo -e "\n=== 容器资源使用 ==="
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# API响应时间测试
echo -e "\n=== API响应时间测试 ==="
curl -w "响应时间: %{time_total}s\n" -o /dev/null -s https://localhost/api/health
EOF

    chmod +x scripts/monitoring/performance_check.sh
    
    log_success "监控脚本创建完成"
}

# 主函数
main() {
    log_info "开始设置StockSchool监控系统..."
    
    # 创建监控脚本目录
    mkdir -p scripts/monitoring
    
    # 执行设置步骤
    create_grafana_dashboards
    setup_prometheus_alerts
    setup_alertmanager_templates
    setup_retention_policies
    create_monitoring_scripts
    
    log_success "监控系统设置完成！"
    
    echo
    echo "下一步操作："
    echo "1. 启动监控服务: docker-compose -f docker-compose.prod.yml up -d prometheus grafana alertmanager"
    echo "2. 访问Grafana: https://localhost/grafana (admin/\$GRAFANA_PASSWORD)"
    echo "3. 访问Prometheus: https://localhost/prometheus"
    echo "4. 运行健康检查: ./scripts/monitoring/health_check.sh"
}

# 执行主函数
main "$@"