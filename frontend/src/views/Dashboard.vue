<template>
  <div class="dashboard">
    <!-- 系统概览卡片 -->
    <el-row :gutter="20" class="overview-cards">
      <el-col :xs="24" :sm="12" :md="6" v-for="metric in overviewMetrics" :key="metric.key">
        <el-card class="metric-card" :class="`metric-${metric.type}`">
          <div class="metric-content">
            <div class="metric-icon">
              <el-icon size="32"><component :is="metric.icon" /></el-icon>
            </div>
            <div class="metric-info">
              <div class="metric-value">{{ metric.value }}</div>
              <div class="metric-label">{{ metric.label }}</div>
              <div class="metric-change" :class="metric.changeType">
                <el-icon size="12">
                  <ArrowUp v-if="metric.changeType === 'increase'" />
                  <ArrowDown v-if="metric.changeType === 'decrease'" />
                  <Minus v-if="metric.changeType === 'stable'" />
                </el-icon>
                {{ metric.change }}
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 系统状态和实时监控 -->
    <el-row :gutter="20" class="status-row">
      <!-- 系统健康状态 -->
      <el-col :xs="24" :md="12">
        <el-card class="status-card">
          <template #header>
            <div class="card-header">
              <span>系统健康状态</span>
              <el-button type="text" @click="refreshSystemStatus">
                <el-icon><Refresh /></el-icon>
              </el-button>
            </div>
          </template>
          <div class="system-status">
            <div 
              v-for="service in systemServices" 
              :key="service.name"
              class="service-item"
            >
              <div class="service-info">
                <span class="status-indicator" :class="`status-${service.status}`"></span>
                <span class="service-name">{{ service.name }}</span>
              </div>
              <div class="service-metrics">
                <span class="metric">响应时间: {{ service.responseTime }}ms</span>
                <span class="metric">CPU: {{ service.cpu }}%</span>
                <span class="metric">内存: {{ service.memory }}%</span>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>

      <!-- 数据同步状态 -->
      <el-col :xs="24" :md="12">
        <el-card class="status-card">
          <template #header>
            <div class="card-header">
              <span>数据同步状态</span>
              <el-button type="text" @click="refreshDataSync">
                <el-icon><Refresh /></el-icon>
              </el-button>
            </div>
          </template>
          <div class="sync-status">
            <div 
              v-for="source in dataSources" 
              :key="source.name"
              class="source-item"
            >
              <div class="source-header">
                <span class="status-indicator" :class="`status-${source.status}`"></span>
                <span class="source-name">{{ source.name }}</span>
                <el-tag :type="source.status === 'success' ? 'success' : 'danger'" size="small">
                  {{ source.statusText }}
                </el-tag>
              </div>
              <div class="source-details">
                <span>最后同步: {{ source.lastSync }}</span>
                <span>数据量: {{ source.recordCount }}</span>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 性能监控图表 -->
    <el-row :gutter="20" class="charts-row">
      <!-- API响应时间趋势 -->
      <el-col :xs="24" :md="12">
        <el-card class="chart-card">
          <template #header>
            <span>API响应时间趋势</span>
          </template>
          <div class="chart-container">
            <v-chart :option="apiResponseTimeOption" autoresize />
          </div>
        </el-card>
      </el-col>

      <!-- 系统资源使用率 -->
      <el-col :xs="24" :md="12">
        <el-card class="chart-card">
          <template #header>
            <span>系统资源使用率</span>
          </template>
          <div class="chart-container">
            <v-chart :option="systemResourceOption" autoresize />
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 告警信息 -->
    <el-row :gutter="20" class="alerts-row">
      <el-col :span="24">
        <el-card class="alerts-card">
          <template #header>
            <div class="card-header">
              <span>活跃告警</span>
              <el-badge :value="activeAlerts.length" :max="99" type="danger">
                <el-button type="text" @click="refreshAlerts">
                  <el-icon><Bell /></el-icon>
                </el-button>
              </el-badge>
            </div>
          </template>
          <div class="alerts-content">
            <el-empty v-if="activeAlerts.length === 0" description="暂无活跃告警" />
            <div v-else class="alerts-list">
              <div 
                v-for="alert in activeAlerts" 
                :key="alert.id"
                class="alert-item"
                :class="`alert-${alert.level}`"
              >
                <div class="alert-icon">
                  <el-icon size="20">
                    <Warning v-if="alert.level === 'warning'" />
                    <CircleClose v-if="alert.level === 'error'" />
                    <InfoFilled v-if="alert.level === 'info'" />
                  </el-icon>
                </div>
                <div class="alert-content">
                  <div class="alert-title">{{ alert.title }}</div>
                  <div class="alert-description">{{ alert.description }}</div>
                  <div class="alert-time">{{ alert.createdAt }}</div>
                </div>
                <div class="alert-actions">
                  <el-button size="small" type="text" @click="acknowledgeAlert(alert.id)">
                    确认
                  </el-button>
                </div>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, GaugeChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
} from 'echarts/components'
import VChart from 'vue-echarts'
import { apiService } from '@/services/api'
import { useWebSocket } from '@/composables/useWebSocket'

// 注册ECharts组件
use([
  CanvasRenderer,
  LineChart,
  GaugeChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
])

const { on, off } = useWebSocket()

// 概览指标
const overviewMetrics = ref([
  {
    key: 'totalStocks',
    label: '监控股票数',
    value: '4,856',
    change: '+12',
    changeType: 'increase',
    icon: 'TrendCharts',
    type: 'primary'
  },
  {
    key: 'activeStrategies',
    label: '活跃策略数',
    value: '23',
    change: '+3',
    changeType: 'increase',
    icon: 'DataAnalysis',
    type: 'success'
  },
  {
    key: 'dailyPredictions',
    label: '今日预测数',
    value: '4,856',
    change: '0',
    changeType: 'stable',
    icon: 'MagicStick',
    type: 'info'
  },
  {
    key: 'systemLoad',
    label: '系统负载',
    value: '68%',
    change: '+5%',
    changeType: 'increase',
    icon: 'Monitor',
    type: 'warning'
  }
])

// 系统服务状态
const systemServices = ref([
  {
    name: 'API服务',
    status: 'success',
    responseTime: 45,
    cpu: 25,
    memory: 68
  },
  {
    name: '数据库',
    status: 'success',
    responseTime: 12,
    cpu: 15,
    memory: 45
  },
  {
    name: 'Redis缓存',
    status: 'success',
    responseTime: 3,
    cpu: 8,
    memory: 32
  },
  {
    name: '模型服务',
    status: 'warning',
    responseTime: 156,
    cpu: 78,
    memory: 85
  }
])

// 数据源同步状态
const dataSources = ref([
  {
    name: 'AKShare行情数据',
    status: 'success',
    statusText: '正常',
    lastSync: '2024-01-15 09:30:00',
    recordCount: '4,856,234'
  },
  {
    name: '财务数据',
    status: 'success',
    statusText: '正常',
    lastSync: '2024-01-15 08:00:00',
    recordCount: '125,678'
  },
  {
    name: '因子数据',
    status: 'success',
    statusText: '正常',
    lastSync: '2024-01-15 09:25:00',
    recordCount: '2,345,123'
  }
])

// 活跃告警
const activeAlerts = ref([
  {
    id: '1',
    level: 'warning',
    title: '模型服务响应时间过长',
    description: '模型预测服务平均响应时间超过100ms阈值',
    createdAt: '2024-01-15 09:45:23'
  },
  {
    id: '2',
    level: 'info',
    title: '数据同步完成',
    description: 'AKShare数据源同步完成，共更新4,856条记录',
    createdAt: '2024-01-15 09:30:15'
  }
])

// API响应时间图表配置
const apiResponseTimeOption = ref({
  tooltip: {
    trigger: 'axis'
  },
  xAxis: {
    type: 'category',
    data: ['09:00', '09:15', '09:30', '09:45', '10:00', '10:15', '10:30']
  },
  yAxis: {
    type: 'value',
    name: '响应时间(ms)'
  },
  series: [
    {
      name: 'API响应时间',
      type: 'line',
      data: [45, 52, 48, 61, 55, 67, 58],
      smooth: true,
      itemStyle: {
        color: '#409eff'
      }
    }
  ]
})

// 系统资源使用率图表配置
const systemResourceOption = ref({
  tooltip: {
    formatter: '{a} <br/>{b} : {c}%'
  },
  series: [
    {
      name: 'CPU使用率',
      type: 'gauge',
      center: ['25%', '50%'],
      radius: '80%',
      data: [{ value: 68, name: 'CPU' }],
      detail: { fontSize: 14 },
      axisLine: {
        lineStyle: {
          color: [[0.7, '#67c23a'], [0.9, '#e6a23c'], [1, '#f56c6c']]
        }
      }
    },
    {
      name: '内存使用率',
      type: 'gauge',
      center: ['75%', '50%'],
      radius: '80%',
      data: [{ value: 45, name: '内存' }],
      detail: { fontSize: 14 },
      axisLine: {
        lineStyle: {
          color: [[0.7, '#67c23a'], [0.9, '#e6a23c'], [1, '#f56c6c']]
        }
      }
    }
  ]
})

// 刷新系统状态
const refreshSystemStatus = async () => {
  try {
    const health = await apiService.getSystemHealth()
    console.log('系统健康状态:', health)
    // 更新系统服务状态
  } catch (error) {
    console.error('获取系统状态失败:', error)
  }
}

// 刷新数据同步状态
const refreshDataSync = async () => {
  try {
    // 这里可以调用具体的数据同步状态API
    console.log('刷新数据同步状态')
  } catch (error) {
    console.error('获取数据同步状态失败:', error)
  }
}

// 刷新告警信息
const refreshAlerts = async () => {
  try {
    const alerts = await apiService.getAlerts({ status: 'active' })
    activeAlerts.value = alerts
  } catch (error) {
    console.error('获取告警信息失败:', error)
  }
}

// 确认告警
const acknowledgeAlert = (alertId: string) => {
  const index = activeAlerts.value.findIndex(alert => alert.id === alertId)
  if (index > -1) {
    activeAlerts.value.splice(index, 1)
  }
}

// 定时刷新数据
let refreshTimer: NodeJS.Timeout

const startAutoRefresh = () => {
  refreshTimer = setInterval(() => {
    refreshSystemStatus()
    refreshDataSync()
    refreshAlerts()
  }, 30000) // 每30秒刷新一次
}

const stopAutoRefresh = () => {
  if (refreshTimer) {
    clearInterval(refreshTimer)
  }
}

// 监听实时数据更新
const handleRealtimeUpdate = (data: any) => {
  console.log('收到实时数据更新:', data)
  // 更新相应的数据
}

const handleAlertUpdate = (alert: any) => {
  console.log('收到告警更新:', alert)
  activeAlerts.value.unshift(alert)
}

onMounted(() => {
  refreshSystemStatus()
  refreshDataSync()
  refreshAlerts()
  startAutoRefresh()
  
  // 监听WebSocket事件
  on('data_update', handleRealtimeUpdate)
  on('alert_update', handleAlertUpdate)
})

onUnmounted(() => {
  stopAutoRefresh()
  
  // 移除WebSocket事件监听
  off('data_update', handleRealtimeUpdate)
  off('alert_update', handleAlertUpdate)
})
</script>

<style scoped>
.dashboard {
  padding: 0;
}

.overview-cards {
  margin-bottom: 20px;
}

.metric-card {
  border: none;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.metric-content {
  display: flex;
  align-items: center;
  gap: 15px;
}

.metric-icon {
  padding: 15px;
  border-radius: 8px;
  background: rgba(64, 158, 255, 0.1);
  color: #409eff;
}

.metric-primary .metric-icon {
  background: rgba(64, 158, 255, 0.1);
  color: #409eff;
}

.metric-success .metric-icon {
  background: rgba(103, 194, 58, 0.1);
  color: #67c23a;
}

.metric-info .metric-icon {
  background: rgba(144, 147, 153, 0.1);
  color: #909399;
}

.metric-warning .metric-icon {
  background: rgba(230, 162, 60, 0.1);
  color: #e6a23c;
}

.metric-info {
  flex: 1;
}

.metric-value {
  font-size: 24px;
  font-weight: bold;
  color: #303133;
  margin-bottom: 5px;
}

.metric-label {
  font-size: 14px;
  color: #606266;
  margin-bottom: 5px;
}

.metric-change {
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 2px;
}

.metric-change.increase {
  color: #67c23a;
}

.metric-change.decrease {
  color: #f56c6c;
}

.metric-change.stable {
  color: #909399;
}

.status-row,
.charts-row,
.alerts-row {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.system-status,
.sync-status {
  space-y: 15px;
}

.service-item,
.source-item {
  padding: 15px;
  border: 1px solid #ebeef5;
  border-radius: 6px;
  margin-bottom: 10px;
}

.service-info,
.source-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}

.service-metrics,
.source-details {
  display: flex;
  gap: 15px;
  font-size: 12px;
  color: #909399;
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}

.status-success {
  background-color: #67c23a;
}

.status-warning {
  background-color: #e6a23c;
}

.status-danger {
  background-color: #f56c6c;
}

.chart-container {
  height: 300px;
}

.alerts-content {
  max-height: 400px;
  overflow-y: auto;
}

.alerts-list {
  space-y: 10px;
}

.alert-item {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 15px;
  border-radius: 6px;
  border-left: 4px solid;
}

.alert-warning {
  background-color: #fdf6ec;
  border-left-color: #e6a23c;
}

.alert-error {
  background-color: #fef0f0;
  border-left-color: #f56c6c;
}

.alert-info {
  background-color: #f4f4f5;
  border-left-color: #909399;
}

.alert-icon {
  margin-top: 2px;
}

.alert-content {
  flex: 1;
}

.alert-title {
  font-weight: 500;
  color: #303133;
  margin-bottom: 5px;
}

.alert-description {
  font-size: 14px;
  color: #606266;
  margin-bottom: 5px;
}

.alert-time {
  font-size: 12px;
  color: #909399;
}

.alert-actions {
  margin-top: 5px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .service-metrics,
  .source-details {
    flex-direction: column;
    gap: 5px;
  }
  
  .chart-container {
    height: 250px;
  }
}
</style>