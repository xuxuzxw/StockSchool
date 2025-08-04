import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { apiService } from '@/services/api'

export interface SystemHealth {
  status: 'healthy' | 'warning' | 'error'
  uptime: number
  version: string
  lastUpdate: string
  services: {
    database: 'online' | 'offline' | 'degraded'
    redis: 'online' | 'offline' | 'degraded'
    api: 'online' | 'offline' | 'degraded'
    websocket: 'online' | 'offline' | 'degraded'
    ml_service: 'online' | 'offline' | 'degraded'
  }
  performance: {
    cpu_usage: number
    memory_usage: number
    disk_usage: number
    response_time: number
  }
}

export interface SystemAlert {
  id: string
  type: 'info' | 'warning' | 'error' | 'success'
  title: string
  message: string
  timestamp: string
  read: boolean
  category: string
  source: string
  actions?: {
    label: string
    action: string
  }[]
}

export interface DataSyncStatus {
  lastSync: string
  nextSync: string
  status: 'syncing' | 'completed' | 'failed' | 'pending'
  progress: number
  sources: {
    name: string
    status: 'syncing' | 'completed' | 'failed' | 'pending'
    lastUpdate: string
    recordCount: number
  }[]
}

export interface SystemMetrics {
  api_requests: {
    total: number
    success_rate: number
    avg_response_time: number
    error_count: number
  }
  user_activity: {
    active_users: number
    total_sessions: number
    avg_session_duration: number
  }
  data_processing: {
    processed_records: number
    processing_rate: number
    queue_size: number
  }
  model_performance: {
    prediction_accuracy: number
    model_latency: number
    predictions_count: number
  }
}

export const useSystemStore = defineStore('system', () => {
  // 状态
  const health = ref<SystemHealth | null>(null)
  const alerts = ref<SystemAlert[]>([])
  const dataSyncStatus = ref<DataSyncStatus | null>(null)
  const metrics = ref<SystemMetrics | null>(null)
  const loading = ref<boolean>(false)
  const lastHealthCheck = ref<string>('')
  const autoRefresh = ref<boolean>(true)
  const refreshInterval = ref<number>(30000) // 30秒
  
  // 计算属性
  const systemStatus = computed(() => {
    if (!health.value) return 'unknown'
    return health.value.status
  })

  const unreadAlertsCount = computed(() => {
    return alerts.value.filter(alert => !alert.read).length
  })

  const criticalAlertsCount = computed(() => {
    return alerts.value.filter(alert => alert.type === 'error' && !alert.read).length
  })

  const servicesStatus = computed(() => {
    if (!health.value) return {}
    return health.value.services
  })

  const performanceMetrics = computed(() => {
    if (!health.value) return null
    return health.value.performance
  })

  const isSystemHealthy = computed(() => {
    return systemStatus.value === 'healthy'
  })

  const syncProgress = computed(() => {
    return dataSyncStatus.value?.progress || 0
  })

  const isSyncing = computed(() => {
    return dataSyncStatus.value?.status === 'syncing'
  })

  // 方法
  const fetchSystemHealth = async () => {
    try {
      const response = await apiService.getSystemHealth()
      health.value = response.health
      lastHealthCheck.value = new Date().toISOString()
      return response
    } catch (error) {
      console.error('获取系统健康状态失败:', error)
      throw error
    }
  }

  const fetchSystemAlerts = async (params?: {
    page?: number
    pageSize?: number
    type?: string
    unreadOnly?: boolean
  }) => {
    try {
      const response = await apiService.getSystemAlerts(params)
      alerts.value = response.alerts
      return response
    } catch (error) {
      console.error('获取系统告警失败:', error)
      throw error
    }
  }

  const fetchDataSyncStatus = async () => {
    try {
      const response = await apiService.getDataSyncStatus()
      dataSyncStatus.value = response.syncStatus
      return response
    } catch (error) {
      console.error('获取数据同步状态失败:', error)
      throw error
    }
  }

  const fetchSystemMetrics = async (timeRange?: string) => {
    try {
      const response = await apiService.getSystemMetrics({ timeRange })
      metrics.value = response.metrics
      return response
    } catch (error) {
      console.error('获取系统指标失败:', error)
      throw error
    }
  }

  const markAlertAsRead = async (alertId: string) => {
    try {
      await apiService.markAlertAsRead(alertId)
      const alert = alerts.value.find(a => a.id === alertId)
      if (alert) {
        alert.read = true
      }
    } catch (error) {
      console.error('标记告警为已读失败:', error)
      throw error
    }
  }

  const markAllAlertsAsRead = async () => {
    try {
      await apiService.markAllAlertsAsRead()
      alerts.value.forEach(alert => {
        alert.read = true
      })
    } catch (error) {
      console.error('标记所有告警为已读失败:', error)
      throw error
    }
  }

  const dismissAlert = async (alertId: string) => {
    try {
      await apiService.dismissAlert(alertId)
      const index = alerts.value.findIndex(a => a.id === alertId)
      if (index !== -1) {
        alerts.value.splice(index, 1)
      }
    } catch (error) {
      console.error('忽略告警失败:', error)
      throw error
    }
  }

  const triggerDataSync = async (sources?: string[]) => {
    try {
      const response = await apiService.triggerDataSync({ sources })
      await fetchDataSyncStatus()
      return response
    } catch (error) {
      console.error('触发数据同步失败:', error)
      throw error
    }
  }

  const restartService = async (serviceName: string) => {
    try {
      const response = await apiService.restartService(serviceName)
      // 重新获取系统健康状态
      setTimeout(() => {
        fetchSystemHealth()
      }, 5000)
      return response
    } catch (error) {
      console.error('重启服务失败:', error)
      throw error
    }
  }

  const getSystemLogs = async (params?: {
    level?: string
    service?: string
    startTime?: string
    endTime?: string
    page?: number
    pageSize?: number
  }) => {
    try {
      const response = await apiService.getSystemLogs(params)
      return response
    } catch (error) {
      console.error('获取系统日志失败:', error)
      throw error
    }
  }

  const exportSystemReport = async (params?: {
    startDate?: string
    endDate?: string
    includeMetrics?: boolean
    includeAlerts?: boolean
    includeLogs?: boolean
  }) => {
    try {
      const response = await apiService.exportSystemReport(params)
      return response
    } catch (error) {
      console.error('导出系统报告失败:', error)
      throw error
    }
  }

  const updateSystemConfig = async (config: Record<string, any>) => {
    try {
      const response = await apiService.updateSystemConfig(config)
      return response
    } catch (error) {
      console.error('更新系统配置失败:', error)
      throw error
    }
  }

  const getSystemConfig = async () => {
    try {
      const response = await apiService.getSystemConfig()
      return response
    } catch (error) {
      console.error('获取系统配置失败:', error)
      throw error
    }
  }

  const refreshAllData = async () => {
    loading.value = true
    try {
      await Promise.all([
        fetchSystemHealth(),
        fetchSystemAlerts({ unreadOnly: false }),
        fetchDataSyncStatus(),
        fetchSystemMetrics()
      ])
    } catch (error) {
      console.error('刷新系统数据失败:', error)
      throw error
    } finally {
      loading.value = false
    }
  }

  const startAutoRefresh = () => {
    if (autoRefresh.value) {
      const interval = setInterval(() => {
        if (autoRefresh.value) {
          refreshAllData()
        } else {
          clearInterval(interval)
        }
      }, refreshInterval.value)
    }
  }

  const stopAutoRefresh = () => {
    autoRefresh.value = false
  }

  const setRefreshInterval = (interval: number) => {
    refreshInterval.value = interval
    if (autoRefresh.value) {
      stopAutoRefresh()
      autoRefresh.value = true
      startAutoRefresh()
    }
  }

  // 添加新告警
  const addAlert = (alert: Omit<SystemAlert, 'id' | 'timestamp' | 'read'>) => {
    const newAlert: SystemAlert = {
      ...alert,
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      read: false
    }
    alerts.value.unshift(newAlert)
  }

  // 更新服务状态
  const updateServiceStatus = (serviceName: string, status: 'online' | 'offline' | 'degraded') => {
    if (health.value?.services) {
      health.value.services[serviceName as keyof typeof health.value.services] = status
    }
  }

  // 更新性能指标
  const updatePerformanceMetrics = (newMetrics: Partial<SystemHealth['performance']>) => {
    if (health.value?.performance) {
      health.value.performance = { ...health.value.performance, ...newMetrics }
    }
  }

  // 获取服务状态颜色
  const getServiceStatusColor = (status: string) => {
    const colorMap: Record<string, string> = {
      online: '#67c23a',
      offline: '#f56c6c',
      degraded: '#e6a23c',
      unknown: '#909399'
    }
    return colorMap[status] || colorMap.unknown
  }

  // 获取告警类型图标
  const getAlertTypeIcon = (type: string) => {
    const iconMap: Record<string, string> = {
      info: 'InfoFilled',
      warning: 'WarningFilled',
      error: 'CircleCloseFilled',
      success: 'CircleCheckFilled'
    }
    return iconMap[type] || iconMap.info
  }

  return {
    // 状态
    health,
    alerts,
    dataSyncStatus,
    metrics,
    loading,
    lastHealthCheck,
    autoRefresh,
    refreshInterval,
    
    // 计算属性
    systemStatus,
    unreadAlertsCount,
    criticalAlertsCount,
    servicesStatus,
    performanceMetrics,
    isSystemHealthy,
    syncProgress,
    isSyncing,
    
    // 方法
    fetchSystemHealth,
    fetchSystemAlerts,
    fetchDataSyncStatus,
    fetchSystemMetrics,
    markAlertAsRead,
    markAllAlertsAsRead,
    dismissAlert,
    triggerDataSync,
    restartService,
    getSystemLogs,
    exportSystemReport,
    updateSystemConfig,
    getSystemConfig,
    refreshAllData,
    startAutoRefresh,
    stopAutoRefresh,
    setRefreshInterval,
    addAlert,
    updateServiceStatus,
    updatePerformanceMetrics,
    getServiceStatusColor,
    getAlertTypeIcon
  }
})