import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useWebSocket } from '@/composables/useWebSocket'
import { useSystemStore } from './system'
import { useDataStore } from './data'

export interface WebSocketMessage {
  id: string
  type: string
  data: any
  timestamp: string
  processed: boolean
}

export interface ConnectionStatus {
  connected: boolean
  connecting: boolean
  lastConnected?: string
  lastDisconnected?: string
  reconnectAttempts: number
  maxReconnectAttempts: number
}

export interface SubscriptionConfig {
  dataUpdates: boolean
  alertUpdates: boolean
  systemUpdates: boolean
  backtestUpdates: boolean
  modelUpdates: boolean
  customChannels: string[]
}

export const useWebSocketStore = defineStore('websocket', () => {
  // 状态
  const messages = ref<WebSocketMessage[]>([])
  const connectionStatus = ref<ConnectionStatus>({
    connected: false,
    connecting: false,
    reconnectAttempts: 0,
    maxReconnectAttempts: 5
  })
  
  const subscriptions = ref<SubscriptionConfig>({
    dataUpdates: true,
    alertUpdates: true,
    systemUpdates: true,
    backtestUpdates: true,
    modelUpdates: true,
    customChannels: []
  })
  
  const messageFilters = ref({
    types: [] as string[],
    dateRange: {
      start: '',
      end: ''
    },
    processed: null as boolean | null
  })
  
  const autoReconnect = ref(true)
  const reconnectInterval = ref(5000) // 5秒
  const messageRetention = ref(1000) // 保留最近1000条消息
  
  // WebSocket实例
  const { 
    isConnected, 
    isConnecting, 
    connect, 
    disconnect, 
    sendMessage,
    on,
    off
  } = useWebSocket()
  
  // 其他store
  const systemStore = useSystemStore()
  const dataStore = useDataStore()

  // 计算属性
  const filteredMessages = computed(() => {
    let result = messages.value
    
    // 按类型筛选
    if (messageFilters.value.types.length > 0) {
      result = result.filter(msg => messageFilters.value.types.includes(msg.type))
    }
    
    // 按处理状态筛选
    if (messageFilters.value.processed !== null) {
      result = result.filter(msg => msg.processed === messageFilters.value.processed)
    }
    
    // 按日期范围筛选
    if (messageFilters.value.dateRange.start) {
      result = result.filter(msg => msg.timestamp >= messageFilters.value.dateRange.start)
    }
    if (messageFilters.value.dateRange.end) {
      result = result.filter(msg => msg.timestamp <= messageFilters.value.dateRange.end)
    }
    
    return result.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
  })

  const unprocessedMessages = computed(() => {
    return messages.value.filter(msg => !msg.processed)
  })

  const messagesByType = computed(() => {
    const grouped: Record<string, WebSocketMessage[]> = {}
    messages.value.forEach(msg => {
      if (!grouped[msg.type]) {
        grouped[msg.type] = []
      }
      grouped[msg.type].push(msg)
    })
    return grouped
  })

  const connectionHealth = computed(() => {
    const { connected, reconnectAttempts, maxReconnectAttempts } = connectionStatus.value
    
    if (connected) return 'healthy'
    if (reconnectAttempts >= maxReconnectAttempts) return 'failed'
    if (reconnectAttempts > 0) return 'reconnecting'
    return 'disconnected'
  })

  const messageTypes = computed(() => {
    const types = new Set(messages.value.map(msg => msg.type))
    return Array.from(types)
  })

  // 方法
  const initializeWebSocket = () => {
    // 更新连接状态
    connectionStatus.value.connected = isConnected.value
    connectionStatus.value.connecting = isConnecting.value
    
    // 监听连接状态变化
    const updateConnectionStatus = () => {
      connectionStatus.value.connected = isConnected.value
      connectionStatus.value.connecting = isConnecting.value
      
      if (isConnected.value) {
        connectionStatus.value.lastConnected = new Date().toISOString()
        connectionStatus.value.reconnectAttempts = 0
        
        // 连接成功后订阅频道
        subscribeToChannels()
      } else {
        connectionStatus.value.lastDisconnected = new Date().toISOString()
        
        // 自动重连
        if (autoReconnect.value && connectionStatus.value.reconnectAttempts < connectionStatus.value.maxReconnectAttempts) {
          setTimeout(() => {
            connectionStatus.value.reconnectAttempts++
            connectWebSocket()
          }, reconnectInterval.value)
        }
      }
    }
    
    // 监听消息
    on('data_update', handleDataUpdate)
    on('alert_update', handleAlertUpdate)
    on('system_update', handleSystemUpdate)
    on('backtest_update', handleBacktestUpdate)
    on('model_update', handleModelUpdate)
    on('custom_message', handleCustomMessage)
    
    // 监听连接事件
    on('connect', updateConnectionStatus)
    on('disconnect', updateConnectionStatus)
    on('reconnect', updateConnectionStatus)
    
    return updateConnectionStatus
  }

  const connectWebSocket = async () => {
    try {
      connectionStatus.value.connecting = true
      await connect()
    } catch (error) {
      console.error('WebSocket连接失败:', error)
      connectionStatus.value.connecting = false
      throw error
    }
  }

  const disconnectWebSocket = () => {
    disconnect()
    connectionStatus.value.connected = false
    connectionStatus.value.connecting = false
  }

  const subscribeToChannels = () => {
    const { dataUpdates, alertUpdates, systemUpdates, backtestUpdates, modelUpdates, customChannels } = subscriptions.value
    
    const subscribeConfig = {
      data_updates: dataUpdates,
      alert_updates: alertUpdates,
      system_updates: systemUpdates,
      backtest_updates: backtestUpdates,
      model_updates: modelUpdates,
      custom_channels: customChannels
    }
    
    sendMessage('subscribe', subscribeConfig)
  }

  const unsubscribeFromChannels = (channels: string[]) => {
    sendMessage('unsubscribe', { channels })
  }

  const updateSubscriptions = (updates: Partial<SubscriptionConfig>) => {
    subscriptions.value = { ...subscriptions.value, ...updates }
    
    if (connectionStatus.value.connected) {
      subscribeToChannels()
    }
  }

  const addMessage = (type: string, data: any) => {
    const message: WebSocketMessage = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      data,
      timestamp: new Date().toISOString(),
      processed: false
    }
    
    messages.value.unshift(message)
    
    // 限制消息数量
    if (messages.value.length > messageRetention.value) {
      messages.value = messages.value.slice(0, messageRetention.value)
    }
    
    return message
  }

  const markMessageAsProcessed = (messageId: string) => {
    const message = messages.value.find(msg => msg.id === messageId)
    if (message) {
      message.processed = true
    }
  }

  const markAllMessagesAsProcessed = () => {
    messages.value.forEach(msg => {
      msg.processed = true
    })
  }

  const deleteMessage = (messageId: string) => {
    const index = messages.value.findIndex(msg => msg.id === messageId)
    if (index !== -1) {
      messages.value.splice(index, 1)
    }
  }

  const clearMessages = (type?: string) => {
    if (type) {
      messages.value = messages.value.filter(msg => msg.type !== type)
    } else {
      messages.value = []
    }
  }

  const sendCustomMessage = (type: string, data: any) => {
    sendMessage(type, data)
  }

  const requestData = (dataType: string, params?: any) => {
    sendMessage('request_data', { type: dataType, params })
  }

  const updateMessageFilters = (filters: Partial<typeof messageFilters.value>) => {
    messageFilters.value = { ...messageFilters.value, ...filters }
  }

  const resetMessageFilters = () => {
    messageFilters.value = {
      types: [],
      dateRange: { start: '', end: '' },
      processed: null
    }
  }

  // 消息处理器
  const handleDataUpdate = (data: any) => {
    const message = addMessage('data_update', data)
    
    // 更新数据store
    if (data.type === 'stock_update') {
      dataStore.fetchStockList()
    } else if (data.type === 'factor_update') {
      dataStore.fetchFactorList()
    }
    
    markMessageAsProcessed(message.id)
  }

  const handleAlertUpdate = (data: any) => {
    const message = addMessage('alert_update', data)
    
    // 更新系统store的告警信息
    systemStore.fetchAlerts()
    
    // 显示通知
    if (data.level === 'critical' || data.level === 'error') {
      // 这里可以集成通知组件
      console.warn('收到重要告警:', data.message)
    }
    
    markMessageAsProcessed(message.id)
  }

  const handleSystemUpdate = (data: any) => {
    const message = addMessage('system_update', data)
    
    // 更新系统状态
    if (data.type === 'health_update') {
      systemStore.fetchSystemHealth()
    } else if (data.type === 'metrics_update') {
      systemStore.fetchSystemMetrics()
    }
    
    markMessageAsProcessed(message.id)
  }

  const handleBacktestUpdate = (data: any) => {
    const message = addMessage('backtest_update', data)
    
    // 更新回测结果
    if (data.type === 'backtest_completed' || data.type === 'backtest_failed') {
      dataStore.fetchBacktestResults()
    }
    
    markMessageAsProcessed(message.id)
  }

  const handleModelUpdate = (data: any) => {
    const message = addMessage('model_update', data)
    
    // 更新模型解释
    if (data.type === 'model_retrained' || data.type === 'explanation_updated') {
      dataStore.fetchModelExplanations()
    }
    
    markMessageAsProcessed(message.id)
  }

  const handleCustomMessage = (data: any) => {
    addMessage('custom_message', data)
  }

  // 配置方法
  const updateConnectionConfig = (config: {
    autoReconnect?: boolean
    reconnectInterval?: number
    maxReconnectAttempts?: number
    messageRetention?: number
  }) => {
    if (config.autoReconnect !== undefined) {
      autoReconnect.value = config.autoReconnect
    }
    if (config.reconnectInterval !== undefined) {
      reconnectInterval.value = config.reconnectInterval
    }
    if (config.maxReconnectAttempts !== undefined) {
      connectionStatus.value.maxReconnectAttempts = config.maxReconnectAttempts
    }
    if (config.messageRetention !== undefined) {
      messageRetention.value = config.messageRetention
    }
  }

  const getConnectionStats = () => {
    return {
      connected: connectionStatus.value.connected,
      totalMessages: messages.value.length,
      unprocessedMessages: unprocessedMessages.value.length,
      messagesByType: Object.keys(messagesByType.value).map(type => ({
        type,
        count: messagesByType.value[type].length
      })),
      lastConnected: connectionStatus.value.lastConnected,
      lastDisconnected: connectionStatus.value.lastDisconnected,
      reconnectAttempts: connectionStatus.value.reconnectAttempts
    }
  }

  const exportMessages = (format: 'json' | 'csv' = 'json') => {
    const data = filteredMessages.value
    
    if (format === 'json') {
      return JSON.stringify(data, null, 2)
    } else {
      // CSV格式
      const headers = ['ID', 'Type', 'Timestamp', 'Processed', 'Data']
      const rows = data.map(msg => [
        msg.id,
        msg.type,
        msg.timestamp,
        msg.processed.toString(),
        JSON.stringify(msg.data)
      ])
      
      return [headers, ...rows].map(row => row.join(',')).join('\n')
    }
  }

  // 清理方法
  const cleanup = () => {
    // 移除所有事件监听器
    off('data_update', handleDataUpdate)
    off('alert_update', handleAlertUpdate)
    off('system_update', handleSystemUpdate)
    off('backtest_update', handleBacktestUpdate)
    off('model_update', handleModelUpdate)
    off('custom_message', handleCustomMessage)
    
    // 断开连接
    disconnectWebSocket()
    
    // 清理消息
    clearMessages()
  }

  return {
    // 状态
    messages,
    connectionStatus,
    subscriptions,
    messageFilters,
    autoReconnect,
    reconnectInterval,
    messageRetention,
    
    // 计算属性
    filteredMessages,
    unprocessedMessages,
    messagesByType,
    connectionHealth,
    messageTypes,
    
    // 方法
    initializeWebSocket,
    connectWebSocket,
    disconnectWebSocket,
    subscribeToChannels,
    unsubscribeFromChannels,
    updateSubscriptions,
    addMessage,
    markMessageAsProcessed,
    markAllMessagesAsProcessed,
    deleteMessage,
    clearMessages,
    sendCustomMessage,
    requestData,
    updateMessageFilters,
    resetMessageFilters,
    updateConnectionConfig,
    getConnectionStats,
    exportMessages,
    cleanup
  }
})