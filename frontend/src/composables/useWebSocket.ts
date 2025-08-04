import { ref, onMounted, onUnmounted } from 'vue'
import { io, Socket } from 'socket.io-client'

interface ConnectionStatus {
  text: string
  color: string
  icon: string
}

export function useWebSocket() {
  const socket = ref<Socket | null>(null)
  const isConnected = ref(false)
  const connectionStatus = ref<ConnectionStatus>({
    text: '连接中...',
    color: '#e6a23c',
    icon: 'Loading'
  })

  const connect = () => {
    try {
      socket.value = io('ws://localhost:8000', {
        transports: ['websocket'],
        timeout: 5000
      })

      socket.value.on('connect', () => {
        isConnected.value = true
        connectionStatus.value = {
          text: '已连接',
          color: '#67c23a',
          icon: 'CircleCheck'
        }
        console.log('WebSocket连接已建立')
      })

      socket.value.on('disconnect', () => {
        isConnected.value = false
        connectionStatus.value = {
          text: '连接断开',
          color: '#f56c6c',
          icon: 'CircleClose'
        }
        console.log('WebSocket连接已断开')
      })

      socket.value.on('connect_error', (error) => {
        isConnected.value = false
        connectionStatus.value = {
          text: '连接失败',
          color: '#f56c6c',
          icon: 'Warning'
        }
        console.error('WebSocket连接错误:', error)
      })

      // 监听实时数据更新
      socket.value.on('data_update', (data) => {
        console.log('收到实时数据更新:', data)
        // 这里可以触发全局状态更新
      })

      socket.value.on('alert_update', (alert) => {
        console.log('收到告警更新:', alert)
        // 这里可以显示告警通知
      })

    } catch (error) {
      console.error('WebSocket初始化失败:', error)
      connectionStatus.value = {
        text: '初始化失败',
        color: '#f56c6c',
        icon: 'Warning'
      }
    }
  }

  const disconnect = () => {
    if (socket.value) {
      socket.value.disconnect()
      socket.value = null
    }
  }

  const emit = (event: string, data?: any) => {
    if (socket.value && isConnected.value) {
      socket.value.emit(event, data)
    }
  }

  const on = (event: string, callback: (data: any) => void) => {
    if (socket.value) {
      socket.value.on(event, callback)
    }
  }

  const off = (event: string, callback?: (data: any) => void) => {
    if (socket.value) {
      socket.value.off(event, callback)
    }
  }

  onMounted(() => {
    connect()
  })

  onUnmounted(() => {
    disconnect()
  })

  return {
    socket,
    isConnected,
    connectionStatus,
    connect,
    disconnect,
    emit,
    on,
    off
  }
}