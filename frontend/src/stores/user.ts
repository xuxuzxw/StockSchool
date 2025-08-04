import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { apiService } from '@/services/api'

export interface UserInfo {
  id: string
  username: string
  email: string
  role: string
  avatar?: string
  preferences: {
    theme: 'light' | 'dark'
    language: 'zh-CN' | 'en-US'
    defaultDashboard: string
    notifications: {
      email: boolean
      push: boolean
      alerts: boolean
    }
    charts: {
      defaultType: string
      colorScheme: string
      animation: boolean
    }
  }
  permissions: string[]
  lastLogin?: string
  createdAt: string
}

export interface UserPreferences {
  theme: 'light' | 'dark'
  language: 'zh-CN' | 'en-US'
  defaultDashboard: string
  notifications: {
    email: boolean
    push: boolean
    alerts: boolean
  }
  charts: {
    defaultType: string
    colorScheme: string
    animation: boolean
  }
}

export const useUserStore = defineStore('user', () => {
  // 状态
  const userInfo = ref<UserInfo | null>(null)
  const token = ref<string>('')
  const isLoggedIn = ref<boolean>(false)
  const loginLoading = ref<boolean>(false)
  const preferences = ref<UserPreferences>({
    theme: 'light',
    language: 'zh-CN',
    defaultDashboard: 'dashboard',
    notifications: {
      email: true,
      push: true,
      alerts: true
    },
    charts: {
      defaultType: 'line',
      colorScheme: 'default',
      animation: true
    }
  })

  // 计算属性
  const hasPermission = computed(() => {
    return (permission: string) => {
      if (!userInfo.value?.permissions) return false
      return userInfo.value.permissions.includes(permission) || userInfo.value.permissions.includes('admin')
    }
  })

  const isAdmin = computed(() => {
    return userInfo.value?.role === 'admin'
  })

  const userName = computed(() => {
    return userInfo.value?.username || '未登录'
  })

  const userAvatar = computed(() => {
    return userInfo.value?.avatar || '/default-avatar.png'
  })

  // 方法
  const login = async (credentials: { username: string; password: string }) => {
    loginLoading.value = true
    try {
      const response = await apiService.login(credentials)
      
      token.value = response.token
      userInfo.value = response.user
      isLoggedIn.value = true
      
      // 保存到localStorage
      localStorage.setItem('token', response.token)
      localStorage.setItem('userInfo', JSON.stringify(response.user))
      
      // 加载用户偏好设置
      await loadUserPreferences()
      
      return response
    } catch (error) {
      console.error('登录失败:', error)
      throw error
    } finally {
      loginLoading.value = false
    }
  }

  const logout = async () => {
    try {
      await apiService.logout()
    } catch (error) {
      console.error('登出失败:', error)
    } finally {
      // 清除状态
      token.value = ''
      userInfo.value = null
      isLoggedIn.value = false
      
      // 清除localStorage
      localStorage.removeItem('token')
      localStorage.removeItem('userInfo')
      localStorage.removeItem('userPreferences')
    }
  }

  const refreshToken = async () => {
    try {
      const response = await apiService.refreshToken()
      token.value = response.token
      localStorage.setItem('token', response.token)
      return response
    } catch (error) {
      console.error('刷新token失败:', error)
      await logout()
      throw error
    }
  }

  const updateUserInfo = async (updates: Partial<UserInfo>) => {
    try {
      const response = await apiService.updateUserInfo(updates)
      userInfo.value = { ...userInfo.value, ...response.user }
      localStorage.setItem('userInfo', JSON.stringify(userInfo.value))
      return response
    } catch (error) {
      console.error('更新用户信息失败:', error)
      throw error
    }
  }

  const changePassword = async (passwordData: {
    currentPassword: string
    newPassword: string
    confirmPassword: string
  }) => {
    try {
      const response = await apiService.changePassword(passwordData)
      return response
    } catch (error) {
      console.error('修改密码失败:', error)
      throw error
    }
  }

  const loadUserPreferences = async () => {
    try {
      // 先从localStorage加载
      const savedPreferences = localStorage.getItem('userPreferences')
      if (savedPreferences) {
        preferences.value = { ...preferences.value, ...JSON.parse(savedPreferences) }
      }
      
      // 如果用户已登录，从服务器加载最新偏好设置
      if (isLoggedIn.value) {
        const response = await apiService.getUserPreferences()
        preferences.value = { ...preferences.value, ...response.preferences }
        localStorage.setItem('userPreferences', JSON.stringify(preferences.value))
      }
    } catch (error) {
      console.error('加载用户偏好设置失败:', error)
    }
  }

  const updatePreferences = async (newPreferences: Partial<UserPreferences>) => {
    try {
      preferences.value = { ...preferences.value, ...newPreferences }
      
      // 保存到localStorage
      localStorage.setItem('userPreferences', JSON.stringify(preferences.value))
      
      // 如果用户已登录，同步到服务器
      if (isLoggedIn.value) {
        await apiService.updateUserPreferences(preferences.value)
      }
    } catch (error) {
      console.error('更新用户偏好设置失败:', error)
      throw error
    }
  }

  const initializeAuth = async () => {
    // 从localStorage恢复状态
    const savedToken = localStorage.getItem('token')
    const savedUserInfo = localStorage.getItem('userInfo')
    
    if (savedToken && savedUserInfo) {
      token.value = savedToken
      userInfo.value = JSON.parse(savedUserInfo)
      isLoggedIn.value = true
      
      try {
        // 验证token是否有效
        await apiService.verifyToken()
        
        // 加载用户偏好设置
        await loadUserPreferences()
      } catch (error) {
        console.error('Token验证失败:', error)
        await logout()
      }
    } else {
      // 加载默认偏好设置
      await loadUserPreferences()
    }
  }

  const getUserStats = async () => {
    try {
      const response = await apiService.getUserStats()
      return response.stats
    } catch (error) {
      console.error('获取用户统计失败:', error)
      throw error
    }
  }

  const getActivityLog = async (params?: {
    page?: number
    pageSize?: number
    startDate?: string
    endDate?: string
  }) => {
    try {
      const response = await apiService.getUserActivityLog(params)
      return response
    } catch (error) {
      console.error('获取活动日志失败:', error)
      throw error
    }
  }

  // 主题相关方法
  const toggleTheme = async () => {
    const newTheme = preferences.value.theme === 'light' ? 'dark' : 'light'
    await updatePreferences({ theme: newTheme })
    
    // 应用主题到DOM
    document.documentElement.setAttribute('data-theme', newTheme)
  }

  const setLanguage = async (language: 'zh-CN' | 'en-US') => {
    await updatePreferences({ language })
  }

  const setDefaultDashboard = async (dashboard: string) => {
    await updatePreferences({ defaultDashboard: dashboard })
  }

  const updateNotificationSettings = async (notifications: Partial<UserPreferences['notifications']>) => {
    await updatePreferences({
      notifications: { ...preferences.value.notifications, ...notifications }
    })
  }

  const updateChartSettings = async (charts: Partial<UserPreferences['charts']>) => {
    await updatePreferences({
      charts: { ...preferences.value.charts, ...charts }
    })
  }

  return {
    // 状态
    userInfo,
    token,
    isLoggedIn,
    loginLoading,
    preferences,
    
    // 计算属性
    hasPermission,
    isAdmin,
    userName,
    userAvatar,
    
    // 方法
    login,
    logout,
    refreshToken,
    updateUserInfo,
    changePassword,
    loadUserPreferences,
    updatePreferences,
    initializeAuth,
    getUserStats,
    getActivityLog,
    
    // 主题和偏好设置方法
    toggleTheme,
    setLanguage,
    setDefaultDashboard,
    updateNotificationSettings,
    updateChartSettings
  }
})