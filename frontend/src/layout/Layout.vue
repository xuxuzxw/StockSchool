<template>
  <el-container class="layout-container">
    <!-- 侧边栏 -->
    <el-aside :width="isCollapse ? '64px' : '200px'" class="sidebar">
      <div class="logo">
        <img src="/logo.svg" alt="StockSchool" v-if="!isCollapse" />
        <span v-if="!isCollapse" class="logo-text">StockSchool</span>
        <el-icon v-else size="24" color="#409eff"><TrendCharts /></el-icon>
      </div>
      
      <el-menu
        :default-active="activeMenu"
        :collapse="isCollapse"
        :unique-opened="true"
        router
        background-color="#304156"
        text-color="#bfcbd9"
        active-text-color="#409eff"
      >
        <el-menu-item
          v-for="route in menuRoutes"
          :key="route.path"
          :index="route.path"
        >
          <el-icon><component :is="route.meta?.icon" /></el-icon>
          <template #title>{{ route.meta?.title }}</template>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <!-- 主内容区 -->
    <el-container>
      <!-- 顶部导航栏 -->
      <el-header class="header">
        <div class="header-left">
          <el-button
            type="text"
            @click="toggleSidebar"
            class="collapse-btn"
          >
            <el-icon size="20">
              <Expand v-if="isCollapse" />
              <Fold v-else />
            </el-icon>
          </el-button>
          
          <el-breadcrumb separator="/" class="breadcrumb">
            <el-breadcrumb-item :to="{ path: '/' }">首页</el-breadcrumb-item>
            <el-breadcrumb-item>{{ currentPageTitle }}</el-breadcrumb-item>
          </el-breadcrumb>
        </div>
        
        <div class="header-right">
          <!-- 实时连接状态 -->
          <div class="connection-status">
            <el-icon :color="connectionStatus.color" size="16">
              <component :is="connectionStatus.icon" />
            </el-icon>
            <span class="status-text">{{ connectionStatus.text }}</span>
          </div>
          
          <!-- 用户菜单 -->
          <el-dropdown>
            <span class="user-dropdown">
              <el-avatar size="small" :src="userAvatar" />
              <span class="username">{{ username }}</span>
              <el-icon><ArrowDown /></el-icon>
            </span>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item>个人设置</el-dropdown-item>
                <el-dropdown-item divided>退出登录</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </el-header>

      <!-- 主内容 -->
      <el-main class="main-content">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { useWebSocket } from '@/composables/useWebSocket'

const route = useRoute()
const { connectionStatus } = useWebSocket()

// 侧边栏折叠状态
const isCollapse = ref(false)

// 用户信息
const username = ref('管理员')
const userAvatar = ref('https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png')

// 菜单路由
const menuRoutes = [
  { path: '/dashboard', meta: { title: '校长驾驶舱', icon: 'Monitor' } },
  { path: '/stock-ranking', meta: { title: '股票评分排名', icon: 'TrendCharts' } },
  { path: '/strategy-analysis', meta: { title: '策略分析', icon: 'DataAnalysis' } },
  { path: '/factor-analysis', meta: { title: '因子分析', icon: 'PieChart' } },
  { path: '/model-explanation', meta: { title: '模型解释', icon: 'MagicStick' } },
  { path: '/backtest-results', meta: { title: '回测结果', icon: 'DataLine' } },
  { path: '/data-query', meta: { title: '数据查询', icon: 'Search' } }
]

// 当前激活的菜单
const activeMenu = computed(() => route.path)

// 当前页面标题
const currentPageTitle = computed(() => {
  const currentRoute = menuRoutes.find(r => r.path === route.path)
  return currentRoute?.meta?.title || '未知页面'
})

// 切换侧边栏
const toggleSidebar = () => {
  isCollapse.value = !isCollapse.value
}

// 监听窗口大小变化
const handleResize = () => {
  if (window.innerWidth < 768) {
    isCollapse.value = true
  }
}

onMounted(() => {
  window.addEventListener('resize', handleResize)
  handleResize()
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
})
</script>

<style scoped>
.layout-container {
  height: 100vh;
}

.sidebar {
  background-color: #304156;
  transition: width 0.3s;
}

.logo {
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 20px;
  background-color: #2b3a4b;
}

.logo img {
  height: 32px;
  margin-right: 10px;
}

.logo-text {
  color: #409eff;
  font-size: 18px;
  font-weight: bold;
}

.header {
  background-color: #fff;
  border-bottom: 1px solid #e4e7ed;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
}

.header-left {
  display: flex;
  align-items: center;
}

.collapse-btn {
  margin-right: 20px;
  color: #606266;
}

.breadcrumb {
  font-size: 14px;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 20px;
}

.connection-status {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
  color: #606266;
}

.status-text {
  font-size: 12px;
}

.user-dropdown {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  color: #606266;
}

.username {
  font-size: 14px;
}

.main-content {
  background-color: #f5f7fa;
  padding: 20px;
  overflow-y: auto;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .breadcrumb {
    display: none;
  }
  
  .header-left {
    flex: 1;
  }
  
  .username {
    display: none;
  }
}
</style>