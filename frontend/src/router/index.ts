import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

// 导入页面组件
import Layout from '@/layout/Layout.vue'
import Dashboard from '@/views/Dashboard.vue'
import StockRanking from '@/views/StockRanking.vue'
import StrategyAnalysis from '@/views/StrategyAnalysis.vue'
import FactorAnalysis from '@/views/FactorAnalysis.vue'
import ModelExplanation from '@/views/ModelExplanation.vue'
import BacktestResults from '@/views/BacktestResults.vue'
import DataQuery from '@/views/DataQuery.vue'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    component: Layout,
    redirect: '/dashboard',
    children: [
      {
        path: 'dashboard',
        name: 'Dashboard',
        component: Dashboard,
        meta: {
          title: '校长驾驶舱',
          icon: 'Monitor'
        }
      },
      {
        path: 'stock-ranking',
        name: 'StockRanking',
        component: StockRanking,
        meta: {
          title: '股票评分排名',
          icon: 'TrendCharts'
        }
      },
      {
        path: 'strategy-analysis',
        name: 'StrategyAnalysis',
        component: StrategyAnalysis,
        meta: {
          title: '策略分析',
          icon: 'DataAnalysis'
        }
      },
      {
        path: 'factor-analysis',
        name: 'FactorAnalysis',
        component: FactorAnalysis,
        meta: {
          title: '因子分析',
          icon: 'PieChart'
        }
      },
      {
        path: 'model-explanation',
        name: 'ModelExplanation',
        component: ModelExplanation,
        meta: {
          title: '模型解释',
          icon: 'MagicStick'
        }
      },
      {
        path: 'backtest-results',
        name: 'BacktestResults',
        component: BacktestResults,
        meta: {
          title: '回测结果',
          icon: 'DataLine'
        }
      },
      {
        path: 'data-query',
        name: 'DataQuery',
        component: DataQuery,
        meta: {
          title: '数据查询',
          icon: 'Search'
        }
      }
    ]
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router