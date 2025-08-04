import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { apiService } from '@/services/api'

export interface StockData {
  code: string
  name: string
  price: number
  change: number
  changePercent: number
  volume: number
  marketCap: number
  pe: number
  pb: number
  industry: string
  score: number
  rank: number
  factors: Record<string, number>
  lastUpdate: string
}

export interface FactorData {
  name: string
  category: string
  value: number
  weight: number
  description: string
  lastUpdate: string
}

export interface BacktestResult {
  id: string
  name: string
  strategy: string
  startDate: string
  endDate: string
  totalReturn: number
  annualizedReturn: number
  maxDrawdown: number
  sharpeRatio: number
  status: 'completed' | 'running' | 'failed'
  createdAt: string
}

export interface ModelExplanation {
  modelId: string
  modelName: string
  explanation: {
    global: {
      featureImportance: { feature: string; importance: number }[]
      summary: string
    }
    local?: {
      stockCode: string
      prediction: number
      shapValues: { feature: string; value: number }[]
      explanation: string
    }
  }
  lastUpdate: string
}

export interface WatchlistItem {
  id: string
  stockCode: string
  stockName: string
  addedAt: string
  notes?: string
  alerts?: {
    priceAbove?: number
    priceBelow?: number
    scoreAbove?: number
    scoreBelow?: number
  }
}

export const useDataStore = defineStore('data', () => {
  // 状态
  const stockList = ref<StockData[]>([])
  const factorList = ref<FactorData[]>([])
  const backtestResults = ref<BacktestResult[]>([])
  const modelExplanations = ref<ModelExplanation[]>([])
  const watchlist = ref<WatchlistItem[]>([])
  const selectedStocks = ref<string[]>([])
  const selectedFactors = ref<string[]>([])
  const currentBacktest = ref<string>('')
  const currentModel = ref<string>('')
  const loading = ref<boolean>(false)
  const lastUpdate = ref<string>('')
  
  // 筛选和排序状态
  const filters = ref({
    industry: '',
    marketCapMin: 0,
    marketCapMax: 0,
    scoreMin: 0,
    scoreMax: 100,
    searchKeyword: ''
  })
  
  const sorting = ref({
    field: 'score',
    order: 'desc' as 'asc' | 'desc'
  })
  
  const pagination = ref({
    page: 1,
    pageSize: 50,
    total: 0
  })

  // 计算属性
  const filteredStocks = computed(() => {
    let result = stockList.value
    
    // 行业筛选
    if (filters.value.industry) {
      result = result.filter(stock => stock.industry === filters.value.industry)
    }
    
    // 市值筛选
    if (filters.value.marketCapMin > 0) {
      result = result.filter(stock => stock.marketCap >= filters.value.marketCapMin)
    }
    if (filters.value.marketCapMax > 0) {
      result = result.filter(stock => stock.marketCap <= filters.value.marketCapMax)
    }
    
    // 评分筛选
    result = result.filter(stock => 
      stock.score >= filters.value.scoreMin && stock.score <= filters.value.scoreMax
    )
    
    // 关键词搜索
    if (filters.value.searchKeyword) {
      const keyword = filters.value.searchKeyword.toLowerCase()
      result = result.filter(stock => 
        stock.code.toLowerCase().includes(keyword) || 
        stock.name.toLowerCase().includes(keyword)
      )
    }
    
    return result
  })

  const sortedStocks = computed(() => {
    const result = [...filteredStocks.value]
    const { field, order } = sorting.value
    
    result.sort((a, b) => {
      let aValue = a[field as keyof StockData]
      let bValue = b[field as keyof StockData]
      
      if (typeof aValue === 'string') {
        aValue = aValue.toLowerCase()
        bValue = (bValue as string).toLowerCase()
      }
      
      if (aValue < bValue) return order === 'asc' ? -1 : 1
      if (aValue > bValue) return order === 'asc' ? 1 : -1
      return 0
    })
    
    return result
  })

  const paginatedStocks = computed(() => {
    const start = (pagination.value.page - 1) * pagination.value.pageSize
    const end = start + pagination.value.pageSize
    return sortedStocks.value.slice(start, end)
  })

  const topPerformingStocks = computed(() => {
    return stockList.value
      .filter(stock => stock.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 10)
  })

  const industryDistribution = computed(() => {
    const distribution: Record<string, number> = {}
    stockList.value.forEach(stock => {
      distribution[stock.industry] = (distribution[stock.industry] || 0) + 1
    })
    return Object.entries(distribution).map(([industry, count]) => ({
      industry,
      count
    }))
  })

  const factorCategories = computed(() => {
    const categories = new Set(factorList.value.map(factor => factor.category))
    return Array.from(categories)
  })

  const watchlistStocks = computed(() => {
    return watchlist.value.map(item => {
      const stock = stockList.value.find(s => s.code === item.stockCode)
      return {
        ...item,
        stock
      }
    })
  })

  const completedBacktests = computed(() => {
    return backtestResults.value.filter(result => result.status === 'completed')
  })

  // 方法
  const fetchStockList = async (params?: {
    page?: number
    pageSize?: number
    industry?: string
    sortBy?: string
    sortOrder?: 'asc' | 'desc'
  }) => {
    loading.value = true
    try {
      const response = await apiService.getStockRanking(params)
      stockList.value = response.stocks
      pagination.value.total = response.total
      lastUpdate.value = new Date().toISOString()
      return response
    } catch (error) {
      console.error('获取股票列表失败:', error)
      throw error
    } finally {
      loading.value = false
    }
  }

  const fetchFactorList = async (category?: string) => {
    try {
      const response = await apiService.getFactorData({ category })
      factorList.value = response.factors
      return response
    } catch (error) {
      console.error('获取因子列表失败:', error)
      throw error
    }
  }

  const fetchBacktestResults = async () => {
    try {
      const response = await apiService.getBacktestList()
      backtestResults.value = response.backtests
      return response
    } catch (error) {
      console.error('获取回测结果失败:', error)
      throw error
    }
  }

  const fetchModelExplanations = async (modelId?: string) => {
    try {
      const response = await apiService.getModelExplanation({ model_id: modelId })
      if (modelId) {
        const index = modelExplanations.value.findIndex(exp => exp.modelId === modelId)
        if (index !== -1) {
          modelExplanations.value[index] = response.explanation
        } else {
          modelExplanations.value.push(response.explanation)
        }
      } else {
        modelExplanations.value = response.explanations || []
      }
      return response
    } catch (error) {
      console.error('获取模型解释失败:', error)
      throw error
    }
  }

  const fetchWatchlist = async () => {
    try {
      const response = await apiService.getWatchlist()
      watchlist.value = response.watchlist
      return response
    } catch (error) {
      console.error('获取关注列表失败:', error)
      throw error
    }
  }

  const addToWatchlist = async (stockCode: string, notes?: string) => {
    try {
      const response = await apiService.addToWatchlist({ stockCode, notes })
      await fetchWatchlist()
      return response
    } catch (error) {
      console.error('添加到关注列表失败:', error)
      throw error
    }
  }

  const removeFromWatchlist = async (itemId: string) => {
    try {
      await apiService.removeFromWatchlist(itemId)
      watchlist.value = watchlist.value.filter(item => item.id !== itemId)
    } catch (error) {
      console.error('从关注列表移除失败:', error)
      throw error
    }
  }

  const updateWatchlistItem = async (itemId: string, updates: Partial<WatchlistItem>) => {
    try {
      const response = await apiService.updateWatchlistItem(itemId, updates)
      const index = watchlist.value.findIndex(item => item.id === itemId)
      if (index !== -1) {
        watchlist.value[index] = { ...watchlist.value[index], ...updates }
      }
      return response
    } catch (error) {
      console.error('更新关注列表项失败:', error)
      throw error
    }
  }

  const searchStocks = async (query: string, limit = 20) => {
    try {
      const response = await apiService.searchStocks({ query, limit })
      return response.stocks
    } catch (error) {
      console.error('搜索股票失败:', error)
      throw error
    }
  }

  const getStockDetail = async (stockCode: string) => {
    try {
      const response = await apiService.getStockDetail(stockCode)
      return response.stock
    } catch (error) {
      console.error('获取股票详情失败:', error)
      throw error
    }
  }

  const getStockHistory = async (stockCode: string, period = '1Y') => {
    try {
      const response = await apiService.getStockHistory({ stockCode, period })
      return response.history
    } catch (error) {
      console.error('获取股票历史数据失败:', error)
      throw error
    }
  }

  const runBacktest = async (config: any) => {
    try {
      const response = await apiService.runBacktest(config)
      await fetchBacktestResults()
      return response
    } catch (error) {
      console.error('运行回测失败:', error)
      throw error
    }
  }

  const deleteBacktest = async (backtestId: string) => {
    try {
      await apiService.deleteBacktest(backtestId)
      backtestResults.value = backtestResults.value.filter(result => result.id !== backtestId)
    } catch (error) {
      console.error('删除回测失败:', error)
      throw error
    }
  }

  const updateFilters = (newFilters: Partial<typeof filters.value>) => {
    filters.value = { ...filters.value, ...newFilters }
    pagination.value.page = 1 // 重置到第一页
  }

  const updateSorting = (field: string, order?: 'asc' | 'desc') => {
    if (sorting.value.field === field && !order) {
      sorting.value.order = sorting.value.order === 'asc' ? 'desc' : 'asc'
    } else {
      sorting.value.field = field
      sorting.value.order = order || 'desc'
    }
    pagination.value.page = 1 // 重置到第一页
  }

  const updatePagination = (updates: Partial<typeof pagination.value>) => {
    pagination.value = { ...pagination.value, ...updates }
  }

  const selectStock = (stockCode: string) => {
    if (!selectedStocks.value.includes(stockCode)) {
      selectedStocks.value.push(stockCode)
    }
  }

  const unselectStock = (stockCode: string) => {
    const index = selectedStocks.value.indexOf(stockCode)
    if (index !== -1) {
      selectedStocks.value.splice(index, 1)
    }
  }

  const toggleStockSelection = (stockCode: string) => {
    if (selectedStocks.value.includes(stockCode)) {
      unselectStock(stockCode)
    } else {
      selectStock(stockCode)
    }
  }

  const clearStockSelection = () => {
    selectedStocks.value = []
  }

  const selectFactor = (factorName: string) => {
    if (!selectedFactors.value.includes(factorName)) {
      selectedFactors.value.push(factorName)
    }
  }

  const unselectFactor = (factorName: string) => {
    const index = selectedFactors.value.indexOf(factorName)
    if (index !== -1) {
      selectedFactors.value.splice(index, 1)
    }
  }

  const toggleFactorSelection = (factorName: string) => {
    if (selectedFactors.value.includes(factorName)) {
      unselectFactor(factorName)
    } else {
      selectFactor(factorName)
    }
  }

  const clearFactorSelection = () => {
    selectedFactors.value = []
  }

  const refreshAllData = async () => {
    loading.value = true
    try {
      await Promise.all([
        fetchStockList(),
        fetchFactorList(),
        fetchBacktestResults(),
        fetchWatchlist()
      ])
    } catch (error) {
      console.error('刷新数据失败:', error)
      throw error
    } finally {
      loading.value = false
    }
  }

  const exportData = async (type: 'stocks' | 'factors' | 'backtests', format = 'csv') => {
    try {
      let response
      switch (type) {
        case 'stocks':
          response = await apiService.exportStockData({ format, filters: filters.value })
          break
        case 'factors':
          response = await apiService.exportFactorData({ format })
          break
        case 'backtests':
          response = await apiService.exportBacktestData({ format })
          break
        default:
          throw new Error('不支持的导出类型')
      }
      return response
    } catch (error) {
      console.error('导出数据失败:', error)
      throw error
    }
  }

  const resetFilters = () => {
    filters.value = {
      industry: '',
      marketCapMin: 0,
      marketCapMax: 0,
      scoreMin: 0,
      scoreMax: 100,
      searchKeyword: ''
    }
    pagination.value.page = 1
  }

  const resetSorting = () => {
    sorting.value = {
      field: 'score',
      order: 'desc'
    }
    pagination.value.page = 1
  }

  return {
    // 状态
    stockList,
    factorList,
    backtestResults,
    modelExplanations,
    watchlist,
    selectedStocks,
    selectedFactors,
    currentBacktest,
    currentModel,
    loading,
    lastUpdate,
    filters,
    sorting,
    pagination,
    
    // 计算属性
    filteredStocks,
    sortedStocks,
    paginatedStocks,
    topPerformingStocks,
    industryDistribution,
    factorCategories,
    watchlistStocks,
    completedBacktests,
    
    // 方法
    fetchStockList,
    fetchFactorList,
    fetchBacktestResults,
    fetchModelExplanations,
    fetchWatchlist,
    addToWatchlist,
    removeFromWatchlist,
    updateWatchlistItem,
    searchStocks,
    getStockDetail,
    getStockHistory,
    runBacktest,
    deleteBacktest,
    updateFilters,
    updateSorting,
    updatePagination,
    selectStock,
    unselectStock,
    toggleStockSelection,
    clearStockSelection,
    selectFactor,
    unselectFactor,
    toggleFactorSelection,
    clearFactorSelection,
    refreshAllData,
    exportData,
    resetFilters,
    resetSorting
  }
})