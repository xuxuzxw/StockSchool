<template>
  <div class="stock-ranking">
    <!-- 筛选和搜索区域 -->
    <el-card class="filter-card">
      <el-row :gutter="20">
        <el-col :xs="24" :sm="12" :md="6">
          <el-input
            v-model="searchQuery"
            placeholder="搜索股票代码或名称"
            clearable
            @input="handleSearch"
          >
            <template #prefix>
              <el-icon><Search /></el-icon>
            </template>
          </el-input>
        </el-col>
        <el-col :xs="24" :sm="12" :md="4">
          <el-select v-model="selectedMarket" placeholder="选择市场" clearable @change="handleFilter">
            <el-option label="全部" value="" />
            <el-option label="沪市主板" value="SH_MAIN" />
            <el-option label="深市主板" value="SZ_MAIN" />
            <el-option label="创业板" value="SZ_GEM" />
            <el-option label="科创板" value="SH_STAR" />
          </el-select>
        </el-col>
        <el-col :xs="24" :sm="12" :md="4">
          <el-select v-model="selectedIndustry" placeholder="选择行业" clearable @change="handleFilter">
            <el-option label="全部" value="" />
            <el-option v-for="industry in industries" :key="industry" :label="industry" :value="industry" />
          </el-select>
        </el-col>
        <el-col :xs="24" :sm="12" :md="4">
          <el-select v-model="scoreRange" placeholder="评分范围" clearable @change="handleFilter">
            <el-option label="全部" value="" />
            <el-option label="90-100分" value="90-100" />
            <el-option label="80-89分" value="80-89" />
            <el-option label="70-79分" value="70-79" />
            <el-option label="60-69分" value="60-69" />
            <el-option label="60分以下" value="0-59" />
          </el-select>
        </el-col>
        <el-col :xs="24" :sm="12" :md="6">
          <el-button type="primary" @click="refreshData" :loading="loading">
            <el-icon><Refresh /></el-icon>
            刷新数据
          </el-button>
          <el-button @click="exportData">
            <el-icon><Download /></el-icon>
            导出
          </el-button>
        </el-col>
      </el-row>
    </el-card>

    <!-- 统计概览 -->
    <el-row :gutter="20" class="stats-row">
      <el-col :xs="12" :sm="6">
        <el-statistic title="总股票数" :value="stats.totalStocks" />
      </el-col>
      <el-col :xs="12" :sm="6">
        <el-statistic title="平均评分" :value="stats.averageScore" :precision="2" />
      </el-col>
      <el-col :xs="12" :sm="6">
        <el-statistic title="推荐股票" :value="stats.recommendedStocks" />
      </el-col>
      <el-col :xs="12" :sm="6">
        <el-statistic title="更新时间" :value="stats.lastUpdate" value-style="font-size: 14px;" />
      </el-col>
    </el-row>

    <!-- 股票排名表格 -->
    <el-card class="table-card">
      <template #header>
        <div class="table-header">
          <span>股票评分排名</span>
          <div class="table-actions">
            <el-radio-group v-model="viewMode" @change="handleViewModeChange">
              <el-radio-button label="table">表格视图</el-radio-button>
              <el-radio-button label="card">卡片视图</el-radio-button>
            </el-radio-group>
          </div>
        </div>
      </template>

      <!-- 表格视图 -->
      <el-table
        v-if="viewMode === 'table'"
        :data="paginatedStocks"
        v-loading="loading"
        stripe
        @sort-change="handleSortChange"
        @row-click="handleRowClick"
        class="stock-table"
      >
        <el-table-column type="index" label="排名" width="80" :index="getRankIndex" />
        
        <el-table-column prop="symbol" label="股票代码" width="120" sortable="custom">
          <template #default="{ row }">
            <el-button type="text" @click="viewStockDetail(row)">
              {{ row.symbol }}
            </el-button>
          </template>
        </el-table-column>
        
        <el-table-column prop="name" label="股票名称" min-width="150" sortable="custom" />
        
        <el-table-column prop="score" label="综合评分" width="120" sortable="custom">
          <template #default="{ row }">
            <div class="score-cell">
              <el-progress
                :percentage="row.score"
                :color="getScoreColor(row.score)"
                :stroke-width="8"
                text-inside
                :format="() => row.score.toFixed(1)"
              />
            </div>
          </template>
        </el-table-column>
        
        <el-table-column prop="price" label="当前价格" width="100" sortable="custom">
          <template #default="{ row }">
            <span :class="getPriceChangeClass(row.priceChange)">¥{{ row.price.toFixed(2) }}</span>
          </template>
        </el-table-column>
        
        <el-table-column prop="priceChange" label="涨跌幅" width="100" sortable="custom">
          <template #default="{ row }">
            <span :class="getPriceChangeClass(row.priceChange)">
              {{ row.priceChange > 0 ? '+' : '' }}{{ (row.priceChange * 100).toFixed(2) }}%
            </span>
          </template>
        </el-table-column>
        
        <el-table-column prop="volume" label="成交量" width="120" sortable="custom">
          <template #default="{ row }">
            {{ formatVolume(row.volume) }}
          </template>
        </el-table-column>
        
        <el-table-column prop="marketCap" label="市值" width="120" sortable="custom">
          <template #default="{ row }">
            {{ formatMarketCap(row.marketCap) }}
          </template>
        </el-table-column>
        
        <el-table-column prop="industry" label="行业" width="120" />
        
        <el-table-column prop="recommendation" label="推荐度" width="100">
          <template #default="{ row }">
            <el-tag :type="getRecommendationType(row.recommendation)" size="small">
              {{ row.recommendation }}
            </el-tag>
          </template>
        </el-table-column>
        
        <el-table-column label="操作" width="150" fixed="right">
          <template #default="{ row }">
            <el-button type="text" size="small" @click="viewStockDetail(row)">
              详情
            </el-button>
            <el-button type="text" size="small" @click="addToWatchlist(row)">
              关注
            </el-button>
            <el-button type="text" size="small" @click="viewFactorAnalysis(row)">
              因子分析
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 卡片视图 -->
      <div v-else class="card-view">
        <el-row :gutter="20">
          <el-col :xs="24" :sm="12" :md="8" :lg="6" v-for="stock in paginatedStocks" :key="stock.symbol">
            <el-card class="stock-card" @click="viewStockDetail(stock)">
              <div class="stock-card-header">
                <div class="stock-info">
                  <div class="stock-symbol">{{ stock.symbol }}</div>
                  <div class="stock-name">{{ stock.name }}</div>
                </div>
                <div class="stock-score">
                  <el-progress
                    type="circle"
                    :percentage="stock.score"
                    :color="getScoreColor(stock.score)"
                    :width="60"
                    :format="() => stock.score.toFixed(0)"
                  />
                </div>
              </div>
              <div class="stock-card-content">
                <div class="price-info">
                  <span class="price" :class="getPriceChangeClass(stock.priceChange)">
                    ¥{{ stock.price.toFixed(2) }}
                  </span>
                  <span class="change" :class="getPriceChangeClass(stock.priceChange)">
                    {{ stock.priceChange > 0 ? '+' : '' }}{{ (stock.priceChange * 100).toFixed(2) }}%
                  </span>
                </div>
                <div class="stock-meta">
                  <div class="meta-item">
                    <span class="label">成交量:</span>
                    <span class="value">{{ formatVolume(stock.volume) }}</span>
                  </div>
                  <div class="meta-item">
                    <span class="label">市值:</span>
                    <span class="value">{{ formatMarketCap(stock.marketCap) }}</span>
                  </div>
                  <div class="meta-item">
                    <span class="label">行业:</span>
                    <span class="value">{{ stock.industry }}</span>
                  </div>
                </div>
                <div class="stock-recommendation">
                  <el-tag :type="getRecommendationType(stock.recommendation)" size="small">
                    {{ stock.recommendation }}
                  </el-tag>
                </div>
              </div>
            </el-card>
          </el-col>
        </el-row>
      </div>

      <!-- 分页 -->
      <div class="pagination-wrapper">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :page-sizes="[20, 50, 100, 200]"
          :total="filteredStocks.length"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { apiService, type StockScore } from '@/services/api'
import { ElMessage, ElMessageBox } from 'element-plus'

// 响应式数据
const loading = ref(false)
const searchQuery = ref('')
const selectedMarket = ref('')
const selectedIndustry = ref('')
const scoreRange = ref('')
const viewMode = ref('table')
const currentPage = ref(1)
const pageSize = ref(50)
const sortField = ref('score')
const sortOrder = ref('desc')

// 股票数据
const stocks = ref<StockScore[]>([])
const industries = ref<string[]>([])

// 统计数据
const stats = ref({
  totalStocks: 0,
  averageScore: 0,
  recommendedStocks: 0,
  lastUpdate: ''
})

// 计算属性
const filteredStocks = computed(() => {
  let result = stocks.value

  // 搜索过滤
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    result = result.filter(stock => 
      stock.symbol.toLowerCase().includes(query) || 
      stock.name.toLowerCase().includes(query)
    )
  }

  // 市场过滤
  if (selectedMarket.value) {
    result = result.filter(stock => stock.market === selectedMarket.value)
  }

  // 行业过滤
  if (selectedIndustry.value) {
    result = result.filter(stock => stock.industry === selectedIndustry.value)
  }

  // 评分范围过滤
  if (scoreRange.value) {
    const [min, max] = scoreRange.value.split('-').map(Number)
    result = result.filter(stock => stock.score >= min && stock.score <= max)
  }

  // 排序
  result.sort((a, b) => {
    const aValue = a[sortField.value as keyof StockScore]
    const bValue = b[sortField.value as keyof StockScore]
    
    if (typeof aValue === 'number' && typeof bValue === 'number') {
      return sortOrder.value === 'desc' ? bValue - aValue : aValue - bValue
    }
    
    if (typeof aValue === 'string' && typeof bValue === 'string') {
      return sortOrder.value === 'desc' 
        ? bValue.localeCompare(aValue)
        : aValue.localeCompare(bValue)
    }
    
    return 0
  })

  return result
})

const paginatedStocks = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  const end = start + pageSize.value
  return filteredStocks.value.slice(start, end)
})

// 方法
const loadStockData = async () => {
  loading.value = true
  try {
    const response = await apiService.getStockScores({
      limit: 5000,
      sort_by: 'score',
      sort_order: 'desc'
    })
    
    stocks.value = response.stocks
    
    // 提取行业列表
    const industrySet = new Set(stocks.value.map(stock => stock.industry))
    industries.value = Array.from(industrySet).sort()
    
    // 更新统计数据
    updateStats()
    
  } catch (error) {
    console.error('加载股票数据失败:', error)
    ElMessage.error('加载股票数据失败')
  } finally {
    loading.value = false
  }
}

const updateStats = () => {
  stats.value = {
    totalStocks: stocks.value.length,
    averageScore: stocks.value.reduce((sum, stock) => sum + stock.score, 0) / stocks.value.length,
    recommendedStocks: stocks.value.filter(stock => stock.recommendation === '强烈推荐' || stock.recommendation === '推荐').length,
    lastUpdate: new Date().toLocaleString()
  }
}

const refreshData = () => {
  loadStockData()
}

const handleSearch = () => {
  currentPage.value = 1
}

const handleFilter = () => {
  currentPage.value = 1
}

const handleSortChange = ({ prop, order }: { prop: string; order: string | null }) => {
  if (prop && order) {
    sortField.value = prop
    sortOrder.value = order === 'ascending' ? 'asc' : 'desc'
  }
}

const handleViewModeChange = () => {
  currentPage.value = 1
}

const handleSizeChange = (size: number) => {
  pageSize.value = size
  currentPage.value = 1
}

const handleCurrentChange = (page: number) => {
  currentPage.value = page
}

const handleRowClick = (row: StockScore) => {
  viewStockDetail(row)
}

const viewStockDetail = (stock: StockScore) => {
  // 跳转到股票详情页面
  console.log('查看股票详情:', stock.symbol)
}

const addToWatchlist = async (stock: StockScore) => {
  try {
    // 调用添加到关注列表的API
    ElMessage.success(`已将 ${stock.name} 添加到关注列表`)
  } catch (error) {
    ElMessage.error('添加到关注列表失败')
  }
}

const viewFactorAnalysis = (stock: StockScore) => {
  // 跳转到因子分析页面
  console.log('查看因子分析:', stock.symbol)
}

const exportData = () => {
  // 导出数据功能
  ElMessage.info('导出功能开发中...')
}

// 工具函数
const getRankIndex = (index: number) => {
  return (currentPage.value - 1) * pageSize.value + index + 1
}

const getScoreColor = (score: number) => {
  if (score >= 90) return '#67c23a'
  if (score >= 80) return '#e6a23c'
  if (score >= 70) return '#409eff'
  if (score >= 60) return '#909399'
  return '#f56c6c'
}

const getPriceChangeClass = (change: number) => {
  if (change > 0) return 'price-up'
  if (change < 0) return 'price-down'
  return 'price-neutral'
}

const getRecommendationType = (recommendation: string) => {
  switch (recommendation) {
    case '强烈推荐': return 'success'
    case '推荐': return 'primary'
    case '中性': return 'info'
    case '不推荐': return 'warning'
    case '强烈不推荐': return 'danger'
    default: return 'info'
  }
}

const formatVolume = (volume: number) => {
  if (volume >= 100000000) {
    return (volume / 100000000).toFixed(2) + '亿'
  }
  if (volume >= 10000) {
    return (volume / 10000).toFixed(2) + '万'
  }
  return volume.toString()
}

const formatMarketCap = (marketCap: number) => {
  if (marketCap >= 100000000) {
    return (marketCap / 100000000).toFixed(2) + '亿'
  }
  if (marketCap >= 10000) {
    return (marketCap / 10000).toFixed(2) + '万'
  }
  return marketCap.toString()
}

// 生命周期
onMounted(() => {
  loadStockData()
})

// 监听筛选条件变化
watch([searchQuery, selectedMarket, selectedIndustry, scoreRange], () => {
  currentPage.value = 1
})
</script>

<style scoped>
.stock-ranking {
  padding: 0;
}

.filter-card {
  margin-bottom: 20px;
}

.stats-row {
  margin-bottom: 20px;
}

.table-card {
  margin-bottom: 20px;
}

.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.stock-table {
  cursor: pointer;
}

.score-cell {
  padding: 5px 0;
}

.price-up {
  color: #f56c6c;
}

.price-down {
  color: #67c23a;
}

.price-neutral {
  color: #909399;
}

.card-view {
  margin-bottom: 20px;
}

.stock-card {
  cursor: pointer;
  transition: all 0.3s;
  height: 200px;
}

.stock-card:hover {
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
  transform: translateY(-2px);
}

.stock-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.stock-info {
  flex: 1;
}

.stock-symbol {
  font-size: 16px;
  font-weight: bold;
  color: #303133;
  margin-bottom: 5px;
}

.stock-name {
  font-size: 14px;
  color: #606266;
}

.stock-score {
  margin-left: 10px;
}

.stock-card-content {
  space-y: 10px;
}

.price-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.price {
  font-size: 18px;
  font-weight: bold;
}

.change {
  font-size: 14px;
}

.stock-meta {
  space-y: 5px;
  margin-bottom: 10px;
}

.meta-item {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  margin-bottom: 5px;
}

.label {
  color: #909399;
}

.value {
  color: #303133;
}

.stock-recommendation {
  text-align: center;
}

.pagination-wrapper {
  display: flex;
  justify-content: center;
  margin-top: 20px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .table-header {
    flex-direction: column;
    gap: 10px;
    align-items: stretch;
  }
  
  .stock-card {
    height: auto;
    min-height: 180px;
  }
  
  .stock-card-header {
    flex-direction: column;
    align-items: center;
    text-align: center;
  }
  
  .stock-score {
    margin-left: 0;
    margin-top: 10px;
  }
}
</style>