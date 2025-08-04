<template>
  <div class="factor-analysis">
    <!-- 因子选择和配置 -->
    <el-card class="factor-config-card">
      <template #header>
        <div class="card-header">
          <span>因子分析配置</span>
          <el-button type="primary" @click="runFactorAnalysis" :loading="analysisLoading">
            <el-icon><DataAnalysis /></el-icon>
            开始分析
          </el-button>
        </div>
      </template>
      
      <el-row :gutter="20">
        <el-col :xs="24" :md="12">
          <el-form-item label="选择因子类别">
            <el-checkbox-group v-model="selectedFactorCategories" @change="handleCategoryChange">
              <el-checkbox label="value">价值因子</el-checkbox>
              <el-checkbox label="growth">成长因子</el-checkbox>
              <el-checkbox label="quality">质量因子</el-checkbox>
              <el-checkbox label="momentum">动量因子</el-checkbox>
              <el-checkbox label="volatility">波动率因子</el-checkbox>
              <el-checkbox label="size">规模因子</el-checkbox>
            </el-checkbox-group>
          </el-form-item>
        </el-col>
        
        <el-col :xs="24" :md="12">
          <el-form-item label="分析周期">
            <el-select v-model="analysisPeriod" placeholder="选择分析周期">
              <el-option label="最近1个月" value="1M" />
              <el-option label="最近3个月" value="3M" />
              <el-option label="最近6个月" value="6M" />
              <el-option label="最近1年" value="1Y" />
              <el-option label="最近2年" value="2Y" />
            </el-select>
          </el-form-item>
        </el-col>
      </el-row>
      
      <!-- 具体因子选择 -->
      <div class="factor-selection">
        <el-divider content-position="left">具体因子选择</el-divider>
        <el-row :gutter="20">
          <el-col :xs="24" :md="8" v-for="category in selectedFactorCategories" :key="category">
            <el-card class="factor-category-card">
              <template #header>
                <span>{{ getFactorCategoryName(category) }}</span>
              </template>
              <el-checkbox-group v-model="selectedFactors[category]">
                <div v-for="factor in availableFactors[category]" :key="factor.code" class="factor-item">
                  <el-checkbox :label="factor.code">
                    <div class="factor-info">
                      <span class="factor-name">{{ factor.name }}</span>
                      <el-tooltip :content="factor.description" placement="top">
                        <el-icon class="factor-help"><QuestionFilled /></el-icon>
                      </el-tooltip>
                    </div>
                  </el-checkbox>
                </div>
              </el-checkbox-group>
            </el-card>
          </el-col>
        </el-row>
      </div>
    </el-card>

    <!-- 因子概览统计 -->
    <el-row :gutter="20" class="factor-overview" v-if="factorStats">
      <el-col :xs="12" :sm="6">
        <el-statistic title="选中因子数" :value="getTotalSelectedFactors()" />
      </el-col>
      <el-col :xs="12" :sm="6">
        <el-statistic title="覆盖股票数" :value="factorStats.coveredStocks" />
      </el-col>
      <el-col :xs="12" :sm="6">
        <el-statistic title="数据完整度" :value="factorStats.dataCompleteness" suffix="%" :precision="1" />
      </el-col>
      <el-col :xs="12" :sm="6">
        <el-statistic title="更新时间" :value="factorStats.lastUpdate" value-style="font-size: 14px;" />
      </el-col>
    </el-row>

    <!-- 因子分析结果 -->
    <div v-if="analysisResults" class="analysis-results">
      <!-- 因子相关性矩阵 -->
      <el-card class="correlation-card">
        <template #header>
          <div class="card-header">
            <span>因子相关性矩阵</span>
            <el-radio-group v-model="correlationViewType" size="small">
              <el-radio-button label="heatmap">热力图</el-radio-button>
              <el-radio-button label="network">网络图</el-radio-button>
            </el-radio-group>
          </div>
        </template>
        <div class="chart-container">
          <v-chart :option="correlationChartOption" autoresize />
        </div>
      </el-card>

      <!-- 因子收益分析 -->
      <el-row :gutter="20" class="factor-returns-section">
        <el-col :xs="24" :lg="12">
          <el-card class="chart-card">
            <template #header>
              <span>因子收益率时序</span>
            </template>
            <div class="chart-container">
              <v-chart :option="factorReturnsOption" autoresize />
            </div>
          </el-card>
        </el-col>
        
        <el-col :xs="24" :lg="12">
          <el-card class="chart-card">
            <template #header>
              <span>因子累计收益</span>
            </template>
            <div class="chart-container">
              <v-chart :option="cumulativeReturnsOption" autoresize />
            </div>
          </el-card>
        </el-col>
      </el-row>

      <!-- 因子有效性分析 -->
      <el-card class="effectiveness-card">
        <template #header>
          <span>因子有效性分析</span>
        </template>
        <el-table :data="analysisResults.factorEffectiveness" stripe>
          <el-table-column prop="factorName" label="因子名称" min-width="150" />
          <el-table-column prop="icMean" label="IC均值" width="100">
            <template #default="{ row }">
              <span :class="getICClass(row.icMean)">{{ row.icMean.toFixed(4) }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="icStd" label="IC标准差" width="100">
            <template #default="{ row }">
              {{ row.icStd.toFixed(4) }}
            </template>
          </el-table-column>
          <el-table-column prop="icIR" label="IC_IR" width="100">
            <template #default="{ row }">
              <span :class="getIRClass(row.icIR)">{{ row.icIR.toFixed(4) }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="winRate" label="胜率" width="100">
            <template #default="{ row }">
              {{ (row.winRate * 100).toFixed(1) }}%
            </template>
          </el-table-column>
          <el-table-column prop="maxDrawdown" label="最大回撤" width="100">
            <template #default="{ row }">
              {{ (row.maxDrawdown * 100).toFixed(2) }}%
            </template>
          </el-table-column>
          <el-table-column prop="sharpeRatio" label="夏普比率" width="100">
            <template #default="{ row }">
              {{ row.sharpeRatio.toFixed(3) }}
            </template>
          </el-table-column>
          <el-table-column label="评级" width="100">
            <template #default="{ row }">
              <el-tag :type="getFactorRatingType(row.rating)" size="small">
                {{ row.rating }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="操作" width="120">
            <template #default="{ row }">
              <el-button type="text" size="small" @click="viewFactorDetail(row)">
                详情
              </el-button>
              <el-button type="text" size="small" @click="addToStrategy(row)">
                加入策略
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-card>

      <!-- 因子分层回测 -->
      <el-card class="layered-backtest-card">
        <template #header>
          <div class="card-header">
            <span>因子分层回测</span>
            <el-select v-model="selectedFactorForLayering" placeholder="选择因子" style="width: 200px;">
              <el-option
                v-for="factor in getSelectedFactorsList()"
                :key="factor.code"
                :label="factor.name"
                :value="factor.code"
              />
            </el-select>
          </div>
        </template>
        <div v-if="layeredBacktestData" class="layered-results">
          <div class="chart-container">
            <v-chart :option="layeredBacktestOption" autoresize />
          </div>
          <el-table :data="layeredBacktestData.layerStats" stripe class="layer-stats-table">
            <el-table-column prop="layer" label="分层" width="80" />
            <el-table-column prop="stockCount" label="股票数" width="100" />
            <el-table-column prop="avgFactorValue" label="平均因子值" width="120">
              <template #default="{ row }">
                {{ row.avgFactorValue.toFixed(4) }}
              </template>
            </el-table-column>
            <el-table-column prop="totalReturn" label="总收益率" width="100">
              <template #default="{ row }">
                <span :class="getPriceChangeClass(row.totalReturn)">
                  {{ (row.totalReturn * 100).toFixed(2) }}%
                </span>
              </template>
            </el-table-column>
            <el-table-column prop="annualizedReturn" label="年化收益" width="100">
              <template #default="{ row }">
                {{ (row.annualizedReturn * 100).toFixed(2) }}%
              </template>
            </el-table-column>
            <el-table-column prop="volatility" label="波动率" width="100">
              <template #default="{ row }">
                {{ (row.volatility * 100).toFixed(2) }}%
              </template>
            </el-table-column>
            <el-table-column prop="sharpeRatio" label="夏普比率" width="100">
              <template #default="{ row }">
                {{ row.sharpeRatio.toFixed(3) }}
              </template>
            </el-table-column>
          </el-table>
        </div>
      </el-card>

      <!-- 因子组合优化 -->
      <el-card class="optimization-card">
        <template #header>
          <div class="card-header">
            <span>因子组合优化</span>
            <el-button @click="optimizeFactorCombination" :loading="optimizationLoading">
              <el-icon><MagicStick /></el-icon>
              优化组合
            </el-button>
          </div>
        </template>
        <div v-if="optimizationResults" class="optimization-results">
          <el-row :gutter="20">
            <el-col :xs="24" :md="12">
              <div class="optimization-metrics">
                <h4>优化结果</h4>
                <div class="metric-grid">
                  <div class="metric-item">
                    <span class="metric-label">预期收益率</span>
                    <span class="metric-value">{{ (optimizationResults.expectedReturn * 100).toFixed(2) }}%</span>
                  </div>
                  <div class="metric-item">
                    <span class="metric-label">预期风险</span>
                    <span class="metric-value">{{ (optimizationResults.expectedRisk * 100).toFixed(2) }}%</span>
                  </div>
                  <div class="metric-item">
                    <span class="metric-label">夏普比率</span>
                    <span class="metric-value">{{ optimizationResults.sharpeRatio.toFixed(3) }}</span>
                  </div>
                  <div class="metric-item">
                    <span class="metric-label">信息比率</span>
                    <span class="metric-value">{{ optimizationResults.informationRatio.toFixed(3) }}</span>
                  </div>
                </div>
              </div>
            </el-col>
            <el-col :xs="24" :md="12">
              <div class="factor-weights">
                <h4>因子权重分配</h4>
                <div class="chart-container" style="height: 300px;">
                  <v-chart :option="factorWeightsOption" autoresize />
                </div>
              </div>
            </el-col>
          </el-row>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, HeatmapChart, PieChart, BarChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  VisualMapComponent
} from 'echarts/components'
import VChart from 'vue-echarts'
import { apiService } from '@/services/api'
import { ElMessage } from 'element-plus'

// 注册ECharts组件
use([
  CanvasRenderer,
  LineChart,
  HeatmapChart,
  PieChart,
  BarChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  VisualMapComponent
])

// 响应式数据
const analysisLoading = ref(false)
const optimizationLoading = ref(false)
const selectedFactorCategories = ref(['value', 'growth'])
const analysisPeriod = ref('1Y')
const correlationViewType = ref('heatmap')
const selectedFactorForLayering = ref('')

// 因子选择
const selectedFactors = ref<Record<string, string[]>>({
  value: [],
  growth: [],
  quality: [],
  momentum: [],
  volatility: [],
  size: []
})

// 可用因子定义
const availableFactors = ref({
  value: [
    { code: 'PE', name: '市盈率', description: '股价与每股收益的比率' },
    { code: 'PB', name: '市净率', description: '股价与每股净资产的比率' },
    { code: 'PS', name: '市销率', description: '股价与每股销售收入的比率' },
    { code: 'PCF', name: '市现率', description: '股价与每股现金流的比率' },
    { code: 'EV_EBITDA', name: 'EV/EBITDA', description: '企业价值与息税折旧摊销前利润的比率' }
  ],
  growth: [
    { code: 'REVENUE_GROWTH', name: '营收增长率', description: '营业收入同比增长率' },
    { code: 'PROFIT_GROWTH', name: '利润增长率', description: '净利润同比增长率' },
    { code: 'EPS_GROWTH', name: 'EPS增长率', description: '每股收益同比增长率' },
    { code: 'ROE_GROWTH', name: 'ROE增长率', description: '净资产收益率同比增长率' }
  ],
  quality: [
    { code: 'ROE', name: '净资产收益率', description: '净利润与净资产的比率' },
    { code: 'ROA', name: '总资产收益率', description: '净利润与总资产的比率' },
    { code: 'GROSS_MARGIN', name: '毛利率', description: '毛利润与营业收入的比率' },
    { code: 'DEBT_RATIO', name: '资产负债率', description: '总负债与总资产的比率' },
    { code: 'CURRENT_RATIO', name: '流动比率', description: '流动资产与流动负债的比率' }
  ],
  momentum: [
    { code: 'RETURN_1M', name: '1个月收益率', description: '过去1个月的股价收益率' },
    { code: 'RETURN_3M', name: '3个月收益率', description: '过去3个月的股价收益率' },
    { code: 'RETURN_6M', name: '6个月收益率', description: '过去6个月的股价收益率' },
    { code: 'RETURN_12M', name: '12个月收益率', description: '过去12个月的股价收益率' },
    { code: 'RSI', name: 'RSI指标', description: '相对强弱指数' }
  ],
  volatility: [
    { code: 'VOLATILITY_1M', name: '1个月波动率', description: '过去1个月的收益率标准差' },
    { code: 'VOLATILITY_3M', name: '3个月波动率', description: '过去3个月的收益率标准差' },
    { code: 'BETA', name: 'Beta系数', description: '相对于市场的系统性风险' },
    { code: 'MAX_DRAWDOWN', name: '最大回撤', description: '历史最大回撤幅度' }
  ],
  size: [
    { code: 'MARKET_CAP', name: '总市值', description: '公司总市值' },
    { code: 'FLOAT_MARKET_CAP', name: '流通市值', description: '流通股市值' },
    { code: 'TOTAL_ASSETS', name: '总资产', description: '公司总资产规模' }
  ]
})

// 分析结果
const analysisResults = ref<any>(null)
const factorStats = ref<any>(null)
const layeredBacktestData = ref<any>(null)
const optimizationResults = ref<any>(null)

// 计算属性
const getTotalSelectedFactors = () => {
  return Object.values(selectedFactors.value).flat().length
}

const getSelectedFactorsList = () => {
  const factors: any[] = []
  Object.entries(selectedFactors.value).forEach(([category, codes]) => {
    codes.forEach(code => {
      const factor = availableFactors.value[category as keyof typeof availableFactors.value]
        .find(f => f.code === code)
      if (factor) {
        factors.push(factor)
      }
    })
  })
  return factors
}

// 图表配置
const correlationChartOption = computed(() => {
  if (!analysisResults.value?.correlationMatrix) return {}
  
  const matrix = analysisResults.value.correlationMatrix
  const factors = matrix.factors
  const data = []
  
  for (let i = 0; i < factors.length; i++) {
    for (let j = 0; j < factors.length; j++) {
      data.push([i, j, matrix.values[i][j]])
    }
  }
  
  return {
    tooltip: {
      position: 'top',
      formatter: (params: any) => {
        return `${factors[params.data[1]]} vs ${factors[params.data[0]]}<br/>相关系数: ${params.data[2].toFixed(3)}`
      }
    },
    grid: {
      height: '50%',
      top: '10%'
    },
    xAxis: {
      type: 'category',
      data: factors,
      splitArea: {
        show: true
      }
    },
    yAxis: {
      type: 'category',
      data: factors,
      splitArea: {
        show: true
      }
    },
    visualMap: {
      min: -1,
      max: 1,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: '15%',
      inRange: {
        color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
      }
    },
    series: [{
      name: '相关系数',
      type: 'heatmap',
      data: data,
      label: {
        show: true,
        formatter: (params: any) => params.data[2].toFixed(2)
      },
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      }
    }]
  }
})

const factorReturnsOption = computed(() => {
  if (!analysisResults.value?.factorReturns) return {}
  
  const data = analysisResults.value.factorReturns
  const series = data.factors.map((factor: string, index: number) => ({
    name: factor,
    type: 'line',
    data: data.returns[index],
    smooth: true
  }))
  
  return {
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      data: data.factors
    },
    xAxis: {
      type: 'category',
      data: data.dates
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: '{value}%'
      }
    },
    series
  }
})

const cumulativeReturnsOption = computed(() => {
  if (!analysisResults.value?.cumulativeReturns) return {}
  
  const data = analysisResults.value.cumulativeReturns
  const series = data.factors.map((factor: string, index: number) => ({
    name: factor,
    type: 'line',
    data: data.returns[index].map((v: number) => (v * 100).toFixed(2)),
    smooth: true
  }))
  
  return {
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        let result = `${params[0].name}<br/>`
        params.forEach((param: any) => {
          result += `${param.seriesName}: ${param.value}%<br/>`
        })
        return result
      }
    },
    legend: {
      data: data.factors
    },
    xAxis: {
      type: 'category',
      data: data.dates
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: '{value}%'
      }
    },
    series
  }
})

const layeredBacktestOption = computed(() => {
  if (!layeredBacktestData.value) return {}
  
  const data = layeredBacktestData.value
  const series = data.layers.map((layer: any) => ({
    name: `第${layer.layer}层`,
    type: 'line',
    data: layer.cumulativeReturns.map((v: number) => (v * 100).toFixed(2)),
    smooth: true
  }))
  
  return {
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      data: series.map((s: any) => s.name)
    },
    xAxis: {
      type: 'category',
      data: data.dates
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: '{value}%'
      }
    },
    series
  }
})

const factorWeightsOption = computed(() => {
  if (!optimizationResults.value?.factorWeights) return {}
  
  const weights = optimizationResults.value.factorWeights
  
  return {
    tooltip: {
      trigger: 'item',
      formatter: '{a} <br/>{b}: {c}% ({d}%)'
    },
    series: [{
      name: '因子权重',
      type: 'pie',
      radius: '50%',
      data: weights.map((item: any) => ({
        value: (item.weight * 100).toFixed(2),
        name: item.factorName
      })),
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      }
    }]
  }
})

// 方法
const handleCategoryChange = () => {
  // 清空未选中类别的因子
  Object.keys(selectedFactors.value).forEach(category => {
    if (!selectedFactorCategories.value.includes(category)) {
      selectedFactors.value[category] = []
    }
  })
}

const runFactorAnalysis = async () => {
  const allSelectedFactors = Object.values(selectedFactors.value).flat()
  if (allSelectedFactors.length === 0) {
    ElMessage.warning('请至少选择一个因子')
    return
  }
  
  analysisLoading.value = true
  try {
    const response = await apiService.getFactorAnalysis({
      factors: allSelectedFactors,
      start_date: getStartDate(),
      end_date: new Date().toISOString().split('T')[0]
    })
    
    analysisResults.value = response
    
    // 更新统计信息
    factorStats.value = {
      coveredStocks: response.coveredStocks,
      dataCompleteness: response.dataCompleteness,
      lastUpdate: new Date().toLocaleString()
    }
    
    ElMessage.success('因子分析完成')
    
  } catch (error) {
    console.error('因子分析失败:', error)
    ElMessage.error('因子分析失败')
  } finally {
    analysisLoading.value = false
  }
}

const optimizeFactorCombination = async () => {
  if (!analysisResults.value) {
    ElMessage.warning('请先运行因子分析')
    return
  }
  
  optimizationLoading.value = true
  try {
    const response = await apiService.optimizeFactorCombination({
      factors: Object.values(selectedFactors.value).flat(),
      objective: 'sharpe_ratio',
      constraints: {
        max_weight: 0.3,
        min_weight: 0.05
      }
    })
    
    optimizationResults.value = response
    ElMessage.success('因子组合优化完成')
    
  } catch (error) {
    console.error('因子组合优化失败:', error)
    ElMessage.error('因子组合优化失败')
  } finally {
    optimizationLoading.value = false
  }
}

const getStartDate = () => {
  const now = new Date()
  const periodMap: Record<string, number> = {
    '1M': 1,
    '3M': 3,
    '6M': 6,
    '1Y': 12,
    '2Y': 24
  }
  
  const months = periodMap[analysisPeriod.value] || 12
  now.setMonth(now.getMonth() - months)
  return now.toISOString().split('T')[0]
}

const getFactorCategoryName = (category: string) => {
  const nameMap: Record<string, string> = {
    value: '价值因子',
    growth: '成长因子',
    quality: '质量因子',
    momentum: '动量因子',
    volatility: '波动率因子',
    size: '规模因子'
  }
  return nameMap[category] || category
}

const getICClass = (ic: number) => {
  if (Math.abs(ic) >= 0.05) return 'ic-excellent'
  if (Math.abs(ic) >= 0.03) return 'ic-good'
  if (Math.abs(ic) >= 0.01) return 'ic-fair'
  return 'ic-poor'
}

const getIRClass = (ir: number) => {
  if (Math.abs(ir) >= 0.5) return 'ir-excellent'
  if (Math.abs(ir) >= 0.3) return 'ir-good'
  if (Math.abs(ir) >= 0.1) return 'ir-fair'
  return 'ir-poor'
}

const getFactorRatingType = (rating: string) => {
  const typeMap: Record<string, string> = {
    'A': 'success',
    'B': 'primary',
    'C': 'warning',
    'D': 'danger'
  }
  return typeMap[rating] || 'info'
}

const getPriceChangeClass = (change: number) => {
  if (change > 0) return 'price-up'
  if (change < 0) return 'price-down'
  return 'price-neutral'
}

const viewFactorDetail = (factor: any) => {
  console.log('查看因子详情:', factor)
  ElMessage.info('因子详情功能开发中...')
}

const addToStrategy = (factor: any) => {
  console.log('添加到策略:', factor)
  ElMessage.success(`已将 ${factor.factorName} 添加到策略`)
}

// 监听选中因子变化，进行分层回测
watch(selectedFactorForLayering, async (newFactor) => {
  if (newFactor && analysisResults.value) {
    try {
      const response = await apiService.getFactorLayeredBacktest({
        factor: newFactor,
        layers: 5,
        start_date: getStartDate(),
        end_date: new Date().toISOString().split('T')[0]
      })
      layeredBacktestData.value = response
    } catch (error) {
      console.error('分层回测失败:', error)
    }
  }
})

// 生命周期
onMounted(() => {
  // 初始化选中一些默认因子
  selectedFactors.value.value = ['PE', 'PB']
  selectedFactors.value.growth = ['REVENUE_GROWTH', 'PROFIT_GROWTH']
})
</script>

<style scoped>
.factor-analysis {
  padding: 0;
}

.factor-config-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.factor-selection {
  margin-top: 20px;
}

.factor-category-card {
  height: 300px;
  overflow-y: auto;
}

.factor-item {
  margin-bottom: 10px;
}

.factor-info {
  display: flex;
  align-items: center;
  gap: 5px;
  width: 100%;
}

.factor-name {
  flex: 1;
}

.factor-help {
  color: #909399;
  cursor: help;
}

.factor-overview {
  margin-bottom: 20px;
}

.analysis-results {
  space-y: 20px;
}

.correlation-card,
.effectiveness-card,
.layered-backtest-card,
.optimization-card {
  margin-bottom: 20px;
}

.factor-returns-section {
  margin-bottom: 20px;
}

.chart-container {
  height: 400px;
}

.layer-stats-table {
  margin-top: 20px;
}

.optimization-results {
  margin-top: 20px;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 20px;
}

.metric-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px;
  background-color: #f8f9fa;
  border-radius: 6px;
}

.metric-label {
  color: #606266;
  font-size: 14px;
}

.metric-value {
  color: #303133;
  font-weight: 500;
  font-size: 16px;
}

/* IC和IR评级颜色 */
.ic-excellent,
.ir-excellent {
  color: #67c23a;
  font-weight: bold;
}

.ic-good,
.ir-good {
  color: #409eff;
  font-weight: bold;
}

.ic-fair,
.ir-fair {
  color: #e6a23c;
}

.ic-poor,
.ir-poor {
  color: #f56c6c;
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

/* 响应式设计 */
@media (max-width: 768px) {
  .chart-container {
    height: 300px;
  }
  
  .metric-grid {
    grid-template-columns: 1fr;
  }
  
  .factor-category-card {
    height: 250px;
  }
}
</style>