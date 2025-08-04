<template>
  <div class="strategy-analysis">
    <!-- 策略选择和配置 -->
    <el-card class="strategy-config-card">
      <template #header>
        <div class="card-header">
          <span>策略配置</span>
          <el-button type="primary" @click="runAnalysis" :loading="analysisLoading">
            <el-icon><DataAnalysis /></el-icon>
            运行分析
          </el-button>
        </div>
      </template>
      
      <el-row :gutter="20">
        <el-col :xs="24" :sm="12" :md="8">
          <el-form-item label="选择策略">
            <el-select v-model="selectedStrategy" placeholder="请选择策略" @change="handleStrategyChange">
              <el-option
                v-for="strategy in strategies"
                :key="strategy.id"
                :label="strategy.name"
                :value="strategy.id"
              >
                <div class="strategy-option">
                  <span>{{ strategy.name }}</span>
                  <el-tag size="small" :type="getStrategyTypeColor(strategy.type)">{{ strategy.type }}</el-tag>
                </div>
              </el-option>
            </el-select>
          </el-form-item>
        </el-col>
        
        <el-col :xs="24" :sm="12" :md="8">
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
        
        <el-col :xs="24" :sm="12" :md="8">
          <el-form-item label="基准指数">
            <el-select v-model="benchmarkIndex" placeholder="选择基准指数">
              <el-option label="沪深300" value="000300" />
              <el-option label="中证500" value="000905" />
              <el-option label="创业板指" value="399006" />
              <el-option label="科创50" value="000688" />
            </el-select>
          </el-form-item>
        </el-col>
      </el-row>
      
      <!-- 策略参数配置 -->
      <div v-if="selectedStrategyConfig" class="strategy-params">
        <el-divider content-position="left">策略参数</el-divider>
        <el-row :gutter="20">
          <el-col :xs="24" :sm="12" :md="8" v-for="param in selectedStrategyConfig.parameters" :key="param.name">
            <el-form-item :label="param.label">
              <el-input-number
                v-if="param.type === 'number'"
                v-model="strategyParams[param.name]"
                :min="param.min"
                :max="param.max"
                :step="param.step"
                :precision="param.precision"
              />
              <el-select
                v-else-if="param.type === 'select'"
                v-model="strategyParams[param.name]"
                :placeholder="param.placeholder"
              >
                <el-option
                  v-for="option in param.options"
                  :key="option.value"
                  :label="option.label"
                  :value="option.value"
                />
              </el-select>
              <el-switch
                v-else-if="param.type === 'boolean'"
                v-model="strategyParams[param.name]"
              />
            </el-form-item>
          </el-col>
        </el-row>
      </div>
    </el-card>

    <!-- 分析结果概览 -->
    <el-row :gutter="20" class="results-overview" v-if="analysisResults">
      <el-col :xs="12" :sm="6">
        <el-statistic title="总收益率" :value="analysisResults.totalReturn" suffix="%" :precision="2">
          <template #prefix>
            <el-icon :color="analysisResults.totalReturn >= 0 ? '#67c23a' : '#f56c6c'">
              <TrendCharts />
            </el-icon>
          </template>
        </el-statistic>
      </el-col>
      <el-col :xs="12" :sm="6">
        <el-statistic title="年化收益率" :value="analysisResults.annualizedReturn" suffix="%" :precision="2">
          <template #prefix>
            <el-icon color="#409eff"><DataAnalysis /></el-icon>
          </template>
        </el-statistic>
      </el-col>
      <el-col :xs="12" :sm="6">
        <el-statistic title="最大回撤" :value="analysisResults.maxDrawdown" suffix="%" :precision="2">
          <template #prefix>
            <el-icon color="#e6a23c"><Warning /></el-icon>
          </template>
        </el-statistic>
      </el-col>
      <el-col :xs="12" :sm="6">
        <el-statistic title="夏普比率" :value="analysisResults.sharpeRatio" :precision="3">
          <template #prefix>
            <el-icon color="#909399"><Odometer /></el-icon>
          </template>
        </el-statistic>
      </el-col>
    </el-row>

    <!-- 图表分析 -->
    <el-row :gutter="20" class="charts-section" v-if="analysisResults">
      <!-- 收益率曲线 -->
      <el-col :xs="24" :lg="12">
        <el-card class="chart-card">
          <template #header>
            <div class="chart-header">
              <span>收益率曲线</span>
              <el-radio-group v-model="returnChartType" size="small">
                <el-radio-button label="cumulative">累计收益</el-radio-button>
                <el-radio-button label="daily">日收益</el-radio-button>
              </el-radio-group>
            </div>
          </template>
          <div class="chart-container">
            <v-chart :option="returnChartOption" autoresize />
          </div>
        </el-card>
      </el-col>
      
      <!-- 回撤分析 -->
      <el-col :xs="24" :lg="12">
        <el-card class="chart-card">
          <template #header>
            <span>回撤分析</span>
          </template>
          <div class="chart-container">
            <v-chart :option="drawdownChartOption" autoresize />
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 详细分析指标 -->
    <el-row :gutter="20" class="metrics-section" v-if="analysisResults">
      <!-- 风险指标 -->
      <el-col :xs="24" :md="12">
        <el-card class="metrics-card">
          <template #header>
            <span>风险指标</span>
          </template>
          <div class="metrics-grid">
            <div class="metric-item">
              <span class="metric-label">波动率</span>
              <span class="metric-value">{{ analysisResults.volatility.toFixed(3) }}</span>
            </div>
            <div class="metric-item">
              <span class="metric-label">下行风险</span>
              <span class="metric-value">{{ analysisResults.downsideRisk.toFixed(3) }}</span>
            </div>
            <div class="metric-item">
              <span class="metric-label">VaR (95%)</span>
              <span class="metric-value">{{ analysisResults.var95.toFixed(3) }}</span>
            </div>
            <div class="metric-item">
              <span class="metric-label">CVaR (95%)</span>
              <span class="metric-value">{{ analysisResults.cvar95.toFixed(3) }}</span>
            </div>
            <div class="metric-item">
              <span class="metric-label">Beta系数</span>
              <span class="metric-value">{{ analysisResults.beta.toFixed(3) }}</span>
            </div>
            <div class="metric-item">
              <span class="metric-label">Alpha系数</span>
              <span class="metric-value">{{ analysisResults.alpha.toFixed(3) }}</span>
            </div>
          </div>
        </el-card>
      </el-col>
      
      <!-- 交易指标 -->
      <el-col :xs="24" :md="12">
        <el-card class="metrics-card">
          <template #header>
            <span>交易指标</span>
          </template>
          <div class="metrics-grid">
            <div class="metric-item">
              <span class="metric-label">总交易次数</span>
              <span class="metric-value">{{ analysisResults.totalTrades }}</span>
            </div>
            <div class="metric-item">
              <span class="metric-label">胜率</span>
              <span class="metric-value">{{ (analysisResults.winRate * 100).toFixed(2) }}%</span>
            </div>
            <div class="metric-item">
              <span class="metric-label">平均盈利</span>
              <span class="metric-value">{{ (analysisResults.avgWin * 100).toFixed(2) }}%</span>
            </div>
            <div class="metric-item">
              <span class="metric-label">平均亏损</span>
              <span class="metric-value">{{ (analysisResults.avgLoss * 100).toFixed(2) }}%</span>
            </div>
            <div class="metric-item">
              <span class="metric-label">盈亏比</span>
              <span class="metric-value">{{ analysisResults.profitLossRatio.toFixed(2) }}</span>
            </div>
            <div class="metric-item">
              <span class="metric-label">换手率</span>
              <span class="metric-value">{{ (analysisResults.turnoverRate * 100).toFixed(2) }}%</span>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 持仓分析 -->
    <el-row :gutter="20" class="holdings-section" v-if="analysisResults">
      <!-- 行业分布 -->
      <el-col :xs="24" :lg="12">
        <el-card class="chart-card">
          <template #header>
            <span>行业分布</span>
          </template>
          <div class="chart-container">
            <v-chart :option="industryDistributionOption" autoresize />
          </div>
        </el-card>
      </el-col>
      
      <!-- 持仓明细 -->
      <el-col :xs="24" :lg="12">
        <el-card class="holdings-card">
          <template #header>
            <div class="card-header">
              <span>当前持仓</span>
              <el-button type="text" @click="exportHoldings">
                <el-icon><Download /></el-icon>
                导出
              </el-button>
            </div>
          </template>
          <el-table :data="analysisResults.currentHoldings" max-height="400">
            <el-table-column prop="symbol" label="股票代码" width="100" />
            <el-table-column prop="name" label="股票名称" min-width="120" />
            <el-table-column prop="weight" label="权重" width="80">
              <template #default="{ row }">
                {{ (row.weight * 100).toFixed(2) }}%
              </template>
            </el-table-column>
            <el-table-column prop="return" label="收益率" width="80">
              <template #default="{ row }">
                <span :class="getPriceChangeClass(row.return)">
                  {{ (row.return * 100).toFixed(2) }}%
                </span>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
    </el-row>

    <!-- 策略对比 -->
    <el-card class="comparison-card" v-if="comparisonData">
      <template #header>
        <div class="card-header">
          <span>策略对比</span>
          <el-button @click="addComparisonStrategy">
            <el-icon><Plus /></el-icon>
            添加对比策略
          </el-button>
        </div>
      </template>
      
      <el-table :data="comparisonData" stripe>
        <el-table-column prop="strategyName" label="策略名称" min-width="150" />
        <el-table-column prop="totalReturn" label="总收益率" width="120">
          <template #default="{ row }">
            <span :class="getPriceChangeClass(row.totalReturn)">
              {{ (row.totalReturn * 100).toFixed(2) }}%
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="annualizedReturn" label="年化收益率" width="120">
          <template #default="{ row }">
            {{ (row.annualizedReturn * 100).toFixed(2) }}%
          </template>
        </el-table-column>
        <el-table-column prop="maxDrawdown" label="最大回撤" width="120">
          <template #default="{ row }">
            {{ (row.maxDrawdown * 100).toFixed(2) }}%
          </template>
        </el-table-column>
        <el-table-column prop="sharpeRatio" label="夏普比率" width="120">
          <template #default="{ row }">
            {{ row.sharpeRatio.toFixed(3) }}
          </template>
        </el-table-column>
        <el-table-column prop="volatility" label="波动率" width="120">
          <template #default="{ row }">
            {{ (row.volatility * 100).toFixed(2) }}%
          </template>
        </el-table-column>
        <el-table-column label="操作" width="100">
          <template #default="{ row, $index }">
            <el-button type="text" size="small" @click="removeComparison($index)">
              移除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, PieChart, BarChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
} from 'echarts/components'
import VChart from 'vue-echarts'
import { apiService } from '@/services/api'
import { ElMessage } from 'element-plus'

// 注册ECharts组件
use([
  CanvasRenderer,
  LineChart,
  PieChart,
  BarChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
])

// 响应式数据
const analysisLoading = ref(false)
const selectedStrategy = ref('')
const analysisPeriod = ref('1Y')
const benchmarkIndex = ref('000300')
const returnChartType = ref('cumulative')
const strategyParams = ref<Record<string, any>>({})

// 策略数据
const strategies = ref([
  {
    id: 'momentum',
    name: '动量策略',
    type: '趋势跟踪',
    description: '基于价格动量的投资策略'
  },
  {
    id: 'mean_reversion',
    name: '均值回归策略',
    type: '反转策略',
    description: '基于价格均值回归的投资策略'
  },
  {
    id: 'factor_model',
    name: '多因子模型',
    type: '因子投资',
    description: '基于多个因子的量化投资策略'
  },
  {
    id: 'ml_prediction',
    name: '机器学习预测',
    type: 'AI策略',
    description: '基于机器学习的股票预测策略'
  }
])

const strategyConfigs = ref({
  momentum: {
    parameters: [
      { name: 'lookback_period', label: '回看周期', type: 'number', min: 5, max: 60, step: 1, precision: 0 },
      { name: 'rebalance_freq', label: '调仓频率', type: 'select', options: [
        { label: '日', value: 'daily' },
        { label: '周', value: 'weekly' },
        { label: '月', value: 'monthly' }
      ]},
      { name: 'top_n', label: '选股数量', type: 'number', min: 10, max: 100, step: 5, precision: 0 }
    ]
  },
  mean_reversion: {
    parameters: [
      { name: 'window_size', label: '窗口大小', type: 'number', min: 10, max: 100, step: 5, precision: 0 },
      { name: 'threshold', label: '阈值', type: 'number', min: 0.1, max: 3.0, step: 0.1, precision: 1 },
      { name: 'holding_period', label: '持有周期', type: 'number', min: 1, max: 30, step: 1, precision: 0 }
    ]
  },
  factor_model: {
    parameters: [
      { name: 'factors', label: '因子选择', type: 'select', options: [
        { label: '价值因子', value: 'value' },
        { label: '成长因子', value: 'growth' },
        { label: '质量因子', value: 'quality' },
        { label: '动量因子', value: 'momentum' }
      ]},
      { name: 'weight_method', label: '权重方法', type: 'select', options: [
        { label: '等权重', value: 'equal' },
        { label: '市值权重', value: 'market_cap' },
        { label: '风险平价', value: 'risk_parity' }
      ]}
    ]
  },
  ml_prediction: {
    parameters: [
      { name: 'model_type', label: '模型类型', type: 'select', options: [
        { label: 'Random Forest', value: 'rf' },
        { label: 'XGBoost', value: 'xgb' },
        { label: 'LSTM', value: 'lstm' }
      ]},
      { name: 'prediction_horizon', label: '预测周期', type: 'number', min: 1, max: 30, step: 1, precision: 0 },
      { name: 'confidence_threshold', label: '置信度阈值', type: 'number', min: 0.5, max: 0.95, step: 0.05, precision: 2 }
    ]
  }
})

// 分析结果
const analysisResults = ref<any>(null)
const comparisonData = ref<any[]>([])

// 计算属性
const selectedStrategyConfig = computed(() => {
  return selectedStrategy.value ? strategyConfigs.value[selectedStrategy.value as keyof typeof strategyConfigs.value] : null
})

// 图表配置
const returnChartOption = computed(() => {
  if (!analysisResults.value) return {}
  
  const data = returnChartType.value === 'cumulative' 
    ? analysisResults.value.cumulativeReturns
    : analysisResults.value.dailyReturns
    
  return {
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        const point = params[0]
        return `${point.name}<br/>${point.seriesName}: ${(point.value * 100).toFixed(2)}%`
      }
    },
    legend: {
      data: ['策略收益', '基准收益']
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
    series: [
      {
        name: '策略收益',
        type: 'line',
        data: data.strategy.map((v: number) => (v * 100).toFixed(2)),
        itemStyle: { color: '#409eff' },
        smooth: true
      },
      {
        name: '基准收益',
        type: 'line',
        data: data.benchmark.map((v: number) => (v * 100).toFixed(2)),
        itemStyle: { color: '#67c23a' },
        smooth: true
      }
    ]
  }
})

const drawdownChartOption = computed(() => {
  if (!analysisResults.value) return {}
  
  return {
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        const point = params[0]
        return `${point.name}<br/>回撤: ${(point.value * 100).toFixed(2)}%`
      }
    },
    xAxis: {
      type: 'category',
      data: analysisResults.value.drawdownData.dates
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: '{value}%'
      }
    },
    series: [
      {
        name: '回撤',
        type: 'line',
        data: analysisResults.value.drawdownData.values.map((v: number) => (v * 100).toFixed(2)),
        itemStyle: { color: '#f56c6c' },
        areaStyle: {
          color: 'rgba(245, 108, 108, 0.3)'
        },
        smooth: true
      }
    ]
  }
})

const industryDistributionOption = computed(() => {
  if (!analysisResults.value) return {}
  
  return {
    tooltip: {
      trigger: 'item',
      formatter: '{a} <br/>{b}: {c}% ({d}%)'
    },
    legend: {
      orient: 'vertical',
      left: 'left'
    },
    series: [
      {
        name: '行业分布',
        type: 'pie',
        radius: '50%',
        data: analysisResults.value.industryDistribution.map((item: any) => ({
          value: (item.weight * 100).toFixed(2),
          name: item.industry
        })),
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }
    ]
  }
})

// 方法
const handleStrategyChange = () => {
  // 重置策略参数
  strategyParams.value = {}
  if (selectedStrategyConfig.value) {
    selectedStrategyConfig.value.parameters.forEach(param => {
      if (param.type === 'number') {
        strategyParams.value[param.name] = param.min
      } else if (param.type === 'boolean') {
        strategyParams.value[param.name] = false
      } else {
        strategyParams.value[param.name] = ''
      }
    })
  }
}

const runAnalysis = async () => {
  if (!selectedStrategy.value) {
    ElMessage.warning('请先选择策略')
    return
  }
  
  analysisLoading.value = true
  try {
    const response = await apiService.runBacktest({
      strategy_id: selectedStrategy.value,
      start_date: getStartDate(),
      end_date: new Date().toISOString().split('T')[0],
      benchmark: benchmarkIndex.value,
      parameters: strategyParams.value
    })
    
    analysisResults.value = response
    ElMessage.success('策略分析完成')
    
  } catch (error) {
    console.error('策略分析失败:', error)
    ElMessage.error('策略分析失败')
  } finally {
    analysisLoading.value = false
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

const addComparisonStrategy = () => {
  // 添加对比策略的逻辑
  ElMessage.info('添加对比策略功能开发中...')
}

const removeComparison = (index: number) => {
  comparisonData.value.splice(index, 1)
}

const exportHoldings = () => {
  // 导出持仓数据
  ElMessage.info('导出功能开发中...')
}

const getStrategyTypeColor = (type: string) => {
  const colorMap: Record<string, string> = {
    '趋势跟踪': 'primary',
    '反转策略': 'success',
    '因子投资': 'warning',
    'AI策略': 'danger'
  }
  return colorMap[type] || 'info'
}

const getPriceChangeClass = (change: number) => {
  if (change > 0) return 'price-up'
  if (change < 0) return 'price-down'
  return 'price-neutral'
}

// 生命周期
onMounted(() => {
  // 初始化默认策略
  if (strategies.value.length > 0) {
    selectedStrategy.value = strategies.value[0].id
    handleStrategyChange()
  }
})
</script>

<style scoped>
.strategy-analysis {
  padding: 0;
}

.strategy-config-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.strategy-option {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.strategy-params {
  margin-top: 20px;
}

.results-overview {
  margin-bottom: 20px;
}

.charts-section,
.metrics-section,
.holdings-section {
  margin-bottom: 20px;
}

.chart-card,
.metrics-card,
.holdings-card,
.comparison-card {
  margin-bottom: 20px;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chart-container {
  height: 400px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
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
  
  .metrics-grid {
    grid-template-columns: 1fr;
  }
  
  .chart-header {
    flex-direction: column;
    gap: 10px;
    align-items: stretch;
  }
}
</style>