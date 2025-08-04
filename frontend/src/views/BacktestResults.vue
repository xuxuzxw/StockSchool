<template>
  <div class="backtest-results">
    <!-- 回测配置和控制 -->
    <el-card class="backtest-config-card">
      <template #header>
        <div class="card-header">
          <span>回测配置</span>
          <div class="header-actions">
            <el-button @click="loadBacktestList" :loading="listLoading">
              <el-icon><Refresh /></el-icon>
              刷新
            </el-button>
            <el-button type="primary" @click="runNewBacktest" :loading="backtestLoading">
              <el-icon><VideoPlay /></el-icon>
              新建回测
            </el-button>
          </div>
        </div>
      </template>
      
      <el-row :gutter="20">
        <el-col :xs="24" :md="8">
          <el-form-item label="选择回测">
            <el-select v-model="selectedBacktest" placeholder="选择回测结果" @change="handleBacktestChange">
              <el-option
                v-for="backtest in backtestList"
                :key="backtest.id"
                :label="`${backtest.name} (${backtest.date})`"
                :value="backtest.id"
              >
                <div class="backtest-option">
                  <span>{{ backtest.name }}</span>
                  <el-tag size="small" :type="getBacktestStatusType(backtest.status)">{{ backtest.status }}</el-tag>
                </div>
              </el-option>
            </el-select>
          </el-form-item>
        </el-col>
        
        <el-col :xs="24" :md="8">
          <el-form-item label="对比基准">
            <el-select v-model="benchmarkIndex" placeholder="选择基准指数">
              <el-option label="沪深300" value="HS300" />
              <el-option label="中证500" value="ZZ500" />
              <el-option label="创业板指" value="CYBZ" />
              <el-option label="上证指数" value="SZZS" />
              <el-option label="深证成指" value="SZCZ" />
            </el-select>
          </el-form-item>
        </el-col>
        
        <el-col :xs="24" :md="8">
          <el-form-item label="分析周期">
            <el-select v-model="analysisPeriod" placeholder="选择分析周期">
              <el-option label="全部" value="all" />
              <el-option label="最近1年" value="1Y" />
              <el-option label="最近6个月" value="6M" />
              <el-option label="最近3个月" value="3M" />
              <el-option label="最近1个月" value="1M" />
            </el-select>
          </el-form-item>
        </el-col>
      </el-row>
    </el-card>

    <!-- 回测概览 -->
    <el-row :gutter="20" class="backtest-overview" v-if="backtestData">
      <el-col :xs="12" :sm="6">
        <el-statistic title="总收益率" :value="backtestData.summary.totalReturn" suffix="%" :precision="2">
          <template #prefix>
            <el-icon :class="getPriceChangeClass(backtestData.summary.totalReturn)">
              <component :is="backtestData.summary.totalReturn >= 0 ? 'TrendChartUp' : 'TrendChartDown'" />
            </el-icon>
          </template>
        </el-statistic>
      </el-col>
      <el-col :xs="12" :sm="6">
        <el-statistic title="年化收益率" :value="backtestData.summary.annualizedReturn" suffix="%" :precision="2" />
      </el-col>
      <el-col :xs="12" :sm="6">
        <el-statistic title="最大回撤" :value="backtestData.summary.maxDrawdown" suffix="%" :precision="2" />
      </el-col>
      <el-col :xs="12" :sm="6">
        <el-statistic title="夏普比率" :value="backtestData.summary.sharpeRatio" :precision="3" />
      </el-col>
    </el-row>

    <!-- 核心指标对比 -->
    <el-card v-if="backtestData" class="metrics-comparison-card">
      <template #header>
        <span>核心指标对比</span>
      </template>
      <el-table :data="metricsComparisonData" stripe>
        <el-table-column prop="metric" label="指标" min-width="120" />
        <el-table-column prop="strategy" label="策略" width="120">
          <template #default="{ row }">
            <span :class="getMetricClass(row.metric, row.strategy, row.benchmark)">
              {{ formatMetricValue(row.metric, row.strategy) }}
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="benchmark" label="基准" width="120">
          <template #default="{ row }">
            {{ formatMetricValue(row.metric, row.benchmark) }}
          </template>
        </el-table-column>
        <el-table-column prop="excess" label="超额" width="120">
          <template #default="{ row }">
            <span :class="getPriceChangeClass(row.excess)">
              {{ formatMetricValue(row.metric, row.excess, true) }}
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="description" label="说明" min-width="200" />
      </el-table>
    </el-card>

    <!-- 收益率曲线 -->
    <el-card v-if="backtestData" class="returns-chart-card">
      <template #header>
        <div class="card-header">
          <span>收益率曲线</span>
          <el-radio-group v-model="returnsChartType" size="small">
            <el-radio-button label="cumulative">累计收益</el-radio-button>
            <el-radio-button label="daily">日收益</el-radio-button>
            <el-radio-button label="rolling">滚动收益</el-radio-button>
          </el-radio-group>
        </div>
      </template>
      <div class="chart-container">
        <v-chart :option="returnsChartOption" autoresize />
      </div>
    </el-card>

    <!-- 回撤分析 -->
    <el-row :gutter="20" class="drawdown-section" v-if="backtestData">
      <el-col :xs="24" :lg="16">
        <el-card class="drawdown-chart-card">
          <template #header>
            <span>回撤分析</span>
          </template>
          <div class="chart-container">
            <v-chart :option="drawdownChartOption" autoresize />
          </div>
        </el-card>
      </el-col>
      
      <el-col :xs="24" :lg="8">
        <el-card class="drawdown-stats-card">
          <template #header>
            <span>回撤统计</span>
          </template>
          <div class="drawdown-stats">
            <div class="stat-item">
              <span class="stat-label">最大回撤</span>
              <span class="stat-value negative">{{ backtestData.drawdownAnalysis.maxDrawdown.toFixed(2) }}%</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">最大回撤期间</span>
              <span class="stat-value">{{ backtestData.drawdownAnalysis.maxDrawdownPeriod }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">回撤次数</span>
              <span class="stat-value">{{ backtestData.drawdownAnalysis.drawdownCount }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">平均回撤</span>
              <span class="stat-value">{{ backtestData.drawdownAnalysis.avgDrawdown.toFixed(2) }}%</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">回撤恢复时间</span>
              <span class="stat-value">{{ backtestData.drawdownAnalysis.avgRecoveryDays }} 天</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">卡玛比率</span>
              <span class="stat-value">{{ backtestData.drawdownAnalysis.calmarRatio.toFixed(3) }}</span>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 风险分析 -->
    <el-card v-if="backtestData" class="risk-analysis-card">
      <template #header>
        <span>风险分析</span>
      </template>
      <el-row :gutter="20">
        <el-col :xs="24" :md="12">
          <div class="risk-metrics">
            <h4>风险指标</h4>
            <div class="metric-grid">
              <div class="metric-item">
                <span class="metric-label">年化波动率</span>
                <span class="metric-value">{{ backtestData.riskAnalysis.volatility.toFixed(2) }}%</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">下行波动率</span>
                <span class="metric-value">{{ backtestData.riskAnalysis.downsideVolatility.toFixed(2) }}%</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">Beta系数</span>
                <span class="metric-value">{{ backtestData.riskAnalysis.beta.toFixed(3) }}</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">Alpha系数</span>
                <span class="metric-value">{{ backtestData.riskAnalysis.alpha.toFixed(3) }}</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">信息比率</span>
                <span class="metric-value">{{ backtestData.riskAnalysis.informationRatio.toFixed(3) }}</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">跟踪误差</span>
                <span class="metric-value">{{ backtestData.riskAnalysis.trackingError.toFixed(2) }}%</span>
              </div>
            </div>
          </div>
        </el-col>
        
        <el-col :xs="24" :md="12">
          <div class="chart-container" style="height: 300px;">
            <h4>收益分布</h4>
            <v-chart :option="returnsDistributionOption" autoresize />
          </div>
        </el-col>
      </el-row>
    </el-card>

    <!-- 交易分析 -->
    <el-card v-if="backtestData" class="trading-analysis-card">
      <template #header>
        <div class="card-header">
          <span>交易分析</span>
          <el-button @click="exportTradingDetails">
            <el-icon><Download /></el-icon>
            导出明细
          </el-button>
        </div>
      </template>
      
      <el-row :gutter="20">
        <el-col :xs="24" :lg="8">
          <div class="trading-stats">
            <h4>交易统计</h4>
            <div class="stat-grid">
              <div class="stat-item">
                <span class="stat-label">总交易次数</span>
                <span class="stat-value">{{ backtestData.tradingAnalysis.totalTrades }}</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">盈利交易</span>
                <span class="stat-value positive">{{ backtestData.tradingAnalysis.profitableTrades }}</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">亏损交易</span>
                <span class="stat-value negative">{{ backtestData.tradingAnalysis.losingTrades }}</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">胜率</span>
                <span class="stat-value">{{ backtestData.tradingAnalysis.winRate.toFixed(2) }}%</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">平均盈利</span>
                <span class="stat-value positive">{{ backtestData.tradingAnalysis.avgProfit.toFixed(2) }}%</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">平均亏损</span>
                <span class="stat-value negative">{{ backtestData.tradingAnalysis.avgLoss.toFixed(2) }}%</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">盈亏比</span>
                <span class="stat-value">{{ backtestData.tradingAnalysis.profitLossRatio.toFixed(2) }}</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">换手率</span>
                <span class="stat-value">{{ backtestData.tradingAnalysis.turnoverRate.toFixed(2) }}%</span>
              </div>
            </div>
          </div>
        </el-col>
        
        <el-col :xs="24" :lg="16">
          <div class="trading-charts">
            <el-row :gutter="10">
              <el-col :xs="24" :md="12">
                <div class="chart-container" style="height: 250px;">
                  <h4>月度收益分布</h4>
                  <v-chart :option="monthlyReturnsOption" autoresize />
                </div>
              </el-col>
              <el-col :xs="24" :md="12">
                <div class="chart-container" style="height: 250px;">
                  <h4>持仓周期分布</h4>
                  <v-chart :option="holdingPeriodOption" autoresize />
                </div>
              </el-col>
            </el-row>
          </div>
        </el-col>
      </el-row>
    </el-card>

    <!-- 持仓分析 -->
    <el-card v-if="backtestData" class="position-analysis-card">
      <template #header>
        <div class="card-header">
          <span>持仓分析</span>
          <el-date-picker
            v-model="positionDate"
            type="date"
            placeholder="选择日期"
            @change="handlePositionDateChange"
            style="width: 200px;"
          />
        </div>
      </template>
      
      <el-row :gutter="20" v-if="positionData">
        <el-col :xs="24" :lg="8">
          <div class="chart-container" style="height: 400px;">
            <h4>行业分布</h4>
            <v-chart :option="industryDistributionOption" autoresize />
          </div>
        </el-col>
        
        <el-col :xs="24" :lg="8">
          <div class="chart-container" style="height: 400px;">
            <h4>市值分布</h4>
            <v-chart :option="marketCapDistributionOption" autoresize />
          </div>
        </el-col>
        
        <el-col :xs="24" :lg="8">
          <div class="position-details">
            <h4>重仓股票</h4>
            <el-table :data="positionData.topHoldings" size="small" max-height="350">
              <el-table-column prop="code" label="代码" width="80" />
              <el-table-column prop="name" label="名称" min-width="100" />
              <el-table-column prop="weight" label="权重" width="80">
                <template #default="{ row }">
                  {{ (row.weight * 100).toFixed(2) }}%
                </template>
              </el-table-column>
              <el-table-column prop="return" label="收益" width="80">
                <template #default="{ row }">
                  <span :class="getPriceChangeClass(row.return)">
                    {{ (row.return * 100).toFixed(2) }}%
                  </span>
                </template>
              </el-table-column>
            </el-table>
          </div>
        </el-col>
      </el-row>
    </el-card>

    <!-- 策略参数敏感性分析 -->
    <el-card v-if="backtestData" class="sensitivity-analysis-card">
      <template #header>
        <div class="card-header">
          <span>参数敏感性分析</span>
          <el-button @click="runSensitivityAnalysis" :loading="sensitivityLoading">
            <el-icon><DataAnalysis /></el-icon>
            运行分析
          </el-button>
        </div>
      </template>
      
      <div v-if="sensitivityData" class="sensitivity-results">
        <el-row :gutter="20">
          <el-col :xs="24" :lg="16">
            <div class="chart-container">
              <v-chart :option="sensitivityHeatmapOption" autoresize />
            </div>
          </el-col>
          
          <el-col :xs="24" :lg="8">
            <div class="sensitivity-insights">
              <h4>敏感性洞察</h4>
              <ul class="insight-list">
                <li v-for="insight in sensitivityData.insights" :key="insight.id">
                  <el-icon><InfoFilled /></el-icon>
                  {{ insight.text }}
                </li>
              </ul>
              
              <div class="optimal-params">
                <h4>最优参数</h4>
                <div v-for="param in sensitivityData.optimalParams" :key="param.name" class="param-item">
                  <span class="param-name">{{ param.name }}</span>
                  <span class="param-value">{{ param.value }}</span>
                </div>
              </div>
            </div>
          </el-col>
        </el-row>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, BarChart, PieChart, HeatmapChart, ScatterChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  DataZoomComponent,
  VisualMapComponent
} from 'echarts/components'
import VChart from 'vue-echarts'
import { apiService } from '@/services/api'
import { ElMessage } from 'element-plus'

// 注册ECharts组件
use([
  CanvasRenderer,
  LineChart,
  BarChart,
  PieChart,
  HeatmapChart,
  ScatterChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  DataZoomComponent,
  VisualMapComponent
])

// 响应式数据
const listLoading = ref(false)
const backtestLoading = ref(false)
const sensitivityLoading = ref(false)
const selectedBacktest = ref('')
const benchmarkIndex = ref('HS300')
const analysisPeriod = ref('all')
const returnsChartType = ref('cumulative')
const positionDate = ref(new Date())

// 数据
const backtestList = ref<any[]>([])
const backtestData = ref<any>(null)
const positionData = ref<any>(null)
const sensitivityData = ref<any>(null)

// 计算属性
const metricsComparisonData = computed(() => {
  if (!backtestData.value) return []
  
  const strategy = backtestData.value.summary
  const benchmark = backtestData.value.benchmark
  
  return [
    {
      metric: '总收益率',
      strategy: strategy.totalReturn,
      benchmark: benchmark.totalReturn,
      excess: strategy.totalReturn - benchmark.totalReturn,
      description: '策略相对基准的总收益率'
    },
    {
      metric: '年化收益率',
      strategy: strategy.annualizedReturn,
      benchmark: benchmark.annualizedReturn,
      excess: strategy.annualizedReturn - benchmark.annualizedReturn,
      description: '年化收益率对比'
    },
    {
      metric: '最大回撤',
      strategy: strategy.maxDrawdown,
      benchmark: benchmark.maxDrawdown,
      excess: strategy.maxDrawdown - benchmark.maxDrawdown,
      description: '最大回撤对比，负值表示策略回撤更小'
    },
    {
      metric: '夏普比率',
      strategy: strategy.sharpeRatio,
      benchmark: benchmark.sharpeRatio,
      excess: strategy.sharpeRatio - benchmark.sharpeRatio,
      description: '风险调整后收益对比'
    },
    {
      metric: '波动率',
      strategy: backtestData.value.riskAnalysis.volatility,
      benchmark: benchmark.volatility,
      excess: backtestData.value.riskAnalysis.volatility - benchmark.volatility,
      description: '收益波动率对比'
    }
  ]
})

// 图表配置
const returnsChartOption = computed(() => {
  if (!backtestData.value) return {}
  
  const data = backtestData.value.returnsData
  let strategyData, benchmarkData
  
  switch (returnsChartType.value) {
    case 'cumulative':
      strategyData = data.cumulativeReturns.strategy
      benchmarkData = data.cumulativeReturns.benchmark
      break
    case 'daily':
      strategyData = data.dailyReturns.strategy
      benchmarkData = data.dailyReturns.benchmark
      break
    case 'rolling':
      strategyData = data.rollingReturns.strategy
      benchmarkData = data.rollingReturns.benchmark
      break
    default:
      strategyData = data.cumulativeReturns.strategy
      benchmarkData = data.cumulativeReturns.benchmark
  }
  
  return {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      }
    },
    legend: {
      data: ['策略', '基准']
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
    dataZoom: [
      {
        type: 'inside',
        start: 0,
        end: 100
      },
      {
        start: 0,
        end: 100
      }
    ],
    series: [
      {
        name: '策略',
        type: 'line',
        data: strategyData.map((v: number) => (v * 100).toFixed(2)),
        smooth: true,
        lineStyle: {
          color: '#5470c6',
          width: 2
        }
      },
      {
        name: '基准',
        type: 'line',
        data: benchmarkData.map((v: number) => (v * 100).toFixed(2)),
        smooth: true,
        lineStyle: {
          color: '#91cc75',
          width: 2
        }
      }
    ]
  }
})

const drawdownChartOption = computed(() => {
  if (!backtestData.value?.drawdownData) return {}
  
  const data = backtestData.value.drawdownData
  
  return {
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        const param = params[0]
        return `${param.name}<br/>回撤: ${param.value}%`
      }
    },
    xAxis: {
      type: 'category',
      data: data.dates
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: '{value}%'
      },
      max: 0
    },
    series: [{
      name: '回撤',
      type: 'line',
      data: data.drawdowns.map((v: number) => (v * 100).toFixed(2)),
      areaStyle: {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [{
            offset: 0, color: 'rgba(238, 102, 102, 0.3)'
          }, {
            offset: 1, color: 'rgba(238, 102, 102, 0.1)'
          }]
        }
      },
      lineStyle: {
        color: '#ee6666'
      }
    }]
  }
})

const returnsDistributionOption = computed(() => {
  if (!backtestData.value?.returnsDistribution) return {}
  
  const data = backtestData.value.returnsDistribution
  
  return {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      }
    },
    xAxis: {
      type: 'category',
      data: data.bins
    },
    yAxis: {
      type: 'value'
    },
    series: [{
      name: '频次',
      type: 'bar',
      data: data.frequencies,
      itemStyle: {
        color: '#5470c6'
      }
    }]
  }
})

const monthlyReturnsOption = computed(() => {
  if (!backtestData.value?.monthlyReturns) return {}
  
  const data = backtestData.value.monthlyReturns
  
  return {
    tooltip: {
      trigger: 'axis'
    },
    xAxis: {
      type: 'category',
      data: data.months
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: '{value}%'
      }
    },
    series: [{
      name: '月收益',
      type: 'bar',
      data: data.returns.map((v: number) => (v * 100).toFixed(2)),
      itemStyle: {
        color: (params: any) => {
          return params.value >= 0 ? '#67c23a' : '#f56c6c'
        }
      }
    }]
  }
})

const holdingPeriodOption = computed(() => {
  if (!backtestData.value?.holdingPeriods) return {}
  
  const data = backtestData.value.holdingPeriods
  
  return {
    tooltip: {
      trigger: 'item',
      formatter: '{a} <br/>{b}: {c} ({d}%)'
    },
    series: [{
      name: '持仓周期',
      type: 'pie',
      radius: '50%',
      data: data.map((item: any) => ({
        value: item.count,
        name: item.period
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

const industryDistributionOption = computed(() => {
  if (!positionData.value?.industryDistribution) return {}
  
  const data = positionData.value.industryDistribution
  
  return {
    tooltip: {
      trigger: 'item',
      formatter: '{a} <br/>{b}: {c}% ({d}%)'
    },
    series: [{
      name: '行业分布',
      type: 'pie',
      radius: ['40%', '70%'],
      data: data.map((item: any) => ({
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
    }]
  }
})

const marketCapDistributionOption = computed(() => {
  if (!positionData.value?.marketCapDistribution) return {}
  
  const data = positionData.value.marketCapDistribution
  
  return {
    tooltip: {
      trigger: 'item',
      formatter: '{a} <br/>{b}: {c}% ({d}%)'
    },
    series: [{
      name: '市值分布',
      type: 'pie',
      radius: '50%',
      data: data.map((item: any) => ({
        value: (item.weight * 100).toFixed(2),
        name: item.category
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

const sensitivityHeatmapOption = computed(() => {
  if (!sensitivityData.value?.heatmapData) return {}
  
  const data = sensitivityData.value.heatmapData
  
  return {
    tooltip: {
      position: 'top',
      formatter: (params: any) => {
        return `${data.yAxis[params.data[1]]} vs ${data.xAxis[params.data[0]]}<br/>收益率: ${params.data[2].toFixed(2)}%`
      }
    },
    grid: {
      height: '50%',
      top: '10%'
    },
    xAxis: {
      type: 'category',
      data: data.xAxis,
      splitArea: {
        show: true
      }
    },
    yAxis: {
      type: 'category',
      data: data.yAxis,
      splitArea: {
        show: true
      }
    },
    visualMap: {
      min: data.minValue,
      max: data.maxValue,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: '15%'
    },
    series: [{
      name: '收益率',
      type: 'heatmap',
      data: data.values,
      label: {
        show: true,
        formatter: (params: any) => params.data[2].toFixed(1)
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

// 方法
const loadBacktestList = async () => {
  listLoading.value = true
  try {
    const response = await apiService.getBacktestList()
    backtestList.value = response.backtests
    
    if (backtestList.value.length > 0 && !selectedBacktest.value) {
      selectedBacktest.value = backtestList.value[0].id
      await handleBacktestChange()
    }
  } catch (error) {
    console.error('获取回测列表失败:', error)
    ElMessage.error('获取回测列表失败')
  } finally {
    listLoading.value = false
  }
}

const handleBacktestChange = async () => {
  if (!selectedBacktest.value) return
  
  try {
    const response = await apiService.getBacktestResults({
      backtest_id: selectedBacktest.value,
      benchmark: benchmarkIndex.value,
      period: analysisPeriod.value
    })
    backtestData.value = response
    
    // 加载默认日期的持仓数据
    await handlePositionDateChange()
    
  } catch (error) {
    console.error('获取回测结果失败:', error)
    ElMessage.error('获取回测结果失败')
  }
}

const handlePositionDateChange = async () => {
  if (!selectedBacktest.value || !positionDate.value) return
  
  try {
    const response = await apiService.getPositionAnalysis({
      backtest_id: selectedBacktest.value,
      date: positionDate.value.toISOString().split('T')[0]
    })
    positionData.value = response
  } catch (error) {
    console.error('获取持仓分析失败:', error)
  }
}

const runNewBacktest = () => {
  ElMessage.info('新建回测功能开发中...')
}

const runSensitivityAnalysis = async () => {
  if (!selectedBacktest.value) {
    ElMessage.warning('请先选择回测结果')
    return
  }
  
  sensitivityLoading.value = true
  try {
    const response = await apiService.getSensitivityAnalysis({
      backtest_id: selectedBacktest.value
    })
    sensitivityData.value = response
    ElMessage.success('敏感性分析完成')
  } catch (error) {
    console.error('敏感性分析失败:', error)
    ElMessage.error('敏感性分析失败')
  } finally {
    sensitivityLoading.value = false
  }
}

const exportTradingDetails = () => {
  ElMessage.info('导出交易明细功能开发中...')
}

const getBacktestStatusType = (status: string) => {
  const typeMap: Record<string, string> = {
    'completed': 'success',
    'running': 'warning',
    'failed': 'danger',
    'pending': 'info'
  }
  return typeMap[status] || 'info'
}

const getPriceChangeClass = (change: number) => {
  if (change > 0) return 'positive'
  if (change < 0) return 'negative'
  return 'neutral'
}

const getMetricClass = (metric: string, strategyValue: number, benchmarkValue: number) => {
  const diff = strategyValue - benchmarkValue
  
  // 对于回撤，负值更好
  if (metric.includes('回撤')) {
    return diff < 0 ? 'positive' : 'negative'
  }
  
  // 对于其他指标，正值更好
  return diff > 0 ? 'positive' : 'negative'
}

const formatMetricValue = (metric: string, value: number, isExcess = false) => {
  if (typeof value !== 'number') return value
  
  if (metric.includes('比率')) {
    return value.toFixed(3)
  }
  
  const suffix = isExcess && value > 0 ? '+' : ''
  return suffix + value.toFixed(2) + '%'
}

// 监听器
watch([benchmarkIndex, analysisPeriod], () => {
  if (selectedBacktest.value) {
    handleBacktestChange()
  }
})

// 生命周期
onMounted(() => {
  loadBacktestList()
})
</script>

<style scoped>
.backtest-results {
  padding: 0;
}

.backtest-config-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-actions {
  display: flex;
  gap: 10px;
}

.backtest-option {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.backtest-overview {
  margin-bottom: 20px;
}

.metrics-comparison-card,
.returns-chart-card,
.risk-analysis-card,
.trading-analysis-card,
.position-analysis-card,
.sensitivity-analysis-card {
  margin-bottom: 20px;
}

.drawdown-section {
  margin-bottom: 20px;
}

.chart-container {
  height: 400px;
}

.drawdown-stats {
  padding: 10px;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid #ebeef5;
}

.stat-item:last-child {
  border-bottom: none;
}

.stat-label {
  color: #606266;
  font-size: 14px;
}

.stat-value {
  color: #303133;
  font-weight: 500;
}

.risk-metrics {
  padding: 10px;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-top: 15px;
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

.trading-stats {
  padding: 10px;
}

.stat-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
  margin-top: 15px;
}

.trading-charts {
  padding: 10px;
}

.position-details {
  padding: 10px;
}

.sensitivity-results {
  margin-top: 20px;
}

.sensitivity-insights {
  padding: 20px;
  background-color: #f8f9fa;
  border-radius: 6px;
}

.insight-list {
  list-style: none;
  padding: 0;
  margin: 15px 0;
}

.insight-list li {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin-bottom: 10px;
  color: #606266;
}

.optimal-params {
  margin-top: 20px;
}

.param-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid #ebeef5;
}

.param-item:last-child {
  border-bottom: none;
}

.param-name {
  color: #606266;
  font-size: 14px;
}

.param-value {
  color: #303133;
  font-weight: 500;
}

/* 颜色类 */
.positive {
  color: #67c23a;
}

.negative {
  color: #f56c6c;
}

.neutral {
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
  
  .stat-grid {
    grid-template-columns: 1fr;
  }
  
  .header-actions {
    flex-direction: column;
    gap: 5px;
  }
}
</style>