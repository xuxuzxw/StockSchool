<template>
  <div class="model-explanation">
    <!-- 模型选择和配置 -->
    <el-card class="model-config-card">
      <template #header>
        <div class="card-header">
          <span>模型解释配置</span>
          <el-button type="primary" @click="generateExplanation" :loading="explanationLoading">
            <el-icon><DataAnalysis /></el-icon>
            生成解释
          </el-button>
        </div>
      </template>
      
      <el-row :gutter="20">
        <el-col :xs="24" :md="8">
          <el-form-item label="选择模型">
            <el-select v-model="selectedModel" placeholder="选择要解释的模型" @change="handleModelChange">
              <el-option
                v-for="model in availableModels"
                :key="model.id"
                :label="model.name"
                :value="model.id"
              >
                <div class="model-option">
                  <span>{{ model.name }}</span>
                  <el-tag size="small" :type="getModelStatusType(model.status)">{{ model.status }}</el-tag>
                </div>
              </el-option>
            </el-select>
          </el-form-item>
        </el-col>
        
        <el-col :xs="24" :md="8">
          <el-form-item label="解释方法">
            <el-select v-model="explanationMethod" placeholder="选择解释方法">
              <el-option label="SHAP值分析" value="shap" />
              <el-option label="LIME解释" value="lime" />
              <el-option label="特征重要性" value="feature_importance" />
              <el-option label="部分依赖图" value="partial_dependence" />
              <el-option label="排列重要性" value="permutation_importance" />
            </el-select>
          </el-form-item>
        </el-col>
        
        <el-col :xs="24" :md="8">
          <el-form-item label="解释范围">
            <el-select v-model="explanationScope" placeholder="选择解释范围">
              <el-option label="全局解释" value="global" />
              <el-option label="局部解释" value="local" />
              <el-option label="样本解释" value="instance" />
            </el-select>
          </el-form-item>
        </el-col>
      </el-row>
      
      <!-- 股票选择（局部解释时显示） -->
      <div v-if="explanationScope === 'local' || explanationScope === 'instance'" class="stock-selection">
        <el-divider content-position="left">股票选择</el-divider>
        <el-row :gutter="20">
          <el-col :xs="24" :md="12">
            <el-form-item label="选择股票">
              <el-select
                v-model="selectedStocks"
                multiple
                filterable
                remote
                reserve-keyword
                placeholder="搜索并选择股票"
                :remote-method="searchStocks"
                :loading="stockSearchLoading"
                style="width: 100%;"
              >
                <el-option
                  v-for="stock in stockOptions"
                  :key="stock.code"
                  :label="`${stock.code} - ${stock.name}`"
                  :value="stock.code"
                />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :xs="24" :md="12">
            <el-form-item label="预测日期">
              <el-date-picker
                v-model="predictionDate"
                type="date"
                placeholder="选择预测日期"
                style="width: 100%;"
                :disabled-date="disabledDate"
              />
          </el-form-item>
        </el-col>
      </el-row>
      </div>
    </el-card>

    <!-- 模型信息概览 -->
    <el-card v-if="selectedModelInfo" class="model-info-card">
      <template #header>
        <span>模型信息</span>
      </template>
      <el-row :gutter="20">
        <el-col :xs="12" :sm="6">
          <el-statistic title="模型类型" :value="selectedModelInfo.type" />
        </el-col>
        <el-col :xs="12" :sm="6">
          <el-statistic title="训练样本数" :value="selectedModelInfo.trainSamples" />
        </el-col>
        <el-col :xs="12" :sm="6">
          <el-statistic title="特征数量" :value="selectedModelInfo.featureCount" />
        </el-col>
        <el-col :xs="12" :sm="6">
          <el-statistic title="模型准确率" :value="selectedModelInfo.accuracy" suffix="%" :precision="2" />
        </el-col>
      </el-row>
      <el-row :gutter="20" style="margin-top: 20px;">
        <el-col :xs="24" :md="12">
          <div class="model-metrics">
            <h4>性能指标</h4>
            <div class="metric-grid">
              <div class="metric-item">
                <span class="metric-label">精确率</span>
                <span class="metric-value">{{ selectedModelInfo.precision?.toFixed(3) }}</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">召回率</span>
                <span class="metric-value">{{ selectedModelInfo.recall?.toFixed(3) }}</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">F1分数</span>
                <span class="metric-value">{{ selectedModelInfo.f1Score?.toFixed(3) }}</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">AUC</span>
                <span class="metric-value">{{ selectedModelInfo.auc?.toFixed(3) }}</span>
              </div>
            </div>
          </div>
        </el-col>
        <el-col :xs="24" :md="12">
          <div class="model-features">
            <h4>主要特征</h4>
            <el-tag
              v-for="feature in selectedModelInfo.topFeatures?.slice(0, 10)"
              :key="feature"
              size="small"
              style="margin: 2px;"
            >
              {{ feature }}
            </el-tag>
          </div>
        </el-col>
      </el-row>
    </el-card>

    <!-- 解释结果 -->
    <div v-if="explanationResults" class="explanation-results">
      <!-- 全局特征重要性 -->
      <el-card v-if="explanationScope === 'global'" class="global-explanation-card">
        <template #header>
          <div class="card-header">
            <span>全局特征重要性</span>
            <el-radio-group v-model="importanceViewType" size="small">
              <el-radio-button label="bar">柱状图</el-radio-button>
              <el-radio-button label="horizontal">水平图</el-radio-button>
              <el-radio-button label="pie">饼图</el-radio-button>
            </el-radio-group>
          </div>
        </template>
        <div class="chart-container">
          <v-chart :option="globalImportanceOption" autoresize />
        </div>
        
        <!-- 特征重要性表格 -->
        <el-table :data="explanationResults.globalImportance" stripe style="margin-top: 20px;">
          <el-table-column prop="rank" label="排名" width="80" />
          <el-table-column prop="feature" label="特征名称" min-width="150" />
          <el-table-column prop="importance" label="重要性" width="120">
            <template #default="{ row }">
              {{ row.importance.toFixed(4) }}
            </template>
          </el-table-column>
          <el-table-column prop="contribution" label="贡献度" width="100">
            <template #default="{ row }">
              {{ (row.contribution * 100).toFixed(2) }}%
            </template>
          </el-table-column>
          <el-table-column prop="description" label="特征描述" min-width="200" />
          <el-table-column label="操作" width="120">
            <template #default="{ row }">
              <el-button type="text" size="small" @click="viewFeatureDetail(row)">
                详情
              </el-button>
              <el-button type="text" size="small" @click="analyzeFeatureCorrelation(row)">
                相关性
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-card>

      <!-- 局部解释结果 -->
      <div v-if="explanationScope === 'local' || explanationScope === 'instance'" class="local-explanation">
        <!-- 股票预测结果 -->
        <el-card class="prediction-results-card">
          <template #header>
            <span>预测结果</span>
          </template>
          <div v-for="stock in explanationResults.stockPredictions" :key="stock.code" class="stock-prediction">
            <el-row :gutter="20">
              <el-col :xs="24" :md="6">
                <div class="stock-info">
                  <h4>{{ stock.code }} - {{ stock.name }}</h4>
                  <el-tag :type="getPredictionType(stock.prediction)" size="large">
                    {{ getPredictionLabel(stock.prediction) }}
                  </el-tag>
                </div>
              </el-col>
              <el-col :xs="24" :md="18">
                <div class="prediction-metrics">
                  <el-row :gutter="10">
                    <el-col :xs="12" :sm="6">
                      <el-statistic title="预测概率" :value="stock.probability" suffix="%" :precision="2" />
                    </el-col>
                    <el-col :xs="12" :sm="6">
                      <el-statistic title="置信度" :value="stock.confidence" suffix="%" :precision="2" />
                    </el-col>
                    <el-col :xs="12" :sm="6">
                      <el-statistic title="预期收益" :value="stock.expectedReturn" suffix="%" :precision="2" />
                    </el-col>
                    <el-col :xs="12" :sm="6">
                      <el-statistic title="风险评级" :value="stock.riskLevel" />
                    </el-col>
                  </el-row>
                </div>
              </el-col>
            </el-row>
          </div>
        </el-card>

        <!-- SHAP值分析 -->
        <el-card v-if="explanationMethod === 'shap'" class="shap-analysis-card">
          <template #header>
            <div class="card-header">
              <span>SHAP值分析</span>
              <el-select v-model="selectedStockForShap" placeholder="选择股票" style="width: 200px;">
                <el-option
                  v-for="stock in selectedStocks"
                  :key="stock"
                  :label="stock"
                  :value="stock"
                />
              </el-select>
            </div>
          </template>
          <div v-if="shapData" class="shap-results">
            <el-row :gutter="20">
              <el-col :xs="24" :lg="12">
                <div class="chart-container">
                  <h4>SHAP值瀑布图</h4>
                  <v-chart :option="shapWaterfallOption" autoresize />
                </div>
              </el-col>
              <el-col :xs="24" :lg="12">
                <div class="chart-container">
                  <h4>特征贡献分布</h4>
                  <v-chart :option="shapDistributionOption" autoresize />
                </div>
              </el-col>
            </el-row>
            
            <!-- SHAP值详细表格 -->
            <el-table :data="shapData.featureContributions" stripe style="margin-top: 20px;">
              <el-table-column prop="feature" label="特征" min-width="150" />
              <el-table-column prop="value" label="特征值" width="120">
                <template #default="{ row }">
                  {{ typeof row.value === 'number' ? row.value.toFixed(4) : row.value }}
                </template>
              </el-table-column>
              <el-table-column prop="shapValue" label="SHAP值" width="120">
                <template #default="{ row }">
                  <span :class="getShapValueClass(row.shapValue)">
                    {{ row.shapValue.toFixed(4) }}
                  </span>
                </template>
              </el-table-column>
              <el-table-column prop="contribution" label="贡献度" width="100">
                <template #default="{ row }">
                  {{ (Math.abs(row.shapValue) / shapData.totalAbsShap * 100).toFixed(2) }}%
                </template>
              </el-table-column>
              <el-table-column prop="impact" label="影响" width="100">
                <template #default="{ row }">
                  <el-tag :type="row.shapValue > 0 ? 'success' : 'danger'" size="small">
                    {{ row.shapValue > 0 ? '正向' : '负向' }}
                  </el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="description" label="解释" min-width="200" />
            </el-table>
          </div>
        </el-card>

        <!-- 部分依赖图 -->
        <el-card v-if="explanationMethod === 'partial_dependence'" class="pdp-card">
          <template #header>
            <div class="card-header">
              <span>部分依赖图</span>
              <el-select v-model="selectedFeatureForPDP" placeholder="选择特征" style="width: 200px;">
                <el-option
                  v-for="feature in topFeatures"
                  :key="feature"
                  :label="feature"
                  :value="feature"
                />
              </el-select>
            </div>
          </template>
          <div v-if="pdpData" class="pdp-results">
            <el-row :gutter="20">
              <el-col :xs="24" :lg="16">
                <div class="chart-container">
                  <v-chart :option="pdpChartOption" autoresize />
                </div>
              </el-col>
              <el-col :xs="24" :lg="8">
                <div class="pdp-insights">
                  <h4>关键洞察</h4>
                  <ul class="insight-list">
                    <li v-for="insight in pdpData.insights" :key="insight.id">
                      <el-icon><InfoFilled /></el-icon>
                      {{ insight.text }}
                    </li>
                  </ul>
                  
                  <div class="pdp-stats">
                    <div class="stat-item">
                      <span class="stat-label">特征范围</span>
                      <span class="stat-value">{{ pdpData.featureRange.min.toFixed(2) }} - {{ pdpData.featureRange.max.toFixed(2) }}</span>
                    </div>
                    <div class="stat-item">
                      <span class="stat-label">预测变化幅度</span>
                      <span class="stat-value">{{ ((pdpData.predictionRange.max - pdpData.predictionRange.min) * 100).toFixed(2) }}%</span>
                    </div>
                    <div class="stat-item">
                      <span class="stat-label">最优特征值</span>
                      <span class="stat-value">{{ pdpData.optimalValue.toFixed(4) }}</span>
                    </div>
                  </div>
                </div>
              </el-col>
            </el-row>
          </div>
        </el-card>
      </div>

      <!-- 模型对比分析 -->
      <el-card class="model-comparison-card">
        <template #header>
          <div class="card-header">
            <span>模型对比分析</span>
            <el-button @click="compareModels" :loading="comparisonLoading">
              <el-icon><Compare /></el-icon>
              对比模型
            </el-button>
          </div>
        </template>
        <div v-if="comparisonResults" class="comparison-results">
          <el-table :data="comparisonResults.models" stripe>
            <el-table-column prop="name" label="模型名称" min-width="150" />
            <el-table-column prop="accuracy" label="准确率" width="100">
              <template #default="{ row }">
                {{ (row.accuracy * 100).toFixed(2) }}%
              </template>
            </el-table-column>
            <el-table-column prop="precision" label="精确率" width="100">
              <template #default="{ row }">
                {{ (row.precision * 100).toFixed(2) }}%
              </template>
            </el-table-column>
            <el-table-column prop="recall" label="召回率" width="100">
              <template #default="{ row }">
                {{ (row.recall * 100).toFixed(2) }}%
              </template>
            </el-table-column>
            <el-table-column prop="f1Score" label="F1分数" width="100">
              <template #default="{ row }">
                {{ row.f1Score.toFixed(3) }}
              </template>
            </el-table-column>
            <el-table-column prop="interpretability" label="可解释性" width="120">
              <template #default="{ row }">
                <el-rate v-model="row.interpretability" disabled show-score text-color="#ff9900" />
              </template>
            </el-table-column>
            <el-table-column prop="complexity" label="复杂度" width="100">
              <template #default="{ row }">
                <el-tag :type="getComplexityType(row.complexity)" size="small">
                  {{ row.complexity }}
                </el-tag>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, BarChart, PieChart, ScatterChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  DataZoomComponent
} from 'echarts/components'
import VChart from 'vue-echarts'
import { apiService } from '@/services/api'
import { ElMessage } from 'element-plus'
import { debounce } from 'lodash-es'

// 注册ECharts组件
use([
  CanvasRenderer,
  LineChart,
  BarChart,
  PieChart,
  ScatterChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  DataZoomComponent
])

// 响应式数据
const explanationLoading = ref(false)
const comparisonLoading = ref(false)
const stockSearchLoading = ref(false)
const selectedModel = ref('')
const explanationMethod = ref('shap')
const explanationScope = ref('global')
const selectedStocks = ref<string[]>([])
const predictionDate = ref(new Date())
const importanceViewType = ref('bar')
const selectedStockForShap = ref('')
const selectedFeatureForPDP = ref('')

// 数据
const availableModels = ref<any[]>([])
const selectedModelInfo = ref<any>(null)
const explanationResults = ref<any>(null)
const stockOptions = ref<any[]>([])
const shapData = ref<any>(null)
const pdpData = ref<any>(null)
const comparisonResults = ref<any>(null)

// 计算属性
const topFeatures = computed(() => {
  if (!selectedModelInfo.value?.topFeatures) return []
  return selectedModelInfo.value.topFeatures.slice(0, 20)
})

// 图表配置
const globalImportanceOption = computed(() => {
  if (!explanationResults.value?.globalImportance) return {}
  
  const data = explanationResults.value.globalImportance.slice(0, 20)
  
  if (importanceViewType.value === 'pie') {
    return {
      tooltip: {
        trigger: 'item',
        formatter: '{a} <br/>{b}: {c} ({d}%)'
      },
      series: [{
        name: '特征重要性',
        type: 'pie',
        radius: '50%',
        data: data.map((item: any) => ({
          value: item.importance,
          name: item.feature
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
  }
  
  const isHorizontal = importanceViewType.value === 'horizontal'
  
  return {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      }
    },
    grid: {
      left: isHorizontal ? '20%' : '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: isHorizontal ? 'value' : 'category',
      data: isHorizontal ? undefined : data.map((item: any) => item.feature),
      axisLabel: isHorizontal ? undefined : {
        rotate: 45,
        interval: 0
      }
    },
    yAxis: {
      type: isHorizontal ? 'category' : 'value',
      data: isHorizontal ? data.map((item: any) => item.feature) : undefined
    },
    series: [{
      name: '重要性',
      type: 'bar',
      data: data.map((item: any) => item.importance),
      itemStyle: {
        color: (params: any) => {
          const colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc']
          return colors[params.dataIndex % colors.length]
        }
      }
    }]
  }
})

const shapWaterfallOption = computed(() => {
  if (!shapData.value?.featureContributions) return {}
  
  const contributions = shapData.value.featureContributions.slice(0, 10)
  const baseValue = shapData.value.baseValue || 0
  
  let cumulative = baseValue
  const data = []
  
  // 基准值
  data.push({
    name: '基准预测',
    value: baseValue,
    itemStyle: { color: '#91cc75' }
  })
  
  // 特征贡献
  contributions.forEach((contrib: any, index: number) => {
    const prevCumulative = cumulative
    cumulative += contrib.shapValue
    
    data.push({
      name: contrib.feature,
      value: contrib.shapValue,
      stack: 'total',
      itemStyle: {
        color: contrib.shapValue > 0 ? '#5470c6' : '#ee6666'
      }
    })
  })
  
  // 最终预测
  data.push({
    name: '最终预测',
    value: cumulative,
    itemStyle: { color: '#fac858' }
  })
  
  return {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      },
      formatter: (params: any) => {
        const param = params[0]
        return `${param.name}<br/>贡献值: ${param.value.toFixed(4)}`
      }
    },
    xAxis: {
      type: 'category',
      data: data.map(item => item.name),
      axisLabel: {
        rotate: 45,
        interval: 0
      }
    },
    yAxis: {
      type: 'value'
    },
    series: [{
      name: 'SHAP值',
      type: 'bar',
      data: data
    }]
  }
})

const shapDistributionOption = computed(() => {
  if (!shapData.value?.featureContributions) return {}
  
  const contributions = shapData.value.featureContributions
  const positive = contributions.filter((c: any) => c.shapValue > 0)
  const negative = contributions.filter((c: any) => c.shapValue < 0)
  
  return {
    tooltip: {
      trigger: 'item'
    },
    legend: {
      orient: 'vertical',
      left: 'left'
    },
    series: [{
      name: '特征贡献',
      type: 'pie',
      radius: '50%',
      data: [
        {
          value: positive.reduce((sum: number, c: any) => sum + c.shapValue, 0),
          name: '正向贡献',
          itemStyle: { color: '#5470c6' }
        },
        {
          value: Math.abs(negative.reduce((sum: number, c: any) => sum + c.shapValue, 0)),
          name: '负向贡献',
          itemStyle: { color: '#ee6666' }
        }
      ],
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

const pdpChartOption = computed(() => {
  if (!pdpData.value) return {}
  
  return {
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        const param = params[0]
        return `${selectedFeatureForPDP.value}: ${param.name}<br/>预测值: ${param.value.toFixed(4)}`
      }
    },
    xAxis: {
      type: 'category',
      data: pdpData.value.featureValues,
      name: selectedFeatureForPDP.value
    },
    yAxis: {
      type: 'value',
      name: '预测值'
    },
    series: [{
      name: '部分依赖',
      type: 'line',
      data: pdpData.value.predictions,
      smooth: true,
      lineStyle: {
        color: '#5470c6',
        width: 3
      },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [{
            offset: 0, color: 'rgba(84, 112, 198, 0.3)'
          }, {
            offset: 1, color: 'rgba(84, 112, 198, 0.1)'
          }]
        }
      }
    }]
  }
})

// 方法
const handleModelChange = async () => {
  if (!selectedModel.value) return
  
  try {
    const response = await apiService.getModelInfo(selectedModel.value)
    selectedModelInfo.value = response
  } catch (error) {
    console.error('获取模型信息失败:', error)
    ElMessage.error('获取模型信息失败')
  }
}

const searchStocks = debounce(async (query: string) => {
  if (!query) {
    stockOptions.value = []
    return
  }
  
  stockSearchLoading.value = true
  try {
    const response = await apiService.searchStocks({ query, limit: 20 })
    stockOptions.value = response.stocks
  } catch (error) {
    console.error('搜索股票失败:', error)
  } finally {
    stockSearchLoading.value = false
  }
}, 300)

const generateExplanation = async () => {
  if (!selectedModel.value) {
    ElMessage.warning('请选择要解释的模型')
    return
  }
  
  if ((explanationScope.value === 'local' || explanationScope.value === 'instance') && selectedStocks.value.length === 0) {
    ElMessage.warning('请选择要解释的股票')
    return
  }
  
  explanationLoading.value = true
  try {
    const params = {
      model_id: selectedModel.value,
      method: explanationMethod.value,
      scope: explanationScope.value,
      stocks: selectedStocks.value,
      date: predictionDate.value.toISOString().split('T')[0]
    }
    
    const response = await apiService.getModelExplanation(params)
    explanationResults.value = response
    
    // 如果是SHAP方法，设置默认选中的股票
    if (explanationMethod.value === 'shap' && selectedStocks.value.length > 0) {
      selectedStockForShap.value = selectedStocks.value[0]
      await loadShapData(selectedStockForShap.value)
    }
    
    // 如果是部分依赖图，设置默认特征
    if (explanationMethod.value === 'partial_dependence' && topFeatures.value.length > 0) {
      selectedFeatureForPDP.value = topFeatures.value[0]
      await loadPDPData(selectedFeatureForPDP.value)
    }
    
    ElMessage.success('模型解释生成完成')
    
  } catch (error) {
    console.error('生成模型解释失败:', error)
    ElMessage.error('生成模型解释失败')
  } finally {
    explanationLoading.value = false
  }
}

const loadShapData = async (stockCode: string) => {
  if (!stockCode || !explanationResults.value) return
  
  try {
    const response = await apiService.getShapValues({
      model_id: selectedModel.value,
      stock_code: stockCode,
      date: predictionDate.value.toISOString().split('T')[0]
    })
    shapData.value = response
  } catch (error) {
    console.error('获取SHAP数据失败:', error)
  }
}

const loadPDPData = async (feature: string) => {
  if (!feature || !selectedModel.value) return
  
  try {
    const response = await apiService.getPartialDependence({
      model_id: selectedModel.value,
      feature: feature,
      stocks: selectedStocks.value
    })
    pdpData.value = response
  } catch (error) {
    console.error('获取部分依赖数据失败:', error)
  }
}

const compareModels = async () => {
  comparisonLoading.value = true
  try {
    const response = await apiService.compareModels({
      models: availableModels.value.map(m => m.id),
      metrics: ['accuracy', 'precision', 'recall', 'f1_score', 'interpretability']
    })
    comparisonResults.value = response
    ElMessage.success('模型对比完成')
  } catch (error) {
    console.error('模型对比失败:', error)
    ElMessage.error('模型对比失败')
  } finally {
    comparisonLoading.value = false
  }
}

const disabledDate = (time: Date) => {
  return time.getTime() > Date.now()
}

const getModelStatusType = (status: string) => {
  const typeMap: Record<string, string> = {
    'active': 'success',
    'training': 'warning',
    'inactive': 'info',
    'error': 'danger'
  }
  return typeMap[status] || 'info'
}

const getPredictionType = (prediction: string) => {
  const typeMap: Record<string, string> = {
    'buy': 'success',
    'hold': 'warning',
    'sell': 'danger'
  }
  return typeMap[prediction] || 'info'
}

const getPredictionLabel = (prediction: string) => {
  const labelMap: Record<string, string> = {
    'buy': '买入',
    'hold': '持有',
    'sell': '卖出'
  }
  return labelMap[prediction] || prediction
}

const getShapValueClass = (value: number) => {
  if (value > 0) return 'shap-positive'
  if (value < 0) return 'shap-negative'
  return 'shap-neutral'
}

const getComplexityType = (complexity: string) => {
  const typeMap: Record<string, string> = {
    'low': 'success',
    'medium': 'warning',
    'high': 'danger'
  }
  return typeMap[complexity] || 'info'
}

const viewFeatureDetail = (feature: any) => {
  console.log('查看特征详情:', feature)
  ElMessage.info('特征详情功能开发中...')
}

const analyzeFeatureCorrelation = (feature: any) => {
  console.log('分析特征相关性:', feature)
  ElMessage.info('特征相关性分析功能开发中...')
}

// 监听器
watch(selectedStockForShap, (newStock) => {
  if (newStock && explanationMethod.value === 'shap') {
    loadShapData(newStock)
  }
})

watch(selectedFeatureForPDP, (newFeature) => {
  if (newFeature && explanationMethod.value === 'partial_dependence') {
    loadPDPData(newFeature)
  }
})

// 生命周期
onMounted(async () => {
  try {
    const response = await apiService.getAvailableModels()
    availableModels.value = response.models
    
    if (availableModels.value.length > 0) {
      selectedModel.value = availableModels.value[0].id
      await handleModelChange()
    }
  } catch (error) {
    console.error('获取可用模型失败:', error)
    ElMessage.error('获取可用模型失败')
  }
})
</script>

<style scoped>
.model-explanation {
  padding: 0;
}

.model-config-card,
.model-info-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.model-option {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.stock-selection {
  margin-top: 20px;
}

.model-metrics,
.model-features {
  padding: 10px;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
}

.metric-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px;
  background-color: #f8f9fa;
  border-radius: 4px;
}

.metric-label {
  color: #606266;
  font-size: 14px;
}

.metric-value {
  color: #303133;
  font-weight: 500;
}

.explanation-results {
  margin-top: 20px;
}

.global-explanation-card,
.prediction-results-card,
.shap-analysis-card,
.pdp-card,
.model-comparison-card {
  margin-bottom: 20px;
}

.chart-container {
  height: 400px;
}

.stock-prediction {
  padding: 15px;
  border: 1px solid #ebeef5;
  border-radius: 6px;
  margin-bottom: 15px;
}

.stock-info h4 {
  margin: 0 0 10px 0;
  color: #303133;
}

.prediction-metrics {
  margin-top: 10px;
}

.shap-results,
.pdp-results {
  margin-top: 20px;
}

.pdp-insights {
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

.pdp-stats {
  margin-top: 20px;
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

.comparison-results {
  margin-top: 20px;
}

/* SHAP值颜色 */
.shap-positive {
  color: #67c23a;
  font-weight: bold;
}

.shap-negative {
  color: #f56c6c;
  font-weight: bold;
}

.shap-neutral {
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
  
  .stock-prediction {
    padding: 10px;
  }
  
  .pdp-insights {
    padding: 15px;
  }
}
</style>