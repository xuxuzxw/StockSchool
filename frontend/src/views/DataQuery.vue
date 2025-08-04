<template>
  <div class="data-query">
    <!-- 查询配置 -->
    <el-card class="query-config-card">
      <template #header>
        <div class="card-header">
          <span>数据查询</span>
          <div class="header-actions">
            <el-button @click="resetQuery">
              <el-icon><Refresh /></el-icon>
              重置
            </el-button>
            <el-button type="primary" @click="executeQuery" :loading="queryLoading">
              <el-icon><Search /></el-icon>
              查询
            </el-button>
          </div>
        </div>
      </template>
      
      <el-form :model="queryForm" label-width="120px" :inline="false">
        <el-row :gutter="20">
          <el-col :xs="24" :md="8">
            <el-form-item label="数据类型">
              <el-select v-model="queryForm.dataType" placeholder="选择数据类型" @change="handleDataTypeChange">
                <el-option label="股票基础数据" value="stock_basic" />
                <el-option label="股票价格数据" value="stock_price" />
                <el-option label="财务数据" value="financial" />
                <el-option label="因子数据" value="factor" />
                <el-option label="行业数据" value="industry" />
                <el-option label="指数数据" value="index" />
                <el-option label="宏观数据" value="macro" />
                <el-option label="资金流向" value="money_flow" />
                <el-option label="技术指标" value="technical" />
              </el-select>
            </el-form-item>
          </el-col>
          
          <el-col :xs="24" :md="8">
            <el-form-item label="时间范围">
              <el-date-picker
                v-model="queryForm.dateRange"
                type="daterange"
                range-separator="至"
                start-placeholder="开始日期"
                end-placeholder="结束日期"
                format="YYYY-MM-DD"
                value-format="YYYY-MM-DD"
              />
            </el-form-item>
          </el-col>
          
          <el-col :xs="24" :md="8">
            <el-form-item label="股票代码">
              <el-select
                v-model="queryForm.stockCodes"
                multiple
                filterable
                remote
                reserve-keyword
                placeholder="输入股票代码或名称"
                :remote-method="searchStocks"
                :loading="stockSearchLoading"
                style="width: 100%;"
              >
                <el-option
                  v-for="stock in stockOptions"
                  :key="stock.code"
                  :label="`${stock.code} ${stock.name}`"
                  :value="stock.code"
                />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
        
        <!-- 动态字段配置 -->
        <div v-if="availableFields.length > 0" class="field-selection">
          <el-form-item label="选择字段">
            <el-checkbox-group v-model="queryForm.selectedFields">
              <el-row :gutter="10">
                <el-col :xs="12" :sm="8" :md="6" :lg="4" v-for="field in availableFields" :key="field.name">
                  <el-checkbox :label="field.name" :value="field.name">
                    <el-tooltip :content="field.description" placement="top">
                      <span>{{ field.label }}</span>
                    </el-tooltip>
                  </el-checkbox>
                </el-col>
              </el-row>
            </el-checkbox-group>
          </el-form-item>
        </div>
        
        <!-- 高级筛选 -->
        <el-collapse v-model="activeCollapse">
          <el-collapse-item title="高级筛选" name="advanced">
            <el-row :gutter="20">
              <el-col :xs="24" :md="8">
                <el-form-item label="行业筛选">
                  <el-select v-model="queryForm.industries" multiple placeholder="选择行业">
                    <el-option
                      v-for="industry in industryOptions"
                      :key="industry.code"
                      :label="industry.name"
                      :value="industry.code"
                    />
                  </el-select>
                </el-form-item>
              </el-col>
              
              <el-col :xs="24" :md="8">
                <el-form-item label="市值范围">
                  <el-slider
                    v-model="queryForm.marketCapRange"
                    range
                    :min="0"
                    :max="10000"
                    :step="100"
                    show-stops
                    :format-tooltip="formatMarketCap"
                  />
                </el-form-item>
              </el-col>
              
              <el-col :xs="24" :md="8">
                <el-form-item label="数据频率">
                  <el-select v-model="queryForm.frequency" placeholder="选择频率">
                    <el-option label="日频" value="daily" />
                    <el-option label="周频" value="weekly" />
                    <el-option label="月频" value="monthly" />
                    <el-option label="季频" value="quarterly" />
                    <el-option label="年频" value="yearly" />
                  </el-select>
                </el-form-item>
              </el-col>
            </el-row>
            
            <!-- 自定义条件 -->
            <div class="custom-conditions">
              <el-form-item label="自定义条件">
                <div v-for="(condition, index) in queryForm.customConditions" :key="index" class="condition-row">
                  <el-row :gutter="10" type="flex" align="middle">
                    <el-col :span="5">
                      <el-select v-model="condition.field" placeholder="字段">
                        <el-option
                          v-for="field in availableFields"
                          :key="field.name"
                          :label="field.label"
                          :value="field.name"
                        />
                      </el-select>
                    </el-col>
                    <el-col :span="4">
                      <el-select v-model="condition.operator" placeholder="操作符">
                        <el-option label="等于" value="=" />
                        <el-option label="不等于" value="!=" />
                        <el-option label="大于" value=">" />
                        <el-option label="大于等于" value=">=" />
                        <el-option label="小于" value="<" />
                        <el-option label="小于等于" value="<=" />
                        <el-option label="包含" value="in" />
                        <el-option label="不包含" value="not in" />
                      </el-select>
                    </el-col>
                    <el-col :span="6">
                      <el-input v-model="condition.value" placeholder="值" />
                    </el-col>
                    <el-col :span="3">
                      <el-select v-model="condition.logic" placeholder="逻辑" v-if="index < queryForm.customConditions.length - 1">
                        <el-option label="AND" value="and" />
                        <el-option label="OR" value="or" />
                      </el-select>
                    </el-col>
                    <el-col :span="6">
                      <el-button @click="addCondition" type="primary" size="small" v-if="index === queryForm.customConditions.length - 1">
                        <el-icon><Plus /></el-icon>
                      </el-button>
                      <el-button @click="removeCondition(index)" type="danger" size="small" v-if="queryForm.customConditions.length > 1">
                        <el-icon><Minus /></el-icon>
                      </el-button>
                    </el-col>
                  </el-row>
                </div>
              </el-form-item>
            </div>
          </el-collapse-item>
          
          <el-collapse-item title="SQL查询" name="sql">
            <el-form-item label="自定义SQL">
              <el-input
                v-model="queryForm.customSQL"
                type="textarea"
                :rows="6"
                placeholder="输入自定义SQL查询语句..."
              />
            </el-form-item>
            <el-alert
              title="注意：自定义SQL查询将覆盖上述所有筛选条件"
              type="warning"
              :closable="false"
              show-icon
            />
          </el-collapse-item>
        </el-collapse>
      </el-form>
    </el-card>

    <!-- 查询结果 -->
    <el-card v-if="queryResults" class="results-card">
      <template #header>
        <div class="card-header">
          <span>查询结果 ({{ queryResults.total }} 条记录)</span>
          <div class="header-actions">
            <el-button @click="downloadData" :loading="downloadLoading">
              <el-icon><Download /></el-icon>
              下载数据
            </el-button>
            <el-button @click="visualizeData">
              <el-icon><DataLine /></el-icon>
              数据可视化
            </el-button>
            <el-button @click="saveQuery">
              <el-icon><Collection /></el-icon>
              保存查询
            </el-button>
          </div>
        </div>
      </template>
      
      <!-- 数据统计概览 -->
      <div class="data-summary" v-if="queryResults.summary">
        <el-row :gutter="20">
          <el-col :xs="12" :sm="6" v-for="(stat, key) in queryResults.summary" :key="key">
            <el-statistic :title="getStatTitle(key)" :value="stat" :precision="getStatPrecision(key)" />
          </el-col>
        </el-row>
      </div>
      
      <!-- 数据表格 -->
      <div class="data-table">
        <el-table
          :data="queryResults.data"
          stripe
          border
          :max-height="500"
          v-loading="queryLoading"
        >
          <el-table-column
            v-for="column in tableColumns"
            :key="column.prop"
            :prop="column.prop"
            :label="column.label"
            :width="column.width"
            :min-width="column.minWidth"
            :formatter="column.formatter"
            :sortable="column.sortable"
            show-overflow-tooltip
          />
        </el-table>
        
        <!-- 分页 -->
        <el-pagination
          v-if="queryResults.total > pageSize"
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :page-sizes="[20, 50, 100, 200]"
          :total="queryResults.total"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
          class="pagination"
        />
      </div>
    </el-card>

    <!-- 数据可视化对话框 -->
    <el-dialog v-model="visualDialogVisible" title="数据可视化" width="80%" :before-close="handleVisualizationClose">
      <div class="visualization-config">
        <el-row :gutter="20">
          <el-col :span="8">
            <el-form-item label="图表类型">
              <el-select v-model="visualConfig.chartType" @change="updateVisualization">
                <el-option label="折线图" value="line" />
                <el-option label="柱状图" value="bar" />
                <el-option label="散点图" value="scatter" />
                <el-option label="饼图" value="pie" />
                <el-option label="热力图" value="heatmap" />
                <el-option label="箱线图" value="boxplot" />
              </el-select>
            </el-form-item>
          </el-col>
          
          <el-col :span="8">
            <el-form-item label="X轴字段">
              <el-select v-model="visualConfig.xField" @change="updateVisualization">
                <el-option
                  v-for="field in numericFields"
                  :key="field.name"
                  :label="field.label"
                  :value="field.name"
                />
              </el-select>
            </el-form-item>
          </el-col>
          
          <el-col :span="8">
            <el-form-item label="Y轴字段">
              <el-select v-model="visualConfig.yField" @change="updateVisualization">
                <el-option
                  v-for="field in numericFields"
                  :key="field.name"
                  :label="field.label"
                  :value="field.name"
                />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
      </div>
      
      <div class="chart-container" v-if="visualizationOption">
        <v-chart :option="visualizationOption" autoresize />
      </div>
    </el-dialog>

    <!-- 保存查询对话框 -->
    <el-dialog v-model="saveDialogVisible" title="保存查询" width="400px">
      <el-form :model="saveForm" label-width="80px">
        <el-form-item label="查询名称" required>
          <el-input v-model="saveForm.name" placeholder="输入查询名称" />
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="saveForm.description" type="textarea" :rows="3" placeholder="输入查询描述" />
        </el-form-item>
        <el-form-item label="标签">
          <el-select v-model="saveForm.tags" multiple allow-create filterable placeholder="添加标签">
            <el-option
              v-for="tag in commonTags"
              :key="tag"
              :label="tag"
              :value="tag"
            />
          </el-select>
        </el-form-item>
      </el-form>
      
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="saveDialogVisible = false">取消</el-button>
          <el-button type="primary" @click="confirmSaveQuery" :loading="saveLoading">保存</el-button>
        </span>
      </template>
    </el-dialog>

    <!-- 已保存查询 -->
    <el-card class="saved-queries-card" v-if="savedQueries.length > 0">
      <template #header>
        <div class="card-header">
          <span>已保存查询</span>
          <el-button @click="loadSavedQueries">
            <el-icon><Refresh /></el-icon>
            刷新
          </el-button>
        </div>
      </template>
      
      <div class="saved-queries-list">
        <el-row :gutter="15">
          <el-col :xs="24" :sm="12" :md="8" :lg="6" v-for="query in savedQueries" :key="query.id">
            <el-card class="saved-query-item" shadow="hover">
              <div class="query-info">
                <h4>{{ query.name }}</h4>
                <p class="query-description">{{ query.description || '无描述' }}</p>
                <div class="query-meta">
                  <el-tag size="small" v-for="tag in query.tags" :key="tag">{{ tag }}</el-tag>
                  <span class="query-date">{{ formatDate(query.created_at) }}</span>
                </div>
              </div>
              <div class="query-actions">
                <el-button size="small" @click="loadSavedQuery(query)">
                  <el-icon><VideoPlay /></el-icon>
                  执行
                </el-button>
                <el-button size="small" @click="editSavedQuery(query)">
                  <el-icon><Edit /></el-icon>
                  编辑
                </el-button>
                <el-button size="small" type="danger" @click="deleteSavedQuery(query.id)">
                  <el-icon><Delete /></el-icon>
                  删除
                </el-button>
              </div>
            </el-card>
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
import { LineChart, BarChart, PieChart, ScatterChart, HeatmapChart, BoxplotChart } from 'echarts/charts'
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
import { ElMessage, ElMessageBox } from 'element-plus'
import dayjs from 'dayjs'

// 注册ECharts组件
use([
  CanvasRenderer,
  LineChart,
  BarChart,
  PieChart,
  ScatterChart,
  HeatmapChart,
  BoxplotChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  DataZoomComponent,
  VisualMapComponent
])

// 响应式数据
const queryLoading = ref(false)
const stockSearchLoading = ref(false)
const downloadLoading = ref(false)
const saveLoading = ref(false)
const visualDialogVisible = ref(false)
const saveDialogVisible = ref(false)
const activeCollapse = ref<string[]>([])
const currentPage = ref(1)
const pageSize = ref(50)

// 表单数据
const queryForm = ref({
  dataType: '',
  dateRange: [] as string[],
  stockCodes: [] as string[],
  selectedFields: [] as string[],
  industries: [] as string[],
  marketCapRange: [0, 10000],
  frequency: 'daily',
  customConditions: [
    { field: '', operator: '=', value: '', logic: 'and' }
  ],
  customSQL: ''
})

const saveForm = ref({
  name: '',
  description: '',
  tags: [] as string[]
})

const visualConfig = ref({
  chartType: 'line',
  xField: '',
  yField: ''
})

// 数据
const stockOptions = ref<any[]>([])
const industryOptions = ref<any[]>([])
const availableFields = ref<any[]>([])
const queryResults = ref<any>(null)
const savedQueries = ref<any[]>([])
const commonTags = ref(['股票', '财务', '因子', '技术指标', '宏观', '行业'])

// 计算属性
const tableColumns = computed(() => {
  if (!queryResults.value?.columns) return []
  
  return queryResults.value.columns.map((col: any) => ({
    prop: col.name,
    label: col.label || col.name,
    width: col.width,
    minWidth: col.minWidth || 120,
    sortable: col.sortable !== false,
    formatter: getColumnFormatter(col.type)
  }))
})

const numericFields = computed(() => {
  return availableFields.value.filter(field => 
    ['number', 'float', 'integer', 'decimal'].includes(field.type)
  )
})

const visualizationOption = computed(() => {
  if (!queryResults.value?.data || !visualConfig.value.xField || !visualConfig.value.yField) {
    return null
  }
  
  const data = queryResults.value.data
  const { chartType, xField, yField } = visualConfig.value
  
  switch (chartType) {
    case 'line':
      return {
        title: {
          text: `${yField} vs ${xField}`,
          left: 'center'
        },
        tooltip: {
          trigger: 'axis'
        },
        xAxis: {
          type: 'category',
          data: data.map((item: any) => item[xField])
        },
        yAxis: {
          type: 'value'
        },
        series: [{
          name: yField,
          type: 'line',
          data: data.map((item: any) => item[yField]),
          smooth: true
        }]
      }
      
    case 'bar':
      return {
        title: {
          text: `${yField} vs ${xField}`,
          left: 'center'
        },
        tooltip: {
          trigger: 'axis'
        },
        xAxis: {
          type: 'category',
          data: data.map((item: any) => item[xField])
        },
        yAxis: {
          type: 'value'
        },
        series: [{
          name: yField,
          type: 'bar',
          data: data.map((item: any) => item[yField])
        }]
      }
      
    case 'scatter':
      return {
        title: {
          text: `${yField} vs ${xField}`,
          left: 'center'
        },
        tooltip: {
          trigger: 'item'
        },
        xAxis: {
          type: 'value',
          name: xField
        },
        yAxis: {
          type: 'value',
          name: yField
        },
        series: [{
          name: 'Data Points',
          type: 'scatter',
          data: data.map((item: any) => [item[xField], item[yField]])
        }]
      }
      
    default:
      return null
  }
})

// 方法
const handleDataTypeChange = async () => {
  if (!queryForm.value.dataType) return
  
  try {
    const response = await apiService.getDataFields(queryForm.value.dataType)
    availableFields.value = response.fields
    queryForm.value.selectedFields = response.defaultFields || []
  } catch (error) {
    console.error('获取字段信息失败:', error)
  }
}

const searchStocks = async (query: string) => {
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
}

const executeQuery = async () => {
  if (!queryForm.value.dataType) {
    ElMessage.warning('请选择数据类型')
    return
  }
  
  queryLoading.value = true
  try {
    const queryParams = {
      ...queryForm.value,
      page: currentPage.value,
      pageSize: pageSize.value
    }
    
    const response = await apiService.queryData(queryParams)
    queryResults.value = response
    
    ElMessage.success(`查询完成，共找到 ${response.total} 条记录`)
  } catch (error) {
    console.error('查询失败:', error)
    ElMessage.error('查询失败，请检查查询条件')
  } finally {
    queryLoading.value = false
  }
}

const resetQuery = () => {
  queryForm.value = {
    dataType: '',
    dateRange: [],
    stockCodes: [],
    selectedFields: [],
    industries: [],
    marketCapRange: [0, 10000],
    frequency: 'daily',
    customConditions: [
      { field: '', operator: '=', value: '', logic: 'and' }
    ],
    customSQL: ''
  }
  queryResults.value = null
  availableFields.value = []
}

const addCondition = () => {
  queryForm.value.customConditions.push({
    field: '',
    operator: '=',
    value: '',
    logic: 'and'
  })
}

const removeCondition = (index: number) => {
  queryForm.value.customConditions.splice(index, 1)
}

const handleSizeChange = (newSize: number) => {
  pageSize.value = newSize
  executeQuery()
}

const handleCurrentChange = (newPage: number) => {
  currentPage.value = newPage
  executeQuery()
}

const downloadData = async () => {
  if (!queryResults.value) return
  
  downloadLoading.value = true
  try {
    const response = await apiService.downloadQueryData({
      ...queryForm.value,
      format: 'csv'
    })
    
    // 创建下载链接
    const blob = new Blob([response.data], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `query_results_${dayjs().format('YYYY-MM-DD_HH-mm-ss')}.csv`
    link.click()
    window.URL.revokeObjectURL(url)
    
    ElMessage.success('数据下载完成')
  } catch (error) {
    console.error('下载失败:', error)
    ElMessage.error('下载失败')
  } finally {
    downloadLoading.value = false
  }
}

const visualizeData = () => {
  if (!queryResults.value?.data?.length) {
    ElMessage.warning('没有可视化的数据')
    return
  }
  
  if (numericFields.value.length < 2) {
    ElMessage.warning('需要至少两个数值字段才能进行可视化')
    return
  }
  
  // 设置默认字段
  if (!visualConfig.value.xField && numericFields.value.length > 0) {
    visualConfig.value.xField = numericFields.value[0].name
  }
  if (!visualConfig.value.yField && numericFields.value.length > 1) {
    visualConfig.value.yField = numericFields.value[1].name
  }
  
  visualDialogVisible.value = true
}

const updateVisualization = () => {
  // 触发计算属性重新计算
}

const handleVisualizationClose = () => {
  visualDialogVisible.value = false
}

const saveQuery = () => {
  if (!queryForm.value.dataType) {
    ElMessage.warning('请先配置查询条件')
    return
  }
  
  saveForm.value = {
    name: '',
    description: '',
    tags: []
  }
  saveDialogVisible.value = true
}

const confirmSaveQuery = async () => {
  if (!saveForm.value.name.trim()) {
    ElMessage.warning('请输入查询名称')
    return
  }
  
  saveLoading.value = true
  try {
    await apiService.saveQuery({
      name: saveForm.value.name,
      description: saveForm.value.description,
      tags: saveForm.value.tags,
      config: queryForm.value
    })
    
    ElMessage.success('查询保存成功')
    saveDialogVisible.value = false
    loadSavedQueries()
  } catch (error) {
    console.error('保存查询失败:', error)
    ElMessage.error('保存查询失败')
  } finally {
    saveLoading.value = false
  }
}

const loadSavedQueries = async () => {
  try {
    const response = await apiService.getSavedQueries()
    savedQueries.value = response.queries
  } catch (error) {
    console.error('加载已保存查询失败:', error)
  }
}

const loadSavedQuery = (query: any) => {
  queryForm.value = { ...query.config }
  handleDataTypeChange()
  ElMessage.success(`已加载查询: ${query.name}`)
}

const editSavedQuery = (query: any) => {
  loadSavedQuery(query)
  saveForm.value = {
    name: query.name,
    description: query.description,
    tags: query.tags
  }
  saveDialogVisible.value = true
}

const deleteSavedQuery = async (queryId: string) => {
  try {
    await ElMessageBox.confirm('确定要删除这个查询吗？', '确认删除', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    
    await apiService.deleteSavedQuery(queryId)
    ElMessage.success('查询删除成功')
    loadSavedQueries()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('删除查询失败:', error)
      ElMessage.error('删除查询失败')
    }
  }
}

const formatMarketCap = (value: number) => {
  return `${value}亿`
}

const getColumnFormatter = (type: string) => {
  switch (type) {
    case 'date':
      return (row: any, column: any, cellValue: any) => {
        return cellValue ? dayjs(cellValue).format('YYYY-MM-DD') : ''
      }
    case 'datetime':
      return (row: any, column: any, cellValue: any) => {
        return cellValue ? dayjs(cellValue).format('YYYY-MM-DD HH:mm:ss') : ''
      }
    case 'number':
    case 'float':
    case 'decimal':
      return (row: any, column: any, cellValue: any) => {
        return typeof cellValue === 'number' ? cellValue.toFixed(2) : cellValue
      }
    case 'percentage':
      return (row: any, column: any, cellValue: any) => {
        return typeof cellValue === 'number' ? `${(cellValue * 100).toFixed(2)}%` : cellValue
      }
    default:
      return undefined
  }
}

const getStatTitle = (key: string) => {
  const titleMap: Record<string, string> = {
    count: '记录数',
    mean: '平均值',
    median: '中位数',
    std: '标准差',
    min: '最小值',
    max: '最大值'
  }
  return titleMap[key] || key
}

const getStatPrecision = (key: string) => {
  return ['count'].includes(key) ? 0 : 2
}

const formatDate = (date: string) => {
  return dayjs(date).format('YYYY-MM-DD')
}

// 生命周期
onMounted(async () => {
  try {
    // 加载行业选项
    const industryResponse = await apiService.getIndustries()
    industryOptions.value = industryResponse.industries
    
    // 加载已保存查询
    loadSavedQueries()
  } catch (error) {
    console.error('初始化失败:', error)
  }
})
</script>

<style scoped>
.data-query {
  padding: 0;
}

.query-config-card,
.results-card,
.saved-queries-card {
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

.field-selection {
  margin-top: 20px;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 6px;
}

.custom-conditions {
  margin-top: 15px;
}

.condition-row {
  margin-bottom: 10px;
}

.data-summary {
  margin-bottom: 20px;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 6px;
}

.data-table {
  margin-top: 20px;
}

.pagination {
  margin-top: 20px;
  text-align: center;
}

.visualization-config {
  margin-bottom: 20px;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 6px;
}

.chart-container {
  height: 500px;
  margin-top: 20px;
}

.saved-queries-list {
  margin-top: 15px;
}

.saved-query-item {
  margin-bottom: 15px;
  cursor: pointer;
  transition: all 0.3s;
}

.saved-query-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.query-info h4 {
  margin: 0 0 8px 0;
  color: #303133;
  font-size: 16px;
}

.query-description {
  margin: 0 0 10px 0;
  color: #606266;
  font-size: 14px;
  line-height: 1.4;
}

.query-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.query-date {
  color: #909399;
  font-size: 12px;
}

.query-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .header-actions {
    flex-direction: column;
    gap: 5px;
  }
  
  .chart-container {
    height: 300px;
  }
  
  .query-actions {
    flex-direction: column;
  }
  
  .condition-row .el-col {
    margin-bottom: 10px;
  }
}
</style>