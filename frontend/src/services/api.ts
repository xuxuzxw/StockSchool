import axios, { type AxiosInstance, type AxiosResponse } from 'axios'
import { ElMessage } from 'element-plus'

// API响应接口
interface ApiResponse<T = any> {
  code: number
  message: string
  data: T
}

// 股票评分接口
interface StockScore {
  symbol: string
  name: string
  score: number
  rank: number
  industry: string
  market_cap: number
  factors: Record<string, number>
  updated_at: string
}

// 因子数据接口
interface FactorData {
  factor_name: string
  factor_value: number
  weight: number
  importance: number
  category: string
}

// 回测结果接口
interface BacktestResult {
  strategy_id: string
  start_date: string
  end_date: string
  total_return: number
  annual_return: number
  sharpe_ratio: number
  max_drawdown: number
  win_rate: number
  returns: Array<{ date: string; value: number }>
}

// 模型解释接口
interface ModelExplanation {
  symbol: string
  prediction: number
  shap_values: Record<string, number>
  feature_importance: Array<{ feature: string; importance: number }>
  decision_path: Array<{ node: string; value: number; threshold: number }>
}

class ApiService {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: '/api/v1',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    })

    // 请求拦截器
    this.client.interceptors.request.use(
      (config) => {
        // 可以在这里添加认证token
        return config
      },
      (error) => {
        return Promise.reject(error)
      }
    )

    // 响应拦截器
    this.client.interceptors.response.use(
      (response: AxiosResponse<ApiResponse>) => {
        const { data } = response
        if (data.code !== 200) {
          ElMessage.error(data.message || '请求失败')
          return Promise.reject(new Error(data.message))
        }
        return data.data
      },
      (error) => {
        const message = error.response?.data?.message || error.message || '网络错误'
        ElMessage.error(message)
        return Promise.reject(error)
      }
    )
  }

  // 系统健康检查
  async getSystemHealth() {
    return this.client.get('/health')
  }

  // 获取股票评分列表
  async getStockScores(params?: {
    limit?: number
    offset?: number
    industry?: string
    min_score?: number
    max_score?: number
  }): Promise<StockScore[]> {
    return this.client.get('/stocks/scores', { params })
  }

  // 获取单只股票详情
  async getStockDetail(symbol: string) {
    return this.client.get(`/stocks/${symbol}`)
  }

  // 获取股票历史数据
  async getStockHistory(symbol: string, params?: {
    start_date?: string
    end_date?: string
    period?: string
  }) {
    return this.client.get(`/stocks/${symbol}/history`, { params })
  }

  // 获取因子数据
  async getFactorData(params?: {
    factor_names?: string[]
    symbols?: string[]
    start_date?: string
    end_date?: string
  }): Promise<FactorData[]> {
    return this.client.get('/factors', { params })
  }

  // 获取因子权重
  async getFactorWeights() {
    return this.client.get('/factors/weights')
  }

  // 更新因子权重
  async updateFactorWeights(weights: Record<string, number>) {
    return this.client.post('/factors/weights', { weights })
  }

  // 运行回测
  async runBacktest(strategy: {
    name: string
    factors: string[]
    weights: Record<string, number>
    start_date: string
    end_date: string
    rebalance_freq: string
    top_n: number
  }): Promise<BacktestResult> {
    return this.client.post('/backtest/run', strategy)
  }

  // 获取回测结果
  async getBacktestResults(strategy_id?: string): Promise<BacktestResult[]> {
    return this.client.get('/backtest/results', {
      params: strategy_id ? { strategy_id } : {}
    })
  }

  // 获取模型解释
  async getModelExplanation(symbol: string): Promise<ModelExplanation> {
    return this.client.get(`/models/explain/${symbol}`)
  }

  // 批量获取模型解释
  async getBatchModelExplanation(symbols: string[]): Promise<ModelExplanation[]> {
    return this.client.post('/models/explain/batch', { symbols })
  }

  // 获取模型性能指标
  async getModelMetrics() {
    return this.client.get('/models/metrics')
  }

  // 获取实时预测
  async getRealtimePredictions(symbols?: string[]) {
    return this.client.get('/predictions/realtime', {
      params: symbols ? { symbols: symbols.join(',') } : {}
    })
  }

  // 获取系统监控数据
  async getSystemMetrics() {
    return this.client.get('/system/metrics')
  }

  // 获取告警信息
  async getAlerts(params?: {
    status?: string
    level?: string
    limit?: number
  }) {
    return this.client.get('/alerts', { params })
  }

  // 数据查询
  async queryData(query: {
    table: string
    columns?: string[]
    conditions?: Record<string, any>
    limit?: number
    offset?: number
  }) {
    return this.client.post('/data/query', query)
  }

  // 导出数据
  async exportData(query: {
    table: string
    columns?: string[]
    conditions?: Record<string, any>
    format: 'csv' | 'excel'
  }) {
    return this.client.post('/data/export', query, {
      responseType: 'blob'
    })
  }
}

export const apiService = new ApiService()
export type { StockScore, FactorData, BacktestResult, ModelExplanation }