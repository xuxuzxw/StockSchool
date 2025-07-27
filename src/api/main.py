import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, Query, Path, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger
import pandas as pd
from sqlalchemy import text
from src.utils.config_loader import config

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.db import get_db_engine
from src.data.tushare_sync import TushareSynchronizer
from src.compute.factor_engine import FactorEngine
from src.compute.quality import DataQualityMonitor
from src.features.feature_store import FeatureStore
from src.strategy.evaluation import StrategyEvaluator
from src.monitoring.performance import performance_manager
from src.monitoring.alerts import AlertEngine

# 创建FastAPI应用
app = FastAPI(
    title="StockSchool量化投研系统API",
    description="专业的量化投研数据分析平台",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
db_engine = None
tushare_syncer = None
factor_engine = None
quality_monitor = None
feature_store = None
strategy_evaluator = None
alert_engine = None

# Pydantic模型
class StockBasicResponse(BaseModel):
    """股票基础信息响应模型"""
    ts_code: str = Field(..., description="股票代码")
    symbol: str = Field(..., description="股票简称")
    name: str = Field(..., description="股票名称")
    area: Optional[str] = Field(None, description="地域")
    industry: Optional[str] = Field(None, description="行业")
    market: Optional[str] = Field(None, description="市场类型")
    list_date: Optional[str] = Field(None, description="上市日期")

class StockDailyResponse(BaseModel):
    """股票日线数据响应模型"""
    ts_code: str = Field(..., description="股票代码")
    trade_date: str = Field(..., description="交易日期")
    open: Optional[float] = Field(None, description="开盘价")
    high: Optional[float] = Field(None, description="最高价")
    low: Optional[float] = Field(None, description="最低价")
    close: Optional[float] = Field(None, description="收盘价")
    pre_close: Optional[float] = Field(None, description="昨收价")
    change: Optional[float] = Field(None, description="涨跌额")
    pct_chg: Optional[float] = Field(None, description="涨跌幅")
    vol: Optional[float] = Field(None, description="成交量")
    amount: Optional[float] = Field(None, description="成交额")

class FactorValueResponse(BaseModel):
    """因子值响应模型"""
    ts_code: str = Field(..., description="股票代码")
    factor_name: str = Field(..., description="因子名称")
    trade_date: str = Field(..., description="交易日期")
    value: float = Field(..., description="因子值")

class DataQualityResponse(BaseModel):
    """数据质量响应模型"""
    ts_code: str = Field(..., description="股票代码")
    completeness_score: float = Field(..., description="完整性评分")
    consistency_score: float = Field(..., description="一致性评分")
    accuracy_score: float = Field(..., description="准确性评分")
    timeliness_score: float = Field(..., description="及时性评分")
    overall_score: float = Field(..., description="总体评分")
    check_date: str = Field(..., description="检查日期")

class StrategyPerformanceResponse(BaseModel):
    """策略表现响应模型"""
    total_return: float = Field(..., description="总收益率")
    annual_return: float = Field(..., description="年化收益率")
    annual_volatility: float = Field(..., description="年化波动率")
    sharpe_ratio: float = Field(..., description="夏普比率")
    max_drawdown: float = Field(..., description="最大回撤")
    win_rate: float = Field(..., description="胜率")

class SystemHealthResponse(BaseModel):
    """系统健康状态响应模型"""
    status: str = Field(..., description="系统状态")
    timestamp: str = Field(..., description="检查时间")
    cpu_usage: Optional[float] = Field(None, description="CPU使用率")
    memory_usage: Optional[float] = Field(None, description="内存使用率")
    disk_usage: Optional[float] = Field(None, description="磁盘使用率")
    issues: List[str] = Field(default_factory=list, description="发现的问题")

# 依赖注入
def get_database():
    """获取数据库连接"""
    global db_engine
    if db_engine is None:
        db_engine = get_db_engine()
    return db_engine

def get_tushare_syncer():
    """获取Tushare同步器"""
    global tushare_syncer
    if tushare_syncer is None:
        tushare_syncer = TushareSynchronizer()
    return tushare_syncer

def get_factor_engine():
    """获取因子引擎"""
    global factor_engine
    if factor_engine is None:
        factor_engine = FactorEngine()
    return factor_engine

def get_quality_monitor():
    """获取数据质量监控器"""
    global quality_monitor
    if quality_monitor is None:
        quality_monitor = DataQualityMonitor()
    return quality_monitor

def get_feature_store():
    """获取特征商店"""
    global feature_store
    if feature_store is None:
        feature_store = FeatureStore()
    return feature_store

def get_strategy_evaluator():
    """获取策略评估器"""
    global strategy_evaluator
    if strategy_evaluator is None:
        strategy_evaluator = StrategyEvaluator()
    return strategy_evaluator

def get_alert_engine():
    """获取告警引擎"""
    global alert_engine
    if alert_engine is None:
        alert_engine = AlertEngine()
        alert_engine.start()
    return alert_engine

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("StockSchool API 启动中...")
    
    # 启动性能监控
    performance_manager.start()
    
    # 初始化告警引擎
    get_alert_engine()
    
    logger.info("StockSchool API 启动完成")

# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("StockSchool API 关闭中...")
    
    # 停止性能监控
    performance_manager.stop()
    
    # 停止告警引擎
    if alert_engine:
        alert_engine.stop()
    
    logger.info("StockSchool API 已关闭")

# 根路径
@app.get("/", summary="API根路径")
async def root():
    """API根路径"""
    return {
        "message": "欢迎使用StockSchool量化投研系统API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

# 健康检查
@app.get("/health", response_model=SystemHealthResponse, summary="系统健康检查")
async def health_check():
    """系统健康检查"""
    try:
        health = performance_manager.get_system_health()
        return SystemHealthResponse(
            status=health['status'],
            timestamp=health['timestamp'],
            cpu_usage=health.get('cpu', {}).get('latest'),
            memory_usage=health.get('memory', {}).get('latest'),
            disk_usage=health.get('disk', {}).get('latest'),
            issues=health['issues']
        )
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail="健康检查失败")

# 股票基础信息API
@app.get("/api/v1/stocks/basic", response_model=List[StockBasicResponse], summary="获取股票基础信息")
async def get_stocks_basic(
    market: Optional[str] = Query(None, description="市场类型"),
    industry: Optional[str] = Query(None, description="行业"),
    limit: int = Query(
        config.get('api_params.default_limit', 100), 
        ge=config.get('api_params.min_limit', 1), 
        le=config.get('api_params.max_limit', 1000), 
        description="返回数量限制"
    ),
    db_engine=Depends(get_database)
):
    """获取股票基础信息"""
    try:
        query = "SELECT * FROM stock_basic WHERE 1=1"
        params = {}
        
        if market:
            query += " AND market = :market"
            params['market'] = market
        
        if industry:
            query += " AND industry = :industry"
            params['industry'] = industry
        
        query += " LIMIT :limit"
        params['limit'] = limit
        
        with db_engine.connect() as conn:
            result = conn.execute(text(query), params)
            stocks = result.fetchall()
        
        return [
            StockBasicResponse(
                ts_code=stock.ts_code,
                symbol=stock.symbol,
                name=stock.name,
                area=stock.area,
                industry=stock.industry,
                market=stock.market,
                list_date=stock.list_date.strftime('%Y%m%d') if stock.list_date else None
            )
            for stock in stocks
        ]
    
    except Exception as e:
        logger.error(f"获取股票基础信息失败: {e}")
        raise HTTPException(status_code=500, detail="获取股票基础信息失败")

# 股票日线数据API
@app.get("/api/v1/stocks/{ts_code}/daily", response_model=List[StockDailyResponse], summary="获取股票日线数据")
async def get_stock_daily(
    ts_code: str = Path(..., description="股票代码"),
    start_date: Optional[str] = Query(None, description="开始日期 (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="结束日期 (YYYY-MM-DD)"),
    limit: int = Query(
        config.get('api_params.default_limit', 100), 
        ge=config.get('api_params.min_limit', 1), 
        le=config.get('api_params.max_limit', 1000), 
        description="返回数量限制"
    ),
    db_engine=Depends(get_database)
):
    """获取股票日线数据"""
    try:
        query = "SELECT * FROM stock_daily WHERE ts_code = :ts_code"
        params = {'ts_code': ts_code}
        
        if start_date:
            query += " AND trade_date >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND trade_date <= :end_date"
            params['end_date'] = end_date
        
        query += " ORDER BY trade_date DESC LIMIT :limit"
        params['limit'] = limit
        
        with db_engine.connect() as conn:
            result = conn.execute(text(query), params)
            daily_data = result.fetchall()
        
        return [
            StockDailyResponse(
                ts_code=data.ts_code,
                trade_date=data.trade_date,
                open=data.open,
                high=data.high,
                low=data.low,
                close=data.close,
                pre_close=data.pre_close,
                change=data.change,
                pct_chg=data.pct_chg,
                vol=data.vol,
                amount=data.amount
            )
            for data in daily_data
        ]
    
    except Exception as e:
        logger.error(f"获取股票日线数据失败: {e}")
        raise HTTPException(status_code=500, detail="获取股票日线数据失败")

# 因子值API
@app.get("/api/v1/factors/{factor_name}/values", response_model=List[FactorValueResponse], summary="获取因子值")
async def get_factor_values(
    factor_name: str = Path(..., description="因子名称"),
    ts_code: Optional[str] = Query(None, description="股票代码"),
    start_date: Optional[str] = Query(None, description="开始日期 (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="结束日期 (YYYY-MM-DD)"),
    limit: int = Query(
        config.get('api_params.default_limit', 100), 
        ge=config.get('api_params.min_limit', 1), 
        le=config.get('api_params.max_limit', 1000), 
        description="返回数量限制"
    ),
    db_engine=Depends(get_database)
):
    """获取因子值"""
    try:
        query = "SELECT * FROM factor_values WHERE factor_name = :factor_name"
        params = {'factor_name': factor_name}
        
        if ts_code:
            query += " AND ts_code = :ts_code"
            params['ts_code'] = ts_code
        
        if start_date:
            query += " AND trade_date >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND trade_date <= :end_date"
            params['end_date'] = end_date
        
        query += " ORDER BY trade_date DESC, ts_code LIMIT :limit"
        params['limit'] = limit
        
        with db_engine.connect() as conn:
            result = conn.execute(text(query), params)
            factor_data = result.fetchall()
        
        return [
            FactorValueResponse(
                ts_code=data.ts_code,
                factor_name=data.factor_name,
                trade_date=data.trade_date,
                value=data.value
            )
            for data in factor_data
        ]
    
    except Exception as e:
        logger.error(f"获取因子值失败: {e}")
        raise HTTPException(status_code=500, detail="获取因子值失败")

# 数据同步API
@app.post("/api/v1/data/sync/basic", summary="同步股票基础信息")
async def sync_stock_basic(
    background_tasks: BackgroundTasks,
    syncer=Depends(get_tushare_syncer)
):
    """同步股票基础信息"""
    try:
        background_tasks.add_task(syncer.sync_stock_basic)
        return {"message": "股票基础信息同步任务已启动"}
    except Exception as e:
        logger.error(f"启动股票基础信息同步失败: {e}")
        raise HTTPException(status_code=500, detail="启动同步任务失败")

@app.post("/api/v1/data/sync/daily/{ts_code}", summary="同步股票日线数据")
async def sync_stock_daily(
    ts_code: str = Path(..., description="股票代码"),
    start_date: Optional[str] = Query(None, description="开始日期(YYYYMMDD)"),
    end_date: Optional[str] = Query(None, description="结束日期(YYYYMMDD)"),
    background_tasks: BackgroundTasks = None,
    syncer=Depends(get_tushare_syncer)
):
    """同步股票日线数据"""
    try:
        if background_tasks:
            background_tasks.add_task(
                syncer.sync_stock_daily, 
                ts_code=ts_code, 
                start_date=start_date, 
                end_date=end_date
            )
            return {"message": f"股票 {ts_code} 日线数据同步任务已启动"}
        else:
            syncer.sync_stock_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            return {"message": f"股票 {ts_code} 日线数据同步完成"}
    except Exception as e:
        logger.error(f"同步股票日线数据失败: {e}")
        raise HTTPException(status_code=500, detail="同步股票日线数据失败")

# 因子计算API
@app.post("/api/v1/factors/calculate/{ts_code}", summary="计算股票因子")
async def calculate_stock_factors(
    ts_code: str = Path(..., description="股票代码"),
    start_date: Optional[str] = Query(None, description="开始日期(YYYYMMDD)"),
    end_date: Optional[str] = Query(None, description="结束日期(YYYYMMDD)"),
    background_tasks: BackgroundTasks = None,
    factor_engine=Depends(get_factor_engine)
):
    """计算股票因子"""
    try:
        if background_tasks:
            background_tasks.add_task(
                factor_engine.calculate_stock_factors,
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            return {"message": f"股票 {ts_code} 因子计算任务已启动"}
        else:
            factor_engine.calculate_stock_factors(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            return {"message": f"股票 {ts_code} 因子计算完成"}
    except Exception as e:
        logger.error(f"计算股票因子失败: {e}")
        raise HTTPException(status_code=500, detail="计算股票因子失败")

# 数据质量检查API
@app.get("/api/v1/quality/check/{ts_code}", response_model=DataQualityResponse, summary="检查数据质量")
async def check_data_quality(
    ts_code: str = Path(..., description="股票代码"),
    quality_monitor=Depends(get_quality_monitor)
):
    """检查数据质量"""
    try:
        quality_result = quality_monitor.check_stock_quality(ts_code)
        
        return DataQualityResponse(
            ts_code=ts_code,
            completeness_score=quality_result['completeness_score'],
            consistency_score=quality_result['consistency_score'],
            accuracy_score=quality_result['accuracy_score'],
            timeliness_score=quality_result['timeliness_score'],
            overall_score=quality_result['overall_score'],
            check_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    except Exception as e:
        logger.error(f"检查数据质量失败: {e}")
        raise HTTPException(status_code=500, detail="检查数据质量失败")

# 策略评估API
@app.post("/api/v1/strategy/evaluate", response_model=StrategyPerformanceResponse, summary="评估策略表现")
async def evaluate_strategy(
    returns_data: List[float],
    benchmark_returns: Optional[List[float]] = None,
    strategy_evaluator=Depends(get_strategy_evaluator)
):
    """评估策略表现"""
    try:
        returns_series = pd.Series(returns_data)
        benchmark_series = pd.Series(benchmark_returns) if benchmark_returns else None
        
        performance = strategy_evaluator.evaluate_strategy(
            returns_series, 
            benchmark_series
        )
        
        return StrategyPerformanceResponse(
            total_return=performance['total_return'],
            annual_return=performance['annual_return'],
            annual_volatility=performance['annual_volatility'],
            sharpe_ratio=performance['sharpe_ratio'],
            max_drawdown=performance['max_drawdown'],
            win_rate=performance['win_rate']
        )
    
    except Exception as e:
        logger.error(f"评估策略表现失败: {e}")
        raise HTTPException(status_code=500, detail="评估策略表现失败")

# 性能监控API
@app.get("/api/v1/monitoring/performance", summary="获取性能报告")
async def get_performance_report():
    """获取性能报告"""
    try:
        report = performance_manager.generate_performance_report()
        return {"report": report}
    except Exception as e:
        logger.error(f"获取性能报告失败: {e}")
        raise HTTPException(status_code=500, detail="获取性能报告失败")

@app.get("/api/v1/monitoring/alerts", summary="获取活跃告警")
async def get_active_alerts(
    alert_engine=Depends(get_alert_engine)
):
    """获取活跃告警"""
    try:
        alerts = alert_engine.get_active_alerts()
        return {
            "alerts": [alert.to_dict() for alert in alerts],
            "count": len(alerts)
        }
    except Exception as e:
        logger.error(f"获取活跃告警失败: {e}")
        raise HTTPException(status_code=500, detail="获取活跃告警失败")

# 错误处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理器"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    # 开发环境运行
    uvicorn.run(
        "main:app",
        host=config.get('api_params.host', '0.0.0.0'),
        port=config.get('api_params.port', 8000),
        reload=config.get('api_params.reload', True),
        log_level=config.get('api_params.log_level', 'info')
    )