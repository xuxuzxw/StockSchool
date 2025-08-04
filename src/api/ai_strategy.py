"""AI策略系统API路由

提供AI策略系统的RESTful API接口，包括：
- 模型管理API
- 预测服务API
- 因子权重API
- 股票评分API
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from ..ai.strategy import (
    AIModelManager,
    FactorWeightEngine, 
    StockScoringEngine,
    PredictionService,
    ModelExplainer,
    BacktestEngine,
    StrategyCustomizer
)
from ..ai.strategy.prediction_service import PredictionRequest, BatchPredictionResult

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/ai-strategy", tags=["AI策略系统"])

# 初始化服务
model_manager = AIModelManager()
factor_weight_engine = FactorWeightEngine()
stock_scoring_engine = StockScoringEngine()
prediction_service = PredictionService()
model_explainer = ModelExplainer()
backtest_engine = BacktestEngine()
strategy_customizer = StrategyCustomizer()

# ==================== 请求/响应模型 ====================

class ModelVersionResponse(BaseModel):
    """模型版本响应"""
    model_name: str
    version: str
    model_type: str
    performance_metrics: Dict[str, float]
    is_active: bool
    created_at: str
    file_path: Optional[str] = None

class ModelListResponse(BaseModel):
    """模型列表响应"""
    models: List[ModelVersionResponse]
    total: int

class PredictionRequestModel(BaseModel):
    """预测请求模型"""
    stock_codes: List[str] = Field(..., description="股票代码列表")
    model_name: str = Field(..., description="模型名称")
    model_version: Optional[str] = Field(None, description="模型版本，不指定则使用活跃版本")
    prediction_date: Optional[str] = Field(None, description="预测日期，格式YYYY-MM-DD")
    return_scores: bool = Field(True, description="是否返回评分")
    return_factors: bool = Field(False, description="是否返回因子贡献")
    cache_ttl: int = Field(300, description="缓存时间（秒）")

class PredictionResultModel(BaseModel):
    """预测结果模型"""
    stock_code: str
    prediction_value: float
    confidence_score: float
    score: Optional[float] = None
    rank: Optional[int] = None
    factor_contributions: Optional[Dict[str, float]] = None
    prediction_date: Optional[str] = None
    model_info: Optional[Dict[str, str]] = None

class BatchPredictionResponse(BaseModel):
    """批量预测响应"""
    request_id: str
    predictions: List[PredictionResultModel]
    model_info: Dict[str, str]
    prediction_date: str
    processing_time: float
    cache_hit_ratio: float
    status: str
    error_message: Optional[str] = None

class FactorWeightResponse(BaseModel):
    """因子权重响应"""
    model_name: str
    model_version: str
    factor_name: str
    weight: float
    importance_score: float
    stability_score: float
    updated_at: str

class StockScoreResponse(BaseModel):
    """股票评分响应"""
    stock_code: str
    stock_name: str
    score: float
    rank: int
    industry_rank: Optional[int] = None
    industry_code: Optional[str] = None
    industry_name: Optional[str] = None
    model_name: str
    model_version: str
    score_date: str

class PerformanceStatsResponse(BaseModel):
    """性能统计响应"""
    total_requests: int
    avg_processing_time: float
    avg_cache_hit_ratio: float
    error_rate: float
    period_days: int

# ==================== 模型管理API ====================

@router.get("/models", response_model=ModelListResponse, summary="获取模型列表")
async def get_models(
    model_name: Optional[str] = Query(None, description="模型名称过滤"),
    is_active: Optional[bool] = Query(None, description="是否只返回活跃模型"),
    limit: int = Query(50, description="返回数量限制")
):
    """获取模型列表"""
    try:
        models = model_manager.get_model_versions(
            model_name=model_name,
            limit=limit
        )
        
        # 过滤活跃模型
        if is_active is not None:
            if is_active:
                active_models = []
                for model in models:
                    active_model = model_manager.get_active_model(model.model_name)
                    if active_model and active_model.version == model.version:
                        active_models.append(model)
                models = active_models
            else:
                # 过滤掉活跃模型
                filtered_models = []
                for model in models:
                    active_model = model_manager.get_active_model(model.model_name)
                    if not active_model or active_model.version != model.version:
                        filtered_models.append(model)
                models = filtered_models
        
        model_responses = []
        for model in models:
            # 检查是否为活跃模型
            active_model = model_manager.get_active_model(model.model_name)
            is_active_model = active_model and active_model.version == model.version
            
            model_responses.append(ModelVersionResponse(
                model_name=model.model_name,
                version=model.version,
                model_type=model.model_type,
                performance_metrics=model.performance_metrics,
                is_active=is_active_model,
                created_at=model.created_at.isoformat(),
                file_path=model.file_path
            ))
        
        return ModelListResponse(
            models=model_responses,
            total=len(model_responses)
        )
        
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")

@router.get("/models/{model_name}/active", response_model=ModelVersionResponse, summary="获取活跃模型")
async def get_active_model(model_name: str):
    """获取指定模型的活跃版本"""
    try:
        active_model = model_manager.get_active_model(model_name)
        if not active_model:
            raise HTTPException(status_code=404, detail=f"没有找到活跃模型: {model_name}")
        
        return ModelVersionResponse(
            model_name=active_model.model_name,
            version=active_model.version,
            model_type=active_model.model_type,
            performance_metrics=active_model.performance_metrics,
            is_active=True,
            created_at=active_model.created_at.isoformat(),
            file_path=active_model.file_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取活跃模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取活跃模型失败: {str(e)}")

@router.post("/models/{model_name}/deploy/{version}", summary="部署模型")
async def deploy_model(model_name: str, version: str):
    """部署指定版本的模型为活跃版本"""
    try:
        success = model_manager.deploy_model(model_name, version)
        if not success:
            raise HTTPException(status_code=400, detail=f"模型部署失败: {model_name} v{version}")
        
        # 清除预测服务缓存
        prediction_service.reload_model(model_name, version)
        
        return {"message": f"模型部署成功: {model_name} v{version}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"部署模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"部署模型失败: {str(e)}")

@router.delete("/models/{model_name}/{version}", summary="删除模型版本")
async def delete_model_version(model_name: str, version: str):
    """删除指定的模型版本"""
    try:
        success = model_manager.delete_model_version(model_name, version)
        if not success:
            raise HTTPException(status_code=400, detail=f"模型删除失败: {model_name} v{version}")
        
        # 清除预测服务缓存
        prediction_service.reload_model(model_name, version)
        
        return {"message": f"模型删除成功: {model_name} v{version}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除模型失败: {str(e)}")

# ==================== 预测服务API ====================

@router.post("/predict", response_model=BatchPredictionResponse, summary="批量预测")
async def predict_stocks(request: PredictionRequestModel):
    """批量预测股票"""
    try:
        # 验证股票代码数量
        if len(request.stock_codes) > 1000:
            raise HTTPException(status_code=400, detail="股票代码数量不能超过1000个")
        
        # 创建预测请求
        prediction_request = PredictionRequest(
            stock_codes=request.stock_codes,
            model_name=request.model_name,
            model_version=request.model_version,
            prediction_date=request.prediction_date,
            return_scores=request.return_scores,
            return_factors=request.return_factors,
            cache_ttl=request.cache_ttl
        )
        
        # 执行预测
        result = prediction_service.predict(prediction_request)
        
        # 转换结果格式
        prediction_results = []
        for pred in result.predictions:
            prediction_results.append(PredictionResultModel(
                stock_code=pred.stock_code,
                prediction_value=pred.prediction_value,
                confidence_score=pred.confidence_score,
                score=pred.score,
                rank=pred.rank,
                factor_contributions=pred.factor_contributions,
                prediction_date=pred.prediction_date,
                model_info=pred.model_info
            ))
        
        return BatchPredictionResponse(
            request_id=result.request_id,
            predictions=prediction_results,
            model_info=result.model_info,
            prediction_date=result.prediction_date,
            processing_time=result.processing_time,
            cache_hit_ratio=result.cache_hit_ratio,
            status=result.status,
            error_message=result.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@router.get("/predict/{stock_code}/history", summary="获取预测历史")
async def get_prediction_history(
    stock_code: str,
    model_name: str = Query(..., description="模型名称"),
    days: int = Query(30, description="历史天数")
):
    """获取股票预测历史"""
    try:
        history_df = prediction_service.get_prediction_history(
            stock_code=stock_code,
            model_name=model_name,
            days=days
        )
        
        if history_df.empty:
            return {"history": [], "total": 0}
        
        # 转换为字典列表
        history_list = history_df.to_dict('records')
        
        # 转换日期格式
        for record in history_list:
            if 'prediction_date' in record and record['prediction_date']:
                record['prediction_date'] = record['prediction_date'].strftime('%Y-%m-%d')
        
        return {
            "history": history_list,
            "total": len(history_list)
        }
        
    except Exception as e:
        logger.error(f"获取预测历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取预测历史失败: {str(e)}")

@router.get("/predict/performance/{model_name}", response_model=PerformanceStatsResponse, summary="获取预测性能统计")
async def get_prediction_performance(
    model_name: str,
    days: int = Query(7, description="统计天数")
):
    """获取预测性能统计"""
    try:
        stats = prediction_service.get_performance_stats(model_name, days)
        
        return PerformanceStatsResponse(
            total_requests=stats.get('total_requests', 0),
            avg_processing_time=stats.get('avg_processing_time', 0),
            avg_cache_hit_ratio=stats.get('avg_cache_hit_ratio', 0),
            error_rate=stats.get('error_rate', 0),
            period_days=stats.get('period_days', days)
        )
        
    except Exception as e:
        logger.error(f"获取性能统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能统计失败: {str(e)}")

@router.post("/predict/cache/clear", summary="清理预测缓存")
async def clear_prediction_cache(
    pattern: str = Query("prediction:*", description="缓存键模式")
):
    """清理预测缓存"""
    try:
        cleared_count = prediction_service.clear_cache(pattern)
        return {
            "message": f"缓存清理完成",
            "cleared_count": cleared_count
        }
        
    except Exception as e:
        logger.error(f"清理缓存失败: {e}")
        raise HTTPException(status_code=500, detail=f"清理缓存失败: {str(e)}")

# ==================== 因子权重API ====================

@router.get("/factors/weights", response_model=List[FactorWeightResponse], summary="获取因子权重")
async def get_factor_weights(
    model_name: str = Query(..., description="模型名称"),
    model_version: Optional[str] = Query(None, description="模型版本"),
    factor_names: Optional[List[str]] = Query(None, description="因子名称列表")
):
    """获取因子权重"""
    try:
        weights = factor_weight_engine.get_factor_weights(
            model_name=model_name,
            model_version=model_version
        )
        
        # 过滤因子
        if factor_names:
            weights = [w for w in weights if w.factor_name in factor_names]
        
        weight_responses = []
        for weight in weights:
            weight_responses.append(FactorWeightResponse(
                model_name=weight.model_name,
                model_version=weight.model_version,
                factor_name=weight.factor_name,
                weight=weight.weight,
                importance_score=weight.importance_score,
                stability_score=weight.stability_score,
                updated_at=weight.updated_at.isoformat()
            ))
        
        return weight_responses
        
    except Exception as e:
        logger.error(f"获取因子权重失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取因子权重失败: {str(e)}")

@router.post("/factors/weights/update", summary="更新因子权重")
async def update_factor_weights(
    model_name: str = Query(..., description="模型名称"),
    model_version: str = Query(..., description="模型版本"),
    recalculate: bool = Query(True, description="是否重新计算权重")
):
    """更新因子权重"""
    try:
        if recalculate:
            # 重新计算权重
            success = factor_weight_engine.calculate_factor_weights(
                model_name=model_name,
                model_version=model_version
            )
        else:
            # 仅更新现有权重
            success = factor_weight_engine.update_factor_weights(
                model_name=model_name,
                model_version=model_version
            )
        
        if not success:
            raise HTTPException(status_code=400, detail="因子权重更新失败")
        
        return {"message": "因子权重更新成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新因子权重失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新因子权重失败: {str(e)}")

@router.get("/factors/weights/history", summary="获取因子权重历史")
async def get_factor_weight_history(
    model_name: str = Query(..., description="模型名称"),
    factor_name: str = Query(..., description="因子名称"),
    days: int = Query(30, description="历史天数")
):
    """获取因子权重变化历史"""
    try:
        history_df = factor_weight_engine.get_weight_history(
            model_name=model_name,
            factor_name=factor_name,
            days=days
        )
        
        if history_df.empty:
            return {"history": [], "total": 0}
        
        # 转换为字典列表
        history_list = history_df.to_dict('records')
        
        # 转换日期格式
        for record in history_list:
            if 'date' in record and record['date']:
                record['date'] = record['date'].strftime('%Y-%m-%d')
            if 'created_at' in record and record['created_at']:
                record['created_at'] = record['created_at'].isoformat()
        
        return {
            "history": history_list,
            "total": len(history_list)
        }
        
    except Exception as e:
        logger.error(f"获取因子权重历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取因子权重历史失败: {str(e)}")

# ==================== 股票评分API ====================

@router.get("/stocks/scores", response_model=List[StockScoreResponse], summary="获取股票评分")
async def get_stock_scores(
    model_name: str = Query(..., description="模型名称"),
    model_version: Optional[str] = Query(None, description="模型版本"),
    score_date: Optional[str] = Query(None, description="评分日期，格式YYYY-MM-DD"),
    stock_codes: Optional[List[str]] = Query(None, description="股票代码列表"),
    min_score: Optional[float] = Query(None, description="最低评分"),
    industry_code: Optional[str] = Query(None, description="行业代码"),
    limit: int = Query(100, description="返回数量限制")
):
    """获取股票评分"""
    try:
        # 设置默认日期
        if score_date is None:
            score_date = datetime.now().strftime('%Y-%m-%d')
        
        scores = stock_scoring_engine.get_stock_scores(
            model_name=model_name,
            model_version=model_version,
            score_date=score_date
        )
        
        # 过滤条件
        if stock_codes:
            scores = [s for s in scores if s.stock_code in stock_codes]
        
        if min_score is not None:
            scores = [s for s in scores if s.score >= min_score]
        
        if industry_code:
            scores = [s for s in scores if s.industry_code == industry_code]
        
        # 限制数量
        scores = scores[:limit]
        
        score_responses = []
        for score in scores:
            score_responses.append(StockScoreResponse(
                stock_code=score.stock_code,
                stock_name=score.stock_name,
                score=score.score,
                rank=score.rank,
                industry_rank=score.industry_rank,
                industry_code=score.industry_code,
                industry_name=score.industry_name,
                model_name=score.model_name,
                model_version=score.model_version,
                score_date=score.score_date.strftime('%Y-%m-%d')
            ))
        
        return score_responses
        
    except Exception as e:
        logger.error(f"获取股票评分失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取股票评分失败: {str(e)}")

@router.get("/stocks/top-scores", response_model=List[StockScoreResponse], summary="获取高评分股票")
async def get_top_stocks(
    model_name: str = Query(..., description="模型名称"),
    model_version: Optional[str] = Query(None, description="模型版本"),
    score_date: Optional[str] = Query(None, description="评分日期，格式YYYY-MM-DD"),
    top_n: int = Query(50, description="返回前N只股票"),
    industry_neutral: bool = Query(False, description="是否行业中性选择")
):
    """获取高评分股票"""
    try:
        # 设置默认日期
        if score_date is None:
            score_date = datetime.now().strftime('%Y-%m-%d')
        
        top_stocks = stock_scoring_engine.get_top_stocks(
            model_name=model_name,
            model_version=model_version,
            score_date=score_date,
            top_n=top_n,
            industry_neutral=industry_neutral
        )
        
        stock_responses = []
        for stock in top_stocks:
            stock_responses.append(StockScoreResponse(
                stock_code=stock.stock_code,
                stock_name=stock.stock_name,
                score=stock.score,
                rank=stock.rank,
                industry_rank=stock.industry_rank,
                industry_code=stock.industry_code,
                industry_name=stock.industry_name,
                model_name=stock.model_name,
                model_version=stock.model_version,
                score_date=stock.score_date.strftime('%Y-%m-%d')
            ))
        
        return stock_responses
        
    except Exception as e:
        logger.error(f"获取高评分股票失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取高评分股票失败: {str(e)}")

@router.post("/stocks/scores/update", summary="更新股票评分")
async def update_stock_scores(
    model_name: str = Query(..., description="模型名称"),
    model_version: Optional[str] = Query(None, description="模型版本"),
    score_date: Optional[str] = Query(None, description="评分日期，格式YYYY-MM-DD")
):
    """更新股票评分"""
    try:
        # 设置默认日期
        if score_date is None:
            score_date = datetime.now().strftime('%Y-%m-%d')
        
        success = stock_scoring_engine.update_daily_scores(
            model_name=model_name,
            model_version=model_version,
            score_date=score_date
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="股票评分更新失败")
        
        return {"message": f"股票评分更新成功: {score_date}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新股票评分失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新股票评分失败: {str(e)}")

# ==================== 系统状态API ====================

@router.get("/health", summary="健康检查")
async def health_check():
    """系统健康检查"""
    try:
        # 检查各个服务状态
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "model_manager": "healthy",
                "factor_weight_engine": "healthy",
                "stock_scoring_engine": "healthy",
                "prediction_service": "healthy"
            }
        }
        
        # 检查缓存状态
        if prediction_service.cache_enabled:
            status["services"]["redis_cache"] = "healthy"
        else:
            status["services"]["redis_cache"] = "disabled"
        
        return status
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/info", summary="系统信息")
async def get_system_info():
    """获取系统信息"""
    try:
        return {
            "name": "AI策略系统",
            "version": "1.0.0",
            "description": "基于AI的量化投资策略系统",
            "features": [
                "AI模型管理",
                "因子权重计算",
                "股票评分排名",
                "实时预测服务",
                "缓存优化",
                "性能监控",
                "模型解释",
                "策略回测",
                "个性化定制"
            ],
            "endpoints": {
                "models": "/ai-strategy/models",
                "predict": "/ai-strategy/predict",
                "factors": "/ai-strategy/factors",
                "stocks": "/ai-strategy/stocks",
                "explain": "/ai-strategy/model/explain",
                "backtest": "/ai-strategy/backtest",
                "strategy": "/ai-strategy/strategy"
            }
        }
        
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统信息失败: {str(e)}")

# ==================== 模型解释API ====================

@router.post("/model/explain", summary="解释模型预测")
async def explain_prediction(
    stock_code: str = Body(..., description="股票代码"),
    model_name: str = Body(..., description="模型名称"),
    model_version: Optional[str] = Body(None, description="模型版本"),
    method: str = Body('shap', description="解释方法")
):
    """解释模型预测"""
    try:
        explanation = model_explainer.explain_prediction(
            stock_code=stock_code,
            model_name=model_name,
            model_version=model_version,
            method=method
        )
        
        if explanation:
            return {
                "message": "解释生成成功",
                "data": explanation
            }
        else:
            raise HTTPException(status_code=404, detail="无法生成解释")
            
    except Exception as e:
        logger.error(f"解释模型预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/explain/global", summary="获取全局模型解释")
async def get_global_explanation(
    model_name: str = Query(..., description="模型名称"),
    model_version: Optional[str] = Query(None, description="模型版本"),
    method: str = Query('shap', description="解释方法")
):
    """获取全局模型解释"""
    try:
        explanation = model_explainer.get_global_explanation(
            model_name=model_name,
            model_version=model_version,
            method=method
        )
        
        if explanation:
            return {
                "message": "获取成功",
                "data": explanation
            }
        else:
            raise HTTPException(status_code=404, detail="全局解释不存在")
            
    except Exception as e:
        logger.error(f"获取全局解释失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/explain/history", summary="获取解释历史")
async def get_explanation_history(
    stock_code: Optional[str] = Query(None, description="股票代码"),
    model_name: Optional[str] = Query(None, description="模型名称"),
    limit: int = Query(20, description="返回数量限制")
):
    """获取解释历史"""
    try:
        explanations = model_explainer.get_explanation_history(
            stock_code=stock_code,
            model_name=model_name,
            limit=limit
        )
        
        return {
            "message": "获取成功",
            "data": {
                "explanations": explanations,
                "total": len(explanations)
            }
        }
        
    except Exception as e:
        logger.error(f"获取解释历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 回测引擎API ====================

@router.post("/backtest/run", summary="运行回测")
async def run_backtest(config_data: dict = Body(..., description="回测配置")):
    """运行回测"""
    try:
        # 运行回测
        result = backtest_engine.run_backtest(config_data)
        
        if result:
            return {
                "message": "回测完成",
                "data": result
            }
        else:
            raise HTTPException(status_code=500, detail="回测失败")
            
    except Exception as e:
        logger.error(f"运行回测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backtest/results", summary="获取回测结果")
async def get_backtest_results(
    strategy_name: Optional[str] = Query(None, description="策略名称"),
    limit: int = Query(10, description="返回数量限制")
):
    """获取回测结果"""
    try:
        results = backtest_engine.get_backtest_results(
            strategy_name=strategy_name,
            limit=limit
        )
        
        return {
            "message": "获取成功",
            "data": {
                "results": results,
                "total": len(results)
            }
        }
        
    except Exception as e:
        logger.error(f"获取回测结果失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backtest/compare", summary="比较策略")
async def compare_strategies(strategy_ids: List[str] = Body(..., description="策略ID列表")):
    """比较策略"""
    try:
        comparison = backtest_engine.compare_strategies(strategy_ids)
        
        if comparison:
            return {
                "message": "比较完成",
                "data": comparison
            }
        else:
            raise HTTPException(status_code=404, detail="策略不存在")
            
    except Exception as e:
        logger.error(f"比较策略失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 策略定制API ====================

@router.post("/strategy/risk-profile", summary="创建用户风险画像")
async def create_risk_profile(
    user_id: str = Body(..., description="用户ID"),
    profile_data: dict = Body(..., description="风险画像数据")
):
    """创建用户风险画像"""
    try:
        risk_profile = strategy_customizer.create_user_risk_profile(
            user_id=user_id,
            profile_data=profile_data
        )
        
        if risk_profile:
            return {
                "message": "风险画像创建成功",
                "data": risk_profile
            }
        else:
            raise HTTPException(status_code=400, detail="创建风险画像失败")
            
    except Exception as e:
        logger.error(f"创建风险画像失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategy/risk-profile/{user_id}", summary="获取用户风险画像")
async def get_risk_profile(user_id: str):
    """获取用户风险画像"""
    try:
        risk_profile = strategy_customizer.get_user_risk_profile(user_id)
        
        if risk_profile:
            return {
                "message": "获取成功",
                "data": risk_profile
            }
        else:
            raise HTTPException(status_code=404, detail="风险画像不存在")
            
    except Exception as e:
        logger.error(f"获取风险画像失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategy/recommend", summary="推荐策略")
async def recommend_strategies(user_id: str = Body(..., description="用户ID")):
    """推荐策略"""
    try:
        recommendation = strategy_customizer.recommend_strategies(user_id)
        
        if recommendation:
            return {
                "message": "推荐生成成功",
                "data": recommendation
            }
        else:
            raise HTTPException(status_code=404, detail="无法生成推荐")
            
    except Exception as e:
        logger.error(f"推荐策略失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategy/templates", summary="获取策略模板")
async def get_strategy_templates(category: Optional[str] = Query(None, description="模板分类")):
    """获取策略模板"""
    try:
        templates = strategy_customizer.get_strategy_templates(category=category)
        
        return {
            "message": "获取成功",
            "data": {
                "templates": templates,
                "total": len(templates)
            }
        }
        
    except Exception as e:
        logger.error(f"获取策略模板失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategy/create", summary="创建用户策略")
async def create_user_strategy(
    user_id: str = Body(..., description="用户ID"),
    strategy_name: str = Body(..., description="策略名称"),
    template_id: str = Body(..., description="模板ID"),
    custom_config: dict = Body(..., description="自定义配置"),
    investment_amount: float = Body(..., description="投资金额")
):
    """创建用户策略"""
    try:
        user_strategy = strategy_customizer.create_user_strategy(
            user_id=user_id,
            strategy_name=strategy_name,
            template_id=template_id,
            custom_config=custom_config,
            investment_amount=investment_amount
        )
        
        if user_strategy:
            return {
                "message": "策略创建成功",
                "data": user_strategy
            }
        else:
            raise HTTPException(status_code=400, detail="创建策略失败")
            
    except Exception as e:
        logger.error(f"创建用户策略失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategy/user/{user_id}", summary="获取用户策略列表")
async def get_user_strategies(
    user_id: str,
    status: Optional[str] = Query(None, description="策略状态")
):
    """获取用户策略列表"""
    try:
        strategies = strategy_customizer.get_user_strategies(
            user_id=user_id,
            status=status
        )
        
        return {
            "message": "获取成功",
            "data": {
                "strategies": strategies,
                "total": len(strategies)
            }
        }
        
    except Exception as e:
        logger.error(f"获取用户策略失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategy/backtest/{strategy_id}", summary="回测用户策略")
async def backtest_user_strategy(strategy_id: str):
    """回测用户策略"""
    try:
        result = strategy_customizer.backtest_user_strategy(strategy_id)
        
        if result:
            return {
                "message": "回测完成",
                "data": result
            }
        else:
            raise HTTPException(status_code=404, detail="策略不存在或回测失败")
            
    except Exception as e:
        logger.error(f"回测用户策略失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))