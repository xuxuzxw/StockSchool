from datetime import date, datetime
from enum import Enum

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field, validator

from src.api.decorators import (  # !/usr/bin/env python3; -*- coding: utf-8 -*-
    API服务,
    因子计算API接口,
    提供因子查询、计算触发、有效性分析等REST,
    Any,
    Dict,
    FactorCalculationService,
    FactorEffectivenessAnalyzer,
    FactorService,
    FactorStandardizer,
    List,
    Optional,
    """,
    asyncio,
    from,
    get_calculation_service,
    get_effectiveness_analyzer,
    get_factor_service,
    get_standardizer,
    import,
    logging,
    require_healthy_database,
    require_permission,
    src.api.dependencies,
    src.compute.factor_effectiveness_analyzer,
    src.compute.factor_standardizer,
    src.services.factor_service,
    typing,
    verify_token,
)

    api_exception_handler, api_response_builder, log_api_call, ResponseBuilder
)

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/v1/factors", tags=["factors"])

# 数据模型定义
class FactorType(str, Enum):
    """因子类型枚举"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    ALL = "all"

class CalculationStatus(str, Enum):
    """计算状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class StandardizationMethod(str, Enum):
    """标准化方法枚举"""
    ZSCORE = "zscore"
    QUANTILE = "quantile"
    RANK = "rank"
    MINMAX = "minmax"
    ROBUST = "robust"

# 请求模型
class FactorQueryRequest(BaseModel):
    """因子查询请求"""
    ts_codes: List[str] = Field(..., description="股票代码列表", max_items=1000)
    factor_names: Optional[List[str]] = Field(None, description="因子名称列表")
    factor_types: Optional[List[FactorType]] = Field(None, description="因子类型列表")
    start_date: Optional[date] = Field(None, description="开始日期")
    end_date: Optional[date] = Field(None, description="结束日期")
    standardized: bool = Field(False, description="是否返回标准化值")

    @validator('ts_codes')
    def validate_ts_codes(cls, v):
        """方法描述"""
            raise ValueError('股票代码列表不能为空')
        for code in v:
            if not code or len(code) != 9 or '.' not in code:
                raise ValueError(f'无效的股票代码格式: {code}')
        return v

class FactorCalculationRequest(BaseModel):
    """因子计算请求"""
    ts_codes: Optional[List[str]] = Field(None, description="股票代码列表")
    factor_names: Optional[List[str]] = Field(None, description="因子名称列表")
    factor_types: List[FactorType] = Field([FactorType.ALL], description="因子类型列表")
    calculation_date: Optional[date] = Field(None, description="计算日期")
    start_date: Optional[date] = Field(None, description="开始日期")
    end_date: Optional[date] = Field(None, description="结束日期")
    force_recalculate: bool = Field(False, description="是否强制重新计算")
    priority: str = Field("normal", description="任务优先级")

    @validator('priority')
    def validate_priority(cls, v):
        """方法描述"""
            raise ValueError('优先级必须是: low, normal, high, urgent')
        return v

class FactorStandardizationRequest(BaseModel):
    """因子标准化请求"""
    factor_names: List[str] = Field(..., description="因子名称列表")
    method: StandardizationMethod = Field(StandardizationMethod.ZSCORE, description="标准化方法")
    calculation_date: date = Field(..., description="计算日期")
    industry_neutral: bool = Field(False, description="是否行业中性化")
    outlier_method: str = Field("clip", description="异常值处理方法")

    @validator('outlier_method')
    def validate_outlier_method(cls, v):
        """方法描述"""
            raise ValueError('异常值处理方法必须是: clip, winsorize, remove')
        return v

class FactorEffectivenessRequest(BaseModel):
    """因子有效性分析请求"""
    factor_names: List[str] = Field(..., description="因子名称列表")
    start_date: date = Field(..., description="开始日期")
    end_date: date = Field(..., description="结束日期")
    return_periods: List[int] = Field([1, 5, 20], description="收益率周期")
    analysis_types: List[str] = Field(["ic", "ir", "layered_backtest"], description="分析类型")

# 响应模型
class APIResponse(BaseModel):
    """API统一响应格式"""
    success: bool = Field(..., description="请求是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")
    message: str = Field("", description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")

class FactorData(BaseModel):
    """因子数据"""
    ts_code: str = Field(..., description="股票代码")
    factor_date: date = Field(..., description="因子日期")
    factor_values: Dict[str, Optional[float]] = Field(..., description="因子值字典")

class CalculationTask(BaseModel):
    """计算任务信息"""
    task_id: str = Field(..., description="任务ID")
    status: CalculationStatus = Field(..., description="任务状态")
    created_at: datetime = Field(..., description="创建时间")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    progress: float = Field(0.0, description="进度百分比")
    message: str = Field("", description="状态消息")

# API端点实现
@router.get("/", response_model=APIResponse, summary="获取因子数据")
@api_exception_handler("获取因子数据")
@log_api_call("获取因子数据")
async def get_factors(
    request: FactorQueryRequest = Depends(),
    user: dict = Depends(verify_token),
    factor_service: FactorService = Depends(get_factor_service),
    _: bool = Depends(require_healthy_database)
):
    """
    获取因子数据

    支持多种查询条件：
    - 按股票代码查询
    - 按因子名称查询
    - 按因子类型查询
    - 按日期范围查询
    - 支持标准化数据查询
    """
    # 转换因子类型枚举为字符串
    factor_types = None
    if request.factor_types:
        factor_types = [ft.value for ft in request.factor_types]

    # 调用服务层
    result_data = factor_service.query_factors(
        ts_codes=request.ts_codes,
        factor_names=request.factor_names,
        factor_types=factor_types,
        start_date=request.start_date,
        end_date=request.end_date,
        standardized=request.standardized
    )

    return ResponseBuilder.success(
        data=result_data,
        message=f"成功获取{result_data['total_count']}条因子数据"
    )

@router.post("/calculate", response_model=APIResponse, summary="触发因子计算")
@api_exception_handler("触发因子计算")
@log_api_call("触发因子计算")
async def calculate_factors(
    request: FactorCalculationRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_permission("write")),
    calculation_service: FactorCalculationService = Depends(get_calculation_service)
):
    """
    触发因子计算任务

    支持多种计算模式：
    - 全市场计算
    - 指定股票计算
    - 指定因子计算
    - 指定日期范围计算
    """
    # 准备请求数据
    request_data = {
        'ts_codes': request.ts_codes,
        'factor_names': request.factor_names,
        'factor_types': [ft.value for ft in request.factor_types],
        'calculation_date': request.calculation_date,
        'start_date': request.start_date,
        'end_date': request.end_date,
        'force_recalculate': request.force_recalculate,
        'priority': request.priority
    }

    # 提交计算任务
    task_id = calculation_service.submit_calculation_request(request_data)

    # 添加后台任务监控
    background_tasks.add_task(monitor_calculation_task, task_id)

    return ResponseBuilder.success(
        data={
            "task_id": task_id,
            "estimated_duration": "5-30分钟",
            "status_url": f"/api/v1/factors/tasks/{task_id}"
        },
        message="因子计算任务已提交"
    )

@router.get("/tasks/{task_id}", response_model=APIResponse, summary="查询计算任务状态")
@api_exception_handler("查询任务状态")
async def get_calculation_task(
    task_id: str,
    user: dict = Depends(verify_token),
    calculation_service: FactorCalculationService = Depends(get_calculation_service)
):
    """查询因子计算任务状态"""
    task_info = calculation_service.get_task_status(task_id)

    if not task_info:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = CalculationTask(
        task_id=task_id,
        status=CalculationStatus(task_info['status']),
        created_at=task_info['created_at'],
        started_at=task_info.get('started_at'),
        completed_at=task_info.get('completed_at'),
        progress=task_info.get('progress', 0.0),
        message=task_info.get('message', '')
    )

    return ResponseBuilder.success(
        data=task.dict(),
        message="任务状态查询成功"
    )

@router.post("/standardize", response_model=APIResponse, summary="因子标准化")
async def standardize_factors(
    request: FactorStandardizationRequest,
    user: dict = Depends(verify_token)
):
    """
    对因子进行标准化处理

    支持多种标准化方法：
    - Z-score标准化
    - 分位数标准化
    - 排名标准化
    - 最小最大值标准化
    - 鲁棒标准化
    """
    try:
        engine = get_db_engine()
        standardizer = FactorStandardizer(engine)

        # 执行标准化
        result = standardizer.standardize_factors(
            factor_names=request.factor_names,
            calculation_date=request.calculation_date,
            method=request.method.value,
            industry_neutral=request.industry_neutral,
            outlier_method=request.outlier_method
        )

        return APIResponse(
            success=True,
            data={
                "standardized_count": len(result),
                "method": request.method.value,
                "calculation_date": request.calculation_date.isoformat(),
                "industry_neutral": request.industry_neutral
            },
            message=f"成功标准化{len(result)}个因子"
        )

    except Exception as e:
        logger.error(f"因子标准化失败: {e}")
        raise HTTPException(status_code=500, detail=f"因子标准化失败: {str(e)}")

@router.post("/effectiveness", response_model=APIResponse, summary="因子有效性分析")
async def analyze_factor_effectiveness(
    request: FactorEffectivenessRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(verify_token)
):
    """
    分析因子有效性

    支持多种分析类型：
    - IC分析（信息系数）
    - IR分析（信息比率）
    - 分层回测分析
    - 因子衰减分析
    """
    try:
        engine = get_db_engine()
        analyzer = FactorEffectivenessAnalyzer(engine)

        # 创建分析任务ID
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 添加后台分析任务
        background_tasks.add_task(
            run_effectiveness_analysis,
            analyzer,
            request,
            analysis_id
        )

        return APIResponse(
            success=True,
            data={
                "analysis_id": analysis_id,
                "factor_count": len(request.factor_names),
                "analysis_types": request.analysis_types,
                "date_range": f"{request.start_date} - {request.end_date}",
                "status_url": f"/api/v1/factors/analysis/{analysis_id}"
            },
            message="因子有效性分析任务已提交"
        )

    except Exception as e:
        logger.error(f"提交有效性分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"提交分析任务失败: {str(e)}")

@router.get("/analysis/{analysis_id}", response_model=APIResponse, summary="查询分析结果")
async def get_analysis_result(
    analysis_id: str,
    user: dict = Depends(verify_token)
):
    """查询因子有效性分析结果"""
    try:
        # TODO: 实现分析结果查询逻辑
        # 这里应该从数据库或缓存中查询分析结果

        return APIResponse(
            success=True,
            data={
                "analysis_id": analysis_id,
                "status": "completed",
                "results": {
                    "ic_analysis": {},
                    "ir_analysis": {},
                    "layered_backtest": {}
                }
            },
            message="分析结果查询成功"
        )

    except Exception as e:
        logger.error(f"查询分析结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询分析结果失败: {str(e)}")

@router.get("/metadata", response_model=APIResponse, summary="获取因子元数据")
async def get_factor_metadata(
    factor_type: Optional[FactorType] = Query(None, description="因子类型"),
    user: dict = Depends(verify_token)
):
    """
    获取因子元数据信息

    包括：
    - 可用因子列表
    - 因子定义和计算公式
    - 因子分类和标签
    - 数据覆盖情况
    """
    try:
        factor_engine = get_factor_engine()
        metadata = factor_engine.get_factor_metadata(factor_type.value if factor_type else None)

        return APIResponse(
            success=True,
            data=metadata,
            message="因子元数据获取成功"
        )

    except Exception as e:
        logger.error(f"获取因子元数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取元数据失败: {str(e)}")

@router.get("/health", response_model=APIResponse, summary="健康检查")
async def health_check():
    """API健康检查"""
    try:
        engine = get_db_engine()

        # 检查数据库连接
        with engine.connect() as conn:
            conn.execute("SELECT 1")

        return APIResponse(
            success=True,
            data={
                "status": "healthy",
                "database": "connected",
                "timestamp": datetime.now().isoformat()
            },
            message="服务运行正常"
        )

    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return APIResponse(
            success=False,
            data={
                "status": "unhealthy",
                "error": str(e)
            },
            message="服务异常"
        )

# 后台任务函数
async def monitor_calculation_task(task_id: str):
    """监控计算任务进度"""
    try:
        manual_trigger = get_manual_trigger()

        while True:
            task_info = manual_trigger.get_task_status(task_id)
            if not task_info:
                break

            status = task_info.get('status')
            if status in ['completed', 'failed']:
                logger.info(f"任务 {task_id} 已完成，状态: {status}")
                break

            await asyncio.sleep(30)  # 每30秒检查一次

    except Exception as e:
        logger.error(f"监控任务 {task_id} 失败: {e}")

async def run_effectiveness_analysis(
    analyzer: FactorEffectivenessAnalyzer,
    request: FactorEffectivenessRequest,
    analysis_id: str
):
    """运行因子有效性分析"""
    try:
        results = {}

        for analysis_type in request.analysis_types:
            if analysis_type == "ic":
                ic_results = analyzer.calculate_ic(
                    factor_names=request.factor_names,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    return_periods=request.return_periods
                )
                results["ic_analysis"] = ic_results

            elif analysis_type == "ir":
                ir_results = analyzer.calculate_ir(
                    factor_names=request.factor_names,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    return_periods=request.return_periods
                )
                results["ir_analysis"] = ir_results

            elif analysis_type == "layered_backtest":
                backtest_results = analyzer.layered_backtest(
                    factor_names=request.factor_names,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    layers=5
                )
                results["layered_backtest"] = backtest_results

        # TODO: 将分析结果保存到数据库或缓存
        logger.info(f"分析任务 {analysis_id} 完成")

    except Exception as e:
        logger.error(f"分析任务 {analysis_id} 失败: {e}")