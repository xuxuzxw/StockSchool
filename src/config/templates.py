from typing import Any, Dict

"""
配置模板和示例

提供不同环境的配置模板
"""


def get_base_config_template() -> Dict[str, Any]:
    """获取基础配置模板"""
    return {
        # 数据同步参数
        "data_sync_params": {
            "batch_size": 1000,
            "retry_times": 3,
            "retry_delay": 1,
            "max_workers": 4,
            "start_year": 2020,
            "max_days": 30,
            "sleep_interval": 0.3,
            # Tushare数据源配置
            "tushare": {"enabled": True, "api_limit": 200, "retry_times": 3, "retry_delay": 1},
            # Akshare数据源配置
            "akshare": {"enabled": True, "api_limit": 100, "retry_times": 3, "retry_delay": 2},
            # 行业数据同步参数
            "industry_batch_size": 500,
            "industry_sleep_interval": 0.3,
            # 财务数据同步参数
            "financial_data": {"start_date": "20200101", "batch_size": 100, "sleep_interval": 0.5},
            # 指标数据同步参数
            "indicator_data": {"start_date": "20200101", "batch_size": 500, "sleep_interval": 0.2},
            # 情绪数据同步参数
            "sentiment_data": {"start_date": "20200101", "sleep_interval": 1.0},
            # 资金流向数据同步参数
            "fund_flow_data": {"start_date": "20200101", "batch_size": 100, "sleep_interval": 1.0},
            # 北向资金数据同步参数
            "north_money_data": {"start_date": "20200101", "sleep_interval": 2.0},
        },
        # 因子计算参数
        "factor_params": {
            "min_data_days": 60,
            # RSI参数
            "rsi": {"window": 14},
            # 移动平均线参数
            "ma": {"windows": [5, 10, 20, 60], "short_window": 5, "long_window": 20},
            # EMA参数
            "ema": {"windows": [12, 26], "short_window": 12, "long_window": 26},
            # MACD参数
            "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            # 布林带参数
            "bollinger": {"window": 20, "num_std": 2},
            # 随机指标参数
            "stochastic": {"k_window": 14, "d_window": 3},
            # ATR参数
            "atr": {"window": 14},
            # 威廉指标参数
            "williams": {"window": 14},
            # 动量指标参数
            "momentum": {"window": 10, "period": 20},
            # ROC参数
            "roc": {"window": 12},
            # MFI参数
            "mfi": {"window": 14},
            # 成交量参数
            "volume": {"sma_windows": [5, 20]},
            # KDJ参数
            "kdj": {"k_period": 9, "d_period": 3, "j_period": 3},
        },
        # 数据质量监控参数
        "data_quality": {
            "outlier_detection": {"enabled": True, "threshold": 3, "methods": ["statistical", "logical"]},  # 3σ原则
            "missing_value_handling": {"method": "forward_fill_industry_mean", "max_fill_days": 5},
            "anomaly_alert": {
                "enabled": True,
                "webhook_url": "${ALERT_WEBHOOK_URL}",
                "alert_threshold": 0.05,  # 异常数据比例阈值
            },
            "data_completeness": {"check_interval": 3600, "min_completeness": 0.95},  # 1小时检查一次  # 最小完整性要求
        },
        # 增量更新策略
        "sync_strategy": {
            "incremental": {"enabled": True, "check_interval": 3600, "max_missing_days": 10},  # 1小时检查一次
            "full": {"enabled": True, "schedule": "0 2 * * 0", "backup_before_sync": True},  # 每周日凌晨2点全量同步
            "dependency_management": {"enabled": True, "max_parallel_tasks": 5, "task_timeout": 1800},  # 30分钟
        },
        # 监控参数
        "monitoring_params": {
            "collection_interval": 60,
            "suppression_duration": 60,
            "slow_query_limit": 20,
            "metric_retention": 100,
            "evaluation_window": 60,
            "alerts": {
                "cpu_threshold": 80.0,
                "memory_threshold": 85.0,
                "error_rate_threshold": 5.0,
                "response_time_threshold": 1000,
            },
            "performance": {"collection_interval": 60, "retention_days": 30},
        },
        # 数据库参数
        "database_params": {
            "connection_pool_size": 10,
            "max_overflow": 20,
            "pool_timeout": 30,
            "pool_recycle": 3600,
            "string_length_short": 20,
            "string_length_medium": 100,
            "default_limit": 100,
            # TimescaleDB优化
            "timescaledb": {
                "chunk_time_interval": "1 month",
                "compression_policy": "3 months",
                "retention_policy": "5 years",
            },
        },
        # API参数
        "api_params": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "reload": False,
            "log_level": "info",
            "cors_origins": ["*"],
            "default_limit": 100,
            "max_limit": 1000,
            "min_limit": 1,
            # 限流配置
            "rate_limiting": {"enabled": True, "requests_per_minute": 100, "burst_size": 20},
        },
        # 模型训练参数
        "training_params": {
            "model_name": "LightGBM",
            "prediction_window": 5,
            "test_size": 0.2,
            "lgbm_params": {
                "objective": "regression_l1",
                "metric": "rmse",
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "random_state": 42,
            },
        },
        # 特征工程参数
        "feature_params": {
            "lookback_period": 14,
            "shap_background_samples": 100,
            "interaction_samples": 100,
            "use_cuda": False,
            "shap_batch_size": 500,
            "max_cache_size": 40960,
            "chunk_size": 10000,
            "max_gpu_memory": 20480,
            "gpu_oom_retry": 3,
            "fallback_to_cpu": True,
            "cache_explanations": True,
            "cache_dir": "./cache/explanations",
            "max_cache_entries": 2000,
            "explanation_timeout": 3600,
            "shap_max_objects": 10000,
            "shap_parallel": True,
            "cuda_device": 0,
            "memory_threshold": 0.8,
            "batch_reduction_factor": 0.5,
            "max_batch_retries": 5,
            "windows_cuda_workaround": True,
            "model_explainer_type": "auto",
        },
        # 任务参数
        "task_params": {
            "timeout_minutes": 30,
            "soft_timeout_minutes": 25,
            "retry_countdown": 60,
            "max_retries": 3,
            "schedule_hour": 20,
        },
        # 策略评估参数
        "strategy_params": {
            "risk_free_rate": 0.03,
            "var_confidence_level": 0.05,
            "trading_days_per_year": 252,
            "rolling_window_days": 60,
            "max_display_items": 20,
        },
        # 质量检查参数
        "quality_params": {"sample_size": 100, "orphan_threshold": 0.05},
        # 常量定义
        "constants": {
            "trading_days_per_year": 252,
            "trading_hours_per_day": 4,
            "market_open_time": "09:30",
            "market_close_time": "15:00",
        },
        # 高级配置
        "advanced": {
            "data_clean": {"auto_clean": False, "alert_threshold": 3, "date_tolerance": 3, "enable_telemetry": True}
        },
    }


def get_development_config_template() -> Dict[str, Any]:
    """获取开发环境配置模板"""
    return {
        "api_params": {"reload": True, "log_level": "debug", "workers": 1},
        "database_params": {"connection_pool_size": 5, "max_overflow": 10},
        "monitoring_params": {"collection_interval": 30, "alerts": {"cpu_threshold": 90.0, "memory_threshold": 90.0}},
        "data_sync_params": {"batch_size": 100, "max_workers": 2, "sleep_interval": 0.5},
        "feature_params": {"use_cuda": False, "shap_batch_size": 100, "max_cache_size": 1024},
    }


def get_testing_config_template() -> Dict[str, Any]:
    """获取测试环境配置模板"""
    return {
        "api_params": {"port": 8001, "log_level": "warning"},
        "database_params": {"connection_pool_size": 3, "max_overflow": 5},
        "data_sync_params": {"batch_size": 50, "max_workers": 1, "retry_times": 1},
        "monitoring_params": {"collection_interval": 10, "metric_retention": 10},
        "feature_params": {
            "use_cuda": False,
            "shap_batch_size": 50,
            "max_cache_size": 512,
            "cache_explanations": False,
        },
    }


def get_production_config_template() -> Dict[str, Any]:
    """获取生产环境配置模板"""
    return {
        "api_params": {
            "reload": False,
            "log_level": "info",
            "workers": 4,
            "rate_limiting": {"enabled": True, "requests_per_minute": 1000, "burst_size": 100},
        },
        "database_params": {"connection_pool_size": 20, "max_overflow": 30, "pool_timeout": 30, "pool_recycle": 3600},
        "monitoring_params": {
            "collection_interval": 60,
            "alerts": {
                "cpu_threshold": 70.0,
                "memory_threshold": 80.0,
                "error_rate_threshold": 1.0,
                "response_time_threshold": 500,
            },
        },
        "data_sync_params": {"batch_size": 2000, "max_workers": 8, "retry_times": 5},
        "sync_strategy": {
            "incremental": {"check_interval": 1800},  # 30分钟检查一次
            "dependency_management": {"max_parallel_tasks": 10},
        },
        "data_quality": {"anomaly_alert": {"enabled": True, "alert_threshold": 0.01}},  # 更严格的阈值
        "feature_params": {
            "use_cuda": True,
            "shap_batch_size": 1000,
            "max_cache_size": 102400,  # 100GB
            "shap_parallel": True,
        },
    }


def get_config_template_by_environment(environment: str) -> Dict[str, Any]:
    """根据环境获取配置模板"""
    base_config = get_base_config_template()

    if environment == "development":
        env_config = get_development_config_template()
    elif environment == "testing":
        env_config = get_testing_config_template()
    elif environment == "production":
        env_config = get_production_config_template()
    else:
        return base_config

    # 合并配置
    def merge_dict(base: Dict, update: Dict):
        """方法描述"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_dict(base[key], value)
            else:
                base[key] = value

    merge_dict(base_config, env_config)
    return base_config


def create_config_files(config_dir: str = "config"):
    """创建配置文件"""
    import os
    from pathlib import Path

    import yaml

    config_path = Path(config_dir)
    config_path.mkdir(parents=True, exist_ok=True)

    # 创建基础配置文件
    base_config = get_base_config_template()
    with open(config_path / "base.yml", "w", encoding="utf-8") as f:
        yaml.dump(base_config, f, default_flow_style=False, allow_unicode=True)

    # 创建环境特定配置文件
    environments = ["development", "testing", "production"]

    for env in environments:
        if env == "development":
            env_config = get_development_config_template()
        elif env == "testing":
            env_config = get_testing_config_template()
        elif env == "production":
            env_config = get_production_config_template()

        with open(config_path / f"{env}.yml", "w", encoding="utf-8") as f:
            yaml.dump(env_config, f, default_flow_style=False, allow_unicode=True)

    print(f"配置文件已创建在 {config_dir} 目录下")


if __name__ == "__main__":
    # 创建示例配置文件
    create_config_files()
