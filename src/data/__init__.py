#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据模块

提供数据获取、同步、处理和质量验证功能

作者: StockSchool Team
创建时间: 2025-01-03
"""

from .sync_manager import DataSyncManager
from .incremental_update import IncrementalUpdateManager
from .data_quality_monitor import DataQualityMonitor
from .factor_data_service import FactorDataService

__all__ = [
    "DataSyncManager",
    "IncrementalUpdateManager",
    "DataQualityMonitor",
    "FactorDataService"
]

__version__ = "1.0.0"
__author__ = "StockSchool Team"
__description__ = "数据获取和处理模块"