#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子API测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.api.main import app
from src.api.factor_api import (
    FactorQueryRequest, FactorCalculationRequest, FactorStandardizationRequest,
    FactorEffectivenessRequest, FactorType, CalculationStatus, StandardizationMethod
)
from src.api.auth import auth_manager

# 创建测试客户端
client = TestClient(app)

class TestFactorAPI:
    """因子API测试类"""
    
    @pytest.fixture
    def mock_auth_token(self):
        """模拟认证token"""
        return "test_token_123"
    
    @pytest.fixture
    def auth_headers(self, mock_auth_token):
        """认证头"""
        return {"Authorization": f"Bearer {mock_auth_token}"}
    
    @pytest.fixture
    def mock_factor_data(self):
        """模拟因子数据"""
        return pd.DataFrame({
            'ts_code': ['000001.SZ', '000001.SZ', '000002.SZ', '000002.SZ'],
            'factor_date': [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 2)],
            'sma_5': [10.5, 10.8, 15.2, 15.5],
            'rsi_14': [45.2, 48.1, 52.3, 50.8],
            'pe_ttm': [12.5, 12.3, 18.7, 18.9]
        })
    
    def test_get_factors_success(self, auth_headers, mock_factor_data):
        """测试成功获取因子数据"""
        with patch('src.api.factor_api.verify_token') as mock_verify:
            mock_verify.return_value = {"user_id": "test_user", "permissions": ["factor:read"]}
            
            with patch('src.api.factor_api.get_factor_engine') as mock_engine:
                mock_engine_instance = Mock()
                mock_engine_instance.get_factors.return_value = mock_factor_data
                mock_engine.return_value = mock_engine_instance
                
                response = client.get(
                    "/api/v1/factors",
                    headers=auth_headers,
                    json={
                        "ts_codes": ["000001.SZ", "000002.SZ"],
                        "factor_names": ["sma_5", "rsi_14"],
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-02"
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["data"]["total_count"] == 4
                assert len(data["data"]["factors"]) == 4
    
    def test_get_factors_unauthorized(self):
        """测试未授权访问"""
        response = client.get(
            "/api/v1/factors",
            json={
                "ts_codes": ["000001.SZ"],
                "factor_names": ["sma_5"]
            }
        )
        
        assert response.status_code == 403  # FastAPI会返回403而不是401
    
    def test_get_factors_invalid_params(self, auth_headers):
        """测试无效参数"""
        with patch('src.api.factor_api.verify_token') as mock_verify:
            mock_verify.return_value = {"user_id": "test_user", "permissions": ["factor:read"]}
            
            response = client.get(
                "/api/v1/factors",
                headers=auth_headers,
                json={
                    "ts_codes": [],  # 空的股票代码列表
                    "factor_names": ["sma_5"]
                }
            )
            
            assert response.status_code == 422  # 参数验证失败
    
    def test_calculate_factors_success(self, auth_headers):
        """测试成功触发因子计算"""
        with patch('src.api.factor_api.verify_token') as mock_verify:
            mock_verify.return_value = {"user_id": "test_user", "permissions": ["factor:calculate"]}
            
            with patch('src.api.factor_api.get_manual_trigger') as mock_trigger:
                mock_trigger_instance = Mock()
                mock_trigger_instance.submit_calculation_request.return_value = "task_123"
                mock_trigger.return_value = mock_trigger_instance
                
                response = client.post(
                    "/api/v1/factors/calculate",
                    headers=auth_headers,
                    json={
                        "ts_codes": ["000001.SZ"],
                        "factor_types": ["technical"],
                        "calculation_date": "2024-01-31",
                        "priority": "high"
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["data"]["task_id"] == "task_123"
                assert "status_url" in data["data"]
    
    def test_get_calculation_task_status(self, auth_headers):
        """测试查询计算任务状态"""
        with patch('src.api.factor_api.verify_token') as mock_verify:
            mock_verify.return_value = {"user_id": "test_user", "permissions": ["factor:read"]}
            
            with patch('src.api.factor_api.get_manual_trigger') as mock_trigger:
                mock_trigger_instance = Mock()
                mock_trigger_instance.get_task_status.return_value = {
                    'status': 'running',
                    'created_at': datetime.now(),
                    'started_at': datetime.now(),
                    'progress': 65.5,
                    'message': '正在计算技术面因子...'
                }
                mock_trigger.return_value = mock_trigger_instance
                
                response = client.get(
                    "/api/v1/factors/tasks/task_123",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["data"]["status"] == "running"
                assert data["data"]["progress"] == 65.5
    
    def test_get_task_not_found(self, auth_headers):
        """测试查询不存在的任务"""
        with patch('src.api.factor_api.verify_token') as mock_verify:
            mock_verify.return_value = {"user_id": "test_user", "permissions": ["factor:read"]}
            
            with patch('src.api.factor_api.get_manual_trigger') as mock_trigger:
                mock_trigger_instance = Mock()
                mock_trigger_instance.get_task_status.return_value = None
                mock_trigger.return_value = mock_trigger_instance
                
                response = client.get(
                    "/api/v1/factors/tasks/nonexistent_task",
                    headers=auth_headers
                )
                
                assert response.status_code == 404
    
    def test_standardize_factors_success(self, auth_headers):
        """测试成功标准化因子"""
        with patch('src.api.factor_api.verify_token') as mock_verify:
            mock_verify.return_value = {"user_id": "test_user", "permissions": ["factor:write"]}
            
            with patch('src.api.factor_api.FactorStandardizer') as mock_standardizer:
                mock_standardizer_instance = Mock()
                mock_standardizer_instance.standardize_factors.return_value = ["sma_5", "rsi_14"]
                mock_standardizer.return_value = mock_standardizer_instance
                
                response = client.post(
                    "/api/v1/factors/standardize",
                    headers=auth_headers,
                    json={
                        "factor_names": ["sma_5", "rsi_14"],
                        "method": "zscore",
                        "calculation_date": "2024-01-31",
                        "industry_neutral": False,
                        "outlier_method": "clip"
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "成功标准化2个因子" in data["message"]
    
    def test_analyze_factor_effectiveness_success(self, auth_headers):
        """测试成功提交因子有效性分析"""
        with patch('src.api.factor_api.verify_token') as mock_verify:
            mock_verify.return_value = {"user_id": "test_user", "permissions": ["factor:read"]}
            
            response = client.post(
                "/api/v1/factors/effectiveness",
                headers=auth_headers,
                json={
                    "factor_names": ["sma_5", "rsi_14"],
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "return_periods": [1, 5, 20],
                    "analysis_types": ["ic", "ir", "layered_backtest"]
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "analysis_id" in data["data"]
            assert data["data"]["factor_count"] == 2
    
    def test_get_factor_metadata_success(self, auth_headers):
        """测试成功获取因子元数据"""
        with patch('src.api.factor_api.verify_token') as mock_verify:
            mock_verify.return_value = {"user_id": "test_user", "permissions": ["factor:read"]}
            
            with patch('src.api.factor_api.get_factor_engine') as mock_engine:
                mock_engine_instance = Mock()
                mock_engine_instance.get_factor_metadata.return_value = {
                    "technical_factors": ["sma_5", "sma_20", "rsi_14"],
                    "fundamental_factors": ["pe_ttm", "pb", "roe"],
                    "sentiment_factors": ["money_flow_5", "attention_score"]
                }
                mock_engine.return_value = mock_engine_instance
                
                response = client.get(
                    "/api/v1/factors/metadata",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "technical_factors" in data["data"]
    
    def test_health_check(self):
        """测试健康检查"""
        with patch('src.api.factor_api.get_db_engine') as mock_engine:
            mock_engine_instance = Mock()
            mock_conn = Mock()
            mock_engine_instance.connect.return_value.__enter__.return_value = mock_conn
            mock_engine.return_value = mock_engine_instance
            
            response = client.get("/api/v1/factors/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["status"] == "healthy"

class TestFactorQueryRequest:
    """因子查询请求模型测试"""
    
    def test_valid_request(self):
        """测试有效请求"""
        request = FactorQueryRequest(
            ts_codes=["000001.SZ", "000002.SZ"],
            factor_names=["sma_5", "rsi_14"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            standardized=True
        )
        
        assert len(request.ts_codes) == 2
        assert request.ts_codes[0] == "000001.SZ"
        assert request.standardized is True
    
    def test_invalid_ts_codes(self):
        """测试无效股票代码"""
        with pytest.raises(ValueError):
            FactorQueryRequest(
                ts_codes=["INVALID_CODE"],
                factor_names=["sma_5"]
            )
    
    def test_empty_ts_codes(self):
        """测试空股票代码列表"""
        with pytest.raises(ValueError):
            FactorQueryRequest(
                ts_codes=[],
                factor_names=["sma_5"]
            )

class TestFactorCalculationRequest:
    """因子计算请求模型测试"""
    
    def test_valid_request(self):
        """测试有效请求"""
        request = FactorCalculationRequest(
            ts_codes=["000001.SZ"],
            factor_types=[FactorType.TECHNICAL],
            calculation_date=date(2024, 1, 31),
            priority="high"
        )
        
        assert request.ts_codes == ["000001.SZ"]
        assert request.factor_types == [FactorType.TECHNICAL]
        assert request.priority == "high"
    
    def test_invalid_priority(self):
        """测试无效优先级"""
        with pytest.raises(ValueError):
            FactorCalculationRequest(
                ts_codes=["000001.SZ"],
                priority="invalid_priority"
            )

class TestFactorStandardizationRequest:
    """因子标准化请求模型测试"""
    
    def test_valid_request(self):
        """测试有效请求"""
        request = FactorStandardizationRequest(
            factor_names=["sma_5", "rsi_14"],
            method=StandardizationMethod.ZSCORE,
            calculation_date=date(2024, 1, 31),
            industry_neutral=True,
            outlier_method="clip"
        )
        
        assert len(request.factor_names) == 2
        assert request.method == StandardizationMethod.ZSCORE
        assert request.industry_neutral is True
    
    def test_invalid_outlier_method(self):
        """测试无效异常值处理方法"""
        with pytest.raises(ValueError):
            FactorStandardizationRequest(
                factor_names=["sma_5"],
                calculation_date=date(2024, 1, 31),
                outlier_method="invalid_method"
            )

class TestFactorEffectivenessRequest:
    """因子有效性分析请求模型测试"""
    
    def test_valid_request(self):
        """测试有效请求"""
        request = FactorEffectivenessRequest(
            factor_names=["sma_5", "rsi_14"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            return_periods=[1, 5, 20],
            analysis_types=["ic", "ir"]
        )
        
        assert len(request.factor_names) == 2
        assert request.return_periods == [1, 5, 20]
        assert "ic" in request.analysis_types

class TestAPIIntegration:
    """API集成测试"""
    
    @pytest.fixture
    def authenticated_client(self):
        """认证客户端"""
        with patch('src.api.auth.auth_manager.verify_token') as mock_verify:
            mock_verify.return_value = Mock(
                user_id="test_user",
                username="test_user",
                roles=["analyst"],
                permissions=["factor:read", "factor:write", "factor:calculate"]
            )
            yield client
    
    def test_complete_factor_workflow(self, authenticated_client):
        """测试完整的因子工作流"""
        headers = {"Authorization": "Bearer test_token"}
        
        # 1. 获取因子元数据
        with patch('src.api.factor_api.get_factor_engine') as mock_engine:
            mock_engine_instance = Mock()
            mock_engine_instance.get_factor_metadata.return_value = {
                "available_factors": ["sma_5", "rsi_14"]
            }
            mock_engine.return_value = mock_engine_instance
            
            response = authenticated_client.get(
                "/api/v1/factors/metadata",
                headers=headers
            )
            assert response.status_code == 200
        
        # 2. 触发因子计算
        with patch('src.api.factor_api.get_manual_trigger') as mock_trigger:
            mock_trigger_instance = Mock()
            mock_trigger_instance.submit_calculation_request.return_value = "task_123"
            mock_trigger.return_value = mock_trigger_instance
            
            response = authenticated_client.post(
                "/api/v1/factors/calculate",
                headers=headers,
                json={
                    "ts_codes": ["000001.SZ"],
                    "factor_types": ["technical"],
                    "calculation_date": "2024-01-31"
                }
            )
            assert response.status_code == 200
            task_id = response.json()["data"]["task_id"]
        
        # 3. 查询任务状态
        with patch('src.api.factor_api.get_manual_trigger') as mock_trigger:
            mock_trigger_instance = Mock()
            mock_trigger_instance.get_task_status.return_value = {
                'status': 'completed',
                'created_at': datetime.now(),
                'completed_at': datetime.now(),
                'progress': 100.0,
                'message': '计算完成'
            }
            mock_trigger.return_value = mock_trigger_instance
            
            response = authenticated_client.get(
                f"/api/v1/factors/tasks/{task_id}",
                headers=headers
            )
            assert response.status_code == 200
            assert response.json()["data"]["status"] == "completed"
        
        # 4. 获取计算结果
        with patch('src.api.factor_api.get_factor_engine') as mock_engine:
            mock_engine_instance = Mock()
            mock_factor_data = pd.DataFrame({
                'ts_code': ['000001.SZ'],
                'factor_date': [date(2024, 1, 31)],
                'sma_5': [10.5],
                'rsi_14': [45.2]
            })
            mock_engine_instance.get_factors.return_value = mock_factor_data
            mock_engine.return_value = mock_engine_instance
            
            response = authenticated_client.get(
                "/api/v1/factors",
                headers=headers,
                json={
                    "ts_codes": ["000001.SZ"],
                    "factor_names": ["sma_5", "rsi_14"],
                    "start_date": "2024-01-31",
                    "end_date": "2024-01-31"
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert data["data"]["total_count"] == 1

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])