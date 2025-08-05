import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.rate_limit_config import RateLimitConfig, RateLimitRule
from src.api.rate_limiter import APIRateLimiter, RateLimitError, UserRateLimiter

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API限流测试
"""


class TestUserRateLimiter:
    """测试用户限流器"""

    def setup_method(self):
        """设置测试环境"""
        self.rate_limiter = UserRateLimiter(window_seconds=60, max_requests=5)

    def test_allow_request_within_limit(self):
        """测试在限制内允许请求"""
        user_id = "test_user"

        # 前5次请求应该被允许
        for i in range(5):
            assert self.rate_limiter.allow_request(user_id) is True

    def test_block_request_exceeding_limit(self):
        """测试超出限制时阻止请求"""
        user_id = "test_user"

        # 发送5次请求
        for i in range(5):
            self.rate_limiter.allow_request(user_id)

        # 第6次请求应该被拒绝
        assert self.rate_limiter.allow_request(user_id) is False

    def test_window_sliding(self):
        """测试滑动窗口机制"""
        user_id = "test_user"

        # 发送5次请求
        for i in range(5):
            self.rate_limiter.allow_request(user_id)

        # 等待窗口滑动
        time.sleep(61)

        # 现在应该允许新的请求
        assert self.rate_limiter.allow_request(user_id) is True

    def test_get_retry_after(self):
        """测试获取重试等待时间"""
        user_id = "test_user"

        # 填满限制
        for i in range(5):
            self.rate_limiter.allow_request(user_id)

        # 获取重试等待时间
        retry_after = self.rate_limiter.get_retry_after(user_id)
        assert 0 <= retry_after <= 60


class TestAPIRateLimiter:
    """测试API限流管理器"""

    def setup_method(self):
        """设置测试环境"""
        self.api_limiter = APIRateLimiter()

    def test_check_rate_limit(self):
        """测试检查限流"""
        user_id = "test_user"
        client_ip = "192.168.1.1"

        # 初始应该允许
        assert self.api_limiter.check_rate_limit(user_id, client_ip) is True

    def test_get_user_id_from_request(self):
        """测试从请求获取用户ID"""
        # 模拟FastAPI请求
        mock_request = AsyncMock()
        mock_request.headers = {"Authorization": "Bearer test_token"}

        # 测试提取用户ID
        user_id = self.api_limiter.get_user_id_from_request(mock_request)
        assert user_id == "test_token"

    def test_get_user_id_from_ip(self):
        """测试从IP获取用户ID"""
        mock_request = AsyncMock()
        mock_request.headers = {}
        mock_request.client.host = "192.168.1.1"

        user_id = self.api_limiter.get_user_id_from_request(mock_request)
        assert user_id == "ip_192.168.1.1"


class TestRateLimitConfig:
    """测试限流配置"""

    def test_rate_limit_rule_creation(self):
        """测试限流规则创建"""
        rule = RateLimitRule(requests_per_minute=60, requests_per_hour=1000, burst_limit=100)

        assert rule.requests_per_minute == 60
        assert rule.requests_per_hour == 1000
        assert rule.burst_limit == 100

    def test_rate_limit_config_loading(self):
        """测试配置加载"""
        config = RateLimitConfig()

        # 测试角色限制
        guest_limits = config.get_role_limits("guest")
        assert guest_limits is not None

        # 测试白名单检查
        assert config.is_whitelisted("127.0.0.1", "ip") is True
        assert config.is_whitelisted("unknown_ip", "ip") is False


class TestRateLimitIntegration:
    """测试限流集成"""

    def setup_method(self):
        """设置测试环境"""
        self.app = FastAPI()

        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        self.client = TestClient(self.app)

    def test_rate_limit_headers(self):
        """测试限流响应头"""
        response = self.client.get("/test")

        # 检查响应头
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time" in response.headers

    def test_rate_limit_error_response(self):
        """测试限流错误响应"""
        # 模拟限流触发
        with patch("src.api.rate_limiter.api_rate_limiter.check_rate_limit", return_value=False):
            with patch("src.api.rate_limiter.api_rate_limiter.get_retry_after", return_value=60):
                response = self.client.get("/test")

                assert response.status_code == 429
                assert response.json()["error_code"] == "RATE_LIMIT_EXCEEDED"
                assert response.headers["Retry-After"] == "60"


@pytest.mark.asyncio
async def test_concurrent_rate_limit():
    """测试并发限流"""
    rate_limiter = UserRateLimiter(window_seconds=60, max_requests=10)

    user_id = "concurrent_user"

    # 并发发送请求
    tasks = []
    for i in range(15):
        task = asyncio.create_task(asyncio.to_thread(rate_limiter.allow_request, user_id))
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    # 验证结果
    allowed_count = sum(1 for r in results if r is True)
    blocked_count = sum(1 for r in results if r is False)

    assert allowed_count <= 10
    assert blocked_count >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
