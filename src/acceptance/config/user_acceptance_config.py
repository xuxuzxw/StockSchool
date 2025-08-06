"""
用户验收测试配置管理
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path


@dataclass
class UserAcceptanceConfig:
    """用户验收测试配置类"""
    
    # 测试用户配置
    test_users: List[str] = field(default_factory=lambda: ['analyst', 'trader', 'manager'])
    
    # 功能开关
    ui_test_enabled: bool = True
    doc_validation_enabled: bool = True
    
    # API限流配置
    api_rate_limit: float = 0.5  # 500ms间隔
    
    # 测试数据配置
    test_stocks_limited: List[str] = field(default_factory=lambda: ['000001.SZ', '000002.SZ'])
    test_date_range_limited: Dict[str, str] = field(default_factory=lambda: {
        'start': '2024-01-01',
        'end': '2024-01-05'
    })
    
    # 测试环境配置
    test_environment: str = 'acceptance'
    
    # 期望的测试结果阈值
    expected_scores: Dict[str, float] = field(default_factory=lambda: {
        'ux_score_min': 85.0,
        'doc_score_min': 90.0,
        'error_ux_score_min': 80.0,
        'viz_score_min': 85.0,
        'value_score_min': 90.0
    })
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UserAcceptanceConfig':
        """从字典创建配置实例"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
    
    def validate(self) -> bool:
        """验证配置的有效性"""
        if not self.test_users:
            raise ValueError("测试用户列表不能为空")
        
        if self.api_rate_limit <= 0:
            raise ValueError("API限流间隔必须大于0")
        
        if not self.test_stocks_limited:
            raise ValueError("测试股票列表不能为空")
        
        return True