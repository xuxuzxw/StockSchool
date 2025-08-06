"""
测试验证混入类
提供通用的验证方法
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging


class TestValidationMixin(ABC):
    """测试验证混入类"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_test_result(self, result: Dict[str, Any], 
                           required_keys: List[str]) -> bool:
        """验证测试结果的完整性"""
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            self.logger.warning(f"测试结果缺少必要键: {missing_keys}")
            return False
        return True
    
    def check_score_threshold(self, score: float, threshold: float, 
                            test_name: str) -> bool:
        """检查分数是否达到阈值"""
        if score < threshold:
            self.logger.warning(
                f"{test_name} 分数 {score} 低于阈值 {threshold}"
            )
            return False
        return True
    
    @abstractmethod
    def _get_expected_score_threshold(self, test_type: str) -> float:
        """获取期望的分数阈值"""
        pass
    
    def validate_boolean_checks(self, checks: Dict[str, bool]) -> bool:
        """验证布尔检查项"""
        failed_checks = [name for name, result in checks.items() if not result]
        if failed_checks:
            self.logger.warning(f"以下检查项失败: {failed_checks}")
            return False
        return True