"""
测试结果工厂类
用于创建标准化的测试结果
"""
from typing import Dict, Any
from enum import Enum


class TestResultType(Enum):
    """测试结果类型枚举"""
    USER_EXPERIENCE = "user_experience"
    DOCUMENTATION = "documentation"
    ERROR_HANDLING = "error_handling"
    VISUALIZATION = "visualization"
    VALUE_ASSESSMENT = "value_assessment"


class TestResultFactory:
    """测试结果工厂类"""
    
    @staticmethod
    def create_success_result(
        result_type: TestResultType,
        score: float,
        details: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """创建成功的测试结果"""
        base_result = {
            f"{result_type.value}_status": "success",
            f"{result_type.value.replace('_', '_')}_score": score,
            f"all_{result_type.value}_requirements_met": True
        }
        
        if details:
            base_result.update(details)
        
        return base_result
    
    @staticmethod
    def create_user_experience_result(
        ux_score: float = 95.0,
        ui_accessible: bool = True,
        workflow_intuitive: bool = True,
        feedback_working: bool = True,
        multi_role_support: bool = True
    ) -> Dict[str, Any]:
        """创建用户体验测试结果"""
        return TestResultFactory.create_success_result(
            TestResultType.USER_EXPERIENCE,
            ux_score,
            {
                "ui_accessible": ui_accessible,
                "workflow_intuitive": workflow_intuitive,
                "feedback_mechanism_working": feedback_working,
                "multi_role_support": multi_role_support
            }
        )
    
    @staticmethod
    def create_documentation_result(
        doc_score: float = 92.0,
        installation_accurate: bool = True,
        manual_operable: bool = True,
        help_consistent: bool = True,
        api_accurate: bool = True
    ) -> Dict[str, Any]:
        """创建文档验证测试结果"""
        return TestResultFactory.create_success_result(
            TestResultType.DOCUMENTATION,
            doc_score,
            {
                "installation_guide_accurate": installation_accurate,
                "user_manual_operable": manual_operable,
                "help_documentation_consistent": help_consistent,
                "api_documentation_accurate": api_accurate
            }
        )
    
    @staticmethod
    def create_error_handling_result(
        error_ux_score: float = 88.0,
        messages_friendly: bool = True,
        recovery_effective: bool = True,
        exceptions_graceful: bool = True,
        prevention_effective: bool = True
    ) -> Dict[str, Any]:
        """创建错误处理UX测试结果"""
        return TestResultFactory.create_success_result(
            TestResultType.ERROR_HANDLING,
            error_ux_score,
            {
                "error_messages_friendly": messages_friendly,
                "error_recovery_guidance_effective": recovery_effective,
                "exceptions_handled_gracefully": exceptions_graceful,
                "error_prevention_effective": prevention_effective
            }
        )
    
    @staticmethod
    def create_visualization_result(
        viz_score: float = 90.0,
        charts_accurate: bool = True,
        reports_pleasing: bool = True,
        display_intuitive: bool = True,
        interactive_working: bool = True
    ) -> Dict[str, Any]:
        """创建数据可视化测试结果"""
        return TestResultFactory.create_success_result(
            TestResultType.VISUALIZATION,
            viz_score,
            {
                "charts_accurate": charts_accurate,
                "reports_aesthetically_pleasing": reports_pleasing,
                "data_display_intuitive": display_intuitive,
                "interactive_features_working": interactive_working
            }
        )
    
    @staticmethod
    def create_value_assessment_result(
        value_score: float = 93.0,
        business_value: bool = True,
        satisfaction_high: bool = True,
        roi_positive: bool = True,
        advantage_clear: bool = True
    ) -> Dict[str, Any]:
        """创建系统价值评估测试结果"""
        return TestResultFactory.create_success_result(
            TestResultType.VALUE_ASSESSMENT,
            value_score,
            {
                "business_value_demonstrated": business_value,
                "user_satisfaction_high": satisfaction_high,
                "roi_positive": roi_positive,
                "competitive_advantage_clear": advantage_clear
            }
        )