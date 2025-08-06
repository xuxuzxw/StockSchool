"""
验收测试核心数据模型
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class TestStatus(Enum):
    """测试状态枚举"""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PhaseType(Enum):
    """测试阶段类型枚举"""
    INFRASTRUCTURE = "infrastructure"
    DATA_SERVICE = "data_service"
    COMPUTE_ENGINE = "compute_engine"
    AI_SERVICE = "ai_service"
    EXTERNAL_AI_ANALYSIS = "external_ai_analysis"
    API_SERVICE = "api_service"
    MONITORING = "monitoring"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    USER_ACCEPTANCE = "user_acceptance"
    CODE_QUALITY = "code_quality"
    SECURITY = "security"


@dataclass
class TestResult:
    """测试结果数据模型"""
    phase: str
    test_name: str
    status: TestStatus
    execution_time: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'phase': self.phase,
            'test_name': self.test_name,
            'status': self.status.value,
            'execution_time': self.execution_time,
            'error_message': self.error_message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class AcceptanceReport:
    """验收报告数据模型"""
    test_session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    overall_result: bool = False
    phase_results: List[TestResult] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    ai_analysis_metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'test_session_id': self.test_session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'skipped_tests': self.skipped_tests,
            'overall_result': self.overall_result,
            'phase_results': [result.to_dict() for result in self.phase_results],
            'performance_metrics': self.performance_metrics,
            'ai_analysis_metrics': self.ai_analysis_metrics,
            'recommendations': self.recommendations
        }