"""
测试阶段工厂 - 使用工厂模式和注册表模式
"""

import logging
from typing import Dict, Type, List, Optional, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .models import TestResult


class TestPhase(Protocol):
    """测试阶段协议"""
    phase_name: str
    
    def execute(self) -> List[TestResult]:
        """执行测试阶段"""
        ...


@dataclass
class PhaseConfig:
    """阶段配置"""
    module_path: str
    class_name: str
    phase_name: str
    enabled: bool = True
    dependencies: List[str] = None
    priority: int = 0


class PhaseRegistry:
    """阶段注册表 - 单例模式"""
    
    _instance = None
    _phases: Dict[str, PhaseConfig] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register_phase(self, phase_id: str, config: PhaseConfig) -> None:
        """注册测试阶段"""
        self._phases[phase_id] = config
    
    def get_phase_config(self, phase_id: str) -> Optional[PhaseConfig]:
        """获取阶段配置"""
        return self._phases.get(phase_id)
    
    def get_all_phases(self) -> Dict[str, PhaseConfig]:
        """获取所有阶段配置"""
        return self._phases.copy()
    
    def get_enabled_phases(self) -> Dict[str, PhaseConfig]:
        """获取启用的阶段配置"""
        return {
            phase_id: config 
            for phase_id, config in self._phases.items() 
            if config.enabled
        }


class PhaseFactory:
    """测试阶段工厂"""
    
    def __init__(self, registry: PhaseRegistry = None):
        self.registry = registry or PhaseRegistry()
        self.logger = logging.getLogger(__name__)
        self._phase_cache: Dict[str, TestPhase] = {}
    
    def create_phase(self, phase_id: str, config: Dict) -> Optional[TestPhase]:
        """
        创建测试阶段实例
        
        Args:
            phase_id: 阶段ID
            config: 配置字典
            
        Returns:
            TestPhase实例或None
        """
        # 检查缓存
        if phase_id in self._phase_cache:
            return self._phase_cache[phase_id]
        
        phase_config = self.registry.get_phase_config(phase_id)
        if not phase_config:
            self.logger.warning(f"未找到阶段配置: {phase_id}")
            return None
        
        if not phase_config.enabled:
            self.logger.info(f"阶段已禁用: {phase_id}")
            return None
        
        try:
            # 动态导入模块
            module = __import__(
                phase_config.module_path, 
                fromlist=[phase_config.class_name]
            )
            
            # 获取类
            phase_class = getattr(module, phase_config.class_name)
            
            # 创建实例
            phase_instance = phase_class(
                phase_name=phase_config.phase_name,
                config=config
            )
            
            # 缓存实例
            self._phase_cache[phase_id] = phase_instance
            
            self.logger.info(f"成功创建阶段: {phase_id}")
            return phase_instance
            
        except ImportError as e:
            self.logger.warning(f"无法导入阶段 {phase_id}: {e}")
            return None
        except AttributeError as e:
            self.logger.warning(f"无法找到阶段类 {phase_id}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"创建阶段 {phase_id} 失败: {e}")
            return None
    
    def create_all_phases(self, config: Dict) -> List[TestPhase]:
        """
        创建所有启用的测试阶段
        
        Args:
            config: 配置字典
            
        Returns:
            测试阶段列表
        """
        phases = []
        enabled_phases = self.registry.get_enabled_phases()
        
        # 按优先级排序
        sorted_phases = sorted(
            enabled_phases.items(),
            key=lambda x: x[1].priority
        )
        
        for phase_id, phase_config in sorted_phases:
            phase = self.create_phase(phase_id, config)
            if phase:
                phases.append(phase)
        
        self.logger.info(f"成功创建 {len(phases)} 个测试阶段")
        return phases
    
    def clear_cache(self) -> None:
        """清理缓存"""
        self._phase_cache.clear()


def register_default_phases():
    """注册默认的测试阶段"""
    registry = PhaseRegistry()
    
    # 基础设施验收阶段
    registry.register_phase("infrastructure", PhaseConfig(
        module_path="src.acceptance.phases.infrastructure",
        class_name="InfrastructurePhase",
        phase_name="基础设施验收",
        priority=1
    ))
    
    # 数据服务验收阶段
    registry.register_phase("data_service", PhaseConfig(
        module_path="src.acceptance.phases.data_service",
        class_name="DataServicePhase",
        phase_name="数据服务验收",
        priority=2
    ))
    
    # 计算引擎验收阶段
    registry.register_phase("compute_engine", PhaseConfig(
        module_path="src.acceptance.phases.compute_engine",
        class_name="ComputeEnginePhase",
        phase_name="计算引擎验收",
        priority=3
    ))
    
    # AI服务验收阶段
    registry.register_phase("ai_service", PhaseConfig(
        module_path="src.acceptance.phases.ai_service",
        class_name="AIServicePhase",
        phase_name="AI服务验收",
        priority=4
    ))
    
    # 外接AI分析验收阶段
    registry.register_phase("external_ai_analysis", PhaseConfig(
        module_path="src.acceptance.phases.external_ai_analysis",
        class_name="ExternalAIAnalysisPhase",
        phase_name="外接AI分析验收",
        priority=5
    ))
    
    # API服务验收阶段
    registry.register_phase("api_service", PhaseConfig(
        module_path="src.acceptance.phases.api_service",
        class_name="APIServicePhase",
        phase_name="API服务验收",
        priority=6
    ))
    
    # 监控系统验收阶段
    registry.register_phase("monitoring", PhaseConfig(
        module_path="src.acceptance.phases.monitoring",
        class_name="MonitoringPhase",
        phase_name="监控系统验收",
        priority=7
    ))
    
    # 性能基准验收阶段
    registry.register_phase("performance", PhaseConfig(
        module_path="src.acceptance.phases.performance",
        class_name="PerformancePhase",
        phase_name="性能基准验收",
        priority=8
    ))
    
    # 集成测试验收阶段
    registry.register_phase("integration", PhaseConfig(
        module_path="src.acceptance.phases.integration",
        class_name="IntegrationPhase",
        phase_name="集成测试验收",
        priority=9
    ))
    
    # 用户验收测试阶段
    registry.register_phase("user_acceptance", PhaseConfig(
        module_path="src.acceptance.phases.user_acceptance",
        class_name="UserAcceptancePhase",
        phase_name="用户验收测试",
        priority=10
    ))
    
    # 代码质量验收阶段
    registry.register_phase("code_quality", PhaseConfig(
        module_path="src.acceptance.phases.code_quality",
        class_name="CodeQualityPhase",
        phase_name="代码质量验收",
        priority=11
    ))
    
    # 安全性验收阶段
    registry.register_phase("security", PhaseConfig(
        module_path="src.acceptance.phases.security",
        class_name="SecurityPhase",
        phase_name="安全性验收",
        priority=12
    ))


# 初始化默认阶段
register_default_phases()