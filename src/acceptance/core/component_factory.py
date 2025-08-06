"""
组件工厂 - 统一管理测试组件的创建
使用工厂模式根据可用性自动选择实现
"""
import importlib
from typing import Type, Any, Dict, Optional
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class ComponentFactory:
    """组件工厂类"""
    
    _component_cache: Dict[str, Any] = {}
    
    @classmethod
    def get_base_test_phase(cls) -> Type:
        """获取BaseTestPhase类"""
        return cls._get_component(
            'BaseTestPhase',
            primary_module='src.acceptance.core.base_phase',
            fallback_module='src.acceptance.core.fallback_models',
            class_name='BaseTestPhase'
        )
    
    @classmethod
    def get_test_result(cls) -> Type:
        """获取TestResult类"""
        return cls._get_component(
            'TestResult',
            primary_module='src.acceptance.core.models',
            fallback_module='src.acceptance.core.fallback_models',
            class_name='TestResult'
        )
    
    @classmethod
    def get_test_status(cls) -> Type:
        """获取TestStatus类"""
        return cls._get_component(
            'TestStatus',
            primary_module='src.acceptance.core.models',
            fallback_module='src.acceptance.core.fallback_models',
            class_name='TestStatus'
        )
    
    @classmethod
    def get_acceptance_test_error(cls) -> Type:
        """获取AcceptanceTestError类"""
        return cls._get_component(
            'AcceptanceTestError',
            primary_module='src.acceptance.core.exceptions',
            fallback_module='src.acceptance.core.fallback_models',
            class_name='AcceptanceTestError'
        )
    
    @classmethod
    def _get_component(cls, cache_key: str, primary_module: str, 
                      fallback_module: str, class_name: str) -> Type:
        """获取组件的通用方法"""
        if cache_key in cls._component_cache:
            return cls._component_cache[cache_key]
        
        # 尝试导入主要模块
        try:
            module = importlib.import_module(primary_module)
            component = getattr(module, class_name)
            cls._component_cache[cache_key] = component
            return component
        except (ImportError, AttributeError) as e:
            # 使用fallback模块
            try:
                fallback_module_obj = importlib.import_module(fallback_module)
                component = getattr(fallback_module_obj, class_name)
                cls._component_cache[cache_key] = component
                return component
            except (ImportError, AttributeError) as fallback_error:
                raise ImportError(
                    f"无法导入 {class_name}，主要模块错误: {e}，"
                    f"fallback模块错误: {fallback_error}"
                )
    
    @classmethod
    def get_config_class(cls, config_name: str) -> Optional[Type]:
        """获取配置类"""
        try:
            module = importlib.import_module(f'src.acceptance.config.{config_name}')
            # 假设配置类名是驼峰命名的配置名
            class_name = ''.join(word.capitalize() for word in config_name.split('_'))
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            return None
    
    @classmethod
    def get_factory_class(cls, factory_name: str) -> Optional[Type]:
        """获取工厂类"""
        try:
            module = importlib.import_module(f'src.acceptance.factories.{factory_name}')
            # 假设工厂类名是驼峰命名的工厂名
            class_name = ''.join(word.capitalize() for word in factory_name.split('_'))
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            return None
    
    @classmethod
    def clear_cache(cls):
        """清空缓存"""
        cls._component_cache.clear()


# 便捷函数
def get_components():
    """获取所有核心组件"""
    factory = ComponentFactory()
    return {
        'BaseTestPhase': factory.get_base_test_phase(),
        'TestResult': factory.get_test_result(),
        'TestStatus': factory.get_test_status(),
        'AcceptanceTestError': factory.get_acceptance_test_error()
    }