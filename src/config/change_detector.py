"""
配置变更检测器
"""

from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ConfigChange:
    """配置变更记录"""
    timestamp: datetime
    path: str
    old_value: Any
    new_value: Any
    source: str = "unknown"
    user: str = "system"


class ChangeDetector:
    """配置变更检测器"""
    
    def detect_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> List[ConfigChange]:
        """检测配置变更"""
        changes = []
        self._compare_configs(old_config, new_config, "", changes)
        return changes
    
    def _compare_configs(self, old_dict: Dict[str, Any], new_dict: Dict[str, Any], 
                        path_prefix: str, changes: List[ConfigChange]) -> None:
        """递归比较配置"""
        # 检查新增和修改
        for key, new_value in new_dict.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else key
            
            if key not in old_dict:
                # 新增
                changes.append(ConfigChange(
                    timestamp=datetime.now(),
                    path=current_path,
                    old_value=None,
                    new_value=new_value,
                    source="file_reload"
                ))
            elif old_dict[key] != new_value:
                if isinstance(old_dict[key], dict) and isinstance(new_value, dict):
                    # 递归比较字典
                    self._compare_configs(old_dict[key], new_value, current_path, changes)
                else:
                    # 修改
                    changes.append(ConfigChange(
                        timestamp=datetime.now(),
                        path=current_path,
                        old_value=old_dict[key],
                        new_value=new_value,
                        source="file_reload"
                    ))
        
        # 检查删除
        for key, old_value in old_dict.items():
            if key not in new_dict:
                current_path = f"{path_prefix}.{key}" if path_prefix else key
                changes.append(ConfigChange(
                    timestamp=datetime.now(),
                    path=current_path,
                    old_value=old_value,
                    new_value=None,
                    source="file_reload"
                ))