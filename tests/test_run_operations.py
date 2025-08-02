#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试run.py中的运维功能

这个脚本用于验证run.py中新增的运维和调试功能是否正常工作
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_import_run_module():
    """测试是否能正确导入run模块"""
    try:
        import run
        print("✅ run模块导入成功")
        return True
    except Exception as e:
        print(f"❌ run模块导入失败: {e}")
        return False

def test_run_command_function():
    """测试run_command函数"""
    try:
        import run
        # 测试简单命令
        returncode = run.run_command("echo test", capture_output=False)
        if returncode == 0:
            print("✅ run_command函数工作正常")
            return True
        else:
            print("❌ run_command函数返回非零退出码")
            return False
    except Exception as e:
        print(f"❌ run_command函数测试失败: {e}")
        return False

def test_capture_output():
    """测试capture_output功能"""
    try:
        import run
        returncode, stdout, stderr = run.run_command("echo hello", capture_output=True)
        if returncode == 0 and "hello" in stdout:
            print("✅ capture_output功能工作正常")
            return True
        else:
            print("❌ capture_output功能异常")
            return False
    except Exception as e:
        print(f"❌ capture_output功能测试失败: {e}")
        return False

def test_operations_functions_exist():
    """测试运维功能函数是否存在"""
    try:
        import run
        functions_to_check = [
            'pre_flight_check',
            'start_celery_worker', 
            'run_daily_workflow',
            'data_quality_check',
            'fix_data_sync',
            'emergency_diagnosis',
            'operations_menu'
        ]
        
        missing_functions = []
        for func_name in functions_to_check:
            if not hasattr(run, func_name):
                missing_functions.append(func_name)
        
        if not missing_functions:
            print("✅ 所有运维功能函数都存在")
            return True
        else:
            print(f"❌ 缺少运维功能函数: {missing_functions}")
            return False
    except Exception as e:
        print(f"❌ 运维功能函数检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== run.py 运维功能测试 ===")
    print()
    
    tests = [
        test_import_run_module,
        test_run_command_function,
        test_capture_output,
        test_operations_functions_exist
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== 测试结果: {passed}/{total} 项通过 ===")
    
    if passed == total:
        print("🎉 所有测试通过！run.py运维功能集成成功")
        return True
    else:
        print("⚠️  部分测试失败，请检查run.py的实现")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)