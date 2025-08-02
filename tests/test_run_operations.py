"""
测试运行操作模块

验证 run.py 中的命令执行功能

作者: StockSchool Team
创建时间: 2025-01-02
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run

def test_import_run_module():
    """测试导入run模块"""
    try:
        import run
        print("Run module imported successfully")
        return True
    except ImportError as e:
        print(f"Failed to import run module: {e}")
        return False

def test_run_command_function():
    """测试run_command函数"""
    try:
        # 测试基本命令执行 (使用Windows兼容命令)
        returncode = run.run_command("cmd /c echo test", capture_output=False)
        if returncode == 0:
            print("run_command function test passed")
            return True
        else:
            print("run_command function test failed")
            return False
    except Exception as e:
        print(f"run_command function test error: {e}")
        return False

def test_capture_output():
    """测试输出捕获功能"""
    try:
        # 测试输出捕获 (使用Windows兼容命令)
        returncode, output = run.run_command("cmd /c echo hello", capture_output=True)
        if returncode == 0 and "hello" in output:
            print("Output capture test passed")
            return True
        else:
            print("Output capture test failed")
            return False
    except Exception as e:
        print(f"Output capture test error: {e}")
        return False

def test_operations_functions_exist():
    """测试操作函数是否存在"""
    required_functions = [
        'run_command',
        'setup_logging',
        'check_data_dependencies',
        'main'
    ]
    
    missing_functions = []
    for func_name in required_functions:
        if not hasattr(run, func_name):
            missing_functions.append(func_name)
    
    if not missing_functions:
        print("All required functions exist")
        return True
    else:
        print(f"Missing functions: {missing_functions}")
        return False

def main():
    """主测试函数"""
    print("Starting run operations test...")
    
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
    
    print(f"\nRun operations test completed: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed!")
        return True
    else:
        print("Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)