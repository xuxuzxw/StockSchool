#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能剖析工具
使用cProfile分析代码性能瓶颈

主要功能：
1. 分析factor_engine.py性能
2. 分析training_pipeline.py性能
3. 生成性能报告
4. 识别性能瓶颈

作者: StockSchool Team
创建时间: 2025
"""

import cProfile
import pstats
import io
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

try:
    from src.compute.factor_engine import FactorEngine
    from src.ai.training_pipeline import ModelTrainingPipeline
except ImportError:
    # 备用导入方式
    import sys
    import os
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, src_path)
    from compute.factor_engine import FactorEngine
    from ai.training_pipeline import ModelTrainingPipeline


class PerformanceProfiler:
    """性能剖析器"""
    
    def __init__(self):
        """初始化性能剖析器"""
        self.results = {}
        print("✅ 性能剖析器初始化完成")
    
    def profile_factor_engine(self, sample_stocks: List[str] = None) -> Dict:
        """分析factor_engine.py的性能
        
        Args:
            sample_stocks: 样本股票列表，如果为None则使用前10只股票
            
        Returns:
            性能分析结果
        """
        print("开始分析 factor_engine.py 性能...")
        
        # 创建性能分析器
        profiler = cProfile.Profile()
        
        # 准备测试数据
        if sample_stocks is None:
            try:
                engine = FactorEngine()
                all_stocks = engine.get_all_stocks()
                sample_stocks = all_stocks[:10]  # 使用前10只股票进行测试
            except Exception as e:
                print(f"❌ 获取股票列表失败: {e}")
                sample_stocks = ['000001.SZ', '000002.SZ', '600000.SH']  # 使用默认股票
        
        print(f"使用 {len(sample_stocks)} 只股票进行性能测试")
        
        # 开始性能分析
        def test_factor_calculation():
            """测试因子计算性能"""
            engine = FactorEngine()
            
            # 测试单只股票因子计算
            for ts_code in sample_stocks:
                try:
                    engine.calculate_stock_factors(ts_code, limit=100)  # 限制100条数据
                except Exception as e:
                    print(f"计算 {ts_code} 因子失败: {e}")
                    continue
        
        # 执行性能分析
        profiler.enable()
        test_factor_calculation()
        profiler.disable()
        
        # 分析结果
        result = self._analyze_profile_results(profiler, "factor_engine")
        self.results['factor_engine'] = result
        
        print("✅ factor_engine.py 性能分析完成")
        return result
    
    def profile_training_pipeline(self, sample_size: int = 100) -> Dict:
        """分析training_pipeline.py的性能
        
        Args:
            sample_size: 样本数据大小
            
        Returns:
            性能分析结果
        """
        print("开始分析 training_pipeline.py 性能...")
        
        # 创建性能分析器
        profiler = cProfile.Profile()
        
        # 开始性能分析
        def test_training_pipeline():
            """测试训练流水线性能"""
            try:
                pipeline = ModelTrainingPipeline()
                
                # 测试数据准备
                pipeline.prepare_training_data(
                    start_date='20240101',
                    end_date='20240131',  # 使用较小的日期范围
                    sample_size=sample_size
                )
                
                # 测试特征工程
                pipeline.feature_engineering()
                
                # 注意：不执行实际训练，只测试数据准备和特征工程
                print("性能测试完成（跳过模型训练）")
                
            except Exception as e:
                print(f"训练流水线测试失败: {e}")
        
        # 执行性能分析
        profiler.enable()
        test_training_pipeline()
        profiler.disable()
        
        # 分析结果
        result = self._analyze_profile_results(profiler, "training_pipeline")
        self.results['training_pipeline'] = result
        
        print("✅ training_pipeline.py 性能分析完成")
        return result
    
    def _analyze_profile_results(self, profiler: cProfile.Profile, module_name: str) -> Dict:
        """分析性能分析结果
        
        Args:
            profiler: cProfile分析器
            module_name: 模块名称
            
        Returns:
            分析结果字典
        """
        # 创建统计对象
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        
        # 按累计时间排序
        stats.sort_stats('cumulative')
        
        # 获取前20个最耗时的函数
        stats.print_stats(20)
        
        # 获取统计信息
        stats_output = stats_stream.getvalue()
        
        # 解析统计信息
        top_functions = self._parse_stats_output(stats_output)
        
        # 获取总体统计
        total_calls = stats.total_calls
        total_time = stats.total_tt
        
        result = {
            'module_name': module_name,
            'total_calls': total_calls,
            'total_time': total_time,
            'top_functions': top_functions,
            'raw_stats': stats_output,
            'analysis_time': datetime.now().isoformat()
        }
        
        return result
    
    def _parse_stats_output(self, stats_output: str) -> List[Dict]:
        """解析统计输出，提取前5个最耗时的函数
        
        Args:
            stats_output: 统计输出字符串
            
        Returns:
            前5个最耗时函数的列表
        """
        lines = stats_output.split('\n')
        top_functions = []
        
        # 查找数据行（跳过头部信息）
        data_started = False
        function_count = 0
        
        for line in lines:
            if 'ncalls' in line and 'tottime' in line:
                data_started = True
                continue
            
            if data_started and line.strip() and function_count < 5:
                try:
                    # 解析统计行
                    parts = line.split()
                    if len(parts) >= 6:
                        ncalls = parts[0]
                        tottime = float(parts[1])
                        percall_tot = float(parts[2]) if parts[2] != '0.000' else 0.0
                        cumtime = float(parts[3])
                        percall_cum = float(parts[4]) if parts[4] != '0.000' else 0.0
                        filename_func = ' '.join(parts[5:])
                        
                        function_info = {
                            'rank': function_count + 1,
                            'ncalls': ncalls,
                            'tottime': tottime,
                            'percall_tot': percall_tot,
                            'cumtime': cumtime,
                            'percall_cum': percall_cum,
                            'filename_func': filename_func
                        }
                        
                        top_functions.append(function_info)
                        function_count += 1
                        
                except (ValueError, IndexError) as e:
                    continue
        
        return top_functions
    
    def generate_performance_report(self, output_file: str = None) -> str:
        """生成性能分析报告
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            报告内容
        """
        if not self.results:
            return "❌ 没有性能分析结果，请先运行性能分析"
        
        report_lines = []
        report_lines.append("# StockSchool 性能分析报告")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        for module_name, result in self.results.items():
            report_lines.append(f"## {module_name} 性能分析")
            report_lines.append("")
            report_lines.append(f"- 总调用次数: {result['total_calls']:,}")
            report_lines.append(f"- 总执行时间: {result['total_time']:.4f} 秒")
            report_lines.append("")
            
            report_lines.append("### 累计耗时最长的前5个函数")
            report_lines.append("")
            
            if result['top_functions']:
                report_lines.append("| 排名 | 调用次数 | 累计时间(秒) | 单次时间(秒) | 函数 |")
                report_lines.append("|------|----------|--------------|--------------|------|")
                
                for func in result['top_functions']:
                    report_lines.append(
                        f"| {func['rank']} | {func['ncalls']} | {func['cumtime']:.4f} | "
                        f"{func['percall_cum']:.6f} | {func['filename_func']} |"
                    )
            else:
                report_lines.append("未找到函数统计信息")
            
            report_lines.append("")
            report_lines.append("### 性能瓶颈分析")
            report_lines.append("")
            
            # 分析性能瓶颈
            bottlenecks = self._identify_bottlenecks(result)
            for bottleneck in bottlenecks:
                report_lines.append(f"- {bottleneck}")
            
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
        
        # 总体建议
        report_lines.append("## 优化建议")
        report_lines.append("")
        suggestions = self._generate_optimization_suggestions()
        for suggestion in suggestions:
            report_lines.append(f"- {suggestion}")
        
        report_content = '\n'.join(report_lines)
        
        # 保存到文件
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                print(f"✅ 性能报告已保存到: {output_file}")
            except Exception as e:
                print(f"❌ 保存报告失败: {e}")
        
        return report_content
    
    def _identify_bottlenecks(self, result: Dict) -> List[str]:
        """识别性能瓶颈
        
        Args:
            result: 性能分析结果
            
        Returns:
            瓶颈描述列表
        """
        bottlenecks = []
        
        if not result['top_functions']:
            return ["无法识别性能瓶颈：缺少函数统计信息"]
        
        top_func = result['top_functions'][0]
        
        # 检查最耗时的函数
        if top_func['cumtime'] > 1.0:
            bottlenecks.append(
                f"最耗时函数 '{top_func['filename_func']}' 累计耗时 {top_func['cumtime']:.4f} 秒，"
                f"占总时间的 {(top_func['cumtime']/result['total_time']*100):.1f}%"
            )
        
        # 检查调用次数过多的函数
        for func in result['top_functions']:
            try:
                ncalls = int(func['ncalls'].split('/')[0])  # 处理 "100/50" 格式
                if ncalls > 10000:
                    bottlenecks.append(
                        f"函数 '{func['filename_func']}' 调用次数过多 ({ncalls:,} 次），"
                        f"可能存在重复计算"
                    )
            except (ValueError, AttributeError):
                continue
        
        # 检查单次调用耗时过长的函数
        for func in result['top_functions']:
            if func['percall_cum'] > 0.1:
                bottlenecks.append(
                    f"函数 '{func['filename_func']}' 单次调用耗时过长 ({func['percall_cum']:.4f} 秒)"
                )
        
        if not bottlenecks:
            bottlenecks.append("未发现明显的性能瓶颈")
        
        return bottlenecks
    
    def _generate_optimization_suggestions(self) -> List[str]:
        """生成优化建议
        
        Returns:
            优化建议列表
        """
        suggestions = [
            "使用数据库索引优化查询性能",
            "考虑使用缓存机制减少重复计算",
            "批量处理数据以减少数据库连接开销",
            "使用向量化操作替代循环计算",
            "考虑使用多进程或多线程并行处理",
            "优化数据结构和算法复杂度",
            "减少不必要的数据复制和转换",
            "使用更高效的数据格式（如Parquet）"
        ]
        
        return suggestions
    
    def run_full_analysis(self, output_dir: str = "performance_reports") -> str:
        """运行完整的性能分析
        
        Args:
            output_dir: 输出目录
            
        Returns:
            报告文件路径
        """
        print("开始完整性能分析...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 分析factor_engine
        try:
            self.profile_factor_engine()
        except Exception as e:
            print(f"❌ factor_engine 性能分析失败: {e}")
        
        # 分析training_pipeline
        try:
            self.profile_training_pipeline()
        except Exception as e:
            print(f"❌ training_pipeline 性能分析失败: {e}")
        
        # 生成报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(output_dir, f"performance_report_{timestamp}.md")
        
        self.generate_performance_report(report_file)
        
        print("✅ 完整性能分析完成")
        return report_file


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='StockSchool 性能分析工具')
    parser.add_argument('--module', choices=['factor_engine', 'training_pipeline', 'all'], 
                       default='all', help='要分析的模块')
    parser.add_argument('--output-dir', type=str, default='performance_reports',
                       help='输出目录')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='样本大小')
    
    args = parser.parse_args()
    
    try:
        profiler = PerformanceProfiler()
        
        if args.module == 'factor_engine':
            profiler.profile_factor_engine()
        elif args.module == 'training_pipeline':
            profiler.profile_training_pipeline(args.sample_size)
        else:
            profiler.run_full_analysis(args.output_dir)
        
        # 生成报告
        if args.module != 'all':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = os.path.join(args.output_dir, f"{args.module}_performance_{timestamp}.md")
            profiler.generate_performance_report(report_file)
        
    except Exception as e:
        print(f"❌ 性能分析过程中发生错误: {e}")
        raise