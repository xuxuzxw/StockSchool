import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.config_loader import config

class StrategyEvaluator:
    """
    策略评估器 - 评估量化策略的表现
    """
    
    def __init__(self, benchmark_return: pd.Series = None):
        """
        初始化策略评估器
        
        Args:
            benchmark_return: 基准收益率序列
        """
        self.benchmark_return = benchmark_return
        self.risk_free_rate = config.get('strategy_params.risk_free_rate', 0.03)
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        计算收益率
        
        Args:
            prices: 价格序列
        
        Returns:
            收益率序列
        """
        return prices.pct_change().dropna()
    
    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """
        计算累计收益率
        
        Args:
            returns: 收益率序列
        
        Returns:
            累计收益率序列
        """
        return (1 + returns).cumprod() - 1
    
    def calculate_annualized_return(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        计算年化收益率
        
        Args:
            returns: 收益率序列
            periods_per_year: 每年的期间数（日频为252）
        
        Returns:
            年化收益率
        """
        if len(returns) == 0:
            return 0.0
        
        total_return = (1 + returns).prod() - 1
        years = len(returns) / periods_per_year
        
        if years <= 0:
            return 0.0
        
        return (1 + total_return) ** (1 / years) - 1
    
    def calculate_annualized_volatility(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        计算年化波动率
        
        Args:
            returns: 收益率序列
            periods_per_year: 每年的期间数
        
        Returns:
            年化波动率
        """
        if len(returns) <= 1:
            return 0.0
        
        return returns.std() * np.sqrt(periods_per_year)
    
    def calculate_sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益率序列
            periods_per_year: 每年的期间数
        
        Returns:
            夏普比率
        """
        if len(returns) <= 1:
            return 0.0
        
        annualized_return = self.calculate_annualized_return(returns, periods_per_year)
        annualized_vol = self.calculate_annualized_volatility(returns, periods_per_year)
        
        if annualized_vol == 0:
            return 0.0
        
        return (annualized_return - self.risk_free_rate) / annualized_vol
    
    def calculate_sortino_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        计算索提诺比率
        
        Args:
            returns: 收益率序列
            periods_per_year: 每年的期间数
        
        Returns:
            索提诺比率
        """
        if len(returns) <= 1:
            return 0.0
        
        annualized_return = self.calculate_annualized_return(returns, periods_per_year)
        
        # 计算下行波动率
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf') if annualized_return > self.risk_free_rate else 0.0
        
        downside_vol = negative_returns.std() * np.sqrt(periods_per_year)
        
        if downside_vol == 0:
            return 0.0
        
        return (annualized_return - self.risk_free_rate) / downside_vol
    
    def calculate_max_drawdown(self, returns: pd.Series) -> Dict[str, float]:
        """
        计算最大回撤
        
        Args:
            returns: 收益率序列
        
        Returns:
            包含最大回撤信息的字典
        """
        if len(returns) == 0:
            return {'max_drawdown': 0.0, 'max_drawdown_duration': 0}
        
        cumulative = self.calculate_cumulative_returns(returns)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / (1 + running_max)
        
        max_drawdown = drawdown.min()
        
        # 计算最大回撤持续期间
        max_dd_end = drawdown.idxmin()
        max_dd_start = cumulative[:max_dd_end].idxmax()
        max_dd_duration = (max_dd_end - max_dd_start).days if hasattr(max_dd_end, 'days') else len(cumulative[max_dd_start:max_dd_end])
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'max_drawdown_start': max_dd_start,
            'max_drawdown_end': max_dd_end
        }
    
    def calculate_calmar_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        计算卡玛比率
        
        Args:
            returns: 收益率序列
            periods_per_year: 每年的期间数
        
        Returns:
            卡玛比率
        """
        annualized_return = self.calculate_annualized_return(returns, periods_per_year)
        max_drawdown = abs(self.calculate_max_drawdown(returns)['max_drawdown'])
        
        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0.0
        
        return annualized_return / max_drawdown
    
    def calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> float:
        """
        计算信息比率
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
        
        Returns:
            信息比率
        """
        if benchmark_returns is None:
            benchmark_returns = self.benchmark_return
        
        if benchmark_returns is None or len(returns) != len(benchmark_returns):
            return 0.0
        
        # 计算超额收益
        excess_returns = returns - benchmark_returns
        
        if len(excess_returns) <= 1:
            return 0.0
        
        tracking_error = excess_returns.std()
        
        if tracking_error == 0:
            return 0.0
        
        return excess_returns.mean() / tracking_error
    
    def calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> float:
        """
        计算贝塔系数
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
        
        Returns:
            贝塔系数
        """
        if benchmark_returns is None:
            benchmark_returns = self.benchmark_return
        
        if benchmark_returns is None or len(returns) != len(benchmark_returns):
            return 1.0
        
        # 去除缺失值
        valid_data = pd.DataFrame({'strategy': returns, 'benchmark': benchmark_returns}).dropna()
        
        if len(valid_data) <= 1:
            return 1.0
        
        covariance = valid_data['strategy'].cov(valid_data['benchmark'])
        benchmark_variance = valid_data['benchmark'].var()
        
        if benchmark_variance == 0:
            return 1.0
        
        return covariance / benchmark_variance
    
    def calculate_alpha(self, returns: pd.Series, benchmark_returns: pd.Series = None, periods_per_year: int = 252) -> float:
        """
        计算阿尔法系数
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            periods_per_year: 每年的期间数
        
        Returns:
            阿尔法系数
        """
        if benchmark_returns is None:
            benchmark_returns = self.benchmark_return
        
        if benchmark_returns is None:
            return 0.0
        
        strategy_return = self.calculate_annualized_return(returns, periods_per_year)
        benchmark_return = self.calculate_annualized_return(benchmark_returns, periods_per_year)
        beta = self.calculate_beta(returns, benchmark_returns)
        
        return strategy_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """
        计算胜率
        
        Args:
            returns: 收益率序列
        
        Returns:
            胜率
        """
        if len(returns) == 0:
            return 0.0
        
        return (returns > 0).sum() / len(returns)
    
    def calculate_profit_loss_ratio(self, returns: pd.Series) -> float:
        """
        计算盈亏比
        
        Args:
            returns: 收益率序列
        
        Returns:
            盈亏比
        """
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return 0.0
        
        avg_profit = positive_returns.mean()
        avg_loss = abs(negative_returns.mean())
        
        if avg_loss == 0:
            return float('inf')
        
        return avg_profit / avg_loss
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = None) -> float:
        """
        计算风险价值(VaR)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
        
        Returns:
            VaR值
        """
        if len(returns) == 0:
            return 0.0
        
        if confidence_level is None:
            confidence_level = config.get('strategy_params.var_confidence_level', 0.05)
        
        return returns.quantile(confidence_level)
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = None) -> float:
        """
        计算条件风险价值(CVaR)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
        
        Returns:
            CVaR值
        """
        if len(returns) == 0:
            return 0.0
        
        if confidence_level is None:
            confidence_level = config.get('strategy_params.var_confidence_level', 0.05)
        
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def evaluate_strategy(self, returns: pd.Series, benchmark_returns: pd.Series = None, 
                         periods_per_year: int = 252) -> Dict[str, Any]:
        """
        综合评估策略
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            periods_per_year: 每年的期间数
        
        Returns:
            评估结果字典
        """
        if len(returns) == 0:
            return {}
        
        # 基础指标
        total_return = (1 + returns).prod() - 1
        annualized_return = self.calculate_annualized_return(returns, periods_per_year)
        annualized_vol = self.calculate_annualized_volatility(returns, periods_per_year)
        
        # 风险调整指标
        sharpe_ratio = self.calculate_sharpe_ratio(returns, periods_per_year)
        sortino_ratio = self.calculate_sortino_ratio(returns, periods_per_year)
        calmar_ratio = self.calculate_calmar_ratio(returns, periods_per_year)
        
        # 回撤指标
        drawdown_info = self.calculate_max_drawdown(returns)
        
        # 交易指标
        win_rate = self.calculate_win_rate(returns)
        profit_loss_ratio = self.calculate_profit_loss_ratio(returns)
        
        # 风险指标
        var_5 = self.calculate_var(returns, 0.05)
        cvar_5 = self.calculate_cvar(returns, 0.05)
        
        evaluation = {
            # 收益指标
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            
            # 风险调整指标
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # 回撤指标
            'max_drawdown': drawdown_info['max_drawdown'],
            'max_drawdown_duration': drawdown_info['max_drawdown_duration'],
            
            # 交易指标
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            
            # 风险指标
            'var_5%': var_5,
            'cvar_5%': cvar_5,
            
            # 统计指标
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'trading_days': len(returns)
        }
        
        # 如果有基准，计算相对指标
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            evaluation.update({
                'beta': self.calculate_beta(returns, benchmark_returns),
                'alpha': self.calculate_alpha(returns, benchmark_returns, periods_per_year),
                'information_ratio': self.calculate_information_ratio(returns, benchmark_returns),
                'benchmark_return': self.calculate_annualized_return(benchmark_returns, periods_per_year),
                'excess_return': annualized_return - self.calculate_annualized_return(benchmark_returns, periods_per_year)
            })
        
        return evaluation
    
    def generate_performance_report(self, returns: pd.Series, benchmark_returns: pd.Series = None,
                                  strategy_name: str = "Strategy") -> str:
        """
        生成策略表现报告
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            strategy_name: 策略名称
        
        Returns:
            报告字符串
        """
        evaluation = self.evaluate_strategy(returns, benchmark_returns)
        
        if not evaluation:
            return "无法生成报告：数据不足"
        
        report = []
        report.append("=" * 60)
        report.append(f"策略表现报告 - {strategy_name}")
        report.append("=" * 60)
        
        # 基本信息
        report.append(f"分析期间: {len(returns)} 个交易日")
        if hasattr(returns.index, 'min') and hasattr(returns.index, 'max'):
            report.append(f"时间范围: {returns.index.min()} 至 {returns.index.max()}")
        
        # 收益指标
        report.append("\n📈 收益指标:")
        report.append(f"  总收益率: {evaluation['total_return']:.2%}")
        report.append(f"  年化收益率: {evaluation['annualized_return']:.2%}")
        report.append(f"  年化波动率: {evaluation['annualized_volatility']:.2%}")
        
        if 'benchmark_return' in evaluation:
            report.append(f"  基准年化收益: {evaluation['benchmark_return']:.2%}")
            report.append(f"  超额收益: {evaluation['excess_return']:.2%}")
        
        # 风险调整指标
        report.append("\n⚖️ 风险调整指标:")
        report.append(f"  夏普比率: {evaluation['sharpe_ratio']:.3f}")
        report.append(f"  索提诺比率: {evaluation['sortino_ratio']:.3f}")
        report.append(f"  卡玛比率: {evaluation['calmar_ratio']:.3f}")
        
        if 'information_ratio' in evaluation:
            report.append(f"  信息比率: {evaluation['information_ratio']:.3f}")
        
        # 回撤指标
        report.append("\n📉 回撤指标:")
        report.append(f"  最大回撤: {evaluation['max_drawdown']:.2%}")
        report.append(f"  最大回撤持续期: {evaluation['max_drawdown_duration']} 天")
        
        # 交易指标
        report.append("\n🎯 交易指标:")
        report.append(f"  胜率: {evaluation['win_rate']:.2%}")
        report.append(f"  盈亏比: {evaluation['profit_loss_ratio']:.2f}")
        
        # 风险指标
        report.append("\n⚠️ 风险指标:")
        report.append(f"  VaR (5%): {evaluation['var_5%']:.2%}")
        report.append(f"  CVaR (5%): {evaluation['cvar_5%']:.2%}")
        
        # 统计特征
        report.append("\n📊 统计特征:")
        report.append(f"  偏度: {evaluation['skewness']:.3f}")
        report.append(f"  峰度: {evaluation['kurtosis']:.3f}")
        
        if 'beta' in evaluation:
            report.append(f"  贝塔系数: {evaluation['beta']:.3f}")
            report.append(f"  阿尔法系数: {evaluation['alpha']:.2%}")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def plot_performance(self, returns: pd.Series, benchmark_returns: pd.Series = None,
                        strategy_name: str = "Strategy", figsize: Tuple[int, int] = (15, 10)):
        """
        绘制策略表现图表
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            strategy_name: 策略名称
            figsize: 图表大小
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'{strategy_name} 策略表现分析', fontsize=16, fontweight='bold')
        
        # 累计收益曲线
        ax1 = axes[0, 0]
        cumulative_returns = self.calculate_cumulative_returns(returns)
        ax1.plot(cumulative_returns.index, cumulative_returns.values, label=strategy_name, linewidth=2)
        
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            benchmark_cumulative = self.calculate_cumulative_returns(benchmark_returns)
            ax1.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                    label='基准', linewidth=2, alpha=0.7)
        
        ax1.set_title('累计收益曲线')
        ax1.set_ylabel('累计收益率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 回撤曲线
        ax2 = axes[0, 1]
        cumulative = self.calculate_cumulative_returns(returns)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / (1 + running_max)
        
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        ax2.set_title('回撤曲线')
        ax2.set_ylabel('回撤幅度')
        ax2.grid(True, alpha=0.3)
        
        # 收益率分布
        ax3 = axes[1, 0]
        ax3.hist(returns.values, bins=50, alpha=0.7, density=True, edgecolor='black')
        ax3.axvline(returns.mean(), color='red', linestyle='--', label=f'均值: {returns.mean():.3f}')
        ax3.axvline(returns.median(), color='green', linestyle='--', label=f'中位数: {returns.median():.3f}')
        ax3.set_title('收益率分布')
        ax3.set_xlabel('日收益率')
        ax3.set_ylabel('密度')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 滚动夏普比率
        ax4 = axes[1, 1]
        default_window = config.get('strategy_params.rolling_window', 60)
        rolling_window = min(default_window, len(returns) // 4)  # 配置天数或1/4数据长度
        if rolling_window >= 10:
            rolling_sharpe = returns.rolling(rolling_window).apply(
                lambda x: self.calculate_sharpe_ratio(x) if len(x) >= 10 else np.nan
            )
            ax4.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
            ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax4.set_title(f'{rolling_window}日滚动夏普比率')
            ax4.set_ylabel('夏普比率')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '数据不足\n无法计算滚动夏普比率', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('滚动夏普比率')
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # 测试代码
    print("测试策略评估模块...")
    
    # 生成模拟数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # 模拟策略收益率
    strategy_returns = pd.Series(
        np.random.normal(0.001, 0.02, len(dates)),  # 日均收益0.1%，波动率2%
        index=dates
    )
    
    # 模拟基准收益率
    benchmark_returns = pd.Series(
        np.random.normal(0.0005, 0.015, len(dates)),  # 日均收益0.05%，波动率1.5%
        index=dates
    )
    
    # 创建评估器
    evaluator = StrategyEvaluator(benchmark_returns)
    
    # 评估策略
    evaluation = evaluator.evaluate_strategy(strategy_returns, benchmark_returns)
    
    print("\n策略评估结果:")
    for key, value in evaluation.items():
        if isinstance(value, float):
            if 'ratio' in key or 'return' in key or 'alpha' in key:
                print(f"  {key}: {value:.3f}")
            elif 'rate' in key or 'drawdown' in key:
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # 生成报告
    report = evaluator.generate_performance_report(
        strategy_returns, benchmark_returns, "测试策略"
    )
    print(f"\n{report}")
    
    print("\n策略评估模块测试完成!")