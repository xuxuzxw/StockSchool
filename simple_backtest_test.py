#!/usr/bin/env python3
"""
回测引擎简化测试脚本
验证投资组合的历史回测功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SimpleBacktestEngine:
    """简化回测引擎"""
    
    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        
    def generate_market_data(self, start_date, end_date, num_stocks=10):
        """生成市场数据"""
        np.random.seed(42)
        
        # 生成股票代码
        stock_codes = [f'00{i:04d}.SZ' for i in range(1, num_stocks + 1)]
        
        # 生成日期序列
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = [d for d in date_range if d.weekday() < 5]  # 工作日
        
        market_data = []
        
        for stock in stock_codes:
            # 为每只股票生成价格序列
            price = 100.0  # 初始价格
            for date in trading_days:
                # 模拟价格变动 (-5% 到 +5%)
                daily_return = np.random.normal(0.001, 0.025)  # 平均0.1%，标准差2.5%
                price *= (1 + daily_return)
                
                market_data.append({
                    'ts_code': stock,
                    'trade_date': date.strftime('%Y%m%d'),
                    'close': max(price, 1.0),  # 确保价格不为负
                    'volume': np.random.randint(100000, 1000000),
                    'return_rate': daily_return
                })
        
        return pd.DataFrame(market_data)
    
    def create_portfolio_signals(self, market_data):
        """创建投资组合信号"""
        # 基于动量策略的简单信号
        signals = []
        
        for stock in market_data['ts_code'].unique():
            stock_data = market_data[market_data['ts_code'] == stock].copy()
            stock_data = stock_data.sort_values('trade_date')
            
            # 计算20日动量
            stock_data['momentum_20'] = stock_data['close'].pct_change(20)
            
            # 生成信号：动量大于阈值买入，小于阈值卖出
            buy_threshold = 0.05
            sell_threshold = -0.05
            
            for _, row in stock_data.iterrows():
                if pd.notna(row['momentum_20']):
                    if row['momentum_20'] > buy_threshold:
                        signal = 'BUY'
                    elif row['momentum_20'] < sell_threshold:
                        signal = 'SELL'
                    else:
                        signal = 'HOLD'
                    
                    signals.append({
                        'ts_code': stock,
                        'trade_date': row['trade_date'],
                        'signal': signal,
                        'price': row['close'],
                        'momentum': row['momentum_20']
                    })
        
        return pd.DataFrame(signals)
    
    def execute_trades(self, signals, max_position_size=0.1):
        """执行交易"""
        trades = []
        
        # 按日期分组处理信号
        signals = signals.sort_values('trade_date')
        
        for date in signals['trade_date'].unique():
            daily_signals = signals[signals['trade_date'] == date]
            
            for _, signal in daily_signals.iterrows():
                stock = signal['ts_code']
                price = signal['price']
                
                if signal['signal'] == 'BUY' and self.cash > 0:
                    # 计算可买入数量
                    max_investment = self.initial_capital * max_position_size
                    shares = int(max_investment / price)
                    
                    if shares > 0:
                        cost = shares * price
                        self.cash -= cost
                        
                        if stock in self.positions:
                            self.positions[stock] += shares
                        else:
                            self.positions[stock] = shares
                        
                        trades.append({
                            'trade_date': date,
                            'ts_code': stock,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price,
                            'value': cost
                        })
                
                elif signal['signal'] == 'SELL' and stock in self.positions:
                    shares = self.positions[stock]
                    if shares > 0:
                        revenue = shares * price
                        self.cash += revenue
                        del self.positions[stock]
                        
                        trades.append({
                            'trade_date': date,
                            'ts_code': stock,
                            'action': 'SELL',
                            'shares': shares,
                            'price': price,
                            'value': revenue
                        })
        
        return trades
    
    def calculate_portfolio_value(self, market_data, date):
        """计算组合价值"""
        portfolio_value = self.cash
        
        for stock, shares in self.positions.items():
            stock_data = market_data[(market_data['ts_code'] == stock) & 
                                   (market_data['trade_date'] == date)]
            
            if not stock_data.empty:
                price = stock_data.iloc[0]['close']
                portfolio_value += shares * price
        
        return portfolio_value
    
    def run_backtest(self, start_date, end_date):
        """运行回测"""
        print("🎯 开始回测...")
        
        # 1. 生成市场数据
        print("📊 生成市场数据...")
        market_data = self.generate_market_data(start_date, end_date)
        
        # 2. 创建交易信号
        print("📈 创建交易信号...")
        signals = self.create_portfolio_signals(market_data)
        
        # 3. 执行交易
        print("💰 执行交易...")
        trades = self.execute_trades(signals)
        
        # 4. 计算每日组合价值
        print("📊 计算组合价值...")
        dates = sorted(market_data['trade_date'].unique())
        
        portfolio_values = []
        for date in dates:
            value = self.calculate_portfolio_value(market_data, date)
            portfolio_values.append({
                'trade_date': date,
                'portfolio_value': value,
                'cash': self.cash,
                'positions': len(self.positions)
            })
        
        # 5. 计算回测指标
        results = self.calculate_backtest_metrics(portfolio_values)
        
        return {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'metrics': results,
            'market_data': market_data
        }
    
    def calculate_backtest_metrics(self, portfolio_values):
        """计算回测指标"""
        if not portfolio_values:
            return {}
        
        df = pd.DataFrame(portfolio_values)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date')
        
        # 计算收益率
        df['daily_return'] = df['portfolio_value'].pct_change()
        
        # 总收益率
        total_return = (df['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # 年化收益率
        days = (df['trade_date'].iloc[-1] - df['trade_date'].iloc[0]).days
        annual_return = ((1 + total_return/100) ** (365/days) - 1) * 100 if days > 0 else 0
        
        # 最大回撤
        df['cumulative'] = (1 + df['daily_return']).cumprod()
        df['running_max'] = df['cumulative'].expanding().max()
        df['drawdown'] = (df['cumulative'] - df['running_max']) / df['running_max']
        max_drawdown = df['drawdown'].min() * 100
        
        # 波动率
        volatility = df['daily_return'].std() * np.sqrt(252) * 100
        
        # 夏普比率
        sharpe_ratio = (annual_return - 3) / volatility if volatility > 0 else 0  # 假设无风险利率3%
        
        # 胜率
        winning_days = (df['daily_return'] > 0).sum()
        total_days = len(df[df['daily_return'].notna()])
        win_rate = (winning_days / total_days * 100) if total_days > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_days': days,
            'final_value': df['portfolio_value'].iloc[-1]
        }

def test_backtest_engine():
    """测试回测引擎"""
    print("🎯 开始回测引擎测试...")
    
    # 初始化回测引擎
    engine = SimpleBacktestEngine(initial_capital=1000000)
    
    # 设置回测时间范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)  # 2个月回测
    
    # 运行回测
    results = engine.run_backtest(start_date, end_date)
    
    # 输出结果
    print("\n" + "="*60)
    print("📊 回测结果汇总")
    print("="*60)
    
    metrics = results['metrics']
    print(f"💰 初始资金: ¥{engine.initial_capital:,.2f}")
    print(f"💵 最终资金: ¥{metrics['final_value']:,.2f}")
    print(f"📈 总收益率: {metrics['total_return']:.2f}%")
    print(f"📊 年化收益率: {metrics['annual_return']:.2f}%")
    print(f"⚠️ 最大回撤: {metrics['max_drawdown']:.2f}%")
    print(f"📊 波动率: {metrics['volatility']:.2f}%")
    print(f"📊 夏普比率: {metrics['sharpe_ratio']:.3f}")
    print(f"✅ 胜率: {metrics['win_rate']:.2f}%")
    print(f"📅 回测天数: {metrics['total_days']}天")
    
    # 交易统计
    trades = results['trades']
    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']
    
    print(f"\n💰 交易统计:")
    print(f"买入交易: {len(buy_trades)}次")
    print(f"卖出交易: {len(sell_trades)}次")
    print(f"总交易: {len(trades)}次")
    
    if trades:
        total_investment = sum(t['value'] for t in buy_trades)
        total_revenue = sum(t['value'] for t in sell_trades)
        print(f"总投资: ¥{total_investment:,.2f}")
        print(f"总回收: ¥{total_revenue:,.2f}")
    
    return {
        'success': len(trades) > 0,
        'metrics': metrics,
        'trade_count': len(trades)
    }

if __name__ == "__main__":
    try:
        result = test_backtest_engine()
        
        if result['success']:
            print("\n🎉 回测引擎测试通过！")
        else:
            print("\n❌ 回测引擎测试失败！")
            
    except Exception as e:
        print(f"\n❌ 测试执行失败: {str(e)}")
        import traceback
        traceback.print_exc()