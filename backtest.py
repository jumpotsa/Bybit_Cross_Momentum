import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CrossSectionalMomentumBacktest:
    """
    Backtests a cross-sectional momentum strategy using Bybit perpetual futures.
    - Long top 10 momentum stocks weighted by inverse volatility
    - Short bottom 10 momentum stocks weighted by inverse volatility
    - Rebalances on the 25th of each month
    - Includes stop-loss functionality
    """
    
    def __init__(self, initial_capital=10000, stop_loss_pct=0.10):
        """
        Args:
            initial_capital: Starting capital in USD
            stop_loss_pct: Stop loss percentage (default 10%)
        """
        self.initial_capital = initial_capital
        self.stop_loss_pct = stop_loss_pct
        self.base_url = "https://api.bybit.com/v5/market/kline"
        self.positions = {}  # Track current positions
        self.portfolio_value = initial_capital
        self.portfolio_history = []
        self.trades = []
        
    def fetch_historical_data(self, symbol, start_date, end_date, category='linear'):
        """
        Fetches historical daily kline data from Bybit.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            start_date: Start date as datetime
            end_date: End date as datetime
            category: 'linear' for USDT perpetuals
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            start_ms = int(start_date.timestamp() * 1000)
            end_ms = int(end_date.timestamp() * 1000)
            
            params = {
                'category': category,
                'symbol': symbol,
                'interval': 'D',
                'start': start_ms,
                'end': end_ms,
                'limit': 1000
            }
            
            all_klines = []
            current_end = end_ms
            
            while True:
                params['end'] = current_end
                response = requests.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if data.get('retCode') != 0:
                    print(f"API Error for {symbol}: {data.get('retMsg')}")
                    break
                
                klines = data.get('result', {}).get('list', [])
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                if len(klines) < 1000:
                    break
                
                oldest_timestamp = int(klines[-1][0])
                if oldest_timestamp <= start_ms:
                    break
                
                current_end = oldest_timestamp
                time.sleep(0.12)  # Rate limiting
            
            if not all_klines:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('date').set_index('date')
            
            # Filter to exact date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_volatility(self, prices, window=30):
        """
        Calculates annualized volatility from price series.
        
        Args:
            prices: Series of prices
            window: Lookback window for volatility calculation
        
        Returns:
            Annualized volatility
        """
        returns = prices.pct_change().dropna()
        volatility = returns.tail(window).std() * np.sqrt(252)  # Annualized
        return volatility if volatility > 0 else 0.01  # Minimum volatility
    
    def calculate_position_weights(self, symbols_data):
        """
        Calculates inverse volatility weights for portfolio positions.
        
        Args:
            symbols_data: Dict of {symbol: price_series}
        
        Returns:
            Dict of {symbol: weight}
        """
        volatilities = {}
        
        for symbol, prices in symbols_data.items():
            if len(prices) > 30:
                vol = self.calculate_volatility(prices)
                volatilities[symbol] = vol
        
        if not volatilities:
            # Equal weight if volatility calculation fails
            n = len(symbols_data)
            return {symbol: 1.0/n for symbol in symbols_data.keys()}
        
        # Inverse volatility weighting
        inv_vols = {symbol: 1.0/vol for symbol, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        weights = {symbol: inv_vol/total_inv_vol for symbol, inv_vol in inv_vols.items()}
        
        return weights
    
    def get_rebalance_dates(self, start_date, end_date):
        """
        Generates list of rebalance dates (25th of each month).
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            List of rebalance dates
        """
        rebalance_dates = []
        current_date = start_date.replace(day=25)
        
        # If start is after 25th, move to next month
        if start_date.day > 25:
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1, day=25)
            else:
                current_date = current_date.replace(month=current_date.month + 1, day=25)
        
        while current_date <= end_date:
            rebalance_dates.append(current_date)
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1, day=25)
            else:
                current_date = current_date.replace(month=current_date.month + 1, day=25)
        
        return rebalance_dates
    
    def check_stop_loss(self, current_price, entry_price, position_type):
        """
        Checks if stop loss is triggered.
        
        Args:
            current_price: Current price
            entry_price: Entry price
            position_type: 'long' or 'short'
        
        Returns:
            Boolean - True if stop loss triggered
        """
        if position_type == 'long':
            loss_pct = (current_price - entry_price) / entry_price
            return loss_pct <= -self.stop_loss_pct
        else:  # short
            loss_pct = (entry_price - current_price) / entry_price
            return loss_pct <= -self.stop_loss_pct
    
    def run_backtest(self, top_10_file, bottom_10_file, start_date=None, end_date=None):
        """
        Runs the cross-sectional momentum backtest.
        
        Args:
            top_10_file: Path to top 10 momentum CSV
            bottom_10_file: Path to bottom 10 momentum CSV
            start_date: Backtest start date (datetime)
            end_date: Backtest end date (datetime)
        
        Returns:
            DataFrame with backtest results
        """
        # Read momentum rankings
        top_10_df = pd.read_csv(top_10_file)
        bottom_10_df = pd.read_csv(bottom_10_file)
        
        # Get symbols
        long_symbols = top_10_df['contract'].tolist()
        short_symbols = bottom_10_df['contract'].tolist()
        all_symbols = long_symbols + short_symbols
        
        print(f"\nLong positions (Top 10): {long_symbols}")
        print(f"Short positions (Bottom 10): {short_symbols}")
        
        # Set default dates if not provided - use maximum available from Bybit
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=1000)  # ~2.7 years of data
        
        print(f"\nBacktest period: {start_date.date()} to {end_date.date()}")
        
        # Fetch historical data for all symbols
        print("\nFetching historical data...")
        historical_data = {}
        
        for i, symbol in enumerate(all_symbols, 1):
            print(f"[{i}/{len(all_symbols)}] Fetching {symbol}...")
            df = self.fetch_historical_data(symbol, start_date, end_date)
            if df is not None and len(df) > 0:
                historical_data[symbol] = df
                time.sleep(0.12)  # Rate limiting
            else:
                print(f"  Warning: No data available for {symbol}")
        
        if not historical_data:
            print("Error: No historical data fetched. Cannot run backtest.")
            return None
        
        # Get common date range
        all_dates = set.intersection(*[set(df.index) for df in historical_data.values()])
        all_dates = sorted(list(all_dates))
        
        if not all_dates:
            print("Error: No common dates across all symbols.")
            return None
        
        print(f"\nCommon trading days: {len(all_dates)}")
        print(f"Date range: {all_dates[0].date()} to {all_dates[-1].date()}")
        
        # Get rebalance dates
        rebalance_dates = self.get_rebalance_dates(all_dates[0], all_dates[-1])
        print(f"Rebalance dates: {len(rebalance_dates)} times")
        
        # Initialize tracking
        self.portfolio_value = self.initial_capital
        self.positions = {}
        daily_portfolio_values = []
        
        # Run backtest day by day
        print("\nRunning backtest...")
        
        for i, date in enumerate(all_dates):
            # Check if rebalance date
            is_rebalance = any(abs((date - rd).days) <= 1 for rd in rebalance_dates)
            
            if is_rebalance or i == 0:
                print(f"\n[{date.date()}] Rebalancing portfolio...")
                
                # Close all existing positions
                for symbol, pos in self.positions.items():
                    if symbol in historical_data:
                        current_price = historical_data[symbol].loc[date, 'close']
                        pnl = self.calculate_position_pnl(pos, current_price)
                        self.portfolio_value += pnl
                        
                        self.trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'close',
                            'type': pos['type'],
                            'entry_price': pos['entry_price'],
                            'exit_price': current_price,
                            'position_size': pos['position_size'],
                            'pnl': pnl,
                            'reason': 'rebalance'
                        })
                
                self.positions = {}
                
                # Calculate new weights
                long_data = {s: historical_data[s].loc[:date, 'close'] 
                            for s in long_symbols if s in historical_data}
                short_data = {s: historical_data[s].loc[:date, 'close'] 
                             for s in short_symbols if s in historical_data}
                
                long_weights = self.calculate_position_weights(long_data)
                short_weights = self.calculate_position_weights(short_data)
                
                # Open new long positions
                for symbol, weight in long_weights.items():
                    entry_price = historical_data[symbol].loc[date, 'close']
                    position_capital = self.portfolio_value * 0.5 * weight  # 50% in longs
                    
                    self.positions[symbol] = {
                        'type': 'long',
                        'entry_price': entry_price,
                        'entry_date': date,
                        'position_size': position_capital / entry_price,
                        'weight': weight
                    }
                    
                    self.trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'open',
                        'type': 'long',
                        'entry_price': entry_price,
                        'position_size': position_capital / entry_price,
                        'capital': position_capital,
                        'weight': weight
                    })
                
                # Open new short positions
                for symbol, weight in short_weights.items():
                    entry_price = historical_data[symbol].loc[date, 'close']
                    position_capital = self.portfolio_value * 0.5 * weight  # 50% in shorts
                    
                    self.positions[symbol] = {
                        'type': 'short',
                        'entry_price': entry_price,
                        'entry_date': date,
                        'position_size': position_capital / entry_price,
                        'weight': weight
                    }
                    
                    self.trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'open',
                        'type': 'short',
                        'entry_price': entry_price,
                        'position_size': position_capital / entry_price,
                        'capital': position_capital,
                        'weight': weight
                    })
            
            # Check stop losses for all positions
            stopped_positions = []
            for symbol, pos in self.positions.items():
                if symbol in historical_data and date in historical_data[symbol].index:
                    current_price = historical_data[symbol].loc[date, 'close']
                    
                    if self.check_stop_loss(current_price, pos['entry_price'], pos['type']):
                        # Close position due to stop loss
                        pnl = self.calculate_position_pnl(pos, current_price)
                        self.portfolio_value += pnl
                        
                        self.trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'close',
                            'type': pos['type'],
                            'entry_price': pos['entry_price'],
                            'exit_price': current_price,
                            'position_size': pos['position_size'],
                            'pnl': pnl,
                            'reason': 'stop_loss'
                        })
                        
                        stopped_positions.append(symbol)
            
            # Remove stopped positions
            for symbol in stopped_positions:
                del self.positions[symbol]
            
            # Calculate current portfolio value
            unrealized_pnl = 0
            for symbol, pos in self.positions.items():
                if symbol in historical_data and date in historical_data[symbol].index:
                    current_price = historical_data[symbol].loc[date, 'close']
                    unrealized_pnl += self.calculate_position_pnl(pos, current_price)
            
            current_portfolio_value = self.portfolio_value + unrealized_pnl
            
            daily_portfolio_values.append({
                'date': date,
                'portfolio_value': current_portfolio_value,
                'cash': self.portfolio_value,
                'unrealized_pnl': unrealized_pnl,
                'num_positions': len(self.positions)
            })
            
            if i % 50 == 0:
                print(f"  Day {i}/{len(all_dates)}: Portfolio Value = ${current_portfolio_value:,.2f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(daily_portfolio_values)
        results_df.set_index('date', inplace=True)
        
        return results_df
    
    def calculate_position_pnl(self, position, current_price):
        """
        Calculates P&L for a position.
        
        Args:
            position: Position dict
            current_price: Current market price
        
        Returns:
            P&L in USD
        """
        entry_price = position['entry_price']
        position_size = position['position_size']
        
        if position['type'] == 'long':
            pnl = (current_price - entry_price) * position_size
        else:  # short
            pnl = (entry_price - current_price) * position_size
        
        return pnl
    
    def calculate_performance_metrics(self, results_df):
        """
        Calculates performance metrics for the backtest.
        
        Args:
            results_df: DataFrame with backtest results
        
        Returns:
            Dict of performance metrics
        """
        # Calculate returns
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        results_df['cumulative_returns'] = (1 + results_df['returns']).cumprod() - 1
        
        # Performance metrics
        total_return = (results_df['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Annualized return
        days = len(results_df)
        years = days / 252
        annualized_return = ((results_df['portfolio_value'].iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Volatility
        annual_volatility = results_df['returns'].std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (annualized_return / annual_volatility) if annual_volatility > 0 else 0
        
        # Max drawdown
        cumulative = (1 + results_df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Drawdown duration
        is_drawdown = drawdown < 0
        drawdown_periods = is_drawdown.astype(int).groupby((~is_drawdown).cumsum()).sum()
        max_drawdown_duration = drawdown_periods.max() if len(drawdown_periods) > 0 else 0
        
        # Win rate from trades
        closed_trades = [t for t in self.trades if t['action'] == 'close' and 'pnl' in t]
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] < 0]
        win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0
        
        # Average win/loss
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        total_wins = sum([t['pnl'] for t in winning_trades])
        total_losses = abs(sum([t['pnl'] for t in losing_trades]))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Stop loss stats
        stop_loss_trades = [t for t in closed_trades if t.get('reason') == 'stop_loss']
        stop_loss_count = len(stop_loss_trades)
        
        # Monthly returns
        monthly_returns = results_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Calmar ratio (annualized return / max drawdown)
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = results_df['returns'][results_df['returns'] < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return / 100) / downside_std if downside_std > 0 else 0
        
        metrics = {
            'Initial Capital': f"${self.initial_capital:,.2f}",
            'Final Portfolio Value': f"${results_df['portfolio_value'].iloc[-1]:,.2f}",
            'Total Return': f"{total_return:.2f}%",
            'Annualized Return': f"{annualized_return:.2f}%",
            'Annualized Volatility': f"{annual_volatility:.2f}%",
            'Sharpe Ratio': f"{sharpe_ratio:.3f}",
            'Sortino Ratio': f"{sortino_ratio:.3f}",
            'Calmar Ratio': f"{calmar_ratio:.3f}",
            'Max Drawdown': f"{max_drawdown:.2f}%",
            'Max Drawdown Duration': f"{max_drawdown_duration} days",
            'Total Trades': len(closed_trades),
            'Winning Trades': len(winning_trades),
            'Losing Trades': len(losing_trades),
            'Win Rate': f"{win_rate:.2f}%",
            'Average Win': f"${avg_win:.2f}",
            'Average Loss': f"${avg_loss:.2f}",
            'Profit Factor': f"{profit_factor:.3f}",
            'Stop Loss Triggered': stop_loss_count,
            'Trading Days': len(results_df),
            'Years': f"{years:.2f}",
            'Best Month': f"{monthly_returns.max()*100:.2f}%",
            'Worst Month': f"{monthly_returns.min()*100:.2f}%"
        }
        
        return metrics
    
    def create_performance_report(self, results_df, output_dir='momentum'):
        """
        Creates comprehensive performance visualization report.
        
        Args:
            results_df: DataFrame with backtest results
            output_dir: Directory to save reports
        """
        print("\nGenerating performance visualizations...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate additional metrics for plotting
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        results_df['cumulative_returns'] = (1 + results_df['returns']).cumprod() - 1
        
        # Calculate drawdown
        cumulative = (1 + results_df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        results_df['drawdown'] = (cumulative - running_max) / running_max * 100
        
        # Monthly returns
        monthly_returns = results_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        # ====================
        # MAIN PERFORMANCE DASHBOARD
        # ====================
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(results_df.index, results_df['portfolio_value'], linewidth=2, color='#2E86AB')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Drawdown Chart
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.fill_between(results_df.index, results_df['drawdown'], 0, color='#A23B72', alpha=0.6)
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Drawdown (%)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        
        # 3. Monthly Returns Heatmap
        ax3 = fig.add_subplot(gs[2, :2])
        monthly_data = monthly_returns.copy()
        monthly_data.index = pd.to_datetime(monthly_data.index)
        
        # Create pivot table for heatmap
        monthly_pivot = pd.DataFrame({
            'Year': monthly_data.index.year,
            'Month': monthly_data.index.month,
            'Return': monthly_data.values
        })
        
        if len(monthly_pivot) > 0:
            heatmap_data = monthly_pivot.pivot(index='Month', columns='Year', values='Return')
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                       ax=ax3, cbar_kws={'label': 'Return (%)'}, linewidths=0.5)
            ax3.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Month', fontsize=11)
            ax3.set_xlabel('Year', fontsize=11)
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax3.set_yticklabels([month_labels[int(i)-1] for i in heatmap_data.index], rotation=0)
        
        # 4. Return Distribution
        ax4 = fig.add_subplot(gs[0, 2])
        returns_pct = results_df['returns'].dropna() * 100
        ax4.hist(returns_pct, bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax4.axvline(x=returns_pct.mean(), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: {returns_pct.mean():.3f}%')
        ax4.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Daily Return (%)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Cumulative Returns
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(results_df.index, results_df['cumulative_returns'] * 100, 
                linewidth=2, color='#06A77D')
        ax5.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Date', fontsize=11)
        ax5.set_ylabel('Cumulative Return (%)', fontsize=11)
        ax5.grid(True, alpha=0.3)
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
        
        # 6. Rolling Sharpe Ratio (90-day)
        ax6 = fig.add_subplot(gs[2, 2])
        rolling_sharpe = (results_df['returns'].rolling(90).mean() / 
                         results_df['returns'].rolling(90).std()) * np.sqrt(252)
        ax6.plot(results_df.index, rolling_sharpe, linewidth=2, color='#C73E1D')
        ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax6.set_title('Rolling 90-Day Sharpe Ratio', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Date', fontsize=11)
        ax6.set_ylabel('Sharpe Ratio', fontsize=11)
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Cross-Sectional Momentum Strategy - Performance Dashboard', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # Save main dashboard
        dashboard_path = os.path.join(output_dir, 'performance_dashboard.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved performance dashboard: {dashboard_path}")
        plt.close()
        
        # ====================
        # TRADE ANALYSIS
        # ====================
        if self.trades:
            closed_trades = [t for t in self.trades if t['action'] == 'close' and 'pnl' in t]
            
            if closed_trades:
                fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                
                # 1. P&L Distribution
                pnls = [t['pnl'] for t in closed_trades]
                axes[0, 0].hist(pnls, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
                axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
                axes[0, 0].set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
                axes[0, 0].set_xlabel('P&L ($)', fontsize=10)
                axes[0, 0].set_ylabel('Frequency', fontsize=10)
                axes[0, 0].grid(True, alpha=0.3)
                
                # 2. Win/Loss by Position Type
                long_trades = [t for t in closed_trades if t['type'] == 'long']
                short_trades = [t for t in closed_trades if t['type'] == 'short']
                
                long_wins = len([t for t in long_trades if t['pnl'] > 0])
                long_losses = len([t for t in long_trades if t['pnl'] <= 0])
                short_wins = len([t for t in short_trades if t['pnl'] > 0])
                short_losses = len([t for t in short_trades if t['pnl'] <= 0])
                
                x = np.arange(2)
                width = 0.35
                axes[0, 1].bar(x - width/2, [long_wins, short_wins], width, 
                              label='Wins', color='#06A77D', alpha=0.8)
                axes[0, 1].bar(x + width/2, [long_losses, short_losses], width, 
                              label='Losses', color='#A23B72', alpha=0.8)
                axes[0, 1].set_title('Wins vs Losses by Position Type', fontsize=12, fontweight='bold')
                axes[0, 1].set_ylabel('Number of Trades', fontsize=10)
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(['Long', 'Short'])
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3, axis='y')
                
                # 3. Cumulative P&L over time
                trades_df = pd.DataFrame(closed_trades)
                trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                axes[1, 0].plot(trades_df['date'], trades_df['cumulative_pnl'], 
                              linewidth=2, color='#F18F01')
                axes[1, 0].set_title('Cumulative P&L from Trades', fontsize=12, fontweight='bold')
                axes[1, 0].set_xlabel('Date', fontsize=10)
                axes[1, 0].set_ylabel('Cumulative P&L ($)', fontsize=10)
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                
                # 4. Stop Loss vs Rebalance Exits
                stop_loss_trades = [t for t in closed_trades if t.get('reason') == 'stop_loss']
                rebalance_trades = [t for t in closed_trades if t.get('reason') == 'rebalance']
                
                exit_types = ['Stop Loss', 'Rebalance']
                exit_counts = [len(stop_loss_trades), len(rebalance_trades)]
                colors = ['#C73E1D', '#2E86AB']
                axes[1, 1].pie(exit_counts, labels=exit_types, autopct='%1.1f%%', 
                              colors=colors, startangle=90)
                axes[1, 1].set_title('Exit Reason Distribution', fontsize=12, fontweight='bold')
                
                plt.suptitle('Trade Analysis Report', fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                # Save trade analysis
                trades_path = os.path.join(output_dir, 'trade_analysis.png')
                plt.savefig(trades_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved trade analysis: {trades_path}")
                plt.close()
        
        # ====================
        # PERFORMANCE METRICS TABLE
        # ====================
        metrics = self.calculate_performance_metrics(results_df)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table data
        table_data = [[k, v] for k, v in metrics.items()]
        
        table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                        cellLoc='left', loc='center', 
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                table[(i, 0)].set_facecolor('#E8F4F8')
                table[(i, 1)].set_facecolor('#E8F4F8')
        
        plt.title('Performance Metrics Summary', fontsize=16, fontweight='bold', pad=20)
        
        # Save metrics table
        metrics_path = os.path.join(output_dir, 'performance_metrics.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved performance metrics: {metrics_path}")
        plt.close()
        
        print(f"\n✓ All visualizations saved to '{output_dir}' folder")

# ====================
# RUN BACKTEST
# ====================

if __name__ == "__main__":
    print("="*80)
    print("CROSS-SECTIONAL MOMENTUM STRATEGY BACKTEST")
    print("Strategy: Long Top 10 / Short Bottom 10 with Volatility Weighting")
    print("="*80)
    
    # Initialize backtest
    backtest = CrossSectionalMomentumBacktest(
        initial_capital=10000,
        stop_loss_pct=0.10  # 10% stop loss
    )
    
    # Use correct file paths from momentum folder
    momentum_dir = 'momentum'
    
    # Check if files exist
    top_10_path = os.path.join(momentum_dir, 'top_10_momentum_perpetual.csv')
    bottom_10_path = os.path.join(momentum_dir, 'bottom_10_momentum_perpetual.csv')
    
    if not os.path.exists(top_10_path):
        print(f"\nError: Cannot find {top_10_path}")
        print("Please make sure you have run ranking.py first to generate the CSV files.")
        exit(1)
    
    if not os.path.exists(bottom_10_path):
        print(f"\nError: Cannot find {bottom_10_path}")
        print("Please make sure you have run ranking.py first to generate the CSV files.")
        exit(1)
    
    # Run backtest
    results = backtest.run_backtest(
        top_10_file=top_10_path,
        bottom_10_file=bottom_10_path
    )
    
    if results is not None:
        # Calculate performance metrics
        metrics = backtest.calculate_performance_metrics(results)
        
        # Display results
        print("\n" + "="*80)
        print("BACKTEST PERFORMANCE METRICS")
        print("="*80)
        for key, value in metrics.items():
            print(f"{key:.<40} {value}")
        print("="*80)
        
        # Create comprehensive visualizations
        backtest.create_performance_report(results, output_dir=momentum_dir)
        
        # Save results to momentum folder
        results_path = os.path.join(momentum_dir, 'backtest_results.csv')
        results.to_csv(results_path)
        print(f"\n✓ Backtest results saved to: {results_path}")
        
        # Save trades
        trades_df = pd.DataFrame(backtest.trades)
        trades_path = os.path.join(momentum_dir, 'backtest_trades.csv')
        trades_df.to_csv(trades_path, index=False)
        print(f"✓ Trade log saved to: {trades_path}")
