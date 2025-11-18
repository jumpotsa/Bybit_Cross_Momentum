import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BybitDataDownloader:
    """
    Downloads and stores all Bybit perpetual futures historical data to CSV files.
    Data is saved in organized folder structure for easy reuse.
    """
    
    def __init__(self, output_dir='data'):
        """
        Args:
            output_dir: Directory to store downloaded data (default: 'data')
        """
        self.output_dir = output_dir
        self.base_url = "https://api.bybit.com/v5/market/kline"
        self.metadata = []
        
        # Create directory structure
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Data directory created: {output_dir}/")
        
    def get_all_perpetual_symbols(self):
        """
        Fetches all available USDT perpetual symbols from Bybit.
        
        Returns:
            List of trading symbols
        """
        try:
            url = "https://api.bybit.com/v5/market/instruments-info"
            params = {
                'category': 'linear',
                'status': 'Trading',
                'limit': 1000
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('retCode') != 0:
                print(f"API Error: {data.get('retMsg')}")
                return []
            
            symbols = []
            for item in data.get('result', {}).get('list', []):
                symbol = item.get('symbol', '')
                # Filter for USDT perpetuals only
                if symbol.endswith('USDT') and item.get('quoteCoin') == 'USDT':
                    symbols.append(symbol)
            
            # Sort alphabetically for organized storage
            symbols.sort()
            
            print(f"✓ Found {len(symbols)} USDT perpetual contracts")
            return symbols
            
        except Exception as e:
            print(f"✗ Error fetching symbols: {e}")
            return []
    
    def fetch_historical_data(self, symbol, start_date, end_date, category='linear'):
        """
        Fetches historical daily kline data from Bybit for a single symbol.
        
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
            
            # Download data in batches
            while True:
                params['end'] = current_end
                response = requests.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if data.get('retCode') != 0:
                    break
                
                klines = data.get('result', {}).get('list', [])
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                # Check if we have all data
                if len(klines) < 1000:
                    break
                
                oldest_timestamp = int(klines[-1][0])
                if oldest_timestamp <= start_ms:
                    break
                
                current_end = oldest_timestamp
                time.sleep(0.1)  # Rate limiting
            
            if not all_klines:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Convert data types
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            df['turnover'] = pd.to_numeric(df['turnover'])
            
            # Add date column
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            # Filter to exact date range
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            return df
            
        except Exception as e:
            return None
    
    def save_symbol_data(self, symbol, df):
        """
        Saves symbol data to CSV file.
        
        Args:
            symbol: Trading symbol
            df: DataFrame with OHLCV data
        
        Returns:
            Boolean indicating success
        """
        try:
            filename = f"{symbol}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save to CSV without index
            df.to_csv(filepath, index=False)
            
            return True
            
        except Exception as e:
            print(f"✗ Error saving {symbol}: {e}")
            return False
    
    def download_all_market_data(self, start_date=None, end_date=None, 
                                  lookback_years=3, min_data_points=370):
        """
        Downloads and saves all market data to CSV files.
        
        Args:
            start_date: Start date (datetime) - if None, calculated from lookback_years
            end_date: End date (datetime) - if None, uses today
            lookback_years: Years of historical data to download
            min_data_points: Minimum data points required (default 370 for 12-1 momentum)
        
        Returns:
            Summary statistics dictionary
        """
        print("\n" + "="*80)
        print("BYBIT MARKET DATA DOWNLOADER")
        print("="*80)
        
        # Set dates
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365 * lookback_years)
        
        print(f"\nDownload Period: {start_date.date()} to {end_date.date()}")
        print(f"Target: {lookback_years} years of data")
        print(f"Minimum required data points: {min_data_points} days")
        
        # Get all symbols
        print("\nFetching available symbols...")
        all_symbols = self.get_all_perpetual_symbols()
        
        if not all_symbols:
            print("✗ No symbols found!")
            return None
        
        # Download data for each symbol
        print(f"\nDownloading data for {len(all_symbols)} symbols...")
        print("="*80)
        
        successful = 0
        failed = 0
        insufficient_data = 0
        
        for symbol in tqdm(all_symbols, desc="Progress"):
            # Fetch data
            df = self.fetch_historical_data(symbol, start_date, end_date)
            
            if df is None:
                failed += 1
                self.metadata.append({
                    'symbol': symbol,
                    'status': 'failed',
                    'data_points': 0,
                    'start_date': None,
                    'end_date': None
                })
                continue
            
            # Check if sufficient data
            if len(df) < min_data_points:
                insufficient_data += 1
                self.metadata.append({
                    'symbol': symbol,
                    'status': 'insufficient_data',
                    'data_points': len(df),
                    'start_date': df['date'].min() if len(df) > 0 else None,
                    'end_date': df['date'].max() if len(df) > 0 else None
                })
                continue
            
            # Save to CSV
            if self.save_symbol_data(symbol, df):
                successful += 1
                self.metadata.append({
                    'symbol': symbol,
                    'status': 'success',
                    'data_points': len(df),
                    'start_date': df['date'].min(),
                    'end_date': df['date'].max(),
                    'filepath': os.path.join(self.output_dir, f"{symbol}.csv")
                })
            else:
                failed += 1
                self.metadata.append({
                    'symbol': symbol,
                    'status': 'save_failed',
                    'data_points': len(df),
                    'start_date': df['date'].min() if len(df) > 0 else None,
                    'end_date': df['date'].max() if len(df) > 0 else None
                })
            
            # Rate limiting
            time.sleep(0.12)
        
        # Save metadata
        metadata_df = pd.DataFrame(self.metadata)
        metadata_path = os.path.join(self.output_dir, '_metadata.csv')
        metadata_df.to_csv(metadata_path, index=False)
        
        # Print summary
        print("\n" + "="*80)
        print("DOWNLOAD SUMMARY")
        print("="*80)
        print(f"Total symbols processed....... {len(all_symbols)}")
        print(f"✓ Successfully downloaded..... {successful}")
        print(f"✗ Failed downloads............ {failed}")
        print(f"⚠ Insufficient data........... {insufficient_data}")
        print(f"✓ Usable symbols.............. {successful}")
        print("="*80)
        print(f"\nData saved to: {os.path.abspath(self.output_dir)}/")
        print(f"Metadata saved to: {os.path.abspath(metadata_path)}")
        
        # Calculate storage size
        total_size = 0
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                if file.endswith('.csv'):
                    filepath = os.path.join(root, file)
                    total_size += os.path.getsize(filepath)
        
        size_mb = total_size / (1024 * 1024)
        print(f"Total storage used: {size_mb:.2f} MB")
        
        return {
            'total_symbols': len(all_symbols),
            'successful': successful,
            'failed': failed,
            'insufficient_data': insufficient_data,
            'storage_mb': size_mb,
            'metadata_df': metadata_df
        }
    
    def load_symbol_data(self, symbol):
        """
        Loads a single symbol's data from CSV.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            DataFrame or None if not found
        """
        filepath = os.path.join(self.output_dir, f"{symbol}.csv")
        
        if not os.path.exists(filepath):
            print(f"✗ Data file not found for {symbol}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            print(f"✗ Error loading {symbol}: {e}")
            return None
    
    def load_all_data(self):
        """
        Loads all downloaded data into memory.
        
        Returns:
            Dictionary of {symbol: DataFrame}
        """
        print(f"\nLoading all data from {self.output_dir}/...")
        
        all_data = {}
        csv_files = [f for f in os.listdir(self.output_dir) 
                    if f.endswith('.csv') and not f.startswith('_')]
        
        for filename in tqdm(csv_files, desc="Loading"):
            symbol = filename.replace('.csv', '')
            df = self.load_symbol_data(symbol)
            
            if df is not None:
                all_data[symbol] = df
        
        print(f"✓ Loaded {len(all_data)} symbols")
        return all_data
    
    def get_metadata(self):
        """
        Loads and returns metadata about downloaded symbols.
        
        Returns:
            DataFrame with metadata
        """
        metadata_path = os.path.join(self.output_dir, '_metadata.csv')
        
        if os.path.exists(metadata_path):
            return pd.read_csv(metadata_path)
        else:
            print("✗ Metadata file not found")
            return None
    
    def get_available_symbols(self):
        """
        Returns list of symbols that have been successfully downloaded.
        
        Returns:
            List of symbol names
        """
        csv_files = [f.replace('.csv', '') for f in os.listdir(self.output_dir) 
                    if f.endswith('.csv') and not f.startswith('_')]
        return sorted(csv_files)


# ====================
# USAGE EXAMPLE
# ====================

if __name__ == "__main__":
    # Initialize downloader
    downloader = BybitDataDownloader(output_dir='data')
    
    # Download all market data
    # This will download 3 years of daily data for all USDT perpetuals
    summary = downloader.download_all_market_data(
        lookback_years=5,           # Download years of data
        min_data_points=370         # Require at least 370 days of data
    )
    
    if summary:
        print("\n" + "="*80)
        print("DATA DOWNLOAD COMPLETE!")
        print("="*80)
        
        # Show successful symbols
        print("\nSuccessfully downloaded symbols:")
        available = downloader.get_available_symbols()
        print(f"Total: {len(available)} symbols")
        
        # Show first 10 as example
        if len(available) > 0:
            print("\nFirst 10 symbols:")
            for sym in available[:10]:
                print(f"  - {sym}")
        
        # Show metadata
        print("\n" + "="*80)
        print("METADATA SUMMARY")
        print("="*80)
        
        metadata = downloader.get_metadata()
        if metadata is not None:
            print("\nStatus Distribution:")
            print(metadata['status'].value_counts())
            
            print("\nData Points Statistics (for successful downloads):")
            success_data = metadata[metadata['status'] == 'success']
            if len(success_data) > 0:
                print(f"  Mean: {success_data['data_points'].mean():.0f} days")
                print(f"  Min:  {success_data['data_points'].min():.0f} days")
                print(f"  Max:  {success_data['data_points'].max():.0f} days")
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. Your data is saved in the 'data/' folder")
        print("2. Use downloader.load_symbol_data('BTCUSDT') to load specific symbol")
        print("3. Use downloader.load_all_data() to load all symbols into memory")
        print("4. Use downloader.get_available_symbols() to see available symbols")
        print("="*80)
