import pandas as pd
import requests
import os
from datetime import datetime

# --- CONFIGURATION ---
# 1. List of stablecoins to exclude from the analysis
STABLECOINS = {
    'usdt', 'usdc', 'busd', 'dai', 'fdusd', 'tusd',
    'usdd', 'usdp', 'paxg', 'gusd', 'pyusd'
}

# 2. Output folder
OUTPUT_DIR = "momentum"
# --- END CONFIGURATION ---

def get_bybit_perpetual_tickers():
    """
    Fetches all perpetual contract tickers from Bybit API.
    Returns data for both linear (USDT) and inverse perpetual contracts.
    """
    print("Fetching perpetual contracts from Bybit API...")
    
    all_contracts = []
    
    # Fetch linear perpetual contracts (USDT perpetuals)
    try:
        url = 'https://api.bybit.com/v5/market/tickers'
        params = {'category': 'linear'}
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('retCode') == 0:
            linear_list = data.get('result', {}).get('list', [])
            print(f"Found {len(linear_list)} linear perpetual contracts")
            all_contracts.extend(linear_list)
        else:
            print(f"Error fetching linear contracts: {data.get('retMsg')}")
    except Exception as e:
        print(f"Error fetching linear contracts from Bybit: {e}")
    
    # Fetch inverse perpetual contracts
    try:
        url = 'https://api.bybit.com/v5/market/tickers'
        params = {'category': 'inverse'}
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('retCode') == 0:
            inverse_list = data.get('result', {}).get('list', [])
            print(f"Found {len(inverse_list)} inverse perpetual contracts")
            all_contracts.extend(inverse_list)
        else:
            print(f"Error fetching inverse contracts: {data.get('retMsg')}")
    except Exception as e:
        print(f"Error fetching inverse contracts from Bybit: {e}")
    
    return all_contracts

def process_bybit_data(contracts):
    """
    Process Bybit contracts data and filter for perpetual contracts only.
    Calculate volume in USD and sort by volume.
    """
    print(f"Processing {len(contracts)} total contracts...")
    
    coins = []
    
    for item in contracts:
        symbol = item.get('symbol', '')
        
        # Filter for perpetual contracts only (exclude futures with expiry dates)
        # Perpetual contracts typically don't have deliveryTime or have it as "0"
        delivery_time = item.get('deliveryTime', '0')
        if delivery_time != '0' and delivery_time != '':
            continue  # Skip futures contracts with delivery date
        
        # Extract base symbol (e.g., BTC from BTCUSDT or BTCUSD)
        base_symbol = ''
        if 'USDT' in symbol:
            base_symbol = symbol.replace('USDT', '').lower()
        elif 'USDC' in symbol:
            base_symbol = symbol.replace('USDC', '').lower()
        elif 'USD' in symbol and symbol.endswith('USD'):
            base_symbol = symbol.replace('USD', '').lower()
        else:
            # Skip non-standard symbols
            continue
        
        # Skip stablecoins
        if base_symbol in STABLECOINS:
            continue
        
        # Skip test or duplicate symbols
        if not base_symbol or len(base_symbol) < 2:
            continue
        
        # Get volume in USD (turnover24h is in USD)
        try:
            turnover_24h = float(item.get('turnover24h', 0))
            volume_24h = float(item.get('volume24h', 0))
            last_price = float(item.get('lastPrice', 0))
            price_change_24h = float(item.get('price24hPcnt', 0)) * 100  # Convert to percentage
            
            if turnover_24h > 0:
                coins.append({
                    'symbol': base_symbol.upper(),
                    'contract': symbol,
                    'last_price': last_price,
                    'volume_24h': volume_24h,
                    'turnover_usd_24h': turnover_24h,
                    'price_change_percentage_24h': price_change_24h
                })
        except (ValueError, TypeError):
            continue
    
    # Sort by turnover (volume in USD) descending
    coins_sorted = sorted(coins, key=lambda x: x['turnover_usd_24h'], reverse=True)
    
    # Remove duplicates - keep the contract with highest volume for each base symbol
    seen_symbols = set()
    unique_coins = []
    for coin in coins_sorted:
        if coin['symbol'] not in seen_symbols:
            seen_symbols.add(coin['symbol'])
            unique_coins.append(coin)
    
    print(f"Found {len(unique_coins)} unique non-stablecoin perpetual contracts.")
    
    # Return top 100
    return unique_coins[:100]

def save_to_csv(coins, filename="top_100_coins.csv"):
    """
    Saves the coin data to a CSV file in the momentum folder.
    """
    if not coins:
        print("No coin data to save.")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")
    
    # Convert to DataFrame
    df = pd.DataFrame(coins)
    
    # Save to CSV
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(coins)} coins to: {filepath}")
    
    # Also save just the symbols to a separate file
    symbols_df = df[['symbol', 'contract']]
    symbols_filepath = os.path.join(OUTPUT_DIR, "coin_names_only.csv")
    symbols_df.to_csv(symbols_filepath, index=False)
    print(f"Saved coin symbols only to: {symbols_filepath}")
    
    return filepath

def main():
    """
    Main function to fetch and save top 100 coins by volume from Bybit perpetual contracts.
    """
    print("="*60)
    print("BYBIT PERPETUAL CONTRACTS FETCHER")
    print("Top 100 by 24h Volume (USD)")
    print("="*60)
    
    # Fetch all perpetual contracts from Bybit
    contracts = get_bybit_perpetual_tickers()
    
    if contracts:
        # Process and filter data
        coins = process_bybit_data(contracts)
        
        if coins:
            # Save to CSV
            save_to_csv(coins)
            
            # Display summary
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"Total coins fetched: {len(coins)}")
            print("\nTop 10 coins by 24h USD volume:")
            for i, coin in enumerate(coins[:10], 1):
                volume_millions = coin['turnover_usd_24h'] / 1_000_000
                print(f"  {i}. {coin['symbol']} ({coin['contract']}) - Volume: ${volume_millions:.2f}M")
            print("="*60)
        else:
            print("\nNo valid perpetual contracts found after filtering.")
    else:
        print("\nFailed to fetch contract data from Bybit. Please try again later.")

if __name__ == "__main__":
    main()
