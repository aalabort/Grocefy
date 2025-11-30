import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import os

class HistoricalPriceTracker:
    def __init__(self, history_dir: str = "backend/data/history"):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def update_history(self, products_csv_path: str, search_results: List[Dict[str, Any]]):
        """
        Update history with results from the current run.
        Creates/Updates a separate CSV for each supermarket found in results.
        
        Args:
            products_csv_path: Path to the master products.csv file
            search_results: List of search result dictionaries from the current run
        """
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 1. Get master list of products
        try:
            master_products_df = pd.read_csv(products_csv_path)
            master_product_names = master_products_df['product_name'].unique()
        except Exception as e:
            print(f"⚠️ Error reading products.csv: {e}")
            return

        # 2. Group results by supermarket
        print(f"   🔍 Tracker received {len(search_results)} search results")
        results_by_market = {}
        for res in search_results:
            market = res.get('supermarket')
            if market:
                if market not in results_by_market:
                    results_by_market[market] = []
                results_by_market[market].append(res)
        
        print(f"   🔍 Found supermarkets: {list(results_by_market.keys())}")
        
        # 3. Process each supermarket
        for market, results in results_by_market.items():
            self._update_supermarket_history(market, results, master_product_names, today)

    def _update_supermarket_history(self, market: str, results: List[Dict[str, Any]], master_product_names: List[str], date_col: str):
        """Update the history file for a specific supermarket"""
        file_path = self.history_dir / f"history_{market}.csv"
        
        # Load or Create DataFrame
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                if 'Product' in df.columns:
                    df.set_index('Product', inplace=True)
            except Exception:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()
            
        # Ensure all master products have rows for Regular and Membership
        # We need 2 rows per product
        required_rows = []
        for p_name in master_product_names:
            required_rows.append(f"{p_name} - Regular")
            required_rows.append(f"{p_name} - Membership")
            
        # Add missing rows
        new_rows = [r for r in required_rows if r not in df.index]
        if new_rows:
            new_df = pd.DataFrame(index=new_rows)
            df = pd.concat([df, new_df])
            
        # Ensure date column exists
        if date_col not in df.columns:
            df[date_col] = None  # Initialize with None/NaN

        # Update prices
        updates = 0
        for res in results:
            p_name = res.get('product')
            if not p_name:
                continue
                
            reg_price = res.get('regular_price')
            mem_price = res.get('membership_price')
            
            # Clean prices (remove currency symbol)
            if reg_price and isinstance(reg_price, str):
                reg_price = reg_price.replace('£', '')
            if mem_price and isinstance(mem_price, str):
                mem_price = mem_price.replace('£', '')
                
            # Update Regular Price
            row_key_reg = f"{p_name} - Regular"
            if row_key_reg in df.index and reg_price:
                df.at[row_key_reg, date_col] = reg_price
                updates += 1
                
            # Update Membership Price
            row_key_mem = f"{p_name} - Membership"
            if row_key_mem in df.index and mem_price:
                df.at[row_key_mem, date_col] = mem_price
                updates += 1
                
        # Save back to CSV
        try:
            df.sort_index(inplace=True) # Sort rows alphabetically
            df.reset_index().rename(columns={'index': 'Product'}).to_csv(file_path, index=False)
            print(f"   💾 Updated {market} history with {updates} prices")
        except Exception as e:
            print(f"❌ Error saving {market} history: {e}")
