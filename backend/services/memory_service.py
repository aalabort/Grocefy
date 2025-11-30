import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import glob
import os

class HistoryCSVMemoryService:
    """
    A memory service that reads directly from the historical price CSVs.
    It acts as a read-only interface to the 'History' data.
    """
    
    def __init__(self, history_dir: str = "backend/data/history"):
        self.history_dir = Path(history_dir)
        
    def get_product_history(self, product_name: str) -> str:
        """
        Scans all history CSVs to find the price history for a specific product.
        Returns a summary string with the lowest recorded price.
        """
        if not self.history_dir.exists():
            return "No price history available yet."
            
        csv_files = list(self.history_dir.glob("history_*.csv"))
        if not csv_files:
            return "No price history files found."
            
        lowest_price = float('inf')
        lowest_details = None
        
        found_records = []
        
        for csv_file in csv_files:
            supermarket = csv_file.stem.replace("history_", "")
            try:
                df = pd.read_csv(csv_file)
                # Filter for rows containing the product name
                # We look for partial matches because the row key is "Name - Regular" or "Name - Membership"
                product_rows = df[df['Product'].str.contains(product_name, case=False, na=False)]
                
                if product_rows.empty:
                    continue
                    
                for _, row in product_rows.iterrows():
                    row_name = row['Product']
                    price_type = "Membership" if "Membership" in row_name else "Regular"
                    
                    # Get all price columns (excluding 'Product')
                    price_cols = [c for c in df.columns if c != 'Product']
                    
                    for date_col in price_cols:
                        price_val = row[date_col]
                        # Check if valid number
                        try:
                            price = float(price_val)
                            if pd.notna(price) and price > 0:
                                found_records.append(f"£{price:.2f} at {supermarket} ({price_type}) on {date_col}")
                                
                                if price < lowest_price:
                                    lowest_price = price
                                    lowest_details = {
                                        "price": price,
                                        "supermarket": supermarket,
                                        "date": date_col,
                                        "type": price_type
                                    }
                        except (ValueError, TypeError):
                            continue
                            
            except Exception as e:
                print(f"⚠️ Error reading history for {supermarket}: {e}")
                continue
                
        if lowest_details:
            return f"Historical Low: £{lowest_details['price']:.2f} at {lowest_details['supermarket']} ({lowest_details['type']}) on {lowest_details['date']}."
        else:
            return "No historical prices found for this product."

    # Legacy method for compatibility if needed, but we should switch to get_product_history
    def search_memory(self, query: str) -> List[Dict[str, Any]]:
        """Adapter to match old interface if needed, but returns list of text dicts"""
        summary = self.get_product_history(query)
        return [{"text": summary}]
