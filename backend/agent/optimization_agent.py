import sys
import os

# Add backend directory to sys.path to allow importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, AsyncGenerator, Optional
import json
from typing_extensions import override
from pydantic import PrivateAttr
from google.adk.agents import Agent
from google.adk.events.event import Event
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.google_llm import Gemini
from google.genai import types
from config import OPTIMIZATION_AGENT_MODEL, USE_MEMBERSHIP_PRICE_FOR_CURRENT

class OptimizationAgent(Agent):
    _result_accumulator: Optional[List] = PrivateAttr(default=None)
    _final_summary: str = PrivateAttr(default="")
    _optimization_results: List = PrivateAttr(default_factory=list)

    def __init__(self, result_accumulator: Optional[List] = None):
        instruction = """
        You are an Optimization Agent.
        Your goal is to analyze a list of products and their prices across different supermarkets to find the best deal.
        
        You will be given:
        1. A list of products with their 'current_price' (what the user pays now).
        2. A list of 'found_prices' for each product from various supermarkets.
        **CRITICAL: USE YOUR MEMORY**
        - Use the `load_memory` tool to check for "historical low" prices for each product.
        - Compare the current best price with the historical low.
        - If the current best price is HIGHER than a historical low, you MUST add a 'historical_low_warning' field to your result.
        - The warning should be formatted like: "⚠️ Cheaper in past: £1.50 at Tesco (2025-11-20)"
        
        Output a summary of the best buying strategy, total savings, and any memory-based insights.
        """
        
        super().__init__(
            name="OptimizationAgent",
            model=OPTIMIZATION_AGENT_MODEL,
            instruction=instruction
        )
        
        self._result_accumulator = result_accumulator
        self._final_summary = ""
        self._optimization_results = []

    @property
    def final_summary(self) -> str:
        return self._final_summary

    @property
    def optimization_results(self) -> List:
        return self._optimization_results

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if not self._result_accumulator:
            print("⚠️ No results to optimize!")
            return

        # 1. Organize results by product
        # We need to reconstruct the structure expected by optimize()
        # The accumulator has flat list of search results.
        # We need to group them by product.
        
        # Get unique products from accumulator to know what we have
        products_map = {}
        for res in self._result_accumulator:
            p_name = res['product']
            if p_name not in products_map:
                # Initialize with product data from the first result that has it
                products_map[p_name] = {
                    **res['product_data'],
                    'found_prices': []
                }
            products_map[p_name]['found_prices'].append(res)
        
        products_data = list(products_map.values())
        
        # 2. Run Optimization Logic
        self._optimization_results = self.optimize(products_data)
        
        # 3. Generate Summary
        self._final_summary = await self.generate_summary(self._optimization_results)
        
        # Save to session state for main.py to pick up
        if ctx.session:
            ctx.session.state['recommendation'] = self._final_summary
            ctx.session.state['optimization_data'] = self._optimization_results
            
        # Yield summary as an event (simulating agent response)
        # This is required to make this method an async generator
        yield Event(
            author="model",
            content=types.Content(parts=[types.Part(text=self._final_summary)])
        )

    def optimize(self, products_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute best prices and compare against current supermarket.
        Returns a list of dicts suitable for CSV output with:
        - Current supermarket and prices (regular + membership if exists)
        - Cheapest option found (regular + membership if exists)
        - Savings compared to current supermarket
        """
        results = []
        for item in products_data:
            product_name = item.get("product_name")
            current_supermarket = item.get("current_supermarket", "Unknown")
            current_regular_price_str = item.get("current_regular_price", "")
            current_membership_price_str = item.get("current_membership_price", "")
            found_prices = item.get("found_prices", [])
            
            # Parse current prices
            current_regular_price = None
            current_membership_price = None
            try:
                if current_regular_price_str:
                    current_regular_price = float(str(current_regular_price_str).replace("£", ""))
            except ValueError:
                pass
            try:
                if current_membership_price_str:
                    current_membership_price = float(str(current_membership_price_str).replace("£", ""))
            except ValueError:
                pass
            
            # Gather all prices (regular and membership) from search results
            price_entries = []  # List of (supermarket, regular_price, membership_price)
            for fp in found_prices:
                if not fp.get("found"):
                    continue
                    
                regular_price = None
                membership_price = None
                
                # Parse regular price
                regular_price_str = fp.get("regular_price")
                if regular_price_str:
                    try:
                        regular_price = float(str(regular_price_str).replace("£", ""))
                    except ValueError:
                        pass
                
                # Parse membership price
                membership_price_str = fp.get("membership_price")
                if membership_price_str:
                    try:
                        membership_price = float(str(membership_price_str).replace("£", ""))
                    except ValueError:
                        pass
                
                # Only add if we have at least one price
                if regular_price is not None or membership_price is not None:
                    price_entries.append((
                        fp["supermarket"],
                        regular_price,
                        membership_price
                    ))
            
            if not price_entries:
                # No price data found, skip this product
                continue
            
            # Find the cheapest option (considering membership price if available)
            def get_best_price(entry):
                """Get the lowest price from regular or membership"""
                supermarket, regular, membership = entry
                prices = [p for p in [regular, membership] if p is not None]
                return min(prices) if prices else float('inf')
            
            cheapest_entry = min(price_entries, key=get_best_price)
            cheapest_supermarket, cheapest_regular, cheapest_membership = cheapest_entry
            cheapest_price = get_best_price(cheapest_entry)
            
            # Calculate savings vs current supermarket
            # Use config preference to determine which price to use for current supermarket
            if USE_MEMBERSHIP_PRICE_FOR_CURRENT:
                current_price = current_membership_price if current_membership_price is not None else current_regular_price
            else:
                current_price = current_regular_price
            savings_val = 0.0
            
            # Check if current price is actually better than what we found
            if current_price is not None and cheapest_price > current_price:
                # Current is cheaper!
                cheapest_supermarket = current_supermarket
                cheapest_regular = current_regular_price
                cheapest_membership = current_membership_price
                cheapest_price = current_price
                savings_val = 0.0
            elif current_price is not None:
                savings_val = max(0.0, current_price - cheapest_price)
            
            # Calculate all savings combinations
            def calc_savings(price1, price2):
                """Calculate savings, return N/A if either price is None"""
                if price1 is not None and price2 is not None:
                    # If price1 (cheapest) is actually more expensive, savings is negative
                    # But we shouldn't really be in this branch if we did the check above correctly
                    # unless we are comparing specific price types (e.g. reg vs mem)
                    return f"£{(price2 - price1):.2f}"
                return "N/A"
            
            saving_cheapest_regular_vs_current_regular = calc_savings(cheapest_regular, current_regular_price)
            saving_cheapest_regular_vs_current_membership = calc_savings(cheapest_regular, current_membership_price)
            saving_cheapest_membership_vs_current_regular = calc_savings(cheapest_membership, current_regular_price)
            saving_cheapest_membership_vs_current_membership = calc_savings(cheapest_membership, current_membership_price)
            
            # Format the result
            result = {
                "product_name": product_name,
                "current_supermarket": current_supermarket,
                "current_regular_price": f"£{current_regular_price:.2f}" if current_regular_price is not None else "N/A",
                "current_membership_price": f"£{current_membership_price:.2f}" if current_membership_price is not None else "N/A",
                "cheapest_supermarket": cheapest_supermarket,
                "cheapest_regular_price": f"£{cheapest_regular:.2f}" if cheapest_regular is not None else "N/A",
                "cheapest_membership_price": f"£{cheapest_membership:.2f}" if cheapest_membership is not None else "N/A",
                "savings_vs_current": f"£{savings_val:.2f}" if current_price is not None else "N/A",
                "saving_cheapest_regular_vs_current_regular": saving_cheapest_regular_vs_current_regular,
                "saving_cheapest_regular_vs_current_membership": saving_cheapest_regular_vs_current_membership,
                "saving_cheapest_membership_vs_current_regular": saving_cheapest_membership_vs_current_regular,
                "saving_cheapest_membership_vs_current_membership": saving_cheapest_membership_vs_current_membership
            }
            results.append(result)
            
        return results

    async def generate_summary(self, optimization_results: List[Dict[str, Any]]) -> str:
        """Generate a concise summary of total savings vs current supermarket."""
        if not optimization_results:
            return "No optimization results found."
            
        total_savings = 0.0
        for r in optimization_results:
            savings_str = r.get("savings_vs_current", "N/A")
            if savings_str != "N/A":
                try:
                    total_savings += float(savings_str.replace("£", ""))
                except ValueError:
                    pass
        
        summary = f"💰 **Total Savings vs Current Supermarket: £{total_savings:.2f}**\n\n"
        summary += "Best options per product:\n"
        for r in optimization_results:
            current_price = r.get("current_membership_price", "N/A")
            if current_price == "N/A":
                current_price = r.get("current_regular_price", "N/A")
            
            cheapest_price = r.get("cheapest_membership_price", "N/A")
            if cheapest_price == "N/A":
                cheapest_price = r.get("cheapest_regular_price", "N/A")
            
            savings = r.get("savings_vs_current", "N/A")
            
            summary += (
                f"- {r['product_name']}: Switch from {r['current_supermarket']} ({current_price}) "
                f"to {r['cheapest_supermarket']} ({cheapest_price}) - Save {savings}\n"
            )
        return summary

if __name__ == "__main__":
    import asyncio
    
    async def test_optimization_agent():
        print("🧪 Testing OptimizationAgent Standalone...")
        agent = OptimizationAgent()
        
        # Dummy Data
        products_data = [
            {
                "product_name": "Test Product A",
                "current_supermarket": "Tesco",
                "current_regular_price": "£2.00",
                "current_membership_price": "£1.80",
                "found_prices": [
                    {"supermarket": "Tesco", "found": True, "regular_price": "£2.00", "membership_price": "£1.80"},
                    {"supermarket": "Sainsburys", "found": True, "regular_price": "£1.50", "membership_price": "£1.40"},
                    {"supermarket": "Aldi", "found": True, "regular_price": "£1.20", "membership_price": None}
                ]
            },
            {
                "product_name": "Test Product B",
                "current_supermarket": "Waitrose",
                "current_regular_price": "£5.00",
                "current_membership_price": None,
                "found_prices": [
                    {"supermarket": "Waitrose", "found": True, "regular_price": "£5.00", "membership_price": None},
                    {"supermarket": "Lidl", "found": True, "regular_price": "£3.00", "membership_price": None}
                ]
            }
        ]
        
        print("\n🧠 Optimizing...")
        results = agent.optimize(products_data)
        print(json.dumps(results, indent=2))
        
        print("\n📝 Generating Summary...")
        summary = await agent.generate_summary(results)
        print("\n" + "="*40)
        print(summary)
        print("="*40)

    asyncio.run(test_optimization_agent())
