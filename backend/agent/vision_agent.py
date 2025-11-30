import sys
import os

# Add backend directory to sys.path to allow imports from tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, AsyncGenerator, Optional
from typing_extensions import override
import asyncio
import re
from pydantic import PrivateAttr
from google.adk.agents import Agent
from google.adk.events.event import Event
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.google_llm import Gemini
from tools.vision_price_fetcher import vision_fetch_product, vision_fetch_product_with_visual_match
from config import VISION_AGENT_MODEL

class VisionAgent(Agent):
    _delay: int = PrivateAttr(default=0)
    _product_name: str = PrivateAttr(default="")
    _supermarket: str = PrivateAttr(default="")
    _product_data: Dict = PrivateAttr(default_factory=dict)
    _output_key: Optional[str] = PrivateAttr(default=None)
    _result_accumulator: Optional[List] = PrivateAttr(default=None)
    _reference_image_path: Optional[str] = PrivateAttr(default=None)

    def __init__(self,
                 product_name: str,
                 supermarket: str,
                 product_data: Dict,
                 delay: int = 0,
                 output_key: Optional[str] = None,
                 result_accumulator: Optional[List] = None,
                 reference_image_path: Optional[str] = None):
        
        # Check for API key
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        # Initialize tools
        # Use enhanced visual matching if reference image is available
        if reference_image_path:
            tools = [vision_fetch_product_with_visual_match]
        else:
            tools = [vision_fetch_product]

        # Instruction string
        tool_name = "vision_fetch_product_with_visual_match" if reference_image_path else "vision_fetch_product"
        
        # Build tool parameters string (no f-string interpolation here, just literals)
        if reference_image_path:
            tool_desc = f"""You have a tool `{tool_name}`. Call it with:
        - `supermarket`: "{supermarket}"
        - `query`: "{product_name}"
        - `reference_image_path`: "{reference_image_path}" (for visual comparison)"""
        else:
            tool_desc = f"""You have a tool `{tool_name}`. Call it with:
        - `supermarket`: "{supermarket}"
        - `query`: "{product_name}\""""
        
        visual_matching_note = """
        VISUAL MATCHING: This tool uses a reference image to verify product matches when names differ across supermarkets. It will try exact name matching first, then simplified name matching with visual verification.""" if reference_image_path else ""
        
        instruction = f"""
        You are a Vision-based Price Fetching Agent.
        Your goal is to find the exact price of '{product_name}' at '{supermarket}' using visual analysis.

        {tool_desc}

        When asked to find a price, always use this tool.
        {visual_matching_note}
        
        IMPORTANT - Handling Results:
        1. If the tool returns a product in 'results', report the 'regular_price' and 'membership_price' (if available).
        2. If the tool returns an 'error':
           - Check if there's a 'reason' field explaining why the product didn't match
           - Check if there are 'similar_products' - these were found but don't exactly match
           - Inform the user that the exact product wasn't found
           - If similar products exist, mention them to help the user
        3. Be clear and helpful - if the exact product isn't available, say so explicitly.
        """

        # Call parent constructor directly, passing instruction
        # Sanitize name to be a valid identifier
        safe_product_name = re.sub(r'[^a-zA-Z0-9_]', '_', product_name)
        safe_supermarket = re.sub(r'[^a-zA-Z0-9_]', '_', supermarket)
        agent_name = f"VisionAgent_{safe_supermarket}_{safe_product_name}"
        
        # Truncate if too long (optional but good practice)
        if len(agent_name) > 60:
            agent_name = agent_name[:60]
            
        super().__init__(
            name=agent_name,
            model=VISION_AGENT_MODEL,
            instruction=instruction,
            tools=tools,
            output_key=output_key
        )
        
        # Store these in private attributes AFTER super().__init__ to avoid reset
        self._product_name = product_name
        self._supermarket = supermarket
        self._product_data = product_data
        self._delay = delay
        self._output_key = output_key
        self._result_accumulator = result_accumulator
        self._reference_image_path = reference_image_path

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # 1. Handle Delay (Rate Limiting)
        if self._delay > 0:
            # print(f"⏳ {self.name} waiting {self._delay}s...")
            await asyncio.sleep(self._delay)
        
        # 2. Run Agent Logic
        final_text_parts = []
        async for event in super()._run_async_impl(ctx):
            yield event
            # Accumulate text response
            if hasattr(event, 'content') and event.content:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        final_text_parts.append(part.text)
        
        # 3. Parse and Yield Results
        final_text = "".join(final_text_parts)
        
        # Parse prices with robust regex
        regular_price = None
        membership_price = None
        
        reg_match = re.search(r'(?:Regular|Standard).*?[£\u00a3]\s*[\*]*\s*(\d+\.\d+)', final_text, re.IGNORECASE)
        mem_match = re.search(r'(?:Member|Clubcard|Nectar|Card).*?[£\u00a3]\s*[\*]*\s*(\d+\.\d+)', final_text, re.IGNORECASE)
        
        if reg_match:
            regular_price = f"£{reg_match.group(1)}"
        if mem_match:
            membership_price = f"£{mem_match.group(1)}"
            
        # Fallback
        if not regular_price and not membership_price:
            prices = re.findall(r'[£\u00a3]\s*[\*]*\s*(\d+\.\d+)', final_text)
            if prices:
                regular_price = f"£{prices[0]}"
                if len(prices) > 1 and ("member" in final_text.lower() or "clubcard" in final_text.lower()):
                    membership_price = f"£{prices[1]}"
        
        is_found = True
        if "could not find" in final_text.lower() and not regular_price:
            is_found = False
        
        result = {
            "product": self._product_name,
            "supermarket": self._supermarket,
            "found": is_found,
            "regular_price": regular_price,
            "membership_price": membership_price,
            "product_data": self._product_data
        }
        
        # Store result in accumulator if provided
        if self._result_accumulator is not None:
            self._result_accumulator.append(result)
        else:
            print(f"WARNING: VisionAgent {self.name} has NO accumulator!")
        
        # Also store as instance variable for direct access
        self._result = result



if __name__ == "__main__":
    import asyncio
    from google.adk.runners import InMemoryRunner
    import re

    async def test_vision_agent():
        print("🧪 Testing VisionAgent Standalone...")
        # Provide dummy product data (empty dict) and no output_key for manual testing
        agent = VisionAgent(product_name="Dummy", supermarket="Dummy", product_data={}, delay=0)
        runner = InMemoryRunner(agent=agent)
        
        product = "Tunnocks Milk Chocolate Caramel Wafer Biscuits"
        supermarket = "Tesco"
        query = f"What is the price of {product} at {supermarket}?"
        
        print(f"🤖 Searching for: {product} at {supermarket}")
        
        try:
            events = await runner.run_debug(
                user_messages=query,
                user_id="test_user",
                verbose=False
            )
            
            # Extract response
            final_text_parts = []
            for event in events:
                if hasattr(event, 'content') and event.content:
                    author = getattr(event, 'author', None)
                    if True: # Relaxed check
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                final_text_parts.append(part.text)
            
            final_text = "".join(final_text_parts)
            print(f"\n📝 Agent Response:\n{final_text}")
            
            # Test Regex
            reg_match = re.search(r'(?:Regular|Standard).*?[£\u00a3]\s*[\*]*\s*(\d+\.\d+)', final_text, re.IGNORECASE)
            if reg_match:
                print(f"\n✅ Extracted Regular Price: £{reg_match.group(1)}")
            else:
                # Fallback
                prices = re.findall(r'[£\u00a3]\s*[\*]*\s*(\d+\.\d+)', final_text)
                if prices:
                    print(f"\n✅ Extracted Price (Fallback): £{prices[0]}")
                else:
                    print("\n❌ Could not extract price.")

        except Exception as e:
            print(f"\n❌ Error: {e}")

    asyncio.run(test_vision_agent())

