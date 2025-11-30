# Dynamic Price Search Coordinator 
# ---------------------------------------------------
# This module provides a factory function that builds a **per‑product**
# pipeline using result_accumulator for data sharing:
#   * VisionAgents store results in shared result_accumulator list
#   * A ParallelAgent runs all VisionAgents for a product
#   * An OptimizationAgent consumes results from the same accumulator
#   * The whole thing is wrapped in a SequentialAgent so it can be
#     executed by the runner.
#
# The number of supermarkets and products is **dynamic** – they come
# from `config.TARGET_SUPERMARKETS` and the CSV input.
#
# Usage (from `main.py`):
#   for product in products:
#       coordinator, results = create_price_search_pipeline(product, TARGET_SUPERMARKETS)
#       await runner.run_debug(...)
#       # Access results from the returned list
#
# The function returns a tuple: (SequentialAgent, result_accumulator)

from typing import List, Dict
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from config import VISION_AGENT_MODEL, OPTIMIZATION_AGENT_MODEL
from .vision_agent import VisionAgent
from .optimization_agent import OptimizationAgent

def _make_output_key(supermarket: str) -> str:
    """Create a safe output key for a supermarket.
    The key is lower‑cased and non‑alphanumeric characters are replaced
    with underscores so it can be used in the session state.
    """
    import re
    safe = re.sub(r"[^a-zA-Z0-9]", "_", supermarket.lower())
    return f"price_{safe}"

from services.memory_service import HistoryCSVMemoryService

def create_price_search_pipeline(product: Dict, supermarkets: List[str], reference_image_path: str = None):
    """Build a per‑product pipeline.

    Parameters
    ----------
    product: dict
        A row from the CSV (must contain at least ``product_name``).
    supermarkets: list[str]
        The list of supermarket names from the config.
    reference_image_path: str, optional
        Path to reference image for visual comparison

    Returns
    -------
    tuple
        (SequentialAgent, result_accumulator, OptimizationAgent) - The coordinator, shared results, and optimizer instance
    """
    product_name = product["product_name"]
    
    # Sanitize product name for agent name (alphanumeric + underscore only)
    import re
    safe_product_name = re.sub(r'[^a-zA-Z0-9_]', '_', product_name)
    # Remove multiple underscores
    safe_product_name = re.sub(r'_+', '_', safe_product_name).strip('_')
    
    # Shared accumulator for results
    result_accumulator = []

    # Initialize Memory Service
    memory_service = HistoryCSVMemoryService()

    # Define tool for agents to access memory
    def load_memory(query: str) -> str:
        """Search historical price data for this product to find past low prices."""
        # The query is likely the product name, but we can pass the product_name directly if needed
        # For now, let's assume the agent asks about the product
        return memory_service.get_product_history(product_name)

    # Create Vision Agents for each supermarket
    vision_agents = []
    for supermarket in supermarkets:
        vision_agents.append(
            VisionAgent(
                product_name=product_name, # Pass original query
                supermarket=supermarket,
                product_data=product,      # Pass full product data
                result_accumulator=result_accumulator,
                reference_image_path=reference_image_path  # NEW: for visual matching
            )
        )

    # Parallel Agent to run searches concurrently
    parallel_search = ParallelAgent(
        name=f"PriceSearch_{safe_product_name}",
        sub_agents=vision_agents,
    )
    # Optimization Agent to analyze results
    optimizer = OptimizationAgent(
        result_accumulator=result_accumulator
    )
    # 4️⃣ SequentialAgent: run parallel search, then optimizer.
    # ------------------------------------------------------------------
    coordinator = SequentialAgent(
        name=f"Coordinator_{safe_product_name}",
        sub_agents=[parallel_search, optimizer],
    )

    return coordinator, result_accumulator, optimizer
