import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# File paths
PRODUCTS_CSV = DATA_DIR / "products.csv"
RESULTS_CSV = RESULTS_DIR / "optimization_results.csv"

# Supermarkets to check
TARGET_SUPERMARKETS = ["Tesco", "Sainsburys", "Aldi", "Lidl", "Morrisons", "Waitrose"]
# TARGET_SUPERMARKETS = ["Tesco", "Morrisons"]


# Model configuration
VISION_MODEL = "gemini-2.0-flash"  # Model for vision-based price fetching tool

# Price Preference Configuration
USE_MEMBERSHIP_PRICE_FOR_CURRENT = False  # If True, use membership price for current supermarket comparison; if False, use regular price

# Model Configuration 
VISION_AGENT_MODEL = "gemini-2.0-flash"
OPTIMIZATION_AGENT_MODEL = "gemini-2.0-flash"
COORDINATOR_MODEL = "gemini-2.0-flash" # For agents that need a model but don't do heavy lifting

# ============================================================================
# RATE LIMITING CONFIGURATION (for quota management)
# ============================================================================
# Set both flags to False when quota limits are removed for full parallel execution

# 1. Vision API Rate Limiting (controls individual vision API calls)
ENABLE_VISION_RATE_LIMITING = False  # Master flag: Enable/disable vision API rate limiting
# Vision rate limiting parameters (only used if ENABLE_VISION_RATE_LIMITING = True):
VISION_MAX_CONCURRENT_CALLS = 1     # Max concurrent vision API calls (1 = sequential, higher = more parallel)
VISION_CALL_DELAY_SECONDS = 5       # Delay between vision API calls

# 2. Batch Processing Rate Limiting (controls product batching)
ENABLE_BATCH_PROCESSING = False      # Master flag: Enable/disable batch processing
# Batch processing parameters (only used if ENABLE_BATCH_PROCESSING = True):
BATCH_SIZE = 1                      # Number of products to process per batch (e.g., 1 product × 2 supermarkets = 2 agents)
BATCH_DELAY_SECONDS = 60            # Delay in seconds between batches to respect rate limits

# ============================================================================




