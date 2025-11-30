"""
Gemini-powered image comparison for visual product matching.
Compares two product images to determine if they represent the same product.
"""

import base64
import os
from typing import Dict
import google.generativeai as genai
from PIL import Image
from io import BytesIO

# Configure Gemini API
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Use Gemini 2.0 Flash for image analysis (same as vision API to share quota)
vision_model = genai.GenerativeModel('gemini-2.0-flash')


async def compare_product_images(
    reference_image: bytes,
    candidate_image: bytes,
    product_description: str
) -> Dict:
    """
    Compare two product images to determine if they're the same product.
    
    Args:
        reference_image: Reference product image (from current supermarket)
        candidate_image: Candidate product image (from search results)
        product_description: Text description of the product being sought
    
    Returns:
        dict: {
            "is_same_product": bool,
            "confidence": float (0.0-1.0),
            "reasoning": str
        }
    """
    try:
        # Convert bytes to PIL Images
        ref_img = Image.open(BytesIO(reference_image))
        cand_img = Image.open(BytesIO(candidate_image))
        
        # Prepare prompt
        prompt = f"""Compare these two product images. Are they the SAME product?

Product description: {product_description}

CRITICAL MATCHING CRITERIA:
1. **Brand**: Must match exactly (e.g., "Nairn's" vs "Store Brand" = DIFFERENT)
2. **Product Type**: Must match (e.g., "Biscuits" vs "Cookies" might be same, but "Biscuits" vs "Chocolate Bar" = DIFFERENT)
3. **Flavor/Variant**: Must match (e.g., "Milk Chocolate" vs "Dark Chocolate" = DIFFERENT)
4. **Package Design**: Similar color scheme and layout (minor redesigns OK)

IGNORE THESE DIFFERENCES:
- Price tags or promotional stickers
- Size/quantity differences (200g vs 250g is acceptable if same product)
- Minor packaging updates (same design, slightly different layout)
- Shelf placement or background

Respond in STRICT JSON format:
{{
    "is_same_product": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your decision"
}}

First image: Reference product
Second image: Candidate product"""

        # Call Gemini with both images
        response = vision_model.generate_content([prompt, ref_img, cand_img])
        result_text = response.text.strip()
        
        # Clean markdown fences if present
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            lines = lines[1:]  # Remove first line
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # Remove last line
            result_text = "\n".join(lines).strip()
        
        # Parse JSON response
        import json
        result = json.loads(result_text)
        
        # Validate response structure
        if not all(k in result for k in ["is_same_product", "confidence", "reasoning"]):
            raise ValueError("Invalid response structure from vision model")
        
        return result
        
    except Exception as e:
        print(f"⚠️ Error comparing images: {e}")
        # Return conservative response on error
        return {
            "is_same_product": False,
            "confidence": 0.0,
            "reasoning": f"Error during comparison: {str(e)}"
        }


# Synchronous wrapper for backwards compatibility
def compare_product_images_sync(
    reference_image: bytes,
    candidate_image: bytes,
    product_description: str
) -> Dict:
    """Synchronous version of compare_product_images"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # If already in async context, use the async version directly
        raise RuntimeError("Use compare_product_images (async) when already in async context")
    
    return loop.run_until_complete(
        compare_product_images(reference_image, candidate_image, product_description)
    )
