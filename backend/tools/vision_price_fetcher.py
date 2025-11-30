"""
Vision-based price fetcher - Proof of Concept for Tesco

Uses Gemini's vision capabilities to:
1. Navigate to Tesco website
2. Take screenshots
3. Visually identify products and prices
4. Handle cookies/popups by "seeing" them
"""

import asyncio
import base64
import os
import sys
from io import BytesIO
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from playwright.async_api import async_playwright

# Add backend directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    VISION_MODEL,
    ENABLE_VISION_RATE_LIMITING,
    VISION_MAX_CONCURRENT_CALLS,
    VISION_CALL_DELAY_SECONDS
)
from utils.image_store import save_product_image, get_product_image
from utils.name_simplifier import simplify_product_name

import cv2
import numpy as np
from difflib import SequenceMatcher
from typing import Optional

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize vision model using configured model
vision_model = genai.GenerativeModel(VISION_MODEL)

# Initialize semaphore for vision API rate limiting
# This limits concurrent vision API calls to avoid quota errors
# Set ENABLE_VISION_RATE_LIMITING=False in config.py to disable
_vision_api_semaphore = asyncio.Semaphore(VISION_MAX_CONCURRENT_CALLS) if ENABLE_VISION_RATE_LIMITING else None


async def take_screenshot(page) -> tuple[bytes, str]:
    """Take a screenshot and return bytes + base64 encoding"""
    screenshot_bytes = await page.screenshot(timeout=60000)
    base64_image = base64.b64encode(screenshot_bytes).decode()
    return screenshot_bytes, base64_image


async def ask_vision_model(screenshot_bytes: bytes, prompt: str) -> str:
    """Ask Gemini to analyze a screenshot"""
    from PIL import Image
    
    # Convert bytes to PIL Image
    image = Image.open(BytesIO(screenshot_bytes))
    
    # Ask Gemini with retry logic
    response_text = await generate_content_with_retry(prompt, image)
    return response_text


async def generate_content_with_retry(prompt, image) -> str:
    """
    Helper to call Gemini with retry logic for rate limits.
    
    When ENABLE_VISION_RATE_LIMITING is True, this function:
    - Limits concurrent calls using a semaphore
    - Adds delays between calls to avoid quota errors
    
    When ENABLE_VISION_RATE_LIMITING is False:
    - Runs at full speed with no artificial delays
    """
    import time
    from google.api_core import exceptions
    
    # Acquire semaphore if rate limiting is enabled
    if _vision_api_semaphore:
        async with _vision_api_semaphore:
            return await _generate_content_with_retry_impl(prompt, image)
    else:
        # No rate limiting - run at full speed
        return await _generate_content_with_retry_impl(prompt, image)


async def _generate_content_with_retry_impl(prompt, image) -> str:
    """Internal implementation of generate_content with retry logic"""
    import time
    from google.api_core import exceptions
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Run the synchronous API call in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: vision_model.generate_content([prompt, image])
            )
            
            # Add delay after successful call if rate limiting is enabled
            if ENABLE_VISION_RATE_LIMITING and VISION_CALL_DELAY_SECONDS > 0:
                await asyncio.sleep(VISION_CALL_DELAY_SECONDS)
            
            return response.text
        except exceptions.ResourceExhausted:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 20  # 20s, 40s, 60s
                print(f"   ⚠️ Quota exceeded, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise
        except Exception as e:
            print(f"   ⚠️ Gemini error: {e}")
            raise
    return ""




async def crop_to_product_image(product_name: str, screenshot_bytes: bytes) -> Optional[bytes]:
    """
    Intelligently crop a product package image from a screenshot using Gemini Vision AI.
    
    This function is designed to be called by LLMs and automated systems. It uses Gemini
    to directly identify and crop product packages from grocery website screenshots,
    eliminating the need for separate OCR and validation steps.
    
    The function asks Gemini to:
    1. Locate the product package for the given name in the screenshot
    2. Provide precise crop coordinates for the package
    3. Validate that the region contains an actual product package
    
    Args:
        product_name: Name of the product to find and crop (e.g., "Ferrero Raffaello 230G")
        screenshot_bytes: Raw bytes of a PNG/JPEG screenshot from a grocery website
    
    Returns:
        Optional[bytes]: PNG-encoded bytes of the cropped product package image, or None if:
            - Product not found in screenshot
            - No valid package region identified
            - Image processing fails
    
    Example:
        >>> screenshot = Path("tesco_search.png").read_bytes()
        >>> package_img = await crop_to_product_image("Nakd Cocoa Bar", screenshot)
        >>> if package_img:
        >>>     Path("nakd_package.png").write_bytes(package_img)
    
    Technical Details:
        - Uses Gemini Vision API for product detection and localization
        - Tries multiple crop strategies (tight, medium, wide) if initial crop fails validation
        - Returns the best validated crop containing the product package
    """
    from PIL import Image
    import json
    
    # Load screenshot
    arr = np.frombuffer(screenshot_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_height, img_width = img.shape[:2]
    
    # Convert to PIL Image for Gemini
    screenshot_image = Image.open(BytesIO(screenshot_bytes))
    
    print(f"   🤖 Using Gemini Vision to locate and crop product package...")
    
    # Ask Gemini to find the product and provide crop coordinates
    localization_prompt = f"""
You are analyzing a grocery website screenshot to find a product package IMAGE.

PRODUCT TO FIND: "{product_name}"

TASK:
Find the CENTER POINT of the PRODUCT PACKAGE IMAGE ONLY.
This is the actual photo/image of the physical product (box, bottle, bag, wrapper).

IMPORTANT - EXCLUDE:
- Product name text on the website
- Price labels on the website  
- "Add to basket" buttons
- Star ratings or reviews
- Any website UI elements

INCLUDE ONLY:
- The actual product package photograph/image

Respond with ONLY a JSON object:
{{
    "found": true or false,
    "confidence": 0.0 to 1.0,
    "product_visible": "brief description",
    "center_point": {{
        "y": vertical center (0-1000),
        "x": horizontal center (0-1000)
    }},
    "reasoning": "why this is the center of the product image"
}}

Use NORMALIZED coordinates (0-1000) where 0,0 is top-left.
Image dimensions: {img_width}x{img_height} pixels.
"""
    
    try:
        response_text = await generate_content_with_retry(localization_prompt, screenshot_image)
        response_text = response_text.strip()
        
        # Clean markdown fences
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        
        print(f"   📊 Gemini Analysis:")
        print(f"      Found: {result.get('found', False)}")
        print(f"      Confidence: {result.get('confidence', 0):.2f}")
        print(f"      Product visible: {result.get('product_visible', 'N/A')}")
        print(f"      Reasoning: {result.get('reasoning', 'N/A')}")
        
        if not result.get("found", False) or result.get("confidence", 0) < 0.5:
            print(f"   ❌ Product not found or low confidence")
            return None
        
        # Extract center point
        center = result.get("center_point", {})
        center_y = int(center.get("y", 500))
        center_x = int(center.get("x", 500))
        
        # Convert to pixels
        center_x_px = int(center_x / 1000 * img_width)
        center_y_px = int(center_y / 1000 * img_height)
        
        # Create a MEDIUM-LARGE coarse crop around the center (800x800)
        # This is a compromise: large enough to capture product, small enough to reduce noise
        crop_size = 800
        half_size = crop_size // 2
        
        x = max(0, center_x_px - half_size)
        y = max(0, center_y_px - half_size)
        width = min(crop_size, img_width - x)
        height = min(crop_size, img_height - y)
        
        print(f"   🐛 DEBUG: Center point (normalized): y={center_y}, x={center_x}")
        print(f"   🐛 DEBUG: Center point (pixels): y={center_y_px}, x={center_x_px}")
        print(f"   🐛 DEBUG: Coarse Crop: x={x}, y={y}, width={width}, height={height}")
        
        print(f"   📐 Coarse crop region (pixels): x={x}, y={y}, width={width}, height={height}")
        
        # Try multiple zoom levels if validation fails (centered on the padded crop)
        # We use the calculated pixel coordinates as the base
        base_x, base_y, base_w, base_h = x, y, width, height
        zoom_levels = [1.0, 1.2, 0.8] # Normal, zoomed out (more context), zoomed in (tighter)

        for i, zoom in enumerate(zoom_levels):
            print(f"   [Attempt {i+1}/{len(zoom_levels)}] Cropping with zoom {zoom}x...")
            
            # Calculate new dimensions based on zoom
            new_width = int(base_w * zoom)
            new_height = int(base_h * zoom)
            
            # Center the new crop on the base crop
            center_x = base_x + base_w / 2
            center_y = base_y + base_h / 2
            
            new_x = int(center_x - new_width / 2)
            new_y = int(center_y - new_height / 2)
            
            # Validate coordinates
            x_curr = max(0, min(new_x, img_width - 1))
            y_curr = max(0, min(new_y, img_height - 1))
            w_curr = max(1, min(new_width, img_width - x_curr))
            h_curr = max(1, min(new_height, img_height - y_curr))

            print(f"   📐 Attempt region: x={x_curr}, y={y_curr}, width={w_curr}, height={h_curr}")

            # Perform crop
            cropped = img[y_curr:y_curr+h_curr, x_curr:x_curr+w_curr]

            if cropped.size == 0:
                print(f"   ❌ Crop resulted in empty image at zoom {zoom}x")
                continue

            # Encode to PNG
            _, buffer = cv2.imencode(".png", cropped)
            crop_bytes = buffer.tobytes()

            # --- STEP 2: REFINE CROP (Two-Pass Strategy) ---
            print(f"   🔨 Refinement: Optimizing crop for '{product_name}'...")
            refined_bytes = await refine_crop(product_name, crop_bytes)
            
            if refined_bytes:
                print(f"   ✨ Refinement successful")
                final_bytes = refined_bytes
            else:
                print(f"   ⚠️ Refinement failed, using original crop")
                final_bytes = crop_bytes
            
            # --- STEP 3: VALIDATION ---
            crop_image = Image.open(BytesIO(final_bytes))
            
            validation_prompt = f"""
Look at this cropped image.

Does this show a PRODUCT PACKAGE (box, bottle, bag, container, wrapper) for "{product_name}"?

Respond with ONLY a JSON object:
{{
    "contains_package": true or false,
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}

Be strict: the image must clearly show the actual physical package/container.
"""
            
            validation_response_text = await generate_content_with_retry(validation_prompt, crop_image)
            validation_text = validation_response_text.strip()
            
            # Clean markdown fences
            if validation_text.startswith("```"):
                lines = validation_text.split("\n")
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                validation_text = "\n".join(lines).strip()
            
            validation = json.loads(validation_text)
            
            contains_package = validation.get("contains_package", False)
            val_confidence = validation.get("confidence", 0.0)
            val_reasoning = validation.get("reasoning", "")
            
            print(f"   ✓ Validation: Package={'✅' if contains_package else '❌'}, Confidence={val_confidence:.2f}")
            print(f"      {val_reasoning}")
            
            if contains_package and val_confidence >= 0.75:
                print(f"   ✅ Successfully cropped and validated product package")
                return final_bytes
            else:
                print(f"   ❌ Validation failed at zoom {zoom}x - crop doesn't contain a valid package")
        
        # If all zoom levels fail
        print(f"   ❌ All cropping attempts failed validation")
        return None
            
    except json.JSONDecodeError as e:
        print(f"   ❌ Failed to parse Gemini response: {e}")
        print(f"      Response: {response_text[:200]}...")
        return None
    except Exception as e:
        print(f"   ❌ Error during Gemini processing: {e}")
        import traceback
        traceback.print_exc()
        return None


async def refine_crop(product_name: str, image_bytes: bytes) -> Optional[bytes]:
    """
    Pass 2: Take a rough crop and ask Gemini to crop it TIGHTLY to the product package.
    This eliminates surrounding whitespace or other elements.
    """
    from PIL import Image
    import json
    
    try:
        # Load image
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img_height, img_width = img.shape[:2]
        
        pil_image = Image.open(BytesIO(image_bytes))
        
        refine_prompt = f"""
You are refining a product image crop.
The image shows a product package for "{product_name}", but it might have extra whitespace or background.

TASK:
Identify the bounding box around the PRODUCT PACKAGE IMAGE ONLY.
This is the actual photograph/image of the physical product (box, bottle, bag, container).

CRITICAL - EXCLUDE from the crop:
- Product name text displayed on the website
- Price text on the website
- Website buttons or UI elements
- Star ratings or review text
- Any text that is NOT part of the actual product packaging

INCLUDE in the crop:
- The complete product package photograph
- Text that is printed ON the actual product packaging (this is OK)

IMPORTANT: Include the ENTIRE package image - do not cut off any edges of the product photo.
It's better to include a bit of extra whitespace than to cut off part of the package or include website text.

Respond with ONLY a JSON object:
{{
    "found": true,
    "crop_region": {{
        "ymin": top edge (0-1000),
        "xmin": left edge (0-1000),
        "ymax": bottom edge (0-1000),
        "xmax": right edge (0-1000)
    }}
}}

Use NORMALIZED coordinates (0-1000) relative to this image.
"""
        text = await generate_content_with_retry(refine_prompt, pil_image)
        text = text.strip()
        
        # Clean markdown
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        
        result = json.loads(text)
        
        if not result.get("found", False):
            return None
            
        crop = result.get("crop_region", {})
        ymin = int(crop.get("ymin", 0))
        xmin = int(crop.get("xmin", 0))
        ymax = int(crop.get("ymax", 1000))
        xmax = int(crop.get("xmax", 1000))
        
        # Convert to pixels
        x = int(xmin / 1000 * img_width)
        y = int(ymin / 1000 * img_height)
        w = int((xmax - xmin) / 1000 * img_width)
        h = int((ymax - ymin) / 1000 * img_height)
        
        # Add 5% padding to ensure we don't cut off edges
        padding_x = int(w * 0.05)
        padding_y = int(h * 0.05)
        
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w = min(img_width - x, w + 2 * padding_x)
        h = min(img_height - y, h + 2 * padding_y)
        
        # Validate
        if w < 10 or h < 10:
            return None
            
        # Crop
        refined = img[y:y+h, x:x+w]
        
        # Encode
        _, buffer = cv2.imencode(".png", refined)
        refined_bytes = buffer.tobytes()
        
        # --- DOUBLE CHECK: Verify and Clean ---
        # The user specifically requested a "double check" to remove UI/text.
        # We send the refined crop back to Gemini to check for unwanted elements.
        
        print(f"   🕵️ Double-checking crop for UI/text artifacts...")
        
        clean_prompt = f"""
You are a Quality Assurance AI for product images.
Analyze this cropped image of "{product_name}".

CHECK FOR:
1. Website UI elements (search bars, buttons, icons)
2. Text *outside* the product package (price tags, product names, reviews)
3. White borders or background noise

DECISION:
- If the image contains ONLY the product package (and maybe its shadow), return "CLEAN".
- If the image contains UI, external text, or excessive background, return "DIRTY" and provide the TIGHT bounding box for the actual product package.

Respond with ONLY a JSON object:
{{
    "status": "CLEAN" or "DIRTY",
    "clean_crop": {{
        "ymin": top edge (0-1000),
        "xmin": left edge (0-1000),
        "ymax": bottom edge (0-1000),
        "xmax": right edge (0-1000)
    }},
    "reason": "why it is clean or dirty"
}}

Use NORMALIZED coordinates (0-1000) relative to THIS image.
"""
        
        try:
            pil_refined = Image.open(BytesIO(refined_bytes))
            clean_text = await generate_content_with_retry(clean_prompt, pil_refined)
            clean_text = clean_text.strip()
            
            if clean_text.startswith("```"):
                clean_text = clean_text.split("```")[1]
                if clean_text.startswith("json"):
                    clean_text = clean_text[4:]
            clean_text = clean_text.strip()
            
            clean_result = json.loads(clean_text)
            print(f"   🕵️ QA Result: {clean_result.get('status')} - {clean_result.get('reason')}")
            
            if clean_result.get("status") == "DIRTY":
                # Perform the cleanup crop
                clean_box = clean_result.get("clean_crop", {})
                cy_min = int(clean_box.get("ymin", 0))
                cx_min = int(clean_box.get("xmin", 0))
                cy_max = int(clean_box.get("ymax", 1000))
                cx_max = int(clean_box.get("xmax", 1000))
                
                # Convert to pixels (relative to the refined crop)
                ref_h, ref_w = refined.shape[:2]
                
                nx = int(cx_min / 1000 * ref_w)
                ny = int(cy_min / 1000 * ref_h)
                nw = int((cx_max - cx_min) / 1000 * ref_w)
                nh = int((cy_max - cy_min) / 1000 * ref_h)
                
                # Validate new crop
                if nw > 10 and nh > 10:
                    print(f"   ✨ Cleaning up image (removing UI/text)...")
                    final_img = refined[ny:ny+nh, nx:nx+nw]
                    _, final_buffer = cv2.imencode(".png", final_img)
                    return final_buffer.tobytes()
                else:
                    print("   ⚠️ Cleanup crop too small, keeping original.")
                    return refined_bytes
            else:
                return refined_bytes
                
        except Exception as e:
            print(f"   ⚠️ Double-check failed: {e}, using refined crop.")
            return refined_bytes

    except Exception as e:
        print(f"      ⚠️ Refinement error: {e}")
        return None



async def capture_reference_image(supermarket: str, query: str) -> str:
    """
    Capture and save a reference image of the product from the given supermarket.
    This image will be used for visual comparison when searching other supermarkets.
    
    Strategy:
    1. Try exact name search
    2. If that fails, try simplified name
    3. Save image from FIRST result found (best effort - we'll verify later during comparison)
    
    Args:
        supermarket: Name of the supermarket
        query: Product to find
    
    Returns:
        str: Path to the saved reference image, or empty string if nothing found
    """
    print(f"\n📸 Capturing reference image for: {query} at {supermarket}")
    
    # Try exact name first
    result = await vision_fetch_product(supermarket, query)
    
    # Check if we got any results (exact or similar)
    has_results = (result.get("results") and len(result["results"]) > 0) or \
                  (result.get("similar_products") and len(result["similar_products"]) > 0)
    
    if not has_results:
        # Try simplified name as fallback
        simplified = simplify_product_name(query)
        if simplified != query:
            print(f"   Trying simplified name: '{simplified}'...")
            result = await vision_fetch_product(supermarket, simplified)
            has_results = (result.get("results") and len(result["results"]) > 0) or \
                         (result.get("similar_products") and len(result["similar_products"]) > 0)
    
    if not has_results:
        print(f"❌ No products found at {supermarket}. Cannot capture reference image.")
        return ""
    
    # We found SOMETHING! Let's save it
    # The screenshot exists at this point
    supermarket_key = supermarket.lower().strip()
    debug_screenshot = f"backend/debug/debug_vision_{supermarket_key}.png"
    
    if not Path(debug_screenshot).exists():
        print(f"⚠️  Screenshot file not found, cannot save reference image")
        return ""
    
    # Read the full screenshot
    with open(debug_screenshot, 'rb') as f:
        full_screenshot = f.read()
    
    # Crop to just the product package image
    print(f"   🔍 Extracting product package from screenshot...")
    try:
        product_image = await crop_to_product_image(query, full_screenshot)
    except Exception as e:
        print(f"   ⚠️  Cropping failed: {e}")
        print(f"   Using full screenshot as reference")
        product_image = full_screenshot
    
    # Save the product image as reference, only if cropping was successful
    if product_image:
        ref_path = save_product_image(query, supermarket, product_image)
        print(f"✅ Reference image saved: {ref_path}")
        
        # Show what we captured
        if result.get("results"):
            product_name = result["results"][0].get("name", "Unknown")
            print(f"   Captured from: {product_name}")
        elif result.get("similar_products"):
            product_name = result["similar_products"][0].get("name", "Unknown")
            print(f"   Captured from similar product: {product_name}")
        
        return ref_path
    else:
        print(f"❌ Could not generate a valid product image for '{query}'")
        return ""



async def vision_fetch_product(supermarket: str, query: str) -> dict:
    """
    Fetch prices from a supermarket using vision-based navigation
    
    Args:
        supermarket: Name of the supermarket (e.g., "Tesco", "Sainsbury's", "Aldi")
        query: Product to search for
    """
    print(f"\n🔍 Vision-based search for: {query} at {supermarket}")
    
    # Supermarket URL mappings
    SUPERMARKET_URLS = {
        "tesco": "https://www.tesco.com/groceries/en-GB/",
        "sainsburys": "https://www.sainsburys.co.uk/shop/gb/groceries",
        "sainsbury's": "https://www.sainsburys.co.uk/shop/gb/groceries",
        "waitrose": "https://www.waitrose.com/ecom/shop/browse/groceries",
        "aldi": "https://groceries.aldi.co.uk/",
        "lidl": "https://www.lidl.co.uk/",
        "morrisons": "https://groceries.morrisons.com/",
        "m&s": "https://www.marksandspencer.com/c/food-to-order",
        "marks and spencer": "https://www.marksandspencer.com/c/food-to-order",
        "cooperative": "https://www.coop.co.uk/",
        "co-op": "https://www.coop.co.uk/",
    }
    
    supermarket_key = supermarket.lower().strip()
    if supermarket_key not in SUPERMARKET_URLS:
        supported = ", ".join(sorted(set(SUPERMARKET_URLS.keys())))
        return {
            "error": f"Supermarket '{supermarket}' not supported. Supported supermarkets: {supported}",
            "results": []
        }
    
    base_url = SUPERMARKET_URLS[supermarket_key]
    
    async with async_playwright() as p:
        # Launch with stealth mode to bypass bot detection
        browser = await p.chromium.launch(
            headless=False,
            args=[
                '--disable-blink-features=AutomationControlled',  # Hide automation
                '--disable-dev-shm-usage',
                '--no-sandbox',
            ]
        )
        
        # Create context with realistic settings
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='en-GB',
            timezone_id='Europe/London',
        )
        
        # Apply stealth techniques
        await context.add_init_script("""
            // Remove webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            
            // Mock plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            
            // Mock languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-GB', 'en']
            });
        """)
        
        page = await context.new_page()
        
        try:
            # Step 1: Navigate to supermarket
            print(f"📍 [{query} @ {supermarket}] Navigating to {supermarket}...")
            await page.goto(base_url, timeout=120000)
            # Replace fixed wait with smart wait for DOM ready
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=10000)
            except:
                pass # Proceed if timeout, page might be usable
            
            # Step 2: Handle cookies - try common selectors first
            print(f"🍪 [{query} @ {supermarket}] Handling cookies...")
            cookie_selectors = [
                'button:has-text("Reject all")',
                'button:has-text("Accept all")',  
                'button:has-text("Continue and accept")',
                'button:has-text("Required only")',
                'button:has-text("Reject")', 
                'button:has-text("Accept")', 
                '#onetrust-reject-all-handler',
                '#onetrust-accept-btn-handler',
            ]
            
            cookie_handled = False
            for selector in cookie_selectors:
                try:
                    # Reduced timeout for checking each cookie button
                    if await page.is_visible(selector, timeout=500):
                        await page.click(selector)
                        print(f"✅ [{query} @ {supermarket}] Clicked cookie button: {selector}")
                        cookie_handled = True
                        break
                except:
                    continue
            
            if not cookie_handled:
                print(f"⚠️  [{query} @ {supermarket}] No cookie button found, continuing anyway")
            
            # Step 2.5: Close any banners/popups (promotional, newsletter, etc.)
            print(f"🚫 [{query} @ {supermarket}] Closing banners and popups...")
            banner_close_selectors = [
                'button[aria-label*="close" i]',
                'button[aria-label*="dismiss" i]',
                '[class*="close"][role="button"]',
                '[class*="dismiss"][role="button"]',
                'button.close',
                'button[title*="close" i]',
                '[data-testid*="close"]',
                '[data-testid*="dismiss"]',
                # Common X button patterns
                'button:has-text("×")',
                'button:has-text("✕")',
                'svg[class*="close"]',
            ]
            
            banners_closed = 0
            for selector in banner_close_selectors:
                try:
                    # Quick check for visibility
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        try:
                            if await element.is_visible():
                                await element.click(timeout=500)
                                banners_closed += 1
                                print(f"✅ Closed banner/popup: {selector}")
                        except:
                            continue
                except:
                    continue
            
            if banners_closed > 0:
                print(f"✅ Closed {banners_closed} banner(s)/popup(s)")
            else:
                print("⚠️  No banners/popups found to close")
            
            # Step 3: Use vision to find the search box
            print(f"🔎 [{query} @ {supermarket}] Looking for search box...")
            screenshot_bytes, _ = await take_screenshot(page)
            
            search_locate_prompt = """
            Look at this grocery website screenshot.
            Can you see a search box or search input field?
            If yes, describe what placeholder text it has (if any) or what icon/label is near it.
            If you see a magnifying glass icon or "Search" text, describe where it is.
            Be specific and brief.
            """
            
            search_description = await ask_vision_model(screenshot_bytes, search_locate_prompt)
            print(f"📦 Vision says: {search_description}")
            
            # Try multiple search strategies
            search_submitted = False
            
            # Strategy 1: Try clicking search icon first
            try:
                # Look for search icon/button
                search_icon = await page.wait_for_selector('[aria-label*="search" i], button[title*="search" i], [data-auto*="search"]', timeout=2000)
                await search_icon.click()
                print("✅ Clicked search icon")
            except:
                print("⚠️  No search icon found")
            
            # Strategy 2: Try to find input field
            try:
                # Try various selectors - expanded list for different supermarkets
                selectors_to_try = [
                    'input[data-auto*="search"]',
                    'input[placeholder*="search" i]',
                    'input[placeholder*="product" i]',
                    'input[placeholder*="find" i]',
                    'input[type="text"][name*="search" i]',
                    'input[type="search"]',
                    'input.search',
                    'input[id*="search" i]',
                    '#search-input',
                    '[role="searchbox"]',
                    'input[aria-label*="search" i]',
                    'input[name="q"]',  # Common search param
                    'input[name="query"]',
                ]
                
                search_input = None
                for selector in selectors_to_try:
                    try:
                        # Reduced timeout for finding search input
                        search_input = await page.wait_for_selector(selector, state="visible", timeout=1000)
                        if search_input:
                            print(f"✅ Found search input: {selector}")
                            break
                    except:
                        continue
                
                if search_input:
                    await search_input.fill(query)
                    await search_input.press("Enter")
                    print(f"✅ [{query} @ {supermarket}] Search submitted")
                    search_submitted = True
                    
                    # Wait for results - smart wait
                    print(f"⏳ [{query} @ {supermarket}] Waiting for results to load...")
                    try:
                        # Wait for network to be idle (meaning results likely loaded)
                        await page.wait_for_load_state("networkidle", timeout=8000)
                    except:
                        print(f"⚠️  [{query} @ {supermarket}] Network idle timeout, proceeding anyway...")
                    
                    # Scroll down gradually to trigger lazy loading of images
                    print(f"📜 [{query} @ {supermarket}] Scrolling to trigger lazy loading...")
                    for i in range(3):
                        await page.evaluate("window.scrollBy(0, 800)")
                        await page.wait_for_timeout(500)
                    
                    # Scroll back up to top to capture the first results
                    await page.evaluate("window.scrollTo(0, 0)")
                    await page.wait_for_timeout(1000)
                    
                else:
                    print("❌ Could not find search input field")
                    
            except Exception as e:
                print(f"❌ Search error: {e}")
            
            # Proceed to screenshot even if search submission had minor issues (might have worked)
            
            # Step 5: Take screenshot of results and extract products
            try:
                # Wait for the page to settle before taking a screenshot
                # Wait specifically for images to be present
                try:
                    await page.wait_for_selector("img", timeout=5000)
                except:
                    pass
                    
                await page.wait_for_timeout(2000)
                
                # CRITICAL: Remove any persistent cookie banners/overlays BEFORE screenshot
                print(f"🧹 [{query} @ {supermarket}] Final cleanup of cookie banners...")
                cleanup_selectors = [
                    # Cookie banners
                    '[id*="cookie" i]',
                    '[class*="cookie" i]',
                    '[data-testid*="cookie" i]',
                    # Consent banners
                    '[id*="consent" i]',
                    '[class*="consent" i]',
                    # Overlays
                    '[class*="overlay" i]',
                    '[class*="modal" i]',
                    # Specific Sainsbury's patterns
                    '[class*="CookieBanner" i]',
                    '[data-testid*="banner" i]',
                ]
                
                for selector in cleanup_selectors:
                    try:
                        elements = await page.query_selector_all(selector)
                        for element in elements:
                            try:
                                if await element.is_visible():
                                    # Try to hide it with JavaScript
                                    await page.evaluate('(el) => el.style.display = "none"', element)
                            except:
                                continue
                    except:
                        continue
                
                # Also try clicking any remaining "Accept" or "Continue" buttons
                final_cookie_buttons = [
                    'button:has-text("Continue and accept")',
                    'button:has-text("Accept all")',
                    'button:has-text("Required only")',
                ]
                for btn_selector in final_cookie_buttons:
                    try:
                        if await page.is_visible(btn_selector, timeout=500):
                            await page.click(btn_selector)
                            await page.wait_for_timeout(500)
                            print(f"✅ Clicked final cookie button: {btn_selector}")
                            break
                    except:
                        continue
                
                screenshot_bytes, _ = await take_screenshot(page)
                
                # Save screenshot for debugging
                screenshot_filename = f"backend/debug/debug_vision_{supermarket_key}.png"
                Path(screenshot_filename).write_bytes(screenshot_bytes)
                print(f"💾 [{query} @ {supermarket}] Saved screenshot to {screenshot_filename}")
            except Exception as e:
                print(f"⚠️ Failed to take screenshot: {e}")
                return {"error": f"Browser error during screenshot: {e}", "results": []}
            
            extraction_prompt = f"""
            You are looking at search results for "{query}" on the {supermarket} website.
            
            IMPORTANT INSTRUCTIONS FOR FINDING PRICES:
            - Look for £ or € symbols - these indicate prices
            - Each product is in its own box/card on the page
            - Prices are ALWAYS shown with the currency symbol (£ or €)
            - Supermarkets often show TWO prices:
              * "Membership price" (discounted price for members, e.g., Clubcard, Nectar)
              * "Regular price" (standard price for non-members)
            - If you see both prices, include BOTH in your response
            
            Please extract ALL visible products from this screenshot.
            For each product, provide:
            1. Product name (full name as shown)
            2. Regular price (the standard TOTAL price with £ symbol, e.g., "£2.50")
            3. Membership price (if shown - the discounted TOTAL price for members, e.g., "£2.00")
            4. Unit price if visible (e.g., "£1.50/kg" or "£2.75/litre")
            
            CRITICAL DISTINCTION:
            - "membership_price" is the TOTAL price for the product (e.g., "£2.00")
            - "unit_price" is the price PER UNIT with a slash (e.g., "£32.61/kg", "£1.50/litre")
            - DO NOT confuse unit prices (with /kg, /litre, etc.) with membership prices
            - Membership prices NEVER have a slash or unit suffix
            
            Format your response as a JSON array like this:
            [
              {{
                "name": "Product Name Here",
                "regular_price": "£2.50",
                "membership_price": "£2.00",
                "unit_price": "£1.25/kg"
              }},
              {{
                "name": "Another Product",
                "regular_price": "£3.00",
                "membership_price": null,
                "unit_price": "£1.50/litre"
              }}
            ]
            
            CRITICAL: Look carefully at each product box for the £ symbol. Don't miss prices!
            Only include products you can clearly see. If you can't see any products, return an empty array [].
            IMPORTANT: Return ONLY the JSON array, no other text or markdown formatting.
            """
            
            products_json = await ask_vision_model(screenshot_bytes, extraction_prompt)
            print(f"\n📊 Gemini's response:\n{products_json}")
            
            # Parse JSON response
            import json
            try:
                # Clean up the response (remove markdown code blocks if present)
                cleaned = products_json.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                cleaned = cleaned.strip()
                
                products = json.loads(cleaned)
                print(f"\n✅ Found {len(products)} products")
                
                # ---- Verification function to ensure exact match ----
                def verify_product_match(query: str, candidate: dict) -> tuple[bool, str]:
                    """Use Gemini to strictly verify if the candidate product matches the query.
                    
                    Args:
                        query: The original search query
                        candidate: The candidate product dict with name, prices, etc.
                    
                    Returns:
                        (is_match, reason): Boolean indicating if it's a match and explanation
                    """
                    verify_prompt = f"""
                    You are a strict product matching verifier.
                    
                    User searched for: "{query}"
                    Product found: "{candidate.get('name', '')}"
                    
                    Your task is to determine if these represent THE SAME PRODUCT.
                    
                    STRICT MATCHING RULES:
                    1. Brand name must match (e.g., "Tunnock's" vs "Tunnocks" is OK, but "Cadbury" vs "Nestle" is NOT)
                    2. Product type must match exactly (e.g., "Mini" vs regular size is DIFFERENT)
                    3. Flavor/variant must match (e.g., "Milk Chocolate" vs "Dark Chocolate" is DIFFERENT)
                    4. Size/quantity differences:
                       - "Multipack" or specific weights (e.g. "240g") ARE ACCEPTABLE if the core product is the same.
                       - "Mini" versions are DIFFERENT.
                       - "Large" vs "Regular" is ACCEPTABLE if not specified in query.
                    5. Minor wording differences are OK (e.g., "Caramel Wafer" vs "Caramel Wafers" is SAME)
                    
                    IMPORTANT: Be strict on BRAND and TYPE, but allow standard packaging variations (multipacks, weights) unless the query specifically excludes them.
                    
                    Respond with a JSON object:
                    {{
                        "is_match": true or false,
                        "reason": "Brief explanation of why it matches or doesn't match"
                    }}
                    
                    Example responses:
                    {{"is_match": false, "reason": "Query asks for regular size but product is Mini version"}}
                    {{"is_match": true, "reason": "Same brand, product type, and variant - minor wording differences only"}}
                    """
                    
                    try:
                        response = vision_model.generate_content([verify_prompt])
                        txt = response.text.strip()
                        print(f"🔍 Verification response: {txt[:200]}...")
                        
                        # Clean markdown fences
                        if txt.startswith("```"):
                            lines = txt.split("\n")
                            lines = lines[1:]  # Remove first line
                            if lines and lines[-1].strip() == "```":
                                lines = lines[:-1]  # Remove last line
                            txt = "\n".join(lines).strip()
                        
                        result = json.loads(txt)
                        is_match = result.get("is_match", False)
                        reason = result.get("reason", "No reason provided")
                        return is_match, reason
                    except Exception as e:
                        print(f"⚠️ Verification error: {e}")
                        # On error, be conservative and reject the match
                        return False, f"Verification failed: {str(e)}"
                
                # ---- Select best match using LLM reasoning ----
                def select_best_match(products, query):
                    """Use Gemini to reason which product best matches the query.
                    Returns the product dict or None if no suitable match.
                    """
                    match_prompt = f"""
                    You are given a search query and a list of product candidates.
                    Query: \"{query}\"
                    Products (each with name, regular_price, membership_price, unit_price):
                    {json.dumps(products, indent=2)}
                    
                    Your task is to pick the product that most closely matches the query.
                    Consider brand, size (ml, l, g, kg), packaging (e.g., 4x250ml), and type.
                    If none of the products seem to match the query, respond with the word NONE.
                    
                    Respond with a JSON object containing the chosen product (exactly as in the list) or the string "NONE".
                    Example response:
                    {{"name": "Arla Big Milk Fresh Whole Milk Vitamin Enriched for kids 4 X250ml", "regular_price": "£3.75", "membership_price": "£2.75", "unit_price": "£2.75/litre"}}
                    or
                    "NONE"
                    """
                    response = vision_model.generate_content([match_prompt])
                    txt = response.text.strip()
                    print(f"🔍 LLM raw response: {txt[:200]}...")  # Debug print
                    # Remove markdown fences properly
                    if txt.startswith("```"):
                        lines = txt.split("\n")
                        # Remove first line (```json or ```)
                        lines = lines[1:]
                        # Remove last line if it's just ```
                        if lines and lines[-1].strip() == "```":
                            lines = lines[:-1]
                        txt = "\n".join(lines).strip()
                    if txt.upper() == "NONE":
                        return None
                    try:
                        parsed = json.loads(txt)
                        if isinstance(parsed, dict):
                            return parsed
                        else:
                            return None
                    except Exception as e:
                        print(f"⚠️ Failed to parse LLM response: {e}")
                        return None
                
                def token_overlap(a: str, b: str) -> int:
                    # Handle None values
                    if a is None:
                        a = ""
                    if b is None:
                        b = ""
                    a_tokens = set(a.lower().split())
                    b_tokens = set(b.lower().split())
                    return len(a_tokens & b_tokens)
                
                # Step 1: Find the best candidate
                best_product = select_best_match(products, query)
                
                if best_product is None:
                    # Fallback: if LLM doesn't find a match, use token overlap
                    print("⚠️ LLM did not find a best match, falling back to token overlap.")
                    best_score = 0
                    for p in products:
                        score = token_overlap(p.get("name", ""), query)
                        if score > best_score:
                            best_score = score
                            best_product = p
                
                # Step 2: Verify the candidate is actually the same product
                if best_product is not None:
                    is_match, reason = verify_product_match(query, best_product)
                    print(f"\n🔍 Verification result: {is_match}")
                    print(f"   Reason: {reason}")
                    
                    if is_match:
                        return {"results": [best_product], "from_vision": True}
                    else:
                        # Product found but doesn't match - return helpful error
                        return {
                            "error": f"Product not found. Found '{best_product.get('name')}' but it doesn't match your search.",
                            "reason": reason,
                            "similar_products": products[:3],  # Show up to 3 similar products
                            "results": []
                        }
                else:
                    return {"error": "No suitable product found", "results": []}
            except json.JSONDecodeError as e:
                print(f"❌ Could not parse JSON: {e}")
                print(f"Raw response: {products_json}")
                return {"error": "Could not parse response", "raw": products_json, "results": []}
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "results": []}
        
        finally:
            await browser.close()


async def vision_fetch_product_with_visual_match(
    supermarket: str,
    query: str,
    reference_image_path: str = None
) -> dict:
    """
    Enhanced product search with visual matching fallback.
    
    Multi-stage matching strategy:
    1. Try exact name match
    2. If that fails and reference image exists, try simplified name + visual verification
    
    Args:
        supermarket: Name of the supermarket
        query: Product to search for
        reference_image_path: Optional path to reference image for visual comparison
    
    Returns:
        dict: Same format as vision_fetch_product
    """
    # Stage 1: Try exact name match
    print(f"\n🔍 Stage 1: Trying exact name match for '{query}'...")
    result = await vision_fetch_product(supermarket, query)
    
    # If we found it, return
    if result.get("results") and len(result["results"]) > 0:
        print(f"✅ Found with exact name!")
        return result
    
    # Stage 2: Try simplified name + visual verification (if reference image available)
    if reference_image_path and Path(reference_image_path).exists():
        print(f"\n🔍 Stage 2: Trying simplified name with visual verification...")
        
        simplified_query = simplify_product_name(query)
        print(f"   Simplified query: '{query}' -> '{simplified_query}'")
        
        # Search with simplified name
        result_simplified = await vision_fetch_product(supermarket, simplified_query)
        
        # If we got candidates, use visual comparison
        if result_simplified.get("similar_products") or (result_simplified.get("results") and len(result_simplified["results"]) > 0):
           
            # Load reference image
            with open(reference_image_path, 'rb') as f:
                ref_image_bytes = f.read()
            
            # Get candidate image from the latest screenshot
            supermarket_key = supermarket.lower().strip()
            candidate_screenshot = f"backend/debug/debug_vision_{supermarket_key}.png"
            
            if Path(candidate_screenshot).exists():
                with open(candidate_screenshot, 'rb') as f:
                    full_candidate_screenshot = f.read()
                
                # Crop candidate to just product image too
                print(f"   🔍 Extracting candidate product image...")
                candidate_image_bytes = await crop_to_product_image(query, full_candidate_screenshot)
                
                # Only proceed if cropping was successful
                if not candidate_image_bytes:
                    print(f"   ❌ Failed to crop candidate image - product not found in screenshot")
                    # Return error - visual match failed
                    return {
                        "error": "Product not found. Could not extract product image from screenshot.",
                        "visual_comparison": {
                            "is_same_product": False,
                            "confidence": 0.0,
                            "reasoning": "Failed to locate product in screenshot"
                        }
                    }
                
                # Save candidate image to temporary debug folder for inspection
                debug_folder = Path("backend/data/product_images/product_images_from_rest_supermarkets")
                debug_folder.mkdir(parents=True, exist_ok=True)
                
                # Create filename: ProductName_Supermarket.png
                safe_product_name = query.replace(" ", "_").replace("/", "_").replace("&", "and")
                safe_supermarket = supermarket.replace(" ", "_")
                debug_image_path = debug_folder / f"{safe_product_name}_{safe_supermarket}.png"
                
                with open(debug_image_path, 'wb') as f:
                    f.write(candidate_image_bytes)
                print(f"   💾 Saved candidate image to: {debug_image_path}")
                
                # Import and use image comparator
                from tools.image_comparator import compare_product_images
                
                print(f"🖼️  Comparing product images...")
                comparison = await compare_product_images(
                    ref_image_bytes,
                    candidate_image_bytes,
                    query
                )
                
                print(f"   Is same product: {comparison['is_same_product']}")
                print(f"   Confidence: {comparison['confidence']:.2f}")
                print(f"   Reasoning: {comparison['reasoning']}")
                
                # If visually confirmed as same product
                if comparison["is_same_product"] and comparison["confidence"] >= 0.75:
                    print(f"✅ Visual match confirmed! Accepting result.")
                    # Return the simplified search result
                    return result_simplified
                else:
                    print(f"❌ Visual match failed. Products don't appear to be the same.")
                    return {
                        "error": f"Product not found. Visual comparison did not confirm a match.",
                        "visual_comparison": comparison,
                        "results": []
                    }
            else:
                print(f"⚠️  No screenshot available for visual comparison")
    
    # If we got here, nothing worked
    print(f"❌ All stages failed. Product not found.")
    return result


# Test function
async def test_vision_fetch():
    result = await vision_fetch_product("Tesco", "Tunnocks Milk Chocolate Caramel Wafer Biscuits")
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    if result.get("results"):
        for i, product in enumerate(result["results"], 1):
            print(f"\n{i}. {product.get('name', 'Unknown')}")
            
            regular_price = product.get('regular_price')
            clubcard_price = product.get('clubcard_price')
            
            if regular_price:
                print(f"   Regular Price: {regular_price}")
            if clubcard_price:
                print(f"   Clubcard Price: {clubcard_price} 💳")
            if not regular_price and not clubcard_price:
                print(f"   Price: Not found")
                
            if product.get('unit_price'):
                print(f"   Unit: {product.get('unit_price')}")
    else:
        print("No products found")
        if result.get("error"):
            print(f"Error: {result['error']}")


if __name__ == "__main__":
    asyncio.run(test_vision_fetch())
