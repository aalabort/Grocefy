"""
Image storage utilities for product reference images.
Handles saving and loading product images for visual comparison.
"""

import os
import re
from pathlib import Path
from typing import Optional

# Storage directory for product images
IMAGES_DIR = Path(__file__).parent.parent / "data" / "product_images"


def _sanitize_filename(text: str) -> str:
    """Convert text to safe filename (alphanumeric + underscore only)"""
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', text)
    # Collapse multiple underscores
    safe = re.sub(r'_+', '_', safe).strip('_')
    # Limit length
    return safe[:100]


def save_product_image(product_name: str, supermarket: str, image_bytes: bytes) -> str:
    """
    Save a product image to disk.
    
    Args:
        product_name: Name of the product
        supermarket: Supermarket where the image was captured
        image_bytes: Raw image data (PNG format expected)
    
    Returns:
        str: Absolute path to the saved image
    """
    # Ensure directory exists
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create safe filename
    safe_product = _sanitize_filename(product_name)
    safe_supermarket = _sanitize_filename(supermarket)
    filename = f"{safe_product}_{safe_supermarket}.png"
    
    filepath = IMAGES_DIR / filename
    
    # Save image
    with open(filepath, 'wb') as f:
        f.write(image_bytes)
    
    print(f"💾 Saved product image: {filepath}")
    return str(filepath.absolute())


def get_product_image(product_name: str, supermarket: str) -> Optional[bytes]:
    """
    Load a product image from disk.
    
    Args:
        product_name: Name of the product
        supermarket: Supermarket where the image was captured
    
    Returns:
        bytes: Image data if found, None otherwise
    """
    safe_product = _sanitize_filename(product_name)
    safe_supermarket = _sanitize_filename(supermarket)
    filename = f"{safe_product}_{safe_supermarket}.png"
    
    filepath = IMAGES_DIR / filename
    
    if not filepath.exists():
        return None
    
    with open(filepath, 'rb') as f:
        return f.read()


def get_product_image_path(product_name: str, supermarket: str) -> Optional[str]:
    """
    Get the path to a product image if it exists.
    
    Args:
        product_name: Name of the product
        supermarket: Supermarket where the image was captured
    
    Returns:
        str: Absolute path if image exists, None otherwise
    """
    safe_product = _sanitize_filename(product_name)
    safe_supermarket = _sanitize_filename(supermarket)
    filename = f"{safe_product}_{safe_supermarket}.png"
    
    filepath = IMAGES_DIR / filename
    
    if filepath.exists():
        return str(filepath.absolute())
    return None


def clear_product_images():
    """Remove all stored product images (for testing/cleanup)"""
    if IMAGES_DIR.exists():
        for file in IMAGES_DIR.glob("*.png"):
            file.unlink()
        print(f"🗑️  Cleared {len(list(IMAGES_DIR.glob('*.png')))} product images")
