"""
Generate reference images for all products in products.csv.

This script processes each product listed in backend/data/products.csv and:
1. Searches for the product on its listed supermarket website
2. Captures and crops the product package image using Gemini-validated cropping
3. Saves the reference image to backend/data/product_images/

These reference images serve as templates for visual comparison when searching
for the same products on other supermarket websites.
"""

import asyncio
import csv
import sys
from pathlib import Path

# Add backend directory to path (parent of utils)
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.vision_price_fetcher import capture_reference_image


async def generate_all_reference_images():
    """Generate reference images for all products in products.csv."""
    
    # Path relative to project root (assuming script run from project root or backend/utils)
    # We'll use the config definitions if possible, but for now let's resolve relative to this file
    base_dir = Path(__file__).parent.parent # backend/
    products_csv = base_dir / "data" / "products.csv"
    
    if not products_csv.exists():
        print(f"❌ Error: {products_csv} not found")
        return
    
    print("=" * 70)
    print("GENERATING REFERENCE IMAGES FOR ALL PRODUCTS")
    print("=" * 70)
    
    # Read products from CSV
    products = []
    with open(products_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            products.append(row)
    
    print(f"\n📋 Found {len(products)} products in products.csv\n")
    
    # Track results
    results = {
        'success': [],
        'failed': []
    }
    
    # Process each product
    for i, product in enumerate(products, 1):
        product_name = product['product_name']
        supermarket = product['current_supermarket']
        
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(products)}] Processing: {product_name}")
        print(f"           Supermarket: {supermarket}")
        print('=' * 70)
        
        try:
            # Capture reference image
            ref_path = await capture_reference_image(supermarket, product_name)
            
            if ref_path:
                results['success'].append({
                    'product': product_name,
                    'supermarket': supermarket,
                    'path': ref_path
                })
                print(f"\n✅ SUCCESS: Reference image saved to {ref_path}")
            else:
                results['failed'].append({
                    'product': product_name,
                    'supermarket': supermarket,
                    'reason': 'No reference image path returned'
                })
                print(f"\n❌ FAILED: Could not generate reference image")
                
        except Exception as e:
            results['failed'].append({
                'product': product_name,
                'supermarket': supermarket,
                'reason': str(e)
            })
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ Successful: {len(results['success'])}")
    for item in results['success']:
        print(f"   - {item['product']}")
        print(f"     Path: {item['path']}")
    
    if results['failed']:
        print(f"\n❌ Failed: {len(results['failed'])}")
        for item in results['failed']:
            print(f"   - {item['product']}")
            print(f"     Reason: {item['reason']}")
    
    print("\n" + "=" * 70)
    print(f"Total: {len(results['success'])}/{len(products)} reference images generated")
    print("=" * 70)
    
    # Save results to JSON for later reference
    import json
    results_file = base_dir / "data" / "reference_image_generation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    asyncio.run(generate_all_reference_images())
