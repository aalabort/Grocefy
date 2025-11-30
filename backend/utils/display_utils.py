"""Display utilities for pretty-printing results and comparisons"""

def print_price_comparison(product_name: str, search_results: list) -> str:
    """Print a nice price comparison table for a product and return the string"""
    output = []
    
    def log(msg=""):
        print(msg)
        output.append(msg)
    
    log(f"\n{'='*70}")
    log(f"🛒 PRICE COMPARISON: {product_name}")
    log(f"{'='*70}")
    
    # Filter results for this product (VisionAgent uses 'product' key, not 'product_name')
    product_results = [r for r in search_results if r.get('product') == product_name]
    
    if not product_results:
        log("❌ No prices found for this product")
        log(f"{'='*70}\n")
        return "\n".join(output)
    
    # Find the cheapest price
    prices_with_supermarket = []
    for result in product_results:
        supermarket = result.get('supermarket', 'Unknown')
        regular = result.get('regular_price')
        membership = result.get('membership_price')
        
        # Determine the best price
        if membership:
            price_val = membership
            price_type = "membership"
        elif regular:
            price_val = regular
            price_type = "regular"
        else:
            price_val = None
            price_type = None
        
        prices_with_supermarket.append({
            'supermarket': supermarket,
            'regular': regular,
            'membership': membership,
            'best_price': price_val,
            'price_type': price_type
        })
    
    # Find cheapest
    valid_prices = [p for p in prices_with_supermarket if p['best_price']]
    if valid_prices:
        cheapest = min(valid_prices, key=lambda x: float(x['best_price'].replace('£', '').replace(',', '')) if x['best_price'] else float('inf'))
        cheapest_supermarket = cheapest['supermarket']
    else:
        cheapest_supermarket = None
    
    # Print each result
    for p in prices_with_supermarket:
        supermarket = p['supermarket']
        is_cheapest = cheapest_supermarket == supermarket
        
        # Emoji based on whether it's the cheapest
        emoji = "🏆" if is_cheapest else "🏪"
        
        if p['regular'] or p['membership']:
            regular_str = p['regular'] if p['regular'] else "N/A"
            membership_str = p['membership'] if p['membership'] else "N/A"
            
            line = f"{emoji} {supermarket:15} │ Regular: {regular_str:8} │ Member: {membership_str:8}"
            
            if is_cheapest:
                line += f" │ ⭐ BEST PRICE!"
            
            log(line)
        else:
            log(f"❌ {supermarket:15} │ Not found")
    
    log(f"{'='*70}\n")
    return "\n".join(output)
