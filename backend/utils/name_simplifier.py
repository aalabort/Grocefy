"""
Product name simplification utilities.
Extracts brand and core product type from full product names.
"""

import re
from typing import List

# Common stopwords to remove
STOPWORDS = {
    'pack', 'multipack', 'box', 'packet', 'bag',
    'gluten', 'free', 'organic', 'natural',
    'fresh', 'frozen', 'chilled',
    'x', 'of', 'the', 'and', '&',
}

# Size/quantity patterns to remove
SIZE_PATTERNS = [
    r'\d+\s*(g|kg|ml|l|oz|lb)s?',  # Weights and volumes
    r'\d+\s*x\s*\d+',  # Pack sizes like "4 x 250ml"
    r'\d+\s*pack',     # "4 pack"
    r'\d+%',           # Percentages
]


def simplify_product_name(full_name: str) -> str:
    """
    Extract brand + core product type from full product name.
    
    Examples:
        "Nairn's Gluten Free Biscuit Breaks - Oats & Chocolate Chip 160g"
        -> "Nairns Biscuit"
        
        "Ferrero Raffaello Coconut & Almond Pralines 230G"
        -> "Ferrero Pralines"
        
        "Nakd Cocoa Orange Bar Multipack"
        -> "Nakd Bar"
    
    Args:
        full_name: Full product name as it appears in products.csv
    
    Returns:
        str: Simplified name (brand + product type)
    """
    # Start with the full name
    name = full_name.lower()
    
    # Remove size/quantity patterns
    for pattern in SIZE_PATTERNS:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)
    
    # Remove hyphens and special characters (except letters, numbers, spaces)
    name = re.sub(r'[^\w\s]', ' ', name)
    
    # Split into words
    words = name.split()
    
    # Remove stopwords
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    
    # Strategy: Take first word (usually brand) + first "product type" word
    # Product type words are typically nouns that describe the product
    if len(words) == 0:
        return full_name  # Fallback to original
    
    # Always keep first word (brand)
    result_words = [words[0]]
    
    # Find the main product type (look for common product words)
    product_types = [
        'biscuit', 'biscuits', 'cookie', 'cookies',
        'chocolate', 'bar', 'bars',
        'praline', 'pralines',
        'wafer', 'wafers',
        'cake', 'cakes',
        'cereal', 'oat', 'oats',
        'milk', 'cheese', 'yogurt', 'butter',
        'bread', 'roll', 'rolls',
        'chip', 'chips', 'crisp', 'crisps',
    ]
    
    for word in words[1:]:
        if word in product_types or len(result_words) < 2:
            result_words.append(word)
            if len(result_words) >= 2:
                break
    
    # Capitalize first letter of each word
    simplified = ' '.join(w.capitalize() for w in result_words)
    
    # If we only got one word, try to get one more
    if len(result_words) == 1 and len(words) > 1:
        simplified += ' ' + words[1].capitalize()
    
    return simplified


# Test function
if __name__ == "__main__":
    test_cases = [
        "Nairn's Gluten Free Biscuit Breaks - Oats & Chocolate Chip 160g",
        "Ferrero Raffaello Coconut & Almond Pralines 230G",
        "Nakd Cocoa Orange Bar Multipack",
        "Tunnocks Milk Chocolate Caramel Wafer Biscuits",
    ]
    
    print("Testing name simplification:\n")
    for name in test_cases:
        simplified = simplify_product_name(name)
        print(f"{name}")
        print(f"  -> {simplified}\n")
