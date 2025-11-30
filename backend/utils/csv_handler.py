import csv
from pathlib import Path
from typing import List, Dict, Any

def read_products_csv(file_path: Path) -> List[Dict[str, str]]:
    """
    Reads the products CSV file and returns a list of dictionaries.
    Each dictionary represents a row in the CSV.
    """
    products = []
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found at {file_path}")

    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            products.append(row)
    return products

def write_results_csv(file_path: Path, results: List[Dict[str, Any]]):
    """
    Writes the optimization results to a CSV file.
    Creates an empty CSV with headers even if results list is empty.
    """
    # Define expected headers for optimization results
    fieldnames = [
        "product_name",
        "current_supermarket",
        "current_regular_price",
        "current_membership_price",
        "cheapest_supermarket",
        "cheapest_regular_price",
        "cheapest_membership_price",
        "savings_vs_current",
        "saving_cheapest_regular_vs_current_regular",
        "saving_cheapest_regular_vs_current_membership",
        "saving_cheapest_membership_vs_current_regular",
        "saving_cheapest_membership_vs_current_membership"
    ]
    
    # If results exist, use their keys (in case format changes)
    if results:
        fieldnames = results[0].keys()

    with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        if results:
            writer.writerows(results)
