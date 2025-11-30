import asyncio
import sys
import os
from pathlib import Path

# Add backend directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import PRODUCTS_CSV, RESULTS_CSV, TARGET_SUPERMARKETS, BASE_DIR, RESULTS_DIR
from utils.csv_handler import read_products_csv, write_results_csv
from utils.display_utils import print_price_comparison
from agent.price_search_coordinator import create_price_search_pipeline
from google.adk.runners import InMemoryRunner

async def main():
    print("🛒 Starting Multi-Agent Grocery Optimization System...")
    print("🤖 Using ADK ParallelAgent & SequentialAgent Architecture")
    
    # 0. Cleanup Debug Folders and Old Reports
    import shutil
    
    debug_folders = [
        BASE_DIR / "debug",
        BASE_DIR / "data/product_images/product_images_from_rest_supermarkets"
    ]
    
    # Files to delete
    files_to_delete = [
        RESULTS_DIR / "OPTIMIZATION_REPORT.md"
    ]
    
    print("\n🧹 Cleaning up debug folders and old reports...")
    
    # Delete old report files
    for file_path in files_to_delete:
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"   ✅ Deleted {file_path}")
            except Exception as e:
                print(f"   ⚠️ Failed to delete {file_path}: {e}")
    
    # Clean debug folders
    for folder in debug_folders:
        if folder.exists():
            try:
                shutil.rmtree(folder)
                print(f"   ✅ Deleted {folder}")
            except Exception as e:
                print(f"   ⚠️ Failed to delete {folder}: {e}")
        
        # Recreate empty folder
        try:
            folder.mkdir(parents=True, exist_ok=True)
            print(f"   ✨ Recreated {folder}")
        except Exception as e:
            print(f"   ⚠️ Failed to recreate {folder}: {e}")
    
    # 1. Load Data
    print(f"\n📂 Loading products from {PRODUCTS_CSV}...")
    try:
        products = read_products_csv(PRODUCTS_CSV)
    except FileNotFoundError:
        print(f"❌ Error: Products file not found at {PRODUCTS_CSV}")
        return
    print(f"✅ Loaded {len(products)} products.")
    
    # Validate columns
    if products and not all(k in products[0] for k in ['product_name', 'current_regular_price', 'current_membership_price', 'current_supermarket']):
        print("❌ Error: CSV missing required columns. Expected: product_name, current_regular_price, current_membership_price, current_supermarket")
        return
    
    # Import batch processing config and results directory
    # Import batch processing config
    from config import ENABLE_BATCH_PROCESSING, BATCH_SIZE, BATCH_DELAY_SECONDS
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Clear previous optimization results
    print(f"\n🧹 Clearing previous optimization results from {RESULTS_CSV}...")
    try:
        # Write just the header row to clear the file
        with open(RESULTS_CSV, 'w') as f:
            f.write("product_name,current_supermarket,current_regular_price,current_membership_price,cheapest_supermarket,cheapest_regular_price,cheapest_membership_price,savings_vs_current,saving_cheapest_regular_vs_current_regular,saving_cheapest_regular_vs_current_membership,current_regular_vs_cheapest_membership,saving_cheapest_membership_vs_current_membership\n")
        print("✅ Previous results cleared")
    except Exception as e:
        print(f"⚠️ Could not clear previous results: {e}")
    
    print("\n🎯 Starting Grocery Optimization Workflow...")
    print("="*60)
    
    # Determine if batch processing is enabled
    if ENABLE_BATCH_PROCESSING and len(products) > BATCH_SIZE:
        print(f"📦 Batch Processing: ENABLED (Batch Size: {BATCH_SIZE} products)")
        print(f"⏱️  Delay between batches: {BATCH_DELAY_SECONDS}s\n")
        
        # Split products into batches
        all_search_results = []
        all_optimization_results = []
        all_summaries = []
        
        num_batches = (len(products) + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
        
        for batch_num in range(num_batches):
            start_idx = batch_num * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(products))
            batch_products = products[start_idx:end_idx]
            
            print(f"\n{'='*60}")
            print(f"📦 Processing Batch {batch_num + 1}/{num_batches}")
            print(f"   Products {start_idx + 1}-{end_idx} of {len(products)}")
            print(f"{'='*60}")
            
            # Process each product in the batch sequentially (or could be parallelized further)
            for product_idx, product in enumerate(batch_products):
                print(f"   🔎 Processing: {product['product_name']}")
                
                # Capture reference image from current supermarket (if not already captured)
                from tools.vision_price_fetcher import capture_reference_image
                from utils.image_store import get_product_image_path
                
                current_supermarket = product.get('current_supermarket', 'Tesco')
                reference_image_path = get_product_image_path(product['product_name'], current_supermarket)
                
                if not reference_image_path:
                    print(f"📸 Capturing reference image from {current_supermarket}...")
                    reference_image_path = await capture_reference_image(current_supermarket, product['product_name'])
                else:
                    print(f"✅ Using existing reference image: {reference_image_path}")
                
                # Create coordinator for this product WITH reference image
                coordinator, result_accumulator, optimizer = create_price_search_pipeline(
                    product,
                    TARGET_SUPERMARKETS,
                    reference_image_path=reference_image_path  # Pass reference for visual matching
                )
                runner = InMemoryRunner(agent=coordinator)
                
                # Use unique session ID for each product to prevent data overwriting
                product_session_id = f"product_{batch_num}_{product_idx}"
                
                # Run the pipeline for this product
                await runner.run_debug(
                    user_messages=f"Find best price for {product['product_name']}",
                    user_id="main_user",
                    session_id=product_session_id,  # Unique session ID
                    verbose=True
                )
                
                # Collect results from result_accumulator (populated by VisionAgents)
                product_search_results = result_accumulator  # This is where VisionAgents store their results
                
                # Optimization recommendation and data
                # DIRECT ACCESS FROM OPTIMIZER INSTANCE (Bypassing session state issues)
                recommendation = optimizer.final_summary
                optimization_data = optimizer.optimization_results
                
                # Store results
                all_search_results.extend(product_search_results)
                if recommendation:
                    # all_optimization_results was storing text summaries, let's keep it for that
                    # But we need a separate list for the structured data
                    all_summaries.append(f"Product: {product['product_name']} -> {recommendation}")
                
                if optimization_data:
                    all_optimization_results.extend(optimization_data)
                    print(f"   ✅ Added {len(optimization_data)} optimization results to collection")
                
                # Print comparison for this product
                if product_search_results:
                    print_price_comparison(product['product_name'], product_search_results)
            
            print(f"\n✅ Batch {batch_num + 1}/{num_batches} completed")
            
            # Wait before next batch (except for the last batch)
            if batch_num < num_batches - 1:
                print(f"⏳ Waiting {BATCH_DELAY_SECONDS}s before next batch...")
                await asyncio.sleep(BATCH_DELAY_SECONDS)
        
        # Aggregate results
        results = {
            'search_results': all_search_results,
            'optimization_data': all_optimization_results,
            'summary': "\n\n".join(all_summaries)
        }
        
    else:
        # No batch processing - run all products at once (original behavior)
        if ENABLE_BATCH_PROCESSING:
            print(f"📦 Batch Processing: ENABLED but skipped (only {len(products)} products, batch size is {BATCH_SIZE})")
        else:
            print(f"📦 Batch Processing: DISABLED (processing all {len(products)} products at once)\n")
        
        search_results = []
        optimization_data_list = []
        summaries = []

        # Process all products sequentially
        for product in products:
            print(f"   🔎 Processing: {product['product_name']}")
            
            # Capture reference image from current supermarket (if not already captured)
            from tools.vision_price_fetcher import capture_reference_image
            from utils.image_store import get_product_image_path
            
            current_supermarket = product.get('current_supermarket', 'Tesco')
            reference_image_path = get_product_image_path(product['product_name'], current_supermarket)
            
            if not reference_image_path:
                print(f"📸 Capturing reference image from {current_supermarket}...")
                reference_image_path = await capture_reference_image(current_supermarket, product['product_name'])
            else:
                print(f"✅ Using existing reference image: {reference_image_path}")
            
            coordinator, result_accumulator, optimizer = create_price_search_pipeline(
                product,
                TARGET_SUPERMARKETS,
                reference_image_path=reference_image_path
            )
            runner = InMemoryRunner(agent=coordinator)
            
            await runner.run_debug(
                user_messages=f"Find best price for {product['product_name']}",
                user_id="main_user",
                verbose=True
            )
            
            # Collect results from result_accumulator (populated by VisionAgents)
            product_search_results = result_accumulator
            
            # Optimization recommendation and data
            # DIRECT ACCESS FROM OPTIMIZER INSTANCE
            recommendation = optimizer.final_summary
            opt_data = optimizer.optimization_results
            
            search_results.extend(product_search_results)
            if recommendation:
                summaries.append(recommendation)
            if opt_data:
                optimization_data_list.extend(opt_data)
            
            # Print comparison
            if product_search_results:
                print_price_comparison(product['product_name'], product_search_results)
                
        results = {
            "search_results": search_results,
            "optimization_data": optimization_data_list,
            "summary": "\n".join(summaries)
        }
    
    summary = results['summary']
    
    # 4. Display Results
    print("\n" + "="*60)
    print("📊 OPTIMIZATION REPORT")
    print("="*60)
    
    # Calculate Global Stats (needed for the report, but report prints last)
    total_potential_savings = 0.0
    best_switch = None
    max_saving = -1.0
    products_with_savings = []
    
    for res in results['optimization_data']:
        try:
            saving_str = res.get('savings_vs_current', 'N/A')
            if saving_str != 'N/A':
                saving_val = float(saving_str.replace('£', ''))
                if saving_val > 0:
                    total_potential_savings += saving_val
                    products_with_savings.append(res)
                    
                    if saving_val > max_saving:
                        max_saving = saving_val
                        best_switch = res
        except ValueError:
            continue

    # Import RESULTS_DIR to ensure it exists
    # from config import RESULTS_DIR  <-- Removed redundant import
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # 5. Save Results to CSV
    print(f"\n💾 Saving results to {RESULTS_CSV}...")
    write_results_csv(RESULTS_CSV, results['optimization_data'])
    print("✅ Results saved to CSV")
    
    # 6. Update Historical Prices
    print(f"\n📅 Updating historical price tracking...")
    from utils.history_tracker import HistoricalPriceTracker
    tracker = HistoricalPriceTracker()
    tracker.update_history(PRODUCTS_CSV, results['search_results'])
    print("✅ Historical prices updated")

    # Construct the full report string for the MD file
    md_report = "# 🛒 Grocery Optimization Report\n\n"
    
    # 1. Optimization Stats (First as requested)
    md_report += "## 📊 Optimization Stats\n\n"
    md_report += f"**💰 GRAND TOTAL POTENTIAL SAVINGS: £{total_potential_savings:.2f}**\n\n"
    
    if best_switch:
        md_report += f"### 🌟 Top Switch\n"
        md_report += f"- **Product**: {best_switch['product_name']}\n"
        md_report += f"- **Savings**: {best_switch['savings_vs_current']}\n"
        md_report += f"- **Switch**: {best_switch['current_supermarket']} ➡️ {best_switch['cheapest_supermarket']}\n\n"
        
    if products_with_savings:
        md_report += "### 📋 Savings Opportunities\n"
        for p in products_with_savings:
            md_report += f"- **{p['product_name']}**: Save **{p['savings_vs_current']}** ({p['current_supermarket']} ➡️ {p['cheapest_supermarket']})\n"
            if 'historical_low_warning' in p:
                md_report += f"  - {p['historical_low_warning']}\n"
    else:
        md_report += "ℹ️  No savings found compared to your current supermarket prices.\n"
    
    md_report += "\n---\n\n"
    
    # 2. Price Comparisons (New Section)
    md_report += "## 🏷️ Price Comparisons\n\n"
    # We need to regenerate these strings since we didn't capture them during the loop
    # Ideally we would have captured them, but re-generating is safe and easy here
    for product_res in results['optimization_data']:
        p_name = product_res['product_name']
        # Find search results for this product
        p_search_results = [r for r in results['search_results'] if r.get('product') == p_name]
        if p_search_results:
            md_report += "```text\n"
            md_report += print_price_comparison(p_name, p_search_results)
            md_report += "```\n\n"

    # Print to terminal (keep original behavior - still showing detailed summaries)
    print("\n" + "="*60)
    print("📝 DETAILED PRODUCT SUMMARIES")
    print("="*60)
    print(summary)
    
    print("\n" + "="*60)
    print("📊 OPTIMIZATION REPORT")
    print("="*60)
    print(f"\n💰 GRAND TOTAL POTENTIAL SAVINGS: £{total_potential_savings:.2f}")
    
    if best_switch:
        print(f"🌟 TOP SWITCH: {best_switch['product_name']}")
        print(f"   Save {best_switch['savings_vs_current']} by switching to {best_switch['cheapest_supermarket']}")
        
    if products_with_savings:
        print(f"\n📋 FOUND {len(products_with_savings)} OPPORTUNITIES TO SAVE:")
        for p in products_with_savings:
            print(f"   • {p['product_name']}: Save {p['savings_vs_current']} ({p['current_supermarket']} -> {p['cheapest_supermarket']})")
            if 'historical_low_warning' in p:
                print(f"     {p['historical_low_warning']}")
    else:
        print("\nℹ️  No savings found compared to your current supermarket prices.")

    print("="*60)
    
    # Save the MD report to backend/results/
    report_path = RESULTS_DIR / "OPTIMIZATION_REPORT.md"
    print(f"\n📄 Saving detailed report to {report_path}...")
    with open(report_path, "w") as f:
        f.write(md_report)
    print("✅ Report saved")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    asyncio.run(main())
