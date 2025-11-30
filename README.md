# Grocefy 🛒

An intelligent multi-agent system that finds the best grocery prices across UK supermarkets using Computer Vision and LLMs.

## 🚀 Features

- **Multi-Supermarket Search**: Searches Tesco, Morrisons, Sainsbury's, Aldi, Lidl, and Waitrose.
- **Vision-Based Price Extraction**: Uses **Gemini 2.0 Flash** and **Playwright** to visually identify products and extract prices from supermarket websites, just like a human would.
- **Intelligent Matching**:
    - **Exact Match**: Finds products by name.
    - **Visual Match**: If text search fails, it compares product images to ensure the correct item is found.
- **Price Optimization**: Automatically calculates the best deal, factoring in membership prices (Clubcard, Nectar, etc.).
- **Historical Price Tracking**: Tracks prices over time to identify trends and potential savings.
- **Cost Tracking**: Monitors API usage and calculates costs per run (Gemini 2.0/2.5 Flash pricing).
- **Agentic Architecture**: Built with the **ADK (Agent Development Kit)**, featuring a hierarchical team of agents.

## 🏗️ Architecture

The system uses a **Sequential Coordinator Agent** that manages:
1.  **Parallel Search Agents**: Scours multiple supermarkets simultaneously.
2.  **Optimization Agent**: Analyzes results to find the best value.

For a detailed breakdown of the agent hierarchy and logic, see **[AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md)**.

## 🛠️ Setup

### Prerequisites
- Python 3.10+
- A Google Cloud Project with Gemini API access

### Installation

1.  **Clone the repository** and navigate to the project folder.

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Playwright browsers**:
    ```bash
    playwright install
    ```

4.  **Configure Environment**:
    Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```

## 🏃‍♂️ Usage

Run the main agent script:

```bash
# Run from the grocefy directory
python backend/main.py
```

The system will:
1.  Load the shopping list from `backend/data/products.csv`.
2.  Dispatch Vision Agents to search for each product.
3.  Display real-time progress of searches and navigations.
4.  Generate a **Savings Report** showing the best prices and total potential savings.
5.  Save results to `backend/results/optimization_results.csv`.
6.  Generate a detailed **Markdown Report** at `backend/results/OPTIMIZATION_REPORT.md`.
7.  Update historical price data in `backend/data/history/`.

## 📂 Project Structure

- **`backend/`**:
    - **`agent/`**: Agent definitions (Coordinator, Vision, Optimization).
    - **`tools/`**: Tools for the agents (Vision Price Fetcher).
    - **`services/`**: Core services (Memory Service).
    - **`utils/`**: Utility modules (CSV handling, Image storage, History tracking).
    - **`data/`**:
        - `products.csv`: Your shopping list.
        - `history/`: CSV files tracking price history per supermarket.
        - `product_images/`: Reference images for visual matching.
    - **`results/`**:
        - `optimization_results.csv`: The latest run results.
        - `OPTIMIZATION_REPORT.md`: Detailed markdown report.
    - `config.py`: Central configuration (Models, Rate Limits, Paths).
    - `main.py`: Application entry point.
- **`docs/`**: Documentation and diagrams.

## 🔮 Future Enhancements

- **Receipt Scanning**: Upload a photo of your receipt to auto-populate your shopping list.
- **Auto-Purchasing**: Integration with supermarket APIs to add items to your cart.
- **Price History Chatbot**: Ask questions like "When is Coke cheapest?"

See [AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md#5-future-enhancements) for more details.
