# RoboTrader

An AI-driven trading platform that analyzes market data, predicts next-hour volatility, and generates actionable trading recommendations via GPT-3.5, all accessible through a Flask web interface for analysis and execution.

## Features

- Dataset generation with `yfinance` and `pandas_ta`
- Next-hour volatility prediction with a pre-trained Keras model
- JSON-based strategy prompts to OpenAI Chat API
- Background processing with real-time status updates
- Demo mode (analysis only) and live trading mode
- Result visualization and trade execution in the UI

## Getting Started

### Prerequisites

- Python 3.10+
- Create a free account and API keys at [Alpaca](https://alpaca.markets)
- OpenAI API key

### Installation

```bash
# Clone repository
git clone https://github.com/<your-username>/RoboTrader.git
cd RoboTrader

# Create and activate virtual environment
python -m venv venv
# Windows PowerShell
evnt\Scripts\Activate.ps1
# or cmd: venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill in your keys:

```ini
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_secret
APCA_API_BASE_URL=https://paper-api.alpaca.markets
OPENAI_API_KEY=your_openai_api_key
```

### Running the App

```bash
# From project root
env/Scripts/python.exe UI/app.py
```

Then open your browser at `http://127.0.0.1:5000`.

## Usage

1. Enter stock symbols (comma-separated).
2. Choose date range.
3. Toggle demo mode on/off.
4. Click **Start Trading Analysis**.
5. Monitor status and view results.
6. Click **Execute Trades** to place live orders (only if demo mode is off).
7. Use **Load Previous Batch Results** to reload last run.

## Project Structure

```
RoboTrader/
├── UI/                       # Flask web interface
│   ├── templates/
│   └── static/
│   └── app.py
├── SMVF/                     # Dataset and model code
├── getAccountInfo.py         # Alpaca account info
├── LLM.py                    # OpenAI LLM integration
├── trade.py                  # Alpaca order placement
├── requirements.txt
├── README.md
└── .gitignore
```

## License

MIT © Zia Ullah Khan
