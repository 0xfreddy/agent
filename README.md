# Crypto Wallet Recommendation Agent

A LangChain-based recommendation system that analyzes crypto wallet behavior to provide actionable trading recommendations using machine learning, statistical analysis, and on-chain data.

## 🚀 Features

- **Wallet Analysis**: Comprehensive analysis of wallet transaction history and holdings
- **Risk Assessment**: Multi-factor risk evaluation including volatility, VaR, and concentration risk
- **Feature Engineering**: Advanced statistical and frequency analysis of trading patterns
- **Personalized Recommendations**: Mood-based recommendations (Degen, Balanced, Saver)
- **Data Integration**: Multiple data sources including on-chain data and market sentiment

## 📋 Prerequisites

- Python 3.10+
- Redis (for caching)
- API Keys:
  - OpenAI API Key
  - LangChain/LangSmith API Key
  - Octav API Key (optional)
  - Nansen API Key (optional)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd crypto-recommendation-agent
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## 🎯 Usage

### Basic Analysis

Analyze a single wallet:
```bash
python src/main.py analyze 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb --mood balanced
```

### Mood Settings

- **degen**: High risk tolerance, aggressive recommendations
- **balanced**: Moderate risk, balanced approach
- **saver**: Low risk, conservative recommendations

### Export Results

Export analysis results to JSON or CSV:
```bash
python src/main.py analyze <wallet_address> --export json --output results.json
```

### Batch Processing

Process multiple wallets from a file:
```bash
python src/main.py batch wallets.txt --mood balanced
```

### System Test

Run a system test with a sample wallet:
```bash
python src/main.py test
```

## 📂 Project Structure

```
crypto-recommendation-agent/
├── src/
│   ├── agents/           # Agent implementations
│   ├── tools/            # LangChain tools for data and analysis
│   ├── chains/           # Chain orchestration
│   ├── models/           # Data models and validators
│   ├── utils/            # Utilities and monitoring
│   ├── config.py         # Configuration management
│   └── main.py           # CLI application
├── tests/                # Test files
├── notebooks/            # Jupyter notebooks for exploration
├── .env                  # Environment variables
├── requirements.txt      # Python dependencies
└── README.md
```

## 🔧 Configuration

Key configuration options in `.env`:

```env
# LangChain Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=crypto-recommendation

# API Keys
OPENAI_API_KEY=your_key
OCTAV_API_KEY=your_key
NANSEN_API_KEY=your_key

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Risk Parameters
RISK_THRESHOLD=0.6
CONFIDENCE_THRESHOLD=0.65
```

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

## 📊 Features Analyzed

### Trading Patterns
- Frequency analysis using FFT
- Autocorrelation patterns
- Trading velocity and consistency

### Risk Metrics
- Portfolio volatility
- Value at Risk (VaR)
- Concentration risk (HHI)
- Protocol-specific risks

### Statistical Features
- Transaction statistics (mean, std, skewness, kurtosis)
- Time-based patterns (hourly, daily, weekly)
- Whale transaction analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

This tool provides recommendations based on historical data and statistical analysis. Always do your own research and consider multiple factors before making investment decisions. Not financial advice.