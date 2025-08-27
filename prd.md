# Product Requirements Document: Crypto Wallet Recommendation Agent

## Project Overview

Build a LangChain-based recommendation agent that analyzes crypto wallet behavior to provide actionable trading recommendations. The system will use machine learning, statistical analysis, and on-chain data to generate personalized, risk-adjusted suggestions.

**IDE**: Cursor AI
**Framework**: LangChain + LangSmith
**Language**: Python 3.10+
**Development Time**: 4 weeks

---

## Project Setup Instructions for Cursor

### Initial Configuration

```bash
# Create project directory
mkdir crypto-recommendation-agent
cd crypto-recommendation-agent

# Initialize git
git init

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create requirements.txt
```

### `requirements.txt`
```txt
langchain==0.1.0
langchain-community==0.0.10
langsmith==0.0.87
langchain-openai==0.0.5
scikit-learn==1.3.0
xgboost==2.0.0
pandas==2.0.3
numpy==1.24.3
requests==2.31.0
python-dotenv==1.0.0
redis==5.0.1
plotly==5.17.0
scipy==1.11.0
```

### `.env` file
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=crypto-recommendation
OPENAI_API_KEY=your_openai_key_here
OCTAV_API_KEY=your_octav_api_key
NANSEN_API_KEY=your_nansen_api_key
REDIS_URL=
```

---

## Project Structure

Create this folder structure in Cursor:

```
crypto-recommendation-agent/
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── main.py                    # Entry point
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py         # Base agent class
│   │   └── recommendation_agent.py # Main recommendation agent
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── data_collection.py    # API integrations
│   │   ├── feature_engineering.py # Feature extraction
│   │   ├── risk_assessment.py    # Risk calculations
│   │   ├── clustering.py         # Wallet clustering
│   │   └── ml_models.py          # ML predictions
│   │
│   ├── chains/
│   │   ├── __init__.py
│   │   └── recommendation_chain.py # Chain orchestration
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py            # Data models
│   │   └── validators.py         # Input validation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── cache.py              # Redis caching
│       └── monitoring.py         # LangSmith tracking
│
├── tests/
│   ├── __init__.py
│   ├── test_tools.py
│   ├── test_agents.py
│   └── test_chains.py
│
├── notebooks/
│   └── exploration.ipynb        # Data exploration
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Development Phases

## Phase 1: Foundation (Week 1)

### Task 1.1: Configuration Setup

**File: `src/config.py`**

```python
"""
Cursor AI Instruction:
Create a configuration class that:
1. Loads environment variables from .env
2. Validates all required API keys exist
3. Sets up default values for model parameters
4. Configures logging
Include error handling for missing keys.
"""

from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

@dataclass
class Config:
    # API Keys
    langchain_api_key: str
    openai_api_key: str
    octav_api_key: str
    
    # Model Parameters
    risk_threshold: float = 0.6
    confidence_threshold: float = 0.65
    
    # Add validation method
    def validate(self):
        """Ensure all required keys are present"""
        pass
```

### Task 1.2: Data Models

**File: `src/models/schemas.py`**

```python
"""
Cursor AI Instruction:
Create Pydantic models for:
1. WalletData: address, transactions list, balances dict
2. RiskProfile: risk_score (0-100), risk_factors dict, risk_category (low/medium/high)
3. Recommendation: action (swap/rebalance/hold), confidence (0-1), reasoning (str), metadata (dict)
4. TransactionHistory: timestamp, protocol, asset_in, asset_out, value_usd, gas_paid

Include validation for:
- Valid Ethereum addresses
- Positive values for amounts
- Timestamps in correct format
"""

from pydantic import BaseModel, validator
from typing import List, Dict, Optional
from datetime import datetime

class WalletData(BaseModel):
    address: str
    transactions: List['TransactionHistory']
    balances: Dict[str, float]
    
    @validator('address')
    def validate_address(cls, v):
        # Add Ethereum address validation
        pass
```

### Task 1.3: Base Agent Setup

**File: `src/agents/base_agent.py`**

```python
"""
Cursor AI Instruction:
Create an abstract base agent class that:
1. Initializes LangSmith tracing
2. Sets up memory using ConversationBufferMemory
3. Provides abstract methods for analyze() and generate_recommendation()
4. Includes error handling and logging
5. Has a method to track metrics to LangSmith

The agent should be inheritable by specific agent implementations.
"""

from abc import ABC, abstractmethod
from langchain.memory import ConversationBufferMemory
from langsmith import Client
import logging

class BaseAgent(ABC):
    def __init__(self, project_name: str):
        self.client = Client()
        self.memory = ConversationBufferMemory()
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def analyze(self, wallet_address: str):
        """Analyze wallet and return insights"""
        pass
```

---

## Phase 2: Data Collection Tools (Week 1-2)

### Task 2.1: API Integration Tools

**File: `src/tools/data_collection.py`**

```python
"""
Cursor AI Instruction:
Create LangChain tools for data collection:

1. OctavDataTool:
   - Fetch wallet transaction history from Octav API
   - Handle pagination if response is large
   - Parse and normalize transaction data
   - Cache responses in Redis for 1 hour
   
2. TVLDataTool:
   - Fetch protocol TVL from DefiLlama
   - Get historical TVL data (30 days)
   - Calculate TVL trend and volatility
   
3. NansenDataTool:
   - Fetch related wallets
   - Get wallet labels if available
   
4. SentimentDataTool:
   - Fetch social sentiment for tokens
   - Get market fear/greed index

Each tool should:
- Inherit from BaseTool
- Have proper error handling
- Log API calls to LangSmith
- Include retry logic for failed requests
"""

from langchain.tools import BaseTool
from typing import Dict, Any
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

class OctavDataTool(BaseTool):
    name = "octav_wallet_data"
    description = "Fetches comprehensive wallet data from Octav"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _run(self, wallet_address: str) -> Dict[str, Any]:
        # Implementation here
        pass
```

### Task 2.2: Feature Engineering Tools

**File: `src/tools/feature_engineering.py`**

```python
"""
Cursor AI Instruction:
Create feature extraction tools:

1. FourierTransformTool:
   - Input: transaction time series
   - Apply FFT to find trading frequency patterns
   - Return: dominant frequencies, pattern type (daily/weekly/monthly)
   - Interpret: map frequencies to human-readable patterns

2. AutocorrelationTool:
   - Calculate autocorrelation for lags 1-30 days
   - Identify periodicity in trading behavior
   - Return: correlation values, identified period

3. StatisticalFeatureTool:
   - Calculate: mean, std, skewness, kurtosis of transaction amounts
   - Transaction count by hour of day, day of week
   - Average time between transactions
   - Whale transaction ratio (>$100k)

Include numpy/scipy for calculations and proper error handling.
"""

import numpy as np
from scipy import stats, signal
from langchain.tools import BaseTool

class FourierTransformTool(BaseTool):
    name = "fourier_analysis"
    description = "Analyzes trading frequency patterns"
    
    def _run(self, transactions: list) -> dict:
        # Convert to time series
        # Apply FFT
        # Find dominant frequencies
        # Return pattern analysis
        pass
```

---

## Phase 3: Risk & ML Tools (Week 2)

### Task 3.1: Risk Assessment Tools

**File: `src/tools/risk_assessment.py`**

```python
"""
Cursor AI Instruction:
Create risk calculation tools:

1. PortfolioVolatilityTool:
   - Calculate portfolio volatility using covariance matrix
   - Formula: σ_p = √(w^T Σ w)
   - Input: token balances and price history
   - Return: volatility score (0-100)

2. VaRCalculatorTool:
   - Calculate Value at Risk at 95% confidence
   - Formula: VaR = μ - σ * 1.645
   - Consider historical returns
   - Return: dollar amount at risk

3. ConcentrationRiskTool:
   - Calculate Herfindahl-Hirschman Index
   - Formula: HHI = Σ(s_i)²
   - Identify over-concentration in single assets
   - Return: HHI score and risk level

4. ProtocolRiskTool:
   - Assess protocol risk based on:
     * TVL trends
     * Audit status
     * Time since launch
     * Recent exploits
   - Return: risk score per protocol

Include proper mathematical calculations using numpy/scipy.
"""

import numpy as np
from scipy import stats
from langchain.tools import BaseTool

class PortfolioVolatilityTool(BaseTool):
    name = "portfolio_volatility"
    description = "Calculates portfolio volatility"
    
    def _run(self, portfolio_data: dict) -> dict:
        # Calculate covariance matrix
        # Apply portfolio weights
        # Return volatility metrics
        pass
```

### Task 3.2: Clustering Tool

**File: `src/tools/clustering.py`**

```python
"""
Cursor AI Instruction:
Create wallet clustering tool:

1. WalletClusteringTool:
   - Use DBSCAN with custom distance metric
   - Distance metric: Dynamic Time Warping for transaction sequences
   - Features to cluster on:
     * Trading frequency
     * Average transaction size
     * Risk score
     * Protocol preferences
   - Return: cluster label and similar wallets

2. Include method to find N most similar wallets
3. Cache cluster assignments in Redis
4. Update clusters daily

Use scikit-learn for DBSCAN and dtaidistance for DTW.
"""

from sklearn.cluster import DBSCAN
from langchain.tools import BaseTool
import numpy as np

class WalletClusteringTool(BaseTool):
    name = "wallet_clustering"
    description = "Clusters wallets by behavior"
    
    def _run(self, wallet_features: dict) -> dict:
        # Prepare feature matrix
        # Apply DBSCAN with DTW distance
        # Find similar wallets
        # Return cluster info
        pass
```

### Task 3.3: ML Prediction Tools

**File: `src/tools/ml_models.py`**

```python
"""
Cursor AI Instruction:
Create ML prediction tools:

1. ActionPredictionTool:
   - Ensemble of Random Forest + XGBoost
   - Predict next likely action (swap/rebalance/hold/add_liquidity)
   - Return: predicted action and probability distribution

2. TimingPredictionTool:
   - Predict optimal timing for action
   - Use time-series features
   - Return: recommended time window

3. SizingPredictionTool:
   - Predict optimal position size
   - Consider risk tolerance and market conditions
   - Return: recommended size as percentage

Load pre-trained models from disk if available.
Include model retraining method.
"""

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import joblib
from langchain.tools import BaseTool

class ActionPredictionTool(BaseTool):
    name = "action_prediction"
    description = "Predicts next wallet action"
    
    def __init__(self):
        super().__init__()
        self.model = self._load_or_train_model()
    
    def _run(self, features: dict) -> dict:
        # Prepare features
        # Make prediction
        # Return action and confidence
        pass
```

---

## Phase 4: Agent & Chain Implementation (Week 3)

### Task 4.1: Main Recommendation Agent

**File: `src/agents/recommendation_agent.py`**

```python
"""
Cursor AI Instruction:
Create the main recommendation agent that:

1. Orchestrates all tools in the correct order:
   - Data collection → Feature engineering → Risk assessment → 
   - Clustering → ML prediction → Recommendation generation

2. Implements three methods:
   - analyze_wallet(): Full analysis pipeline
   - generate_recommendation(): Create final recommendation
   - explain_recommendation(): Generate human-readable explanation

3. Handles three mood settings:
   - Degen: High risk tolerance, aggressive recommendations
   - Balanced: Moderate risk, balanced approach  
   - Saver: Low risk, conservative recommendations

4. Includes SHAP values for explainability
5. Tracks all operations to LangSmith
6. Caches results in Redis

The agent should handle errors gracefully and provide fallback recommendations.
"""

from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from langchain.agents import AgentExecutor, create_openai_tools_agent
import shap

class RecommendationAgent(BaseAgent):
    def __init__(self):
        super().__init__(project_name="crypto-recommendation")
        self.tools = self._initialize_tools()
        self.mood_settings = {
            "degen": {"risk_multiplier": 1.5, "confidence_threshold": 0.5},
            "balanced": {"risk_multiplier": 1.0, "confidence_threshold": 0.65},
            "saver": {"risk_multiplier": 0.5, "confidence_threshold": 0.8}
        }
    
    def analyze_wallet(self, wallet_address: str, mood: str = "balanced") -> Dict:
        # Full analysis pipeline
        pass
```

### Task 4.2: Recommendation Chain

**File: `src/chains/recommendation_chain.py`**

```python
"""
Cursor AI Instruction:
Create a LangChain Sequential Chain that:

1. Chains together all analysis steps:
   - DataCollectionChain: Gather all API data
   - FeatureEngineeringChain: Extract features
   - RiskAssessmentChain: Calculate risk metrics
   - PredictionChain: Generate predictions
   - RecommendationChain: Final recommendation

2. Each chain should:
   - Have clear input/output keys
   - Include error handling
   - Log to LangSmith
   - Support streaming responses

3. Implement caching between chains
4. Allow for chain interruption/resumption
"""

from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from typing import Dict, Any

class RecommendationChain:
    def __init__(self):
        self.chain = self._build_chain()
    
    def _build_chain(self) -> SequentialChain:
        # Build and connect all sub-chains
        pass
    
    def run(self, wallet_address: str, mood: str) -> Dict[str, Any]:
        # Execute chain with monitoring
        pass
```

---

## Phase 5: Monitoring & Testing (Week 4)

### Task 5.1: Monitoring Setup

**File: `src/utils/monitoring.py`**

```python
"""
Cursor AI Instruction:
Create monitoring utilities:

1. MetricsTracker:
   - Track prediction accuracy
   - Monitor API latency
   - Count tool usage
   - Track error rates

2. DriftDetector:
   - Calculate Population Stability Index (PSI)
   - Detect feature drift
   - Alert when model performance degrades
   - Trigger retraining when needed

3. LangSmithLogger:
   - Custom callbacks for detailed logging
   - Track costs (API calls, compute)
   - Log user feedback
   - Create custom dashboards

Include methods to export metrics to Grafana/Prometheus.
"""

from langsmith import Client
from langchain.callbacks import BaseCallbackHandler
import numpy as np

class MetricsTracker:
    def __init__(self):
        self.client = Client()
    
    def track_prediction(self, prediction: dict, actual: dict = None):
        # Log to LangSmith
        pass

class DriftDetector:
    def calculate_psi(self, baseline: np.array, current: np.array) -> float:
        # PSI calculation
        pass
```

### Task 5.2: Testing Suite

**File: `tests/test_agents.py`**

```python
"""
Cursor AI Instruction:
Create comprehensive tests:

1. Unit tests for each tool:
   - Test with mock data
   - Verify output format
   - Test error handling

2. Integration tests:
   - Test full pipeline with test wallet
   - Verify chain execution
   - Test caching behavior

3. Performance tests:
   - Measure latency
   - Test concurrent requests
   - Memory usage monitoring

4. Evaluation tests:
   - Test prediction accuracy on historical data
   - A/B testing framework
   - User satisfaction metrics

Use pytest and pytest-asyncio for testing.
"""

import pytest
from src.agents.recommendation_agent import RecommendationAgent

@pytest.fixture
def test_agent():
    return RecommendationAgent()

def test_wallet_analysis(test_agent):
    # Test wallet analysis
    pass
```

---

## Phase 6: Main Application (Week 4)

### Task 6.1: CLI Application

**File: `src/main.py`**

```python
"""
Cursor AI Instruction:
Create the main application that:

1. Provides CLI interface with commands:
   - analyze: Analyze a single wallet
   - batch: Process multiple wallets
   - monitor: Real-time monitoring mode
   - evaluate: Run evaluation on test set

2. Includes:
   - Progress bars for long operations
   - Colored output for better readability
   - Export results to JSON/CSV
   - Interactive mode for parameter adjustment

3. Error handling:
   - Graceful degradation on API failures
   - Fallback to cached data when available
   - Clear error messages for users

Use click for CLI and rich for formatting.
"""

import click
from rich.console import Console
from rich.progress import track
from src.agents.recommendation_agent import RecommendationAgent

console = Console()

@click.group()
def cli():
    """Crypto Wallet Recommendation Agent"""
    pass

@cli.command()
@click.argument('wallet_address')
@click.option('--mood', default='balanced', type=click.Choice(['degen', 'balanced', 'saver']))
def analyze(wallet_address: str, mood: str):
    """Analyze a wallet and generate recommendations"""
    console.print(f"[bold blue]Analyzing wallet: {wallet_address}[/bold blue]")
    # Implementation
    pass

if __name__ == "__main__":
    cli()
```

### Task 6.2: API Server (Optional)

**File: `src/api.py`**

```python
"""
Cursor AI Instruction:
Create FastAPI server that:

1. Endpoints:
   - POST /analyze: Analyze wallet
   - GET /recommendation/{wallet}: Get cached recommendation
   - POST /feedback: Submit user feedback
   - GET /metrics: System metrics

2. Features:
   - Rate limiting per API key
   - Request validation
   - Response caching
   - WebSocket for real-time updates

3. Security:
   - API key authentication
   - Input sanitization
   - SQL injection prevention
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Crypto Recommendation API")

@app.post("/analyze")
async def analyze_wallet(request: WalletRequest):
    # Implementation
    pass
```

---

## Cursor AI Specific Instructions

### For Each File Creation:

1. **Use Cursor's AI Chat**:
   - Copy the docstring instruction at the top of each file
   - Ask Cursor to implement the full file based on the instruction
   - Review and iterate on the generated code

2. **Use Cursor's Composer**:
   - For complex files, use Composer mode (Cmd+K)
   - Reference multiple files for context
   - Ask for specific improvements

3. **Testing Pattern**:
   ```python
   # After creating each file, test it immediately:
   # In Cursor terminal:
   python -m pytest tests/test_[component].py -v
   ```

4. **Debugging with Cursor**:
   - Use inline chat (Cmd+L) for fixing specific errors
   - Highlight problematic code and ask for fixes
   - Use "Fix with AI" for error messages

---

## Development Checklist

### Week 1: Foundation
- [ ] Project setup and structure
- [ ] Configuration management
- [ ] Data models and validation
- [ ] Base agent class
- [ ] API integration tools

### Week 2: Core Tools
- [ ] Feature engineering tools
- [ ] Risk assessment tools
- [ ] Clustering implementation
- [ ] ML model tools
- [ ] Caching layer

### Week 3: Agent & Orchestration
- [ ] Main recommendation agent
- [ ] Chain implementation
- [ ] Explainability (SHAP)
- [ ] Mood adjustment logic
- [ ] Redis integration

### Week 4: Production Ready
- [ ] Monitoring setup
- [ ] Testing suite
- [ ] CLI application
- [ ] API server (optional)
- [ ] Documentation
- [ ] Deployment scripts

---

## Testing & Validation

### Test Wallets for Development

```python
# Use these test wallets for development
TEST_WALLETS = {
    "whale": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
    "trader": "0x123...",  # Add active trader wallet
    "farmer": "0x456...",  # Add yield farmer wallet
    "degen": "0x789..."   # Add high-risk wallet
}
```

### Success Metrics

```python
# Minimum acceptance criteria
ACCEPTANCE_CRITERIA = {
    "prediction_accuracy": 0.65,  # 65% minimum
    "api_latency_ms": 200,        # <200ms p99
    "error_rate": 0.01,           # <1% errors
    "user_satisfaction": 4.0      # >4/5 rating
}
```

---

## Deployment Instructions

### Local Development
```bash
# Run locally
python src/main.py analyze 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb --mood balanced
```

### Docker Deployment
```dockerfile
# Dockerfile will be created by Cursor
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "src/main.py"]
```

### Environment Variables
```bash
# Production secrets (never commit!)
export LANGCHAIN_API_KEY=ls__xxx
export OPENAI_API_KEY=sk-xxx
export OCTAV_API_KEY=xxx
```

---

## Cursor Productivity Tips

1. **Multi-file edits**: Select multiple files in sidebar, then use Cmd+K
2. **Context awareness**: Keep relevant files open for better AI suggestions
3. **Terminal integration**: Use Cursor's terminal for immediate testing
4. **Error fixing**: Copy error messages and paste in chat for fixes
5. **Code review**: Ask Cursor to review your code for improvements
