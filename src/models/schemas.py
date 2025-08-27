from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Literal, Any
from datetime import datetime
from enum import Enum
import re

class RiskCategory(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class Action(str, Enum):
    SWAP = "swap"
    REBALANCE = "rebalance"
    HOLD = "hold"
    ADD_LIQUIDITY = "add_liquidity"
    REMOVE_LIQUIDITY = "remove_liquidity"
    STAKE = "stake"
    UNSTAKE = "unstake"

class Mood(str, Enum):
    DEGEN = "degen"
    BALANCED = "balanced"
    SAVER = "saver"

class TransactionHistory(BaseModel):
    timestamp: datetime
    protocol: str
    asset_in: Optional[str] = None
    asset_out: Optional[str] = None
    value_usd: float = Field(ge=0)
    gas_paid: float = Field(ge=0)
    tx_hash: str
    method: Optional[str] = None
    
    @validator('tx_hash')
    def validate_tx_hash(cls, v):
        if not re.match(r'^0x[a-fA-F0-9]{64}$', v):
            raise ValueError('Invalid transaction hash format')
        return v

class TokenBalance(BaseModel):
    symbol: str
    address: str
    balance: float = Field(ge=0)
    value_usd: float = Field(ge=0)
    price_usd: float = Field(ge=0)
    allocation_percentage: float = Field(ge=0, le=100)

class WalletData(BaseModel):
    address: str
    transactions: List[TransactionHistory]
    balances: Dict[str, TokenBalance]
    total_value_usd: float = Field(ge=0)
    transaction_count: int = Field(ge=0)
    first_transaction: Optional[datetime] = None
    last_transaction: Optional[datetime] = None
    
    @validator('address')
    def validate_address(cls, v):
        if not re.match(r'^0x[a-fA-F0-9]{40}$', v):
            raise ValueError('Invalid Ethereum address format')
        return v.lower()

class RiskProfile(BaseModel):
    risk_score: float = Field(ge=0, le=100)
    risk_factors: Dict[str, float]
    risk_category: RiskCategory
    portfolio_volatility: float = Field(ge=0)
    value_at_risk: float
    concentration_risk: float = Field(ge=0, le=1)
    protocol_risks: Dict[str, float]
    
    @validator('risk_category')
    def validate_risk_category(cls, v, values):
        if 'risk_score' in values:
            score = values['risk_score']
            if score < 25:
                return RiskCategory.LOW
            elif score < 50:
                return RiskCategory.MEDIUM
            elif score < 75:
                return RiskCategory.HIGH
            else:
                return RiskCategory.VERY_HIGH
        return v

class WalletCluster(BaseModel):
    cluster_id: int
    similar_wallets: List[str]
    cluster_characteristics: Dict[str, Any]
    distance_to_centroid: float
    cluster_size: int

class Prediction(BaseModel):
    action: Action
    confidence: float = Field(ge=0, le=1)
    probability_distribution: Dict[Action, float]
    timing_window: Optional[Dict[str, datetime]] = None
    recommended_size_percentage: float = Field(ge=0, le=100)
    expected_outcome: Dict[str, float]

class Recommendation(BaseModel):
    wallet_address: str
    timestamp: datetime = Field(default_factory=datetime.now)
    mood: Mood
    action: Action
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    metadata: Dict[str, Any]
    risk_assessment: RiskProfile
    prediction_details: Prediction
    similar_wallet_actions: Optional[List[Dict[str, Any]]] = None
    shap_values: Optional[Dict[str, float]] = None
    
    tokens_involved: Optional[List[str]] = None
    protocols_suggested: Optional[List[str]] = None
    estimated_gas: Optional[float] = None
    estimated_return: Optional[float] = None
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v

class FeatureSet(BaseModel):
    trading_frequency: float
    avg_transaction_size: float
    transaction_count_24h: int
    transaction_count_7d: int
    transaction_count_30d: int
    whale_transaction_ratio: float
    
    dominant_frequency: Optional[float] = None
    pattern_type: Optional[str] = None
    autocorrelation_lag_1: Optional[float] = None
    autocorrelation_lag_7: Optional[float] = None
    autocorrelation_lag_30: Optional[float] = None
    
    mean_transaction_amount: float
    std_transaction_amount: float
    skewness_transaction_amount: float
    kurtosis_transaction_amount: float
    
    hour_distribution: Dict[int, int]
    weekday_distribution: Dict[int, int]
    avg_time_between_transactions: float
    
    portfolio_diversity: float
    top_token_concentration: float
    protocol_count: int
    unique_tokens_traded: int

class MarketData(BaseModel):
    timestamp: datetime
    btc_price: float
    eth_price: float
    gas_price: float
    fear_greed_index: int = Field(ge=0, le=100)
    market_cap_total: float
    volume_24h: float
    dominance_btc: float
    dominance_eth: float

class ProtocolData(BaseModel):
    protocol_name: str
    tvl: float
    tvl_change_24h: float
    tvl_change_7d: float
    tvl_change_30d: float
    audit_status: bool
    time_since_launch_days: int
    recent_exploits: List[Dict[str, Any]]
    risk_score: float = Field(ge=0, le=100)

class APIResponse(BaseModel):
    status: Literal["success", "error"]
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    cached: bool = False

class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []

class BacktestResult(BaseModel):
    strategy_name: str
    period_start: datetime
    period_end: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    average_return_per_trade: float
    best_trade: Dict[str, Any]
    worst_trade: Dict[str, Any]