# DefiLlama API Endpoints for Risk Agent

After reviewing the DefiLlama API documentation, here are the most valuable endpoints for your risk assessment agent:

## 🔴 **Critical Endpoints for Risk Assessment**

### 1. **Protocol TVL & Health Monitoring**

#### `/protocol/{name}` - GET Protocol TVL
```python
# Why: Monitor protocol health and TVL trends
# Risk Signal: Rapid TVL decline = exit risk
endpoint = "https://api.llama.fi/protocol/aave"
returns = {
    "tvl": [{"date": 1609459200, "totalLiquidityUSD": 1000000}],
    "chainTvls": {...},  # TVL breakdown by chain
    "tokensInUsd": [...],  # Token composition
    "tokens": [...]  # Detailed token breakdown
}
# Use for: Protocol risk scoring, migration detection
```

#### `/protocols` - GET All Protocols TVL
```python
# Why: Compare protocols, find safer alternatives
# Risk Signal: Identify protocols with better TVL stability
endpoint = "https://api.llama.fi/protocols"
# Use for: Relative protocol ranking, diversification recommendations
```

### 2. **Hack & Exploit Detection**

#### `/hacks` - GET Hacks History 🚨
```python
# Why: CRITICAL for risk assessment
# Risk Signal: Recent hacks, protocol vulnerability patterns
endpoint = "https://api.llama.fi/hacks"
returns = [{
    "date": timestamp,
    "name": "Protocol Name",
    "classification": "exploit type",
    "technique": "how it happened",
    "amount": 1000000,  # $ lost
    "chain": ["Ethereum"],
    "bridgeHack": false
}]
# Use for: Blacklisting risky protocols, hack pattern analysis
```

### 3. **Stablecoin Risk Monitoring**

#### `/stablecoins` - GET Stablecoin Market Caps
```python
# Why: Monitor stablecoin depegs and risks
# Risk Signal: Declining mcap, concentration risk
endpoint = "https://api.llama.fi/stablecoins"
# Use for: Stablecoin safety scoring, depeg risk assessment
```

#### `/stablecoin/{asset}` - GET Specific Stablecoin History
```python
# Why: Track individual stablecoin health
endpoint = "https://api.llama.fi/stablecoin/1"  # USDT
# Use for: Historical stability analysis, depeg detection
```

### 4. **Liquidity & Slippage Risk**

#### `/dexs/{protocol}` - GET DEX Volume
```python
# Why: Volume = liquidity = lower slippage risk
# Risk Signal: Low volume = high slippage on exit
endpoint = "https://api.llama.fi/dexs/uniswap"
# Use for: Exit cost estimation, liquidity risk scoring
```

### 5. **Yield & APY Monitoring**

#### `/pools` - GET Yield Pools
```python
# Why: Identify unsustainable yields (ponzi risk)
# Risk Signal: APY >100% = high risk
endpoint = "https://api.llama.fi/pools"
returns = [{
    "pool": "pool_id",
    "apy": 15.5,  # Annual percentage yield
    "tvl": 1000000,
    "ilRisk": "high",  # Impermanent loss risk
    "exposure": "multi",  # Asset exposure
}]
# Use for: Yield sustainability analysis, IL risk assessment
```

### 6. **Chain Risk Assessment**

#### `/chains` - GET Current TVL of All Chains
```python
# Why: Chain concentration risk
# Risk Signal: Over-exposure to single chain
endpoint = "https://api.llama.fi/v2/chains"
# Use for: Chain diversification recommendations
```

### 7. **Bridge Risk Monitoring**

#### `/bridges` - GET Bridge Volume
```python
# Why: Bridge hacks are common
# Risk Signal: Using vulnerable bridges
endpoint = "https://api.llama.fi/bridges"
# Use for: Bridge risk scoring, safer route recommendations
```

### 8. **Oracle & Price Risk**

#### `/oracles` - GET Oracle Secured Value
```python
# Why: Oracle attacks are common exploit vectors
# Risk Signal: Low oracle security = high risk
endpoint = "https://api.llama.fi/oracles"
# Use for: Protocol oracle risk assessment
```

---

## 📊 **Risk Score Calculation Implementation**

```python
class DefiLlamaRiskAnalyzer:
    def __init__(self):
        self.base_url = "https://api.llama.fi"
        
    def calculate_protocol_risk_score(self, protocol_name: str) -> dict:
        risk_factors = {}
        
        # 1. TVL Trend Risk (30% weight)
        tvl_data = self.get_protocol_tvl(protocol_name)
        risk_factors['tvl_risk'] = self.calculate_tvl_risk(tvl_data)
        # Declining TVL = higher risk
        
        # 2. Hack History Risk (25% weight)
        hack_data = self.get_hack_history()
        risk_factors['hack_risk'] = self.calculate_hack_risk(protocol_name, hack_data)
        # Recent hacks or similar protocols hacked = higher risk
        
        # 3. Concentration Risk (15% weight)
        token_distribution = self.get_token_distribution(protocol_name)
        risk_factors['concentration_risk'] = self.calculate_concentration(token_distribution)
        # Single token dominance = higher risk
        
        # 4. Yield Sustainability Risk (15% weight)
        yield_data = self.get_pool_apys(protocol_name)
        risk_factors['yield_risk'] = self.calculate_yield_risk(yield_data)
        # APY >50% = higher risk
        
        # 5. Oracle Risk (10% weight)
        oracle_data = self.get_oracle_security(protocol_name)
        risk_factors['oracle_risk'] = self.calculate_oracle_risk(oracle_data)
        # Low oracle TVS = higher risk
        
        # 6. Bridge Exposure Risk (5% weight)
        bridge_usage = self.get_bridge_exposure(protocol_name)
        risk_factors['bridge_risk'] = self.calculate_bridge_risk(bridge_usage)
        # High bridge dependence = higher risk
        
        # Weighted average
        weights = {
            'tvl_risk': 0.30,
            'hack_risk': 0.25,
            'concentration_risk': 0.15,
            'yield_risk': 0.15,
            'oracle_risk': 0.10,
            'bridge_risk': 0.05
        }
        
        total_risk = sum(risk_factors[key] * weights[key] for key in weights)
        
        return {
            'total_risk_score': total_risk,
            'risk_factors': risk_factors,
            'risk_level': self.categorize_risk(total_risk),
            'recommendations': self.generate_risk_recommendations(risk_factors)
        }
```

---

## 🎯 **Priority Implementation Order**

### Must Have (Week 1):
1. `/hacks` - Critical for safety
2. `/protocol/{name}` - Core protocol health
3. `/pools` - Yield risk assessment

### Should Have (Week 2):
4. `/stablecoins` - Stablecoin risk
5. `/chains` - Chain concentration
6. `/dexs/{protocol}` - Liquidity risk

### Nice to Have (Week 3):
7. `/bridges` - Bridge risk
8. `/oracles` - Oracle security
9. `/stablecoin/{asset}` - Specific stablecoin monitoring

---

## 💡 **Specific Risk Signals to Monitor**

```python
RISK_THRESHOLDS = {
    'tvl_decline': -20,  # 20% TVL drop in 7 days
    'hack_recency': 90,  # Hack within 90 days
    'apy_unsustainable': 100,  # APY over 100%
    'concentration_high': 0.5,  # 50%+ in single token
    'oracle_tvs_low': 1000000,  # <$1M secured
    'stablecoin_depeg': 0.02,  # 2% off peg
    'liquidity_low': 100000,  # <$100k daily volume
}
```

## 🔄 **Update Frequency**

```python
UPDATE_SCHEDULE = {
    'tvl_data': '1_hour',  # Critical, monitor closely
    'hack_data': '6_hours',  # Check for new exploits
    'yield_data': '4_hours',  # APYs change frequently
    'stablecoin_data': '30_minutes',  # Depeg risk
    'oracle_data': 'daily',  # Slower changes
    'bridge_data': 'daily'  # Volume patterns
}
```

These endpoints provide comprehensive risk assessment capabilities. The hack database is particularly valuable - it's rare to have such detailed exploit history available via API. Combined with TVL trends and yield monitoring, you can build a robust risk scoring system that anticipates problems before they occur.