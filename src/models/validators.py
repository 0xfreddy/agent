from typing import Any, Dict, List
import re
from datetime import datetime, timedelta

class InputValidator:
    
    @staticmethod
    def validate_ethereum_address(address: str) -> bool:
        pattern = r'^0x[a-fA-F0-9]{40}$'
        return bool(re.match(pattern, address))
    
    @staticmethod
    def validate_transaction_hash(tx_hash: str) -> bool:
        pattern = r'^0x[a-fA-F0-9]{64}$'
        return bool(re.match(pattern, tx_hash))
    
    @staticmethod
    def validate_positive_amount(amount: float) -> bool:
        try:
            return float(amount) > 0
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_percentage(value: float) -> bool:
        try:
            return 0 <= float(value) <= 100
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_confidence_score(score: float) -> bool:
        try:
            return 0 <= float(score) <= 1
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_risk_score(score: float) -> bool:
        try:
            return 0 <= float(score) <= 100
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_timestamp(timestamp: Any) -> bool:
        if isinstance(timestamp, datetime):
            return True
        if isinstance(timestamp, str):
            try:
                datetime.fromisoformat(timestamp)
                return True
            except ValueError:
                return False
        if isinstance(timestamp, (int, float)):
            try:
                datetime.fromtimestamp(timestamp)
                return True
            except (ValueError, OSError):
                return False
        return False
    
    @staticmethod
    def validate_mood(mood: str) -> bool:
        valid_moods = ['degen', 'balanced', 'saver']
        return mood.lower() in valid_moods
    
    @staticmethod
    def validate_action(action: str) -> bool:
        valid_actions = [
            'swap', 'rebalance', 'hold', 'add_liquidity',
            'remove_liquidity', 'stake', 'unstake'
        ]
        return action.lower() in valid_actions
    
    @staticmethod
    def validate_token_symbol(symbol: str) -> bool:
        pattern = r'^[A-Z0-9]{2,12}$'
        return bool(re.match(pattern, symbol.upper()))
    
    @staticmethod
    def validate_protocol_name(name: str) -> bool:
        if not name or len(name) < 2 or len(name) > 50:
            return False
        pattern = r'^[a-zA-Z0-9\-\.]+$'
        return bool(re.match(pattern, name))

class DataValidator:
    
    @staticmethod
    def validate_wallet_data(data: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        warnings = []
        
        if 'address' not in data:
            errors.append("Missing wallet address")
        elif not InputValidator.validate_ethereum_address(data['address']):
            errors.append("Invalid wallet address format")
        
        if 'transactions' in data:
            if not isinstance(data['transactions'], list):
                errors.append("Transactions must be a list")
            else:
                for i, tx in enumerate(data['transactions']):
                    tx_errors = DataValidator._validate_transaction(tx, i)
                    errors.extend(tx_errors)
        
        if 'balances' in data:
            if not isinstance(data['balances'], dict):
                errors.append("Balances must be a dictionary")
            else:
                for token, balance in data['balances'].items():
                    if not InputValidator.validate_positive_amount(balance):
                        warnings.append(f"Invalid balance for token {token}")
        
        if 'total_value_usd' in data:
            if not InputValidator.validate_positive_amount(data['total_value_usd']):
                warnings.append("Total value should be positive")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    @staticmethod
    def _validate_transaction(tx: Dict[str, Any], index: int) -> List[str]:
        errors = []
        prefix = f"Transaction {index}: "
        
        if 'tx_hash' not in tx:
            errors.append(f"{prefix}Missing transaction hash")
        elif not InputValidator.validate_transaction_hash(tx['tx_hash']):
            errors.append(f"{prefix}Invalid transaction hash")
        
        if 'timestamp' not in tx:
            errors.append(f"{prefix}Missing timestamp")
        elif not InputValidator.validate_timestamp(tx['timestamp']):
            errors.append(f"{prefix}Invalid timestamp format")
        
        if 'value_usd' in tx and not InputValidator.validate_positive_amount(tx['value_usd']):
            errors.append(f"{prefix}Invalid USD value")
        
        if 'gas_paid' in tx and not InputValidator.validate_positive_amount(tx['gas_paid']):
            errors.append(f"{prefix}Invalid gas amount")
        
        return errors
    
    @staticmethod
    def validate_recommendation_request(data: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        warnings = []
        
        if 'wallet_address' not in data:
            errors.append("Missing wallet address")
        elif not InputValidator.validate_ethereum_address(data['wallet_address']):
            errors.append("Invalid wallet address")
        
        if 'mood' in data and not InputValidator.validate_mood(data['mood']):
            errors.append(f"Invalid mood: {data['mood']}")
        
        if 'risk_threshold' in data:
            if not InputValidator.validate_confidence_score(data['risk_threshold']):
                errors.append("Risk threshold must be between 0 and 1")
        
        if 'confidence_threshold' in data:
            if not InputValidator.validate_confidence_score(data['confidence_threshold']):
                errors.append("Confidence threshold must be between 0 and 1")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    @staticmethod
    def validate_risk_profile(data: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        warnings = []
        
        if 'risk_score' not in data:
            errors.append("Missing risk score")
        elif not InputValidator.validate_risk_score(data['risk_score']):
            errors.append("Risk score must be between 0 and 100")
        
        if 'portfolio_volatility' in data and data['portfolio_volatility'] < 0:
            errors.append("Portfolio volatility cannot be negative")
        
        if 'concentration_risk' in data:
            if not InputValidator.validate_confidence_score(data['concentration_risk']):
                errors.append("Concentration risk must be between 0 and 1")
        
        if 'risk_category' not in data:
            warnings.append("Risk category not specified")
        elif data['risk_category'] not in ['low', 'medium', 'high', 'very_high']:
            errors.append("Invalid risk category")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

class TimeValidator:
    
    @staticmethod
    def is_recent(timestamp: datetime, max_age_hours: int = 24) -> bool:
        now = datetime.now()
        age = now - timestamp
        return age <= timedelta(hours=max_age_hours)
    
    @staticmethod
    def is_trading_hours(timestamp: datetime) -> bool:
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        if weekday >= 5:
            return True
        
        return True
    
    @staticmethod
    def validate_time_range(start: datetime, end: datetime) -> bool:
        return start < end
    
    @staticmethod
    def validate_data_freshness(data_timestamp: datetime, max_staleness_minutes: int = 60) -> bool:
        now = datetime.now()
        age = now - data_timestamp
        return age <= timedelta(minutes=max_staleness_minutes)