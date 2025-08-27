from dataclasses import dataclass
from typing import Optional
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class Config:
    langchain_api_key: str
    openai_api_key: str
    octav_api_key: str
    nansen_api_key: Optional[str] = None
    redis_url: str = "redis://localhost:6379"
    
    risk_threshold: float = 0.6
    confidence_threshold: float = 0.65
    cache_ttl: int = 3600  # 1 hour
    max_retries: int = 3
    request_timeout: int = 30
    
    # Octav API specific configuration
    octav_base_url: str = "https://api.octav.fi"
    octav_portfolio_timeout: int = 45  # Longer timeout for portfolio requests
    octav_include_nfts: bool = False
    octav_include_images: bool = False
    octav_include_explorer_urls: bool = False
    octav_wait_for_sync: bool = False
    
    model_name: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    cluster_min_samples: int = 5
    cluster_eps: float = 0.3
    
    psi_threshold: float = 0.2
    drift_check_frequency: int = 100
    
    def validate(self) -> bool:
        required_keys = {
            'langchain_api_key': self.langchain_api_key,
            'openai_api_key': self.openai_api_key,
            'octav_api_key': self.octav_api_key,
        }
        
        missing_keys = []
        for key, value in required_keys.items():
            if not value or value == f"your_{key}_here":
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        if not 0 <= self.risk_threshold <= 1:
            raise ValueError("Risk threshold must be between 0 and 1")
        
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        if self.cache_ttl < 0:
            raise ValueError("Cache TTL must be positive")
        
        if self.max_retries < 0:
            raise ValueError("Max retries must be non-negative")
        
        if self.request_timeout <= 0:
            raise ValueError("Request timeout must be positive")
        
        logging.info("Configuration validated successfully")
        return True
    
    @classmethod
    def from_env(cls) -> 'Config':
        config = cls(
            langchain_api_key=os.getenv('LANGCHAIN_API_KEY', ''),
            openai_api_key=os.getenv('OPENAI_API_KEY', ''),
            octav_api_key=os.getenv('OCTAV_API_KEY', ''),
            nansen_api_key=os.getenv('NANSEN_API_KEY'),
            redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379'),
            
            risk_threshold=float(os.getenv('RISK_THRESHOLD', '0.6')),
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.65')),
            cache_ttl=int(os.getenv('CACHE_TTL', '3600')),
            max_retries=int(os.getenv('MAX_RETRIES', '3')),
            request_timeout=int(os.getenv('REQUEST_TIMEOUT', '30')),
            
            # Octav API specific configuration
            octav_base_url=os.getenv('OCTAV_BASE_URL', 'https://api.octav.fi'),
            octav_portfolio_timeout=int(os.getenv('OCTAV_PORTFOLIO_TIMEOUT', '45')),
            octav_include_nfts=os.getenv('OCTAV_INCLUDE_NFTS', 'false').lower() == 'true',
            octav_include_images=os.getenv('OCTAV_INCLUDE_IMAGES', 'false').lower() == 'true',
            octav_include_explorer_urls=os.getenv('OCTAV_INCLUDE_EXPLORER_URLS', 'false').lower() == 'true',
            octav_wait_for_sync=os.getenv('OCTAV_WAIT_FOR_SYNC', 'false').lower() == 'true',
            
            model_name=os.getenv('MODEL_NAME', 'gpt-4-turbo-preview'),
            temperature=float(os.getenv('TEMPERATURE', '0.7')),
            max_tokens=int(os.getenv('MAX_TOKENS', '2000')),
            
            cluster_min_samples=int(os.getenv('CLUSTER_MIN_SAMPLES', '5')),
            cluster_eps=float(os.getenv('CLUSTER_EPS', '0.3')),
            
            psi_threshold=float(os.getenv('PSI_THRESHOLD', '0.2')),
            drift_check_frequency=int(os.getenv('DRIFT_CHECK_FREQUENCY', '100')),
        )
        
        return config
    
    def get_logger(self, name: str) -> logging.Logger:
        return logging.getLogger(name)

def get_config() -> Config:
    config = Config.from_env()
    try:
        config.validate()
    except ValueError as e:
        logging.warning(f"Configuration validation warning: {e}")
    return config