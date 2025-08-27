from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from langchain.memory import ConversationBufferMemory
from langsmith import Client
import logging
from datetime import datetime
import json
import traceback
from src.config import get_config

class BaseAgent(ABC):
    
    def __init__(self, project_name: str = "crypto-recommendation"):
        self.config = get_config()
        self.project_name = project_name
        
        try:
            self.client = Client(
                api_key=self.config.langchain_api_key
            ) if self.config.langchain_api_key else None
        except Exception as e:
            logging.warning(f"Failed to initialize LangSmith client: {e}")
            self.client = None
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        self._metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_recommendations': 0,
            'avg_confidence': 0.0,
            'errors': []
        }
        
        self._initialize()
        
    def _initialize(self):
        self.logger.info(f"Initializing {self.__class__.__name__}")
        
    @abstractmethod
    def analyze(self, wallet_address: str, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def generate_recommendation(self, analysis_results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        pass
    
    def track_metrics(self, metric_name: str, value: Any, metadata: Optional[Dict] = None):
        try:
            if self.client:
                self.client.create_run(
                    project_name=self.project_name,
                    run_type="chain",
                    name=metric_name,
                    inputs={"value": value, "metadata": metadata or {}},
                    outputs={"timestamp": datetime.now().isoformat()}
                )
            
            if metric_name == 'analysis_complete':
                self._metrics['total_analyses'] += 1
                if value.get('success'):
                    self._metrics['successful_analyses'] += 1
                else:
                    self._metrics['failed_analyses'] += 1
            
            elif metric_name == 'recommendation_generated':
                self._metrics['total_recommendations'] += 1
                confidence = value.get('confidence', 0)
                n = self._metrics['total_recommendations']
                self._metrics['avg_confidence'] = (
                    (self._metrics['avg_confidence'] * (n - 1) + confidence) / n
                )
            
            self.logger.info(f"Metric tracked: {metric_name} = {value}")
            
        except Exception as e:
            self.logger.error(f"Failed to track metric {metric_name}: {e}")
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self._metrics['errors'].append(error_data)
        
        self.logger.error(f"Error in {self.__class__.__name__}: {error_data}")
        
        if self.client:
            try:
                self.client.create_run(
                    project_name=self.project_name,
                    name="error_logged",
                    inputs=error_data,
                    error=str(error)
                )
            except:
                pass
    
    def add_to_memory(self, key: str, value: Any):
        self.memory.save_context(
            {"input": key},
            {"output": json.dumps(value) if not isinstance(value, str) else value}
        )
    
    def get_from_memory(self, key: str) -> Optional[Any]:
        messages = self.memory.chat_memory.messages
        for message in reversed(messages):
            if hasattr(message, 'content') and key in message.content:
                try:
                    return json.loads(message.content)
                except:
                    return message.content
        return None
    
    def clear_memory(self):
        self.memory.clear()
        self.logger.info("Memory cleared")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        return {
            **self._metrics,
            'success_rate': (
                self._metrics['successful_analyses'] / max(self._metrics['total_analyses'], 1)
            ),
            'error_count': len(self._metrics['errors'])
        }
    
    def validate_input(self, wallet_address: str) -> bool:
        from src.models.validators import InputValidator
        
        if not InputValidator.validate_ethereum_address(wallet_address):
            self.logger.error(f"Invalid wallet address: {wallet_address}")
            return False
        
        return True
    
    def handle_rate_limit(self, retry_after: int = 60):
        import time
        self.logger.warning(f"Rate limit hit, waiting {retry_after} seconds")
        time.sleep(retry_after)
    
    def with_retry(self, func, max_retries: int = None, *args, **kwargs):
        max_retries = max_retries or self.config.max_retries
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    self.log_error(e, {"function": func.__name__, "args": args, "kwargs": kwargs})
                    raise
                
                if "rate limit" in str(e).lower():
                    self.handle_rate_limit()
                else:
                    import time
                    time.sleep(2 ** attempt)
        
        return None
    
    @abstractmethod
    def explain_recommendation(self, recommendation: Dict[str, Any]) -> str:
        pass
    
    def shutdown(self):
        self.logger.info(f"Shutting down {self.__class__.__name__}")
        self.logger.info(f"Final metrics: {self.get_metrics_summary()}")
        
        if self.client:
            try:
                self.client.create_run(
                    project_name=self.project_name,
                    name="agent_shutdown",
                    inputs={"metrics": self.get_metrics_summary()},
                    outputs={"timestamp": datetime.now().isoformat()}
                )
            except:
                pass