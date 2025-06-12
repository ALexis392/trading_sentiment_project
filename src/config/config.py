"""
Configuraciones y constantes del sistema de trading
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import warnings

warnings.filterwarnings('ignore')

@dataclass
class TradingConfig:
    """Configuración principal del sistema de trading"""
    # Parámetros del mercado
    LOOKBACK_PERIOD: int = 252
    INITIAL_CAPITAL: float = 10000
    TRANSACTION_COST: float = 0.001
    MIN_LOOKBACK_DAYS: int = 30
    DCA_FREQUENCY: int = 30
    RISK_FREE_RATE: float = 0.02
    
    # Parámetros de indicadores técnicos
    RSI_PERIOD: int = 14
    VOLATILITY_PERIODS: List[int] = None
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: int = 2
    VIX_SMA_PERIOD: int = 10
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    # Parámetros del modelo ML
    ML_TEST_SIZE: float = 0.3
    ML_RANDOM_STATE: int = 42
    RF_N_ESTIMATORS: int = 100
    RF_MAX_DEPTH: int = 10
    
    # Features para el modelo
    BASE_FEATURES: List[str] = None
    
    def __post_init__(self):
        if self.VOLATILITY_PERIODS is None:
            self.VOLATILITY_PERIODS = [20, 60]
        
        if self.BASE_FEATURES is None:
            self.BASE_FEATURES = [
                'VIX', 'RSI', 'Volatility_20', 'Volatility_60', 
                'VIX_SMA', 'Price_Change_20', 'BB_Position', 'MACD_Histogram'
            ]

# Configuración por defecto
DEFAULT_CONFIG = TradingConfig()
