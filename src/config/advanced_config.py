# src/config/advanced_config.py
"""
Configuraciones avanzadas del sistema de trading con múltiples estrategias
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import warnings

warnings.filterwarnings('ignore')

@dataclass
class BacktestingThresholds:
    """Configuración de umbrales para backtesting"""
    
    # Umbrales para VIX
    vix_fear_threshold: float = 0.90  # Percentil para miedo (comprar)
    vix_euphoria_threshold: float = 0.10  # Percentil para euforia (vender)
    vix_moderate_fear: float = 0.85  # Percentil moderado para miedo
    vix_moderate_euphoria: float = 0.15  # Percentil moderado para euforia
    
    # Umbrales para RSI
    rsi_oversold: float = 0.10  # Percentil para sobreventa
    rsi_overbought: float = 0.90  # Percentil para sobrecompra
    rsi_moderate_oversold: float = 0.15  # Percentil moderado sobreventa
    rsi_moderate_overbought: float = 0.85  # Percentil moderado sobrecompra
    
    # Umbrales para Bollinger Bands
    bb_lower_threshold: float = 0.05  # Posición cerca del límite inferior
    bb_upper_threshold: float = 0.95  # Posición cerca del límite superior
    bb_moderate_lower: float = 0.10  # Posición moderada inferior
    bb_moderate_upper: float = 0.90  # Posición moderada superior
    
    # Umbrales para cambio de precios
    price_change_extreme: float = 0.90  # Percentil para cambios extremos
    price_change_moderate: float = 0.80  # Percentil para cambios moderados
    
    # Volatilidad
    volatility_high: float = 0.85  # Percentil para alta volatilidad
    volatility_extreme: float = 0.90  # Percentil para volatilidad extrema
    
    def get_strategy_name(self) -> str:
        """Retorna nombre descriptivo de la estrategia"""
        if (self.vix_fear_threshold >= 0.95 and self.rsi_oversold <= 0.05 and 
            self.bb_lower_threshold <= 0.03):
            return "Muy Estricta"
        elif (self.vix_fear_threshold >= 0.90 and self.rsi_oversold <= 0.10 and 
              self.bb_lower_threshold <= 0.05):
            return "Estricta (Default)"
        elif (self.vix_fear_threshold >= 0.85 and self.rsi_oversold <= 0.15 and 
              self.bb_lower_threshold <= 0.10):
            return "Moderada"
        else:
            return "Laxa/Personalizada"

# Estrategias predefinidas
STRICT_STRATEGY = BacktestingThresholds(
    vix_fear_threshold=0.95,
    vix_euphoria_threshold=0.05,
    vix_moderate_fear=0.90,
    vix_moderate_euphoria=0.10,
    rsi_oversold=0.05,
    rsi_overbought=0.95,
    rsi_moderate_oversold=0.10,
    rsi_moderate_overbought=0.90,
    bb_lower_threshold=0.03,
    bb_upper_threshold=0.97,
    bb_moderate_lower=0.05,
    bb_moderate_upper=0.95,
    price_change_extreme=0.95,
    price_change_moderate=0.85,
    volatility_high=0.90,
    volatility_extreme=0.95
)

MODERATE_STRATEGY = BacktestingThresholds(
    vix_fear_threshold=0.85,
    vix_euphoria_threshold=0.15,
    vix_moderate_fear=0.80,
    vix_moderate_euphoria=0.20,
    rsi_oversold=0.15,
    rsi_overbought=0.85,
    rsi_moderate_oversold=0.20,
    rsi_moderate_overbought=0.80,
    bb_lower_threshold=0.10,
    bb_upper_threshold=0.90,
    bb_moderate_lower=0.15,
    bb_moderate_upper=0.85,
    price_change_extreme=0.85,
    price_change_moderate=0.75,
    volatility_high=0.80,
    volatility_extreme=0.85
)

LAX_STRATEGY = BacktestingThresholds(
    vix_fear_threshold=0.75,
    vix_euphoria_threshold=0.25,
    vix_moderate_fear=0.70,
    vix_moderate_euphoria=0.30,
    rsi_oversold=0.25,
    rsi_overbought=0.75,
    rsi_moderate_oversold=0.30,
    rsi_moderate_overbought=0.70,
    bb_lower_threshold=0.20,
    bb_upper_threshold=0.80,
    bb_moderate_lower=0.25,
    bb_moderate_upper=0.75,
    price_change_extreme=0.75,
    price_change_moderate=0.65,
    volatility_high=0.70,
    volatility_extreme=0.75
)

DEFAULT_STRATEGY = BacktestingThresholds()  # Valores por defecto

@dataclass
class AdvancedTradingConfig:
    """Configuración avanzada del sistema de trading - Compatible con TradingConfig original"""
    
    # Parámetros básicos del sistema original (TODOS los atributos necesarios)
    LOOKBACK_PERIOD: int = 252  # Período de lookback para cálculos (días)
    INITIAL_CAPITAL: float = 10000
    TRANSACTION_COST: float = 0.001
    MIN_LOOKBACK_DAYS: int = 30
    DCA_FREQUENCY: int = 30
    RISK_FREE_RATE: float = 0.02  # Tasa libre de riesgo para Sharpe ratio
    
    # NUEVA: Tasa libre de riesgo anual para estrategias mejoradas
    RISK_FREE_RATE_ANNUAL: float = 0.05  # 5% por defecto
    
    # NUEVA: Ahorro mensual para estrategia 2 (opcional, se calcula automáticamente si es None)
    MONTHLY_SAVINGS: Optional[float] = None
    
    # Parámetros de indicadores técnicos
    RSI_PERIOD: int = 14
    VOLATILITY_PERIODS: List[int] = field(default_factory=lambda: [20, 60])
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
    
    # Features para el modelo - Compatible con TradingConfig original
    BASE_FEATURES: List[str] = field(default_factory=lambda: [
        'VIX', 'RSI', 'Volatility_20', 'Volatility_60', 
        'VIX_SMA', 'Price_Change_20', 'BB_Position', 'MACD_Histogram'
    ])
    
    # Estrategia de backtesting (NUEVA funcionalidad)
    backtesting_thresholds: BacktestingThresholds = field(default_factory=lambda: DEFAULT_STRATEGY)
    
    # Look-ahead bias prevention (NUEVA funcionalidad)
    PREVENT_LOOKAHEAD_BIAS: bool = True
    EXPANDING_WINDOW: bool = True
    
    # NUEVA: Configuración de gráficos
    GENERATE_CHARTS: bool = True
    SAVE_CHARTS: bool = True
    
    def set_backtesting_strategy(self, strategy_type: str = "default"):
        """Establece la estrategia de backtesting"""
        strategies = {
            "strict": STRICT_STRATEGY,
            "moderate": MODERATE_STRATEGY, 
            "lax": LAX_STRATEGY,
            "default": DEFAULT_STRATEGY
        }
        
        if strategy_type.lower() in strategies:
            self.backtesting_thresholds = strategies[strategy_type.lower()]
        else:
            print(f"⚠️  Estrategia '{strategy_type}' no encontrada. Usando 'default'")
            self.backtesting_thresholds = DEFAULT_STRATEGY
    
    def set_risk_free_rate(self, annual_rate: float):
        """Establece la tasa libre de riesgo anual"""
        if 0 <= annual_rate <= 1:
            self.RISK_FREE_RATE_ANNUAL = annual_rate
            # Actualizar también la tasa para Sharpe ratio si es menor
            if annual_rate < self.RISK_FREE_RATE:
                self.RISK_FREE_RATE = annual_rate
            print(f"✅ Tasa libre de riesgo establecida: {annual_rate:.1%} anual")
        else:
            raise ValueError("La tasa libre de riesgo debe estar entre 0 y 1 (0% y 100%)")
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Retorna resumen de la estrategia actual"""
        thresholds = self.backtesting_thresholds
        return {
            'strategy_name': thresholds.get_strategy_name(),
            'vix_thresholds': {
                'fear': f">{thresholds.vix_fear_threshold:.0%} percentil",
                'euphoria': f"<{thresholds.vix_euphoria_threshold:.0%} percentil"
            },
            'rsi_thresholds': {
                'oversold': f"<{thresholds.rsi_oversold:.0%} percentil", 
                'overbought': f">{thresholds.rsi_overbought:.0%} percentil"
            },
            'bb_thresholds': {
                'lower': f"<{thresholds.bb_lower_threshold:.0%} posición",
                'upper': f">{thresholds.bb_upper_threshold:.0%} posición"
            },
            'lookahead_prevention': self.PREVENT_LOOKAHEAD_BIAS,
            'expanding_window': self.EXPANDING_WINDOW,
            'risk_free_rate_annual': self.RISK_FREE_RATE_ANNUAL
        }

@dataclass 
class AnalysisRequest:
    """Configuración para solicitud de análisis"""
    
    # Símbolos a analizar
    symbols: List[str] = field(default_factory=lambda: ['SPYG'])
    
    # Períodos específicos (en años)
    periods: List[int] = field(default_factory=lambda: [3, 5])
    
    # Período personalizado (para yfinance: '1y', '2y', '5y', '10y', 'max')
    custom_periods: List[str] = field(default_factory=list)
    
    # Si incluir período máximo
    include_max_period: bool = True
    
    # Configuración de trading
    trading_config: AdvancedTradingConfig = field(default_factory=AdvancedTradingConfig)
    
    # Tipo de análisis
    analysis_type: str = "complete"  # "complete", "quick", "comparison_only"
    
    # Estrategia de backtesting
    backtesting_strategy: str = "default"  # "strict", "moderate", "lax", "default", "custom"
    
    def validate(self) -> bool:
        """Valida la configuración de análisis"""
        if not self.symbols:
            raise ValueError("Debe especificar al menos un símbolo")
        
        if not self.periods and not self.custom_periods and not self.include_max_period:
            raise ValueError("Debe especificar al menos un período de análisis")
        
        valid_strategies = ["strict", "moderate", "lax", "default", "custom"]
        if self.backtesting_strategy not in valid_strategies:
            raise ValueError(f"Estrategia debe ser una de: {valid_strategies}")
        
        return True
    
    def get_all_periods(self) -> List[str]:
        """Retorna todos los períodos a analizar"""
        periods = []
        
        # Agregar períodos en años
        for years in self.periods:
            periods.append(f"{years}y")
        
        # Agregar períodos personalizados  
        periods.extend(self.custom_periods)
        
        # Agregar período máximo
        if self.include_max_period:
            periods.append("max")
        
        return periods

# Configuraciones predefinidas mejoradas
QUICK_ANALYSIS_ENHANCED = AnalysisRequest(
    symbols=['SPYG'],
    periods=[3],
    include_max_period=False,
    analysis_type="quick",
    backtesting_strategy="default",
    trading_config=AdvancedTradingConfig(
        RISK_FREE_RATE_ANNUAL=0.05,
        GENERATE_CHARTS=True
    )
)

COMPREHENSIVE_ANALYSIS_ENHANCED = AnalysisRequest(
    symbols=['SPYG', 'QQQ', 'BTC-USD'],
    periods=[1, 3, 5, 10],
    include_max_period=True,
    analysis_type="complete", 
    backtesting_strategy="moderate",
    trading_config=AdvancedTradingConfig(
        RISK_FREE_RATE_ANNUAL=0.05,
        GENERATE_CHARTS=True,
        INITIAL_CAPITAL=25000  # Capital mayor para análisis comprehensivo
    )
)

CRYPTO_ANALYSIS_ENHANCED = AnalysisRequest(
    symbols=['BTC-USD', 'ETH-USD'],
    periods=[1, 2, 5],
    include_max_period=True,
    analysis_type="complete",
    backtesting_strategy="lax",  # Criptos son más volátiles
    trading_config=AdvancedTradingConfig(
        RISK_FREE_RATE_ANNUAL=0.05,
        GENERATE_CHARTS=True,
        TRANSACTION_COST=0.002  # Mayor costo para criptos
    )
)