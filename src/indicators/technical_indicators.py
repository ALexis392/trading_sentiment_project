"""
Módulo para cálculo de indicadores técnicos
"""
import pandas as pd
import numpy as np
from typing import Tuple
from src.config.config import TradingConfig


class TechnicalIndicators:
    """Clase para calcular indicadores técnicos"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula el RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, period: int = 20) -> pd.Series:
        """Calcula volatilidad histórica anualizada"""
        return returns.rolling(window=period).std() * np.sqrt(252) * 100
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula Bandas de Bollinger"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band, sma
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame, config: TradingConfig) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Calcula todos los indicadores técnicos
        
        Args:
            data: DataFrame con datos de precios
            config: Configuración del sistema
            
        Returns:
            DataFrame con indicadores y percentiles del VIX
        """
        df = data.copy()
        
        # RSI
        df['RSI'] = TechnicalIndicators.calculate_rsi(df['Close'], config.RSI_PERIOD)
        
        # Retornos y volatilidad
        df['Returns'] = df['Close'].pct_change()
        for period in config.VOLATILITY_PERIODS:
            df[f'Volatility_{period}'] = TechnicalIndicators.calculate_volatility(df['Returns'], period)
        
        # VIX suavizado
        df['VIX_SMA'] = df['VIX'].rolling(window=config.VIX_SMA_PERIOD).mean()
        
        # Cambios de precio
        df['Price_Change_5'] = df['Close'].pct_change(5) * 100
        df['Price_Change_20'] = df['Close'].pct_change(20) * 100
        
        # Bandas de Bollinger
        bb_upper, bb_lower, bb_sma = TechnicalIndicators.calculate_bollinger_bands(
            df['Close'], config.BOLLINGER_PERIOD, config.BOLLINGER_STD
        )
        df['SMA_20'] = bb_sma
        df['BB_Upper'] = bb_upper
        df['BB_Lower'] = bb_lower
        df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # MACD
        macd, macd_signal, macd_histogram = TechnicalIndicators.calculate_macd(
            df['Close'], config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL
        )
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Histogram'] = macd_histogram
        
        # Percentiles del VIX
        vix_percentiles = df['VIX'].quantile([0.1, 0.9]).values
        
        return df, vix_percentiles
