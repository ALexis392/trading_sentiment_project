"""
Módulo para obtención y limpieza de datos de mercado
"""
import yfinance as yf
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime


class DataFetcher:
    """Clase para obtener y procesar datos de mercado"""
    
    @staticmethod
    def fetch_stock_data(symbol: str, period: str = '5y') -> pd.DataFrame:
        """
        Obtiene datos históricos de un símbolo
        
        Args:
            symbol: Símbolo del activo
            period: Período de datos ('1y', '5y', 'max', etc.)
            
        Returns:
            DataFrame con datos históricos limpios
        """
        print(f"Obteniendo datos para {symbol}...")
        
        try:
            # Obtener datos del stock
            stock = yf.download(symbol, period=period, progress=False)
            
            # Obtener datos del VIX
            vix = yf.download('^VIX', period=period, progress=False)
            
            # Combinar datos
            data = pd.DataFrame()
            data['Close'] = stock['Close']
            data['Volume'] = stock['Volume']
            data['High'] = stock['High']
            data['Low'] = stock['Low']
            data['VIX'] = vix['Close']
            
            # Limpiar datos
            data = data.dropna()
            
            if data.empty:
                raise ValueError(f"No se pudieron obtener datos para {symbol}")
                
            return data
            
        except Exception as e:
            raise RuntimeError(f"Error obteniendo datos para {symbol}: {str(e)}")
    
    @staticmethod
    def validate_data(data: pd.DataFrame) -> bool:
        """
        Valida que los datos sean correctos
        
        Args:
            data: DataFrame a validar
            
        Returns:
            True si los datos son válidos
        """
        required_columns = ['Close', 'Volume', 'High', 'Low', 'VIX']
        
        if data.empty:
            return False
            
        if not all(col in data.columns for col in required_columns):
            return False
            
        if data.isnull().all().any():
            return False
            
        return True

