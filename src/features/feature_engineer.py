# feature_engineer.py
"""
Módulo para ingeniería de características (features)
"""
import pandas as pd
from typing import List


class FeatureEngineer:
    """Clase para crear y preparar features para ML"""
    
    @staticmethod
    def create_additional_features(data: pd.DataFrame) -> pd.DataFrame:
        """Crea features adicionales para el modelo"""
        df = data.copy()
        
        # Features adicionales
        df['VIX_RSI_Ratio'] = df['VIX'] / (df['RSI'] + 1)
        df['Vol_Spread'] = df['Volatility_20'] - df['Volatility_60']
        df['VIX_Change'] = df['VIX'].pct_change(5) * 100
        df['RSI_Momentum'] = df['RSI'].diff(5)
        df['MACD_Signal_Cross'] = (df['MACD'] > df['MACD_Signal']).astype(int)
        
        return df
    
    @staticmethod
    def prepare_features(data: pd.DataFrame, base_features: List[str]) -> pd.DataFrame:
        """
        Prepara features para el modelo ML
        
        Args:
            data: DataFrame con datos
            base_features: Lista de features base
            
        Returns:
            DataFrame con features preparados
        """
        # Crear features adicionales
        df_with_features = FeatureEngineer.create_additional_features(data)
        
        # Seleccionar features base
        all_features = base_features + ['VIX_RSI_Ratio', 'Vol_Spread', 'VIX_Change', 'RSI_Momentum', 'MACD_Signal_Cross']
        
        # Seleccionar solo las columnas que existen
        available_features = [f for f in all_features if f in df_with_features.columns]
        feature_df = df_with_features[available_features].copy()
        
        return feature_df.dropna()
