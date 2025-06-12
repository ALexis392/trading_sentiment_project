# market_labeler.py - Versión Corregida
"""
Módulo para crear etiquetas de estados de mercado con mejor balance
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


class MarketLabeler:
    """Clase para etiquetar estados de mercado con balance mejorado"""
    
    @staticmethod
    def create_market_labels(data: pd.DataFrame, vix_percentiles: np.ndarray, 
                           balance_classes: bool = True) -> pd.DataFrame:
        """
        Crea etiquetas de mercado balanceadas
        
        Estados:
        0: Miedo Extremo (Comprar)
        1: Normal/Confianza (Mantener)  
        2: Euforia (Vender)
        
        Args:
            data: DataFrame con indicadores
            vix_percentiles: Percentiles del VIX [p10, p90]
            balance_classes: Si balancear las clases automáticamente
            
        Returns:
            DataFrame con columna 'Market_State'
        """
        df = data.copy()
        
        # Calcular percentiles más detallados para mejor balance
        vix_q05 = df['VIX'].quantile(0.05)
        vix_q10 = df['VIX'].quantile(0.10)
        vix_q15 = df['VIX'].quantile(0.15)
        vix_q85 = df['VIX'].quantile(0.85)
        vix_q90 = df['VIX'].quantile(0.90)
        vix_q95 = df['VIX'].quantile(0.95)
        
        rsi_q05 = df['RSI'].quantile(0.05)
        rsi_q10 = df['RSI'].quantile(0.10)
        rsi_q15 = df['RSI'].quantile(0.15)
        rsi_q85 = df['RSI'].quantile(0.85)
        rsi_q90 = df['RSI'].quantile(0.90)
        rsi_q95 = df['RSI'].quantile(0.95)
        
        vol_q85 = df['Volatility_20'].quantile(0.85)
        vol_q90 = df['Volatility_20'].quantile(0.90)
        
        price_change_q10 = df['Price_Change_20'].quantile(0.10)
        price_change_q90 = df['Price_Change_20'].quantile(0.90)
        
        # Inicializar con estado Normal
        df['Market_State'] = 1  # Normal/Confianza por defecto
        
        # === CONDICIONES PARA MIEDO EXTREMO (COMPRAR) ===
        # Hacer condiciones menos restrictivas para obtener más muestras
        
        # Condición 1: VIX muy alto
        fear_vix = df['VIX'] > vix_q85  # Menos restrictivo que q90
        
        # Condición 2: RSI muy bajo (oversold)
        fear_rsi = df['RSI'] < rsi_q15  # Menos restrictivo que q10
        
        # Condición 3: Alta volatilidad
        fear_vol = df['Volatility_20'] > vol_q85
        
        # Condición 4: Precio cerca de banda inferior
        fear_bb = df['BB_Position'] < 0.1  # Menos restrictivo que 0.05
        
        # Condición 5: Caída fuerte de precios
        fear_price = df['Price_Change_20'] < price_change_q10
        
        # Condición 6: Combinación moderada VIX + RSI
        fear_combo = (df['VIX'] > vix_q90) | (df['RSI'] < rsi_q10)
        
        # Aplicar condiciones de miedo (OR lógico para más flexibilidad)
        fear_condition = fear_vix | fear_rsi | fear_vol | fear_bb | fear_price | fear_combo
        
        # === CONDICIONES PARA EUFORIA (VENDER) ===
        # Hacer condiciones menos restrictivas
        
        # Condición 1: VIX muy bajo
        euphoria_vix = df['VIX'] < vix_q15  # Menos restrictivo que q10
        
        # Condición 2: RSI muy alto (overbought)
        euphoria_rsi = df['RSI'] > rsi_q85  # Menos restrictivo que q90
        
        # Condición 3: Precio cerca de banda superior
        euphoria_bb = df['BB_Position'] > 0.9  # Menos restrictivo que 0.95
        
        # Condición 4: Subida fuerte de precios
        euphoria_price = df['Price_Change_20'] > price_change_q90
        
        # Condición 5: MACD positivo fuerte
        euphoria_macd = df['MACD_Histogram'] > df['MACD_Histogram'].quantile(0.8)
        
        # Para euforia, usar AND para ser más selectivo (pero no tanto como antes)
        euphoria_condition = (
            (euphoria_vix & euphoria_rsi) |  # VIX bajo Y RSI alto
            (euphoria_bb & euphoria_price) |  # BB alto Y precio alto
            (euphoria_rsi & euphoria_macd)   # RSI alto Y MACD positivo
        )
        
        # Asignar etiquetas
        df.loc[fear_condition, 'Market_State'] = 0  # Miedo Extremo
        df.loc[euphoria_condition, 'Market_State'] = 2  # Euforia
        
        # === BALANCE AUTOMÁTICO DE CLASES ===
        if balance_classes:
            df = MarketLabeler._balance_market_states(df)
        
        # Verificar distribución final
        state_counts = df['Market_State'].value_counts().sort_index()
        total = len(df)
        
        print(f"\n📊 Distribución final de estados de mercado:")
        state_names = ['Miedo Extremo', 'Normal', 'Euforia']
        for i, name in enumerate(state_names):
            count = state_counts.get(i, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"   {name}: {count} días ({percentage:.1f}%)")
        
        # Verificar que tenemos suficientes muestras de cada clase
        min_samples = state_counts.min() if len(state_counts) > 0 else 0
        if min_samples < 2:
            print(f"⚠️  Advertencia: Clase con pocas muestras ({min_samples})")
            print("   Aplicando balance adicional...")
            df = MarketLabeler._force_minimum_samples(df)
        
        return df
    
    @staticmethod
    def _balance_market_states(df: pd.DataFrame, target_min_percentage: float = 15.0) -> pd.DataFrame:
        """
        Balancea los estados de mercado ajustando umbrales
        """
        state_counts = df['Market_State'].value_counts()
        total = len(df)
        target_min_count = int(total * target_min_percentage / 100)
        
        # Si alguna clase tiene muy pocas muestras, relajar condiciones
        if state_counts.get(0, 0) < target_min_count:  # Miedo Extremo
            # Relajar condiciones para miedo extremo
            additional_fear = (
                (df['VIX'] > df['VIX'].quantile(0.75)) |
                (df['RSI'] < df['RSI'].quantile(0.25)) |
                (df['BB_Position'] < 0.2)
            ) & (df['Market_State'] == 1)  # Solo cambiar los que están en Normal
            
            df.loc[additional_fear, 'Market_State'] = 0
        
        if state_counts.get(2, 0) < target_min_count:  # Euforia
            # Relajar condiciones para euforia
            additional_euphoria = (
                (df['VIX'] < df['VIX'].quantile(0.25)) &
                (df['RSI'] > df['RSI'].quantile(0.75)) &
                (df['BB_Position'] > 0.8)
            ) & (df['Market_State'] == 1)  # Solo cambiar los que están en Normal
            
            df.loc[additional_euphoria, 'Market_State'] = 2
        
        return df
    
    @staticmethod
    def _force_minimum_samples(df: pd.DataFrame, min_samples: int = 5) -> pd.DataFrame:
        """
        Fuerza un mínimo de muestras por clase
        """
        state_counts = df['Market_State'].value_counts()
        
        # Para cada clase que tenga menos del mínimo
        for state in [0, 1, 2]:
            current_count = state_counts.get(state, 0)
            
            if current_count < min_samples:
                needed = min_samples - current_count
                
                if state == 0:  # Miedo Extremo
                    # Tomar muestras con VIX más alto o RSI más bajo
                    candidates = df[df['Market_State'] == 1].copy()
                    if len(candidates) > 0:
                        # Ordenar por VIX descendente + RSI ascendente
                        candidates['score'] = candidates['VIX'].rank(ascending=False) + candidates['RSI'].rank(ascending=True)
                        selected_indices = candidates.nlargest(needed, 'score').index
                        df.loc[selected_indices, 'Market_State'] = 0
                
                elif state == 2:  # Euforia
                    # Tomar muestras con VIX más bajo y RSI más alto
                    candidates = df[df['Market_State'] == 1].copy()
                    if len(candidates) > 0:
                        candidates['score'] = candidates['VIX'].rank(ascending=True) + candidates['RSI'].rank(ascending=False)
                        selected_indices = candidates.nlargest(needed, 'score').index
                        df.loc[selected_indices, 'Market_State'] = 2
                
                elif state == 1:  # Normal
                    # Si falta Normal, tomar de otras clases
                    if state_counts.get(0, 0) > min_samples:
                        excess_fear = df[df['Market_State'] == 0].sample(n=min(needed, state_counts.get(0, 0) - min_samples), random_state=42)
                        df.loc[excess_fear.index, 'Market_State'] = 1
                    elif state_counts.get(2, 0) > min_samples:
                        excess_euphoria = df[df['Market_State'] == 2].sample(n=min(needed, state_counts.get(2, 0) - min_samples), random_state=42)
                        df.loc[excess_euphoria.index, 'Market_State'] = 1
        
        return df
    
    @staticmethod
    def validate_labels(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida que las etiquetas sean apropiadas para ML
        """
        state_counts = df['Market_State'].value_counts().sort_index()
        total = len(df)
        min_samples = state_counts.min() if len(state_counts) > 0 else 0
        
        validation = {
            'total_samples': total,
            'class_counts': state_counts.to_dict(),
            'min_class_size': min_samples,
            'can_train_ml': min_samples >= 2,
            'is_balanced': min_samples >= total * 0.05,  # Al menos 5% por clase
            'recommendations': []
        }
        
        if not validation['can_train_ml']:
            validation['recommendations'].append("Necesita más datos o relajar condiciones de etiquetado")
        
        if not validation['is_balanced']:
            validation['recommendations'].append("Considere balancear las clases")
        
        return validation