"""
Clase principal que integra todos los módulos
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from src.config.config import TradingConfig, DEFAULT_CONFIG
from src.data.data_fetcher import DataFetcher
from src.indicators.technical_indicators import TechnicalIndicators
from src.labeling.market_labeler import MarketLabeler
from src.features.feature_engineer import FeatureEngineer
from src.models.ml_model import MLModel
from src.backtesting.backtester_advanced import AdvancedBacktester



class MarketSentimentTrader:
    """
    Sistema de trading basado en sentimiento de mercado - Versión Robusta
    """
    
    def __init__(self, stock_symbol: str = 'SPYG', config: Optional[TradingConfig] = None):
        """
        Inicializa el sistema de trading
        """
        self.stock_symbol = stock_symbol
        self.config = config or DEFAULT_CONFIG
        
        # Inicializar módulos
        self.data_fetcher = DataFetcher()
        self.technical_indicators = TechnicalIndicators()
        self.market_labeler = MarketLabeler()
        self.feature_engineer = FeatureEngineer()
        self.ml_model = MLModel(self.config)
        self.backtester = AdvancedBacktester(self.config)
        
        # Estado interno
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.vix_percentiles: Optional[np.ndarray] = None
        self.is_data_loaded = False
        self.is_model_trained = False
    
    def fetch_data(self, period: str = '5y') -> pd.DataFrame:
        """Obtiene datos históricos con validación mejorada"""
        try:
            self.raw_data = self.data_fetcher.fetch_stock_data(self.stock_symbol, period)
            
            if not self.data_fetcher.validate_data(self.raw_data):
                raise ValueError("Datos inválidos")
            
            # Verificar que tenemos suficientes datos
            min_required_days = max(300, self.config.LOOKBACK_PERIOD)  # Mínimo 300 días
            if len(self.raw_data) < min_required_days:
                print(f"⚠️  Advertencia: Solo {len(self.raw_data)} días de datos (recomendado: {min_required_days}+)")
                
                # Intentar obtener más datos
                try:
                    print("🔄 Intentando obtener período máximo de datos...")
                    self.raw_data = self.data_fetcher.fetch_stock_data(self.stock_symbol, 'max')
                    if len(self.raw_data) < 100:  # Mínimo absoluto
                        raise ValueError(f"Datos insuficientes: solo {len(self.raw_data)} días disponibles")
                except:
                    if len(self.raw_data) < 100:
                        raise ValueError(f"Datos insuficientes para análisis confiable")
            
            self.is_data_loaded = True
            print(f"✅ Datos obtenidos: {len(self.raw_data)} registros desde {self.raw_data.index[0].strftime('%Y-%m-%d')} hasta {self.raw_data.index[-1].strftime('%Y-%m-%d')}")
            return self.raw_data
            
        except Exception as e:
            raise RuntimeError(f"Error al obtener datos para {self.stock_symbol}: {str(e)}")
    
    def calculate_indicators(self, data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """Calcula indicadores técnicos"""
        if data is None:
            if not self.is_data_loaded or self.raw_data is None:
                raise ValueError("Primero debe cargar los datos con fetch_data()")
            data = self.raw_data
        
        try:
            self.processed_data, self.vix_percentiles = self.technical_indicators.calculate_all_indicators(
                data, self.config
            )
            
            # Verificar que los indicadores se calcularon correctamente
            required_indicators = ['RSI', 'Volatility_20', 'BB_Position', 'MACD_Histogram']
            missing_indicators = [ind for ind in required_indicators if ind not in self.processed_data.columns]
            
            if missing_indicators:
                print(f"⚠️  Indicadores faltantes: {missing_indicators}")
            
            # Limpiar datos nulos
            initial_length = len(self.processed_data)
            self.processed_data = self.processed_data.dropna()
            final_length = len(self.processed_data)
            
            if final_length < initial_length * 0.8:  # Perdimos más del 20% de datos
                print(f"⚠️  Se perdieron {initial_length - final_length} registros por valores nulos")
            
            if final_length < 100:
                raise ValueError(f"Datos insuficientes después de calcular indicadores: {final_length} registros")
            
            print(f"✅ Indicadores calculados: {final_length} registros válidos")
            return self.processed_data, self.vix_percentiles
            
        except Exception as e:
            raise RuntimeError(f"Error al calcular indicadores: {str(e)}")
    
    def create_market_labels(self, data: Optional[pd.DataFrame] = None, 
                           vix_percentiles: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Crea etiquetas de mercado con validación"""
        if data is None:
            data = self.processed_data
        if vix_percentiles is None:
            vix_percentiles = self.vix_percentiles
            
        if data is None or vix_percentiles is None:
            raise ValueError("Primero debe calcular los indicadores con calculate_indicators()")
        
        try:
            # Crear etiquetas con balance automático
            labeled_data = self.market_labeler.create_market_labels(
                data, vix_percentiles, balance_classes=True
            )
            
            # Validar las etiquetas
            validation = self.market_labeler.validate_labels(labeled_data)
            
            print(f"\n📊 Validación de etiquetas:")
            print(f"   Total de muestras: {validation['total_samples']}")
            print(f"   Mínimo por clase: {validation['min_class_size']}")
            print(f"   ¿Puede entrenar ML?: {'✅ Sí' if validation['can_train_ml'] else '❌ No'}")
            print(f"   ¿Está balanceado?: {'✅ Sí' if validation['is_balanced'] else '⚠️ Parcialmente'}")
            
            if validation['recommendations']:
                print(f"   💡 Recomendaciones:")
                for rec in validation['recommendations']:
                    print(f"     - {rec}")
            
            if not validation['can_train_ml']:
                raise ValueError("❌ Las etiquetas generadas no son suficientes para entrenar ML")
            
            return labeled_data
            
        except Exception as e:
            raise RuntimeError(f"Error al crear etiquetas: {str(e)}")
    
    def train_model(self, data: pd.DataFrame) -> float:
        """Entrena el modelo de ML con manejo robusto"""
        try:
            print(f"\n🤖 Preparando entrenamiento del modelo...")
            
            # Verificar que tenemos las columnas necesarias
            if 'Market_State' not in data.columns:
                raise ValueError("Los datos no contienen etiquetas de mercado")
            
            # Preparar features
            features_df = self.feature_engineer.prepare_features(data, self.config.BASE_FEATURES)
            
            if features_df.empty:
                raise ValueError("No se pudieron generar features válidos")
            
            # Obtener etiquetas correspondientes
            labels = data.loc[features_df.index, 'Market_State']
            
            print(f"📊 Datos para entrenamiento:")
            print(f"   Features: {features_df.shape[1]} columnas, {features_df.shape[0]} filas")
            print(f"   Features disponibles: {list(features_df.columns)}")
            
            # Verificar alineación
            if len(features_df) != len(labels):
                raise ValueError("Desalineación entre features y etiquetas")
            
            # Entrenar modelo
            accuracy = self.ml_model.train(features_df, labels)
            self.is_model_trained = True
            
            # Mostrar información del modelo
            model_info = self.ml_model.get_model_info()
            if model_info.get('class_distribution'):
                dist = model_info['class_distribution']
                if dist.get('needs_rebalancing'):
                    print("⚠️  Nota: El modelo utilizó rebalanceo de clases")
            
            return accuracy
            
        except Exception as e:
            print(f"❌ Error detallado en entrenamiento: {str(e)}")
            raise RuntimeError(f"Error al entrenar modelo: {str(e)}")
    
    def predict_market_state(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Predice estado actual del mercado"""
        if not self.is_model_trained:
            raise ValueError("Modelo no entrenado. Ejecutar train_model() primero.")
        
        if data is None:
            data = self.processed_data
            
        if data is None:
            raise ValueError("No hay datos disponibles")
        
        try:
            features = self.feature_engineer.prepare_features(data, self.config.BASE_FEATURES)
            
            if features.empty:
                raise ValueError("No se pudieron generar features para predicción")
            
            prediction = self.ml_model.predict(features)
            return prediction
            
        except Exception as e:
            print(f"⚠️  Error en predicción: {str(e)}")
            # Retornar predicción conservadora en caso de error
            return {
                'prediction': 'Normal/Confianza',
                'probabilities': {'Miedo Extremo': 0.33, 'Normal/Confianza': 0.34, 'Euforia': 0.33},
                'signal': 'MANTENER',
                'prediction_code': 1,
                'note': 'Predicción conservadora debido a error'
            }
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Obtiene importancia de features"""
        return self.ml_model.get_feature_importance()
    
    def backtest_strategy(self, data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Ejecuta backtesting de la estrategia"""
        if data is None:
            data = self.processed_data
            
        if data is None:
            raise ValueError("No hay datos disponibles para backtesting")
        
        try:
            return self.backtester.run_backtest(data)
            
        except Exception as e:
            raise RuntimeError(f"Error en backtesting: {str(e)}")
    
    def run_complete_analysis(self, period: str = '5y') -> Dict[str, Any]:
        """
        Ejecuta análisis completo del sistema con manejo robusto de errores
        """
        print(f"🚀 ANÁLISIS COMPLETO PARA {self.stock_symbol}")
        print("=" * 60)
        
        try:
            # 1. Obtener datos
            print("\n1️⃣ Obteniendo datos históricos...")
            raw_data = self.fetch_data(period)
            
            # 2. Calcular indicadores
            print("\n2️⃣ Calculando indicadores técnicos...")
            processed_data, vix_percentiles = self.calculate_indicators()
            
            # 3. Crear etiquetas
            print("\n3️⃣ Creando etiquetas de mercado...")
            labeled_data = self.create_market_labels()
            
            # 4. Entrenar modelo
            print("\n4️⃣ Entrenando modelo ML...")
            try:
                accuracy = self.train_model(labeled_data)
            except Exception as e:
                print(f"⚠️  Error en entrenamiento ML: {str(e)}")
                print("🔄 Continuando sin modelo ML...")
                accuracy = 0.0
                self.is_model_trained = False
            
            # 5. Predicción actual (solo si el modelo está entrenado)
            print("\n5️⃣ Generando predicción actual...")
            if self.is_model_trained:
                try:
                    current_prediction = self.predict_market_state()
                except Exception as e:
                    print(f"⚠️  Error en predicción: {str(e)}")
                    current_prediction = {
                        'prediction': 'No disponible',
                        'probabilities': {},
                        'signal': 'MANTENER',
                        'note': 'Error en predicción'
                    }
            else:
                current_prediction = {
                    'prediction': 'No disponible (modelo no entrenado)',
                    'probabilities': {},
                    'signal': 'MANTENER',
                    'note': 'Modelo no pudo ser entrenado'
                }
            
            # 6. Importancia de features
            print("\n6️⃣ Analizando importancia de features...")
            feature_importance = self.get_feature_importance()
            
            # 7. Backtesting
            print("\n7️⃣ Ejecutando backtesting...")
            try:
                backtest_data, performance_metrics = self.backtest_strategy()
            except Exception as e:
                print(f"⚠️  Error en backtesting: {str(e)}")
                print("🔄 Usando métricas básicas...")
                backtest_data = labeled_data
                performance_metrics = self._create_fallback_metrics(labeled_data)
            
            # Compilar resultados
            results = {
                'symbol': self.stock_symbol,
                'period': period,
                'data_points': len(labeled_data),
                'model_accuracy': accuracy,
                'model_trained': self.is_model_trained,
                'current_prediction': current_prediction,
                'feature_importance': feature_importance,
                'backtest_results': backtest_data,
                'performance_metrics': performance_metrics,
                'vix_percentiles': {
                    'low': float(vix_percentiles[0]),
                    'high': float(vix_percentiles[1])
                },
                'analysis_status': 'completed_with_warnings' if not self.is_model_trained else 'completed_successfully'
            }
            
            print(f"\n✅ Análisis completado para {self.stock_symbol}")
            if not self.is_model_trained:
                print("⚠️  Nota: Análisis completado con limitaciones en ML")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Error crítico en análisis: {str(e)}")
            raise RuntimeError(f"Error en análisis completo: {str(e)}")
    
    def _create_fallback_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Crea métricas básicas en caso de error en backtesting"""
        period_days = len(data)
        period_years = period_days / 252
        
        # Calcular retorno básico Buy & Hold
        if 'Close' in data.columns:
            total_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
            annual_return = total_return / period_years if period_years > 0 else total_return
        else:
            total_return = 0.0
            annual_return = 0.0
        
        return {
            'period': {
                'start_date': data.index[0].strftime('%Y-%m-%d'),
                'end_date': data.index[-1].strftime('%Y-%m-%d'),
                'days': period_days,
                'years': period_years
            },
            'strategy': {
                'final_value': self.config.INITIAL_CAPITAL,
                'total_return': 0.0,
                'annual_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_trades': 0,
                'win_rate': 0.0
            },
            'buyhold': {
                'final_value': self.config.INITIAL_CAPITAL * (1 + total_return/100),
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            },
            'dca': {
                'final_value': self.config.INITIAL_CAPITAL,
                'total_return': 0.0,
                'annual_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_investments': 0
            },
            'comparison': {
                'outperformance_vs_buyhold': -total_return,
                'outperformance_vs_dca': 0.0,
                'risk_adjusted_return_vs_buyhold': 0.0,
                'risk_adjusted_return_vs_dca': 0.0
            },
            'trades': [],
            'dca_investments': [],
            'note': 'Métricas básicas debido a error en backtesting completo'
        }