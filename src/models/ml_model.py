# ml_model.py - Versión Corregida
"""
Módulo para modelo de Machine Learning con manejo robusto de clases desbalanceadas
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from typing import Dict, Any, Tuple, Optional, List
import warnings


class MLModel:
    """Clase para modelo de Machine Learning con manejo robusto de datos"""
    
    def __init__(self, config):
        self.config = config
        self.model: Optional[RandomForestClassifier] = None
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self.is_trained = False
        self.class_distribution = None
        self.min_samples_per_class = 5  # Mínimo de muestras por clase
    
    def _check_class_distribution(self, labels: pd.Series) -> Dict[str, Any]:
        """
        Verifica la distribución de clases y determina la estrategia de entrenamiento
        """
        class_counts = labels.value_counts().sort_index()
        min_class_count = class_counts.min()
        total_samples = len(labels)
        
        distribution_info = {
            'class_counts': class_counts.to_dict(),
            'min_class_count': min_class_count,
            'total_samples': total_samples,
            'can_stratify': min_class_count >= 2,
            'sufficient_data': min_class_count >= self.min_samples_per_class,
            'needs_rebalancing': min_class_count < 10
        }
        
        print(f"\n📊 Distribución de clases:")
        for class_label, count in class_counts.items():
            class_name = ['Miedo Extremo', 'Normal', 'Euforia'][class_label]
            percentage = (count / total_samples) * 100
            print(f"   {class_name}: {count} muestras ({percentage:.1f}%)")
        
        if not distribution_info['sufficient_data']:
            print(f"⚠️  Advertencia: Clase con pocas muestras ({min_class_count})")
        
        return distribution_info
    
    def _rebalance_data(self, features_df: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Rebalancea los datos si hay clases con muy pocas muestras
        """
        from sklearn.utils import resample
        
        # Combinar features y labels
        data_combined = features_df.copy()
        data_combined['target'] = labels
        
        # Separar por clase
        class_data = {}
        for class_label in data_combined['target'].unique():
            class_data[class_label] = data_combined[data_combined['target'] == class_label]
        
        # Encontrar el tamaño objetivo (mínimo viable o mediana)
        class_sizes = [len(class_data[c]) for c in class_data.keys()]
        target_size = max(self.min_samples_per_class, int(np.median(class_sizes)))
        
        # Resamplear cada clase
        resampled_data = []
        for class_label, class_df in class_data.items():
            if len(class_df) < target_size:
                # Oversample clases pequeñas
                resampled_class = resample(
                    class_df, 
                    replace=True, 
                    n_samples=target_size, 
                    random_state=self.config.ML_RANDOM_STATE
                )
                print(f"   📈 Clase {class_label}: {len(class_df)} → {target_size} muestras (oversampling)")
            elif len(class_df) > target_size * 3:
                # Undersample clases muy grandes
                resampled_class = resample(
                    class_df, 
                    replace=False, 
                    n_samples=target_size * 2, 
                    random_state=self.config.ML_RANDOM_STATE
                )
                print(f"   📉 Clase {class_label}: {len(class_df)} → {target_size * 2} muestras (undersampling)")
            else:
                resampled_class = class_df
                print(f"   ✅ Clase {class_label}: {len(class_df)} muestras (sin cambios)")
            
            resampled_data.append(resampled_class)
        
        # Combinar datos rebalanceados
        balanced_data = pd.concat(resampled_data, ignore_index=True)
        
        # Mezclar los datos
        balanced_data = balanced_data.sample(frac=1, random_state=self.config.ML_RANDOM_STATE).reset_index(drop=True)
        
        # Separar features y labels
        balanced_features = balanced_data.drop('target', axis=1)
        balanced_labels = balanced_data['target']
        
        return balanced_features, balanced_labels
    
    def train(self, features_df: pd.DataFrame, labels: pd.Series) -> float:
        """
        Entrena el modelo de ML con manejo robusto de clases desbalanceadas
        """
        print("Entrenando modelo de Machine Learning...")
        
        # Verificar distribución de clases
        distribution_info = self._check_class_distribution(labels)
        self.class_distribution = distribution_info
        
        # Guardar nombres de features
        self.feature_names = list(features_df.columns)
        
        # Manejar datos insuficientes
        if not distribution_info['sufficient_data']:
            if distribution_info['min_class_count'] == 0:
                raise ValueError("❌ Error: Una o más clases no tienen muestras")
            
            print(f"⚠️  Datos insuficientes detectados. Aplicando rebalanceo...")
            features_df, labels = self._rebalance_data(features_df, labels)
            
            # Re-verificar distribución después del rebalanceo
            distribution_info = self._check_class_distribution(labels)
            self.class_distribution = distribution_info
        
        # Determinar estrategia de división de datos
        if distribution_info['can_stratify'] and len(labels) >= 10:
            # División estratificada normal
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    features_df, labels, 
                    test_size=min(self.config.ML_TEST_SIZE, 0.4),  # Máximo 40% para test
                    random_state=self.config.ML_RANDOM_STATE, 
                    stratify=labels
                )
                print("✅ División estratificada exitosa")
            except ValueError:
                # Fallback a división simple
                X_train, X_test, y_train, y_test = train_test_split(
                    features_df, labels, 
                    test_size=min(self.config.ML_TEST_SIZE, 0.3),
                    random_state=self.config.ML_RANDOM_STATE
                )
                print("⚠️  Usando división simple (no estratificada)")
        else:
            # División simple para casos extremos
            test_size = min(0.2, max(0.1, distribution_info['min_class_count'] / len(labels)))
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, labels, 
                test_size=test_size,
                random_state=self.config.ML_RANDOM_STATE
            )
            print(f"⚠️  División simple con {test_size:.1%} para test")
        
        # Verificar que tenemos datos suficientes para entrenar
        if len(X_train) < 3:
            raise ValueError("❌ Error: Datos insuficientes para entrenamiento")
        
        # Escalar features
        try:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        except Exception as e:
            raise ValueError(f"❌ Error al escalar features: {str(e)}")
        
        # Configurar Random Forest con parámetros adaptativos
        n_estimators = min(self.config.RF_N_ESTIMATORS, len(X_train) * 2)  # Adaptar según datos
        max_depth = min(self.config.RF_MAX_DEPTH, max(3, len(X_train) // 10))  # Evitar overfitting
        
        self.model = RandomForestClassifier(
            n_estimators=max(10, n_estimators),  # Mínimo 10 árboles
            max_depth=max(3, max_depth),  # Mínimo profundidad 3
            random_state=self.config.ML_RANDOM_STATE,
            class_weight='balanced',  # Importante para clases desbalanceadas
            min_samples_split=max(2, len(X_train) // 20),  # Adaptar según tamaño
            min_samples_leaf=max(1, len(X_train) // 50),   # Adaptar según tamaño
            bootstrap=True,
            oob_score=True if len(X_train) > 10 else False
        )
        
        # Entrenar modelo
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(X_train_scaled, y_train)
        except Exception as e:
            raise ValueError(f"❌ Error al entrenar modelo: {str(e)}")
        
        # Evaluar modelo
        try:
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"✅ Precisión del modelo: {accuracy:.3f}")
            
            # Mostrar reporte de clasificación si hay suficientes datos
            if len(X_test) > 5:
                try:
                    target_names = ['Miedo Extremo', 'Normal', 'Euforia']
                    report = classification_report(
                        y_test, y_pred, 
                        target_names=target_names, 
                        zero_division=0,
                        output_dict=False
                    )
                    print("\n📊 Reporte de clasificación:")
                    print(report)
                except:
                    print("⚠️  No se pudo generar reporte detallado")
            
            # Mostrar OOB score si está disponible
            if hasattr(self.model, 'oob_score_') and self.model.oob_score_ is not None:
                print(f"📊 OOB Score: {self.model.oob_score_:.3f}")
            
        except Exception as e:
            print(f"⚠️  Error en evaluación: {str(e)}")
            accuracy = 0.5  # Valor por defecto
        
        self.is_trained = True
        return accuracy
    
    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Realiza predicción del estado de mercado
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        try:
            # Preparar features
            latest_features = features.iloc[-1:].values
            latest_scaled = self.scaler.transform(latest_features)
            
            # Predecir
            prediction = self.model.predict(latest_scaled)[0]
            probabilities = self.model.predict_proba(latest_scaled)[0]
            
            states = ['Miedo Extremo', 'Normal/Confianza', 'Euforia']
            signals = {0: 'COMPRAR', 1: 'MANTENER', 2: 'VENDER'}
            
            return {
                'prediction': states[prediction],
                'probabilities': dict(zip(states, probabilities)),
                'signal': signals[prediction],
                'prediction_code': prediction
            }
            
        except Exception as e:
            # Predicción de fallback
            print(f"⚠️  Error en predicción: {str(e)}. Usando predicción conservadora.")
            return {
                'prediction': 'Normal/Confianza',
                'probabilities': {'Miedo Extremo': 0.33, 'Normal/Confianza': 0.34, 'Euforia': 0.33},
                'signal': 'MANTENER',
                'prediction_code': 1
            }
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Obtiene importancia de features"""
        if not self.is_trained or self.model is None or self.feature_names is None:
            return None
        
        try:
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            print(f"⚠️  Error obteniendo importancia de features: {str(e)}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtiene información del modelo entrenado"""
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        info = {
            'status': 'trained',
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'class_distribution': self.class_distribution,
            'model_params': self.model.get_params() if self.model else None
        }
        
        if hasattr(self.model, 'oob_score_'):
            info['oob_score'] = self.model.oob_score_
        
        return info