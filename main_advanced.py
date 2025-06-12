#!/usr/bin/env python3
"""
main_advanced.py - Sistema Avanzado de Trading con Tasa Libre de Riesgo

Main mejorado con estrategias de tasa libre de riesgo y visualizaciones
"""
import sys
import os
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Importar configuraciones básicas
    from src.config.config import TradingConfig, DEFAULT_CONFIG
    
    # Importar configuraciones avanzadas
    from src.config.advanced_config import (
        AdvancedTradingConfig, AnalysisRequest, BacktestingThresholds,
        STRICT_STRATEGY, MODERATE_STRATEGY, LAX_STRATEGY, DEFAULT_STRATEGY,
        QUICK_ANALYSIS_ENHANCED, COMPREHENSIVE_ANALYSIS_ENHANCED, CRYPTO_ANALYSIS_ENHANCED
    )
    
    # Importar módulos del sistema
    from src.core.market_sentiment_trader import MarketSentimentTrader
    from src.analysis.analysis_runner import AnalysisRunner
    
    # Intentar importar backtester mejorado
    try:
        from src.backtesting.backtester_enhanced import EnhancedBacktester
        from src.visualization.chart_generator import ChartGenerator
        ENHANCED_BACKTESTER_AVAILABLE = True
        print("✅ Backtester mejorado con tasa libre de riesgo disponible")
    except ImportError:
        try:
            from src.backtesting.backtester_advanced import AdvancedBacktester as EnhancedBacktester
            ENHANCED_BACKTESTER_AVAILABLE = False
            print("⚠️  Usando backtester avanzado estándar")
        except ImportError:
            from src.backtesting.backtester import Backtester as EnhancedBacktester
            ENHANCED_BACKTESTER_AVAILABLE = False
            print("⚠️  Usando backtester básico")
    
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    print("💡 Verifica que la estructura de carpetas sea correcta")
    print("📁 Estructura esperada:")
    print("   src/config/advanced_config.py")
    print("   src/backtesting/backtester_enhanced.py")
    print("   src/visualization/chart_generator.py")
    sys.exit(1)


class EnhancedAnalysisRunner:
    """Ejecutor mejorado de análisis con tasa libre de riesgo y gráficos"""
    
    def __init__(self):
        self.results_cache = {}
        self.chart_generator = ChartGenerator() if ENHANCED_BACKTESTER_AVAILABLE else None
    
    def print_header(self):
        """Imprime header mejorado"""
        print("=" * 110)
        print("🚀 SISTEMA AVANZADO DE TRADING CON TASA LIBRE DE RIESGO Y VISUALIZACIONES")
        print("=" * 110)
        print("📊 Características Mejoradas:")
        print("   • Análisis multi-símbolo y multi-período")
        print("   • Prevención de look-ahead bias")
        print("   • Estrategias configurables (estricta/moderada/laxa)")
        print("   • ✨ NUEVO: Estrategias con tasa libre de riesgo")
        print("   • ✨ NUEVO: Ahorro mensual vs capital completo")
        print("   • ✨ NUEVO: Gráficos comparativos automáticos")
        print("   • ✨ NUEVO: Análisis detallado de umbrales VIX/RSI")
        if ENHANCED_BACKTESTER_AVAILABLE:
            print("   • 🎯 Backtester mejorado disponible ✅")
        else:
            print("   • ⚠️  Usando backtester estándar")
        print("=" * 110)
    
    def show_enhanced_strategy_info(self, config: AdvancedTradingConfig):
        """Muestra información detallada de la estrategia mejorada"""
        strategy_info = config.get_strategy_summary()
        
        print(f"\n📋 CONFIGURACIÓN DE ESTRATEGIA MEJORADA:")
        print(f"   🎯 Estrategia: {strategy_info['strategy_name']}")
        print(f"   💰 Capital inicial: ${config.INITIAL_CAPITAL:,.2f}")
        print(f"   📈 Tasa libre de riesgo: {config.RISK_FREE_RATE_ANNUAL:.1%} anual")
        print(f"   💸 Costo de transacción: {config.TRANSACTION_COST:.3%}")
        print(f"   📊 VIX - Miedo: {strategy_info['vix_thresholds']['fear']}")
        print(f"   📊 VIX - Euforia: {strategy_info['vix_thresholds']['euphoria']}")
        print(f"   📈 RSI - Sobreventa: {strategy_info['rsi_thresholds']['oversold']}")
        print(f"   📈 RSI - Sobrecompra: {strategy_info['rsi_thresholds']['overbought']}")
        print(f"   📉 BB - Límite Inferior: {strategy_info['bb_thresholds']['lower']}")
        print(f"   📉 BB - Límite Superior: {strategy_info['bb_thresholds']['upper']}")
        print(f"   🔒 Prevención Look-ahead: {'✅ Sí' if strategy_info['lookahead_prevention'] else '❌ No'}")
        print(f"   📊 Ventana Expandida: {'✅ Sí' if strategy_info['expanding_window'] else '❌ No'}")
        print(f"   📈 Generar gráficos: {'✅ Sí' if config.GENERATE_CHARTS else '❌ No'}")
    
    def run_enhanced_single_analysis(self, symbol: str, years: int, 
                                   config: AdvancedTradingConfig) -> Dict[str, Any]:
        """Ejecuta análisis mejorado para un símbolo y período específico"""
        
        print(f"\n🔍 ANÁLISIS MEJORADO: {symbol} - {years} años")
        print("-" * 60)
        
        try:
            # Crear trader con configuración mejorada
            trader = MarketSentimentTrader(symbol, config)
            
            # Obtener datos completos para entrenar modelo
            print("📊 Obteniendo datos históricos completos...")
            full_data = trader.fetch_data(period='max')
            full_data_with_indicators, vix_percentiles = trader.calculate_indicators(full_data)
            full_labeled_data = trader.create_market_labels(full_data_with_indicators, vix_percentiles)
            
            # Entrenar modelo con todos los datos
            print("🤖 Entrenando modelo ML...")
            accuracy = trader.train_model(full_labeled_data)
            
            # Filtrar datos para período de inversión
            end_date = full_labeled_data.index[-1]
            start_investment_date = end_date - pd.DateOffset(years=years)
            investment_period_data = full_labeled_data[full_labeled_data.index >= start_investment_date].copy()
            
            print(f"📅 Período de análisis: {investment_period_data.index[0].strftime('%Y-%m-%d')} a {investment_period_data.index[-1].strftime('%Y-%m-%d')}")
            print(f"📊 Datos: {len(investment_period_data)} días ({len(investment_period_data)/252:.2f} años)")
            
            # Usar backtester mejorado si está disponible
            if ENHANCED_BACKTESTER_AVAILABLE:
                # Reemplazar el backtester del trader con el mejorado
                trader.backtester = EnhancedBacktester(config)
                
                # Ejecutar backtesting mejorado
                backtest_data, performance_metrics = trader.backtester.run_enhanced_backtest(
                    investment_period_data, symbol
                )
            else:
                # Usar backtesting estándar
                backtest_data, performance_metrics = trader.backtest_strategy(investment_period_data)
            
            # Predicción actual
            current_prediction = trader.predict_market_state(full_labeled_data)
            
            # Análisis de condiciones de mercado
            market_analysis = self.analyze_market_conditions_enhanced(
                trader, investment_period_data, symbol
            )
            
            # Compilar resultados
            results = {
                'trader': trader,
                'full_data': full_labeled_data,
                'investment_data': investment_period_data,
                'backtest_data': backtest_data,
                'performance_metrics': performance_metrics,
                'current_prediction': current_prediction,
                'market_analysis': market_analysis,
                'model_accuracy': accuracy,
                'feature_importance': trader.get_feature_importance(),
                'enhanced_features': {
                    'risk_free_rate_strategies': ENHANCED_BACKTESTER_AVAILABLE,
                    'charts_generated': config.GENERATE_CHARTS and ENHANCED_BACKTESTER_AVAILABLE
                }
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error en análisis mejorado: {str(e)}")
            raise
    
    def analyze_market_conditions_enhanced(self, trader: MarketSentimentTrader, 
                                         data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Análisis mejorado de condiciones de mercado con valores específicos de umbrales"""
        
        if data.empty or len(data) < 100:
            return {'error': 'Datos insuficientes para análisis'}
        
        try:
            # Obtener valores actuales
            recent_data = data.tail(30)
            last_valid_idx = data.dropna().index[-1]
            
            current_conditions = {
                'vix_current': float(data['VIX'].iloc[-1]),
                'vix_recent_avg': float(recent_data['VIX'].mean()),
                'vix_percentile': float(data['VIX'].rank(pct=True).iloc[-1] * 100),
                
                'rsi_current': float(data['RSI'].iloc[-1]),
                'rsi_recent_avg': float(recent_data['RSI'].mean()),
                'rsi_percentile': float(data['RSI'].rank(pct=True).iloc[-1] * 100),
                
                'bb_position_current': float(data['BB_Position'].iloc[-1]),
                'bb_position_avg': float(recent_data['BB_Position'].mean()),
                
                'volatility_current': float(data['Volatility_20'].iloc[-1]),
                'volatility_recent_avg': float(recent_data['Volatility_20'].mean()),
                'volatility_percentile': float(data['Volatility_20'].rank(pct=True).iloc[-1] * 100),
                
                'price_change_20_current': float(data['Price_Change_20'].iloc[-1]),
                'price_change_percentile': float(data['Price_Change_20'].rank(pct=True).iloc[-1] * 100)
            }
            
            # Obtener umbrales específicos calculados
            thresholds = trader.config.backtesting_thresholds
            
            # Calcular umbrales específicos (valores reales, no percentiles)
            calculated_thresholds = {}
            
            if trader.config.PREVENT_LOOKAHEAD_BIAS and 'VIX_Q_Fear' in data.columns:
                # Usar umbrales dinámicos calculados
                calculated_thresholds = {
                    'vix_fear_value': float(data.loc[last_valid_idx, 'VIX_Q_Fear']) if 'VIX_Q_Fear' in data.columns else data['VIX'].quantile(thresholds.vix_fear_threshold),
                    'vix_euphoria_value': float(data.loc[last_valid_idx, 'VIX_Q_Euphoria']) if 'VIX_Q_Euphoria' in data.columns else data['VIX'].quantile(thresholds.vix_euphoria_threshold),
                    'rsi_oversold_value': float(data.loc[last_valid_idx, 'RSI_Q_Oversold']) if 'RSI_Q_Oversold' in data.columns else data['RSI'].quantile(thresholds.rsi_oversold),
                    'rsi_overbought_value': float(data.loc[last_valid_idx, 'RSI_Q_Overbought']) if 'RSI_Q_Overbought' in data.columns else data['RSI'].quantile(thresholds.rsi_overbought),
                }
            else:
                # Usar umbrales fijos basados en toda la data
                calculated_thresholds = {
                    'vix_fear_value': float(data['VIX'].quantile(thresholds.vix_fear_threshold)),
                    'vix_euphoria_value': float(data['VIX'].quantile(thresholds.vix_euphoria_threshold)),
                    'rsi_oversold_value': float(data['RSI'].quantile(thresholds.rsi_oversold)),
                    'rsi_overbought_value': float(data['RSI'].quantile(thresholds.rsi_overbought)),
                }
            
            # Evaluar condiciones específicas
            fear_conditions = {
                'vix_extreme': current_conditions['vix_current'] > calculated_thresholds['vix_fear_value'],
                'rsi_oversold': current_conditions['rsi_current'] < calculated_thresholds['rsi_oversold_value'],
                'bb_lower': current_conditions['bb_position_current'] < thresholds.bb_lower_threshold,
                'volatility_high': current_conditions['volatility_percentile'] > (thresholds.volatility_high * 100)
            }
            
            euphoria_conditions = {
                'vix_low': current_conditions['vix_current'] < calculated_thresholds['vix_euphoria_value'],
                'rsi_overbought': current_conditions['rsi_current'] > calculated_thresholds['rsi_overbought_value'],
                'bb_upper': current_conditions['bb_position_current'] > thresholds.bb_upper_threshold,
                'price_change_high': current_conditions['price_change_percentile'] > (thresholds.price_change_extreme * 100)
            }
            
            # Señal final
            fear_score = sum(fear_conditions.values())
            euphoria_score = sum(euphoria_conditions.values())
            
            if fear_score >= 2:
                market_signal = "COMPRAR"
                market_state = "Miedo/Oportunidad"
                signal_strength = min(fear_score / len(fear_conditions), 1.0)
            elif euphoria_score >= 3:
                market_signal = "VENDER"
                market_state = "Euforia/Sobrevalorado"
                signal_strength = min(euphoria_score / len(euphoria_conditions), 1.0)
            else:
                market_signal = "MANTENER"
                market_state = "Normal/Neutral"
                signal_strength = 0.5
            
            return {
                'symbol': symbol,
                'current_conditions': current_conditions,
                'calculated_thresholds': calculated_thresholds,
                'fear_conditions': fear_conditions,
                'euphoria_conditions': euphoria_conditions,
                'fear_score': fear_score,
                'euphoria_score': euphoria_score,
                'market_signal': market_signal,
                'market_state': market_state,
                'signal_strength': signal_strength,
                'threshold_percentiles': {
                    'vix_fear': thresholds.vix_fear_threshold,
                    'vix_euphoria': thresholds.vix_euphoria_threshold,
                    'rsi_oversold': thresholds.rsi_oversold,
                    'rsi_overbought': thresholds.rsi_overbought,
                    'bb_lower': thresholds.bb_lower_threshold,
                    'bb_upper': thresholds.bb_upper_threshold
                }
            }
            
        except Exception as e:
            return {'error': f'Error en análisis de condiciones: {str(e)}'}
    
    def print_enhanced_market_analysis(self, market_analysis: Dict[str, Any]):
        """Imprime análisis detallado mejorado con valores específicos"""
        
        if 'error' in market_analysis:
            print(f"⚠️  Error en análisis: {market_analysis['error']}")
            return
        
        symbol = market_analysis['symbol']
        conditions = market_analysis['current_conditions']
        thresholds = market_analysis['calculated_thresholds']
        fear_cond = market_analysis['fear_conditions']
        euphoria_cond = market_analysis['euphoria_conditions']
        threshold_pcts = market_analysis['threshold_percentiles']
        
        print(f"\n🔍 ANÁLISIS DETALLADO DE CONDICIONES - {symbol}")
        print("=" * 70)
        
        # Estado general
        print(f"📊 Estado del Mercado: {market_analysis['market_state']}")
        print(f"🎯 Señal: {market_analysis['market_signal']}")
        print(f"💪 Fuerza de Señal: {market_analysis['signal_strength']:.0%}")
        
        # Condiciones actuales con umbrales específicos
        print(f"\n📈 CONDICIONES ACTUALES vs UMBRALES ESPECÍFICOS:")
        print(f"   VIX: {conditions['vix_current']:.1f} | Miedo: >{thresholds['vix_fear_value']:.1f} ({threshold_pcts['vix_fear']:.0%}) | Euforia: <{thresholds['vix_euphoria_value']:.1f} ({threshold_pcts['vix_euphoria']:.0%})")
        print(f"   RSI: {conditions['rsi_current']:.1f} | Sobreventa: <{thresholds['rsi_oversold_value']:.1f} ({threshold_pcts['rsi_oversold']:.0%}) | Sobrecompra: >{thresholds['rsi_overbought_value']:.1f} ({threshold_pcts['rsi_overbought']:.0%})")
        print(f"   BB Position: {conditions['bb_position_current']:.2f} | Límites: <{threshold_pcts['bb_lower']:.0%} y >{threshold_pcts['bb_upper']:.0%}")
        print(f"   Volatilidad: {conditions['volatility_current']:.1f}% (percentil {conditions['volatility_percentile']:.0f}%)")
        print(f"   Cambio 20d: {conditions['price_change_20_current']:+.1f}% (percentil {conditions['price_change_percentile']:.0f}%)")
        
        # Condiciones activas
        active_fear = [k for k, v in fear_cond.items() if v]
        if active_fear:
            print(f"\n😰 CONDICIONES DE MIEDO ACTIVAS ({len(active_fear)}):")
            condition_explanations = {
                'vix_extreme': f'VIX {conditions["vix_current"]:.1f} > {thresholds["vix_fear_value"]:.1f}',
                'rsi_oversold': f'RSI {conditions["rsi_current"]:.1f} < {thresholds["rsi_oversold_value"]:.1f}',
                'bb_lower': f'BB Position {conditions["bb_position_current"]:.2f} < {threshold_pcts["bb_lower"]:.0%}',
                'volatility_high': f'Volatilidad en percentil {conditions["volatility_percentile"]:.0f}%'
            }
            for condition in active_fear:
                explanation = condition_explanations.get(condition, condition.replace('_', ' ').title())
                print(f"   ✅ {explanation}")
        
        active_euphoria = [k for k, v in euphoria_cond.items() if v]
        if active_euphoria:
            print(f"\n🚀 CONDICIONES DE EUFORIA ACTIVAS ({len(active_euphoria)}):")
            condition_explanations = {
                'vix_low': f'VIX {conditions["vix_current"]:.1f} < {thresholds["vix_euphoria_value"]:.1f}',
                'rsi_overbought': f'RSI {conditions["rsi_current"]:.1f} > {thresholds["rsi_overbought_value"]:.1f}',
                'bb_upper': f'BB Position {conditions["bb_position_current"]:.2f} > {threshold_pcts["bb_upper"]:.0%}',
                'price_change_high': f'Cambio precio en percentil {conditions["price_change_percentile"]:.0f}%'
            }
            for condition in active_euphoria:
                explanation = condition_explanations.get(condition, condition.replace('_', ' ').title())
                print(f"   ✅ {explanation}")
        
        if not active_fear and not active_euphoria:
            print(f"\n😐 ESTADO NEUTRAL - No hay condiciones extremas activas")
    
    def run_enhanced_analysis(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Ejecuta análisis mejorado según la configuración"""
        
        print(f"\n🚀 INICIANDO ANÁLISIS MEJORADO CON TASA LIBRE DE RIESGO")
        print(f"   📊 Símbolos: {', '.join(request.symbols)}")
        print(f"   📅 Períodos: {', '.join(map(str, request.periods))} años")
        print(f"   📈 Incluir máximo: {'Sí' if request.include_max_period else 'No'}")
        print(f"   🎯 Estrategia: {request.backtesting_strategy}")
        print(f"   💰 Tasa libre de riesgo: {request.trading_config.RISK_FREE_RATE_ANNUAL:.1%}")
        
        # Configurar estrategia
        request.trading_config.set_backtesting_strategy(request.backtesting_strategy)
        self.show_enhanced_strategy_info(request.trading_config)
        
        all_results = {}
        
        for symbol in request.symbols:
            print(f"\n" + "="*90)
            print(f"📊 ANALIZANDO {symbol} CON ESTRATEGIAS MEJORADAS")
            print("="*90)
            
            symbol_results = {}
            
            try:
                # Análisis por período
                for years in request.periods:
                    print(f"\n📅 Período: {years} años")
                    
                    try:
                        results = self.run_enhanced_single_analysis(symbol, years, request.trading_config)
                        
                        # Mostrar análisis detallado
                        self.print_enhanced_market_analysis(results['market_analysis'])
                        
                        # Mostrar resumen de estrategias mejoradas
                        self.print_enhanced_performance_summary(results['performance_metrics'], years)
                        
                        symbol_results[f"{years}y"] = results
                        
                    except Exception as e:
                        print(f"❌ Error en período {years} años: {str(e)}")
                        symbol_results[f"{years}y"] = {'error': str(e)}
                
                # Análisis de período máximo si se solicita
                if request.include_max_period:
                    print(f"\n📅 Período: MÁXIMO")
                    print("-" * 50)
                    
                    try:
                        # Para período máximo, usar configuración especial
                        max_config = request.trading_config
                        max_config.INITIAL_CAPITAL = request.trading_config.INITIAL_CAPITAL
                        
                        trader = MarketSentimentTrader(symbol, max_config)
                        
                        if ENHANCED_BACKTESTER_AVAILABLE:
                            trader.backtester = EnhancedBacktester(max_config)
                            
                        max_results = trader.run_complete_analysis(period='max')
                        
                        if 'backtest_results' in max_results:
                            market_analysis = self.analyze_market_conditions_enhanced(
                                trader, max_results['backtest_results'], symbol
                            )
                            max_results['market_analysis'] = market_analysis
                            self.print_enhanced_market_analysis(market_analysis)
                        
                        symbol_results['max'] = max_results
                        
                        # Mostrar resumen
                        if 'performance_metrics' in max_results:
                            metrics = max_results['performance_metrics']
                            period_years = metrics['period']['years']
                            self.print_enhanced_performance_summary(metrics, f"máximo ({period_years:.1f} años)")
                        
                    except Exception as e:
                        print(f"❌ Error en período máximo: {str(e)}")
                        symbol_results['max'] = {'error': str(e)}
                
            except Exception as e:
                print(f"❌ Error general para {symbol}: {str(e)}")
                symbol_results = {'error': str(e)}
            
            all_results[symbol] = symbol_results
        
        # Resumen final mejorado
        self.print_enhanced_final_summary(all_results, request)
        
        return {
            'request_config': request,
            'trading_config': request.trading_config,
            'results': all_results,
            'strategy_info': request.trading_config.get_strategy_summary(),
            'enhanced_features': {
                'risk_free_rate_strategies': ENHANCED_BACKTESTER_AVAILABLE,
                'visualization_available': ENHANCED_BACKTESTER_AVAILABLE
            }
        }
    
    def print_enhanced_performance_summary(self, metrics: Dict[str, Any], period_desc: str):
        """Imprime resumen mejorado de rendimiento"""
        
        print(f"\n💰 RESUMEN DE RENDIMIENTO - {period_desc}:")
        
        # Verificar si tenemos estrategias mejoradas
        if 'strategy1' in metrics:
            print(f"   🔵 Estrategia 1 (Capital + RF): {metrics['strategy1']['annual_return']:+.1f}% anual")
        if 'strategy2' in metrics:
            print(f"   🟢 Estrategia 2 (Ahorro + RF): {metrics['strategy2']['annual_return']:+.1f}% anual")
        
        # Estrategias tradicionales
        if 'buyhold' in metrics:
            print(f"   🟠 Buy & Hold: {metrics['buyhold']['annual_return']:+.1f}% anual")
        if 'dca' in metrics:
            print(f"   🔴 DCA: {metrics['dca']['annual_return']:+.1f}% anual")
        
        # Estrategia clásica (si existe)
        if 'strategy' in metrics:
            print(f"   ⚪ Estrategia Clásica: {metrics['strategy']['annual_return']:+.1f}% anual")
        
        # Comparaciones mejoradas
        if 'comparisons' in metrics:
            comparisons = metrics['comparisons']
            print(f"\n🏆 OUTPERFORMANCES:")
            if 'strategy1_vs_buyhold' in comparisons:
                print(f"      Estrategia 1 vs Buy & Hold: {comparisons['strategy1_vs_buyhold']:+.1f}%")
            if 'strategy2_vs_buyhold' in comparisons:
                print(f"      Estrategia 2 vs Buy & Hold: {comparisons['strategy2_vs_buyhold']:+.1f}%")
            if 'strategy1_vs_dca' in comparisons:
                print(f"      Estrategia 1 vs DCA: {comparisons['strategy1_vs_dca']:+.1f}%")
            if 'strategy2_vs_dca' in comparisons:
                print(f"      Estrategia 2 vs DCA: {comparisons['strategy2_vs_dca']:+.1f}%")
        
        # Información adicional de estrategias mejoradas
        if 'simulation_details' in metrics:
            details = metrics['simulation_details']
            print(f"\n⚡ DETALLES DE OPERACIONES:")
            if 'strategy1_trades' in details:
                print(f"      Estrategia 1: {len(details['strategy1_trades'])} operaciones")
            if 'strategy2_trades' in details:
                print(f"      Estrategia 2: {len(details['strategy2_trades'])} operaciones")
            if 'monthly_savings' in details and details['monthly_savings'] > 0:
                print(f"      Ahorro mensual: ${details['monthly_savings']:,.2f}")
            if 'risk_free_rate' in details:
                print(f"      Tasa libre de riesgo: {details['risk_free_rate']:.1%}")
            if 'chart_path' in details:
                print(f"      📈 Gráfico generado: {details['chart_path']}")

    def print_enhanced_final_summary(self, all_results: Dict[str, Any], request: AnalysisRequest):
        """Imprime resumen final mejorado de todos los análisis"""
        
        print(f"\n🎯 RESUMEN FINAL DEL ANÁLISIS MEJORADO")
        print("=" * 90)
        
        # Contar resultados exitosos
        total_symbols = len(request.symbols)
        successful_symbols = len([s for s, r in all_results.items() if 'error' not in r])
        
        print(f"📊 Símbolos analizados: {successful_symbols}/{total_symbols}")
        print(f"📅 Períodos por símbolo: {len(request.periods)} + {'máximo' if request.include_max_period else 'sin máximo'}")
        print(f"💰 Capital base: ${request.trading_config.INITIAL_CAPITAL:,.2f}")
        print(f"📈 Tasa libre de riesgo: {request.trading_config.RISK_FREE_RATE_ANNUAL:.1%}")
        print(f"🎯 Estrategia: {request.backtesting_strategy}")
        
        # Resumen por símbolo
        for symbol, symbol_results in all_results.items():
            if 'error' in symbol_results:
                print(f"\n❌ {symbol}: {symbol_results['error']}")
                continue
                
            print(f"\n📊 {symbol} - Resumen:")
            
            for period_key, period_results in symbol_results.items():
                if 'error' in period_results:
                    print(f"   ❌ {period_key}: {period_results['error']}")
                    continue
                    
                if 'performance_metrics' in period_results:
                    metrics = period_results['performance_metrics']
                    
                    # Mejor estrategia
                    strategies = {}
                    if 'strategy1' in metrics:
                        strategies['Estrategia 1'] = metrics['strategy1']['annual_return']
                    if 'strategy2' in metrics:
                        strategies['Estrategia 2'] = metrics['strategy2']['annual_return']
                    if 'buyhold' in metrics:
                        strategies['Buy & Hold'] = metrics['buyhold']['annual_return']
                    if 'dca' in metrics:
                        strategies['DCA'] = metrics['dca']['annual_return']
                    
                    if strategies:
                        best_strategy = max(strategies, key=strategies.get)
                        best_return = strategies[best_strategy]
                        print(f"   🏆 {period_key}: {best_strategy} ({best_return:+.1f}% anual)")
        
        print(f"\n✅ Análisis completado con estrategias mejoradas de tasa libre de riesgo")
        if ENHANCED_BACKTESTER_AVAILABLE:
            print(f"📈 Gráficos generados automáticamente para cada análisis")
        print(f"💡 Los resultados incluyen prevención de look-ahead bias")
    

def create_enhanced_custom_request() -> AnalysisRequest:
    """Crea configuración personalizada mejorada"""
    
    print("\n🛠️  CONFIGURACIÓN PERSONALIZADA MEJORADA")
    print("=" * 60)
    
    # Símbolos
    symbols_input = input("📊 Símbolos a analizar (ej: SPYG,QQQ,BTC-USD): ").strip()
    symbols = [s.strip().upper() for s in symbols_input.split(',')] if symbols_input else ['SPYG']
    
    # Períodos en años
    periods_input = input("📅 Períodos en años (ej: 1,3,5): ").strip()
    if periods_input:
        try:
            periods = [int(p.strip()) for p in periods_input.split(',')]
        except ValueError:
            print("⚠️  Error en períodos, usando [3,5] por defecto")
            periods = [3, 5]
    else:
        periods = [3, 5]
    
    # Período máximo
    include_max = input("📈 ¿Incluir período máximo? (s/n): ").strip().lower()
    include_max_period = include_max in ['s', 'si', 'sí', 'y', 'yes']
    
    # Capital inicial
    capital_input = input(f"💰 Capital inicial (default 10000): ").strip()
    initial_capital = float(capital_input) if capital_input else 10000
    
    # NUEVA: Tasa libre de riesgo
    rf_rate_input = input(f"📈 Tasa libre de riesgo anual % (default 5.0): ").strip()
    if rf_rate_input:
        try:
            rf_rate = float(rf_rate_input) / 100  # Convertir porcentaje a decimal
            if rf_rate < 0 or rf_rate > 0.20:  # Límite razonable 0-20%
                raise ValueError("Tasa fuera del rango razonable")
        except ValueError:
            print("⚠️  Tasa inválida, usando 5% por defecto")
            rf_rate = 0.05
    else:
        rf_rate = 0.05
    
    # Estrategia de backtesting
    print("\n🎯 Estrategias disponibles:")
    print("   1. strict    - Muy estricta (señales menos frecuentes pero más confiables)")
    print("   2. moderate  - Moderada (balance entre frecuencia y confiabilidad)")
    print("   3. lax       - Laxa (más señales, menos restrictiva)")
    print("   4. default   - Por defecto (estricta estándar)")
    print("   5. custom    - Personalizada (definir umbrales manualmente)")
    
    strategy_choice = input("Selecciona estrategia (1-5): ").strip()
    
    strategy_map = {
        '1': 'strict',
        '2': 'moderate', 
        '3': 'lax',
        '4': 'default',
        '5': 'custom'
    }
    
    backtesting_strategy = strategy_map.get(strategy_choice, 'default')
    
    # Crear configuración mejorada
    trading_config = AdvancedTradingConfig(
        INITIAL_CAPITAL=initial_capital,
        RISK_FREE_RATE_ANNUAL=rf_rate,
        GENERATE_CHARTS=True,
        SAVE_CHARTS=True
    )
    
    # Configuración personalizada de umbrales si se eligió custom
    if backtesting_strategy == 'custom':
        print("\n⚙️  CONFIGURACIÓN PERSONALIZADA DE UMBRALES:")
        print("   (Presiona Enter para usar valores por defecto)")
        
        try:
            vix_fear = input(f"VIX percentil para miedo (default 90): ").strip()
            vix_euphoria = input(f"VIX percentil para euforia (default 10): ").strip()
            rsi_oversold = input(f"RSI percentil sobreventa (default 10): ").strip()
            rsi_overbought = input(f"RSI percentil sobrecompra (default 90): ").strip()
            bb_lower = input(f"BB posición límite inferior (default 5): ").strip()
            bb_upper = input(f"BB posición límite superior (default 95): ").strip()
            
            custom_thresholds = BacktestingThresholds()
            
            if vix_fear:
                custom_thresholds.vix_fear_threshold = float(vix_fear) / 100
            if vix_euphoria:
                custom_thresholds.vix_euphoria_threshold = float(vix_euphoria) / 100
            if rsi_oversold:
                custom_thresholds.rsi_oversold = float(rsi_oversold) / 100
            if rsi_overbought:
                custom_thresholds.rsi_overbought = float(rsi_overbought) / 100
            if bb_lower:
                custom_thresholds.bb_lower_threshold = float(bb_lower) / 100
            if bb_upper:
                custom_thresholds.bb_upper_threshold = float(bb_upper) / 100
            
            trading_config.backtesting_thresholds = custom_thresholds
            
        except ValueError:
            print("⚠️  Error en configuración personalizada, usando valores por defecto")
            backtesting_strategy = 'default'
    
    print(f"\n✅ Configuración creada:")
    print(f"   💰 Capital: ${trading_config.INITIAL_CAPITAL:,.2f}")
    print(f"   📈 Tasa RF: {trading_config.RISK_FREE_RATE_ANNUAL:.1%}")
    print(f"   📊 Gráficos: {'Sí' if trading_config.GENERATE_CHARTS else 'No'}")
    
    return AnalysisRequest(
        symbols=symbols,
        periods=periods,
        include_max_period=include_max_period,
        trading_config=trading_config,
        analysis_type="complete",
        backtesting_strategy=backtesting_strategy
    )


def show_enhanced_strategy_comparison():
    """Muestra comparación detallada de estrategias mejoradas"""
    
    print("\n📊 COMPARACIÓN DE ESTRATEGIAS MEJORADAS")
    print("=" * 100)
    
    strategies = {
        'Estricta': STRICT_STRATEGY,
        'Moderada': MODERATE_STRATEGY,
        'Laxa': LAX_STRATEGY,
        'Default': DEFAULT_STRATEGY
    }
    
    print(f"{'Estrategia':<12} {'VIX Miedo':<12} {'VIX Euforia':<12} {'RSI Over':<12} {'RSI Under':<12} {'BB Límites':<15}")
    print("-" * 100)
    
    for name, strategy in strategies.items():
        vix_fear = f">{strategy.vix_fear_threshold:.0%}"
        vix_euphoria = f"<{strategy.vix_euphoria_threshold:.0%}"
        rsi_over = f"<{strategy.rsi_oversold:.0%}"
        rsi_under = f">{strategy.rsi_overbought:.0%}"
        bb_limits = f"{strategy.bb_lower_threshold:.0%}-{strategy.bb_upper_threshold:.0%}"
        
        print(f"{name:<12} {vix_fear:<12} {vix_euphoria:<12} {rsi_over:<12} {rsi_under:<12} {bb_limits:<15}")
    
    print(f"\n🔄 NUEVAS ESTRATEGIAS CON TASA LIBRE DE RIESGO:")
    print(f"   💰 Estrategia 1: Capital completo esperando señales (ganando RF mientras espera)")
    print(f"   💰 Estrategia 2: Ahorro mensual acumulando en RF hasta señales de compra")
    print(f"   📈 Tasa por defecto: 5% anual (configurable)")
    print(f"   📊 Ambas incluyen costos de transacción y look-ahead bias prevention")
    
    print(f"\n💡 RECOMENDACIONES:")
    print(f"   🎯 Estricta + RF: Para máxima conservación, pocas operaciones")
    print(f"   ⚖️  Moderada + RF: Balance óptimo para la mayoría")
    print(f"   🚀 Laxa + RF: Para mercados volátiles o traders activos")
    print(f"   📋 Estrategia 1: Si tienes capital completo desde el inicio")
    print(f"   📋 Estrategia 2: Si vas ahorrando gradualmente")


def show_risk_free_rate_info():
    """Explica las estrategias con tasa libre de riesgo"""
    
    print("\n💰 ESTRATEGIAS CON TASA LIBRE DE RIESGO")
    print("=" * 70)
    
    print("❓ ¿Qué son las estrategias con tasa libre de riesgo?")
    print("   Las nuevas estrategias invierten el dinero no utilizado en una")
    print("   tasa libre de riesgo mientras esperan señales de trading.")
    
    print("\n🔵 ESTRATEGIA 1: CAPITAL COMPLETO + TASA LIBRE DE RIESGO")
    print("   • Tienes $10,000 desde el inicio")
    print("   • Mientras esperas señal de compra: dinero gana 5% anual")
    print("   • Al recibir señal de compra: inviertes todo en el activo")
    print("   • Al recibir señal de venta: vuelves a tasa libre de riesgo")
    print("   • Ventaja: Siempre estás ganando algo, no hay dinero inactivo")
    
    print("\n🟢 ESTRATEGIA 2: AHORRO MENSUAL + TASA LIBRE DE RIESGO")
    print("   • Ahorras gradualmente hasta llegar a $10,000")
    print("   • Cada mes: depositas en cuenta que gana 5% anual")
    print("   • Al recibir señal de compra: inviertes todo lo ahorrado")
    print("   • Sigues ahorrando mensualmente durante la inversión")
    print("   • Al vender: todo va a tasa libre de riesgo + sigues ahorrando")
    print("   • Ventaja: Simula ahorro real + inversión por oportunidades")
    
    print("\n📊 COMPARACIÓN CON ESTRATEGIAS TRADICIONALES:")
    print("   📈 Buy & Hold: Compra al inicio y mantiene")
    print("   💰 DCA: Invierte cantidad fija periódicamente")
    print("   🔵 Estrategia 1: Espera oportunidades ganando tasa RF")
    print("   🟢 Estrategia 2: Ahorra + espera oportunidades ganando tasa RF")
    
    print("\n⚙️ CONFIGURACIÓN:")
    print("   • Tasa libre de riesgo por defecto: 5% anual")
    print("   • Configurable entre 0% y 20%")
    print("   • Se aplica diariamente: (1 + tasa_anual)^(1/252) - 1")
    print("   • Incluye costos de transacción al entrar/salir del mercado")


def show_visualization_info():
    """Explica las visualizaciones disponibles"""
    
    print("\n📈 VISUALIZACIONES Y GRÁFICOS")
    print("=" * 60)
    
    print("📊 GRÁFICOS AUTOMÁTICOS GENERADOS:")
    if ENHANCED_BACKTESTER_AVAILABLE:
        print("   ✅ Evolución comparativa de portfolios")
        print("   ✅ Análisis de drawdown por estrategia")
        print("   ✅ Métricas de rendimiento en barras")
        print("   ✅ Señales de trading sobre precio del activo")
        print("   ✅ Análisis de umbrales dinámicos (VIX, RSI, BB)")
        print("   ✅ Gráficos guardados automáticamente en PNG")
    else:
        print("   ❌ Módulo de visualización no disponible")
        print("   💡 Instala matplotlib y seaborn para habilitar gráficos")
    
    print("\n📈 TIPOS DE GRÁFICOS:")
    print("   1. Gráfico Principal: Compara todas las estrategias en el tiempo")
    print("   2. Análisis de Drawdown: Muestra pérdidas máximas")
    print("   3. Métricas de Barras: Retorno anual vs Sharpe ratio")
    print("   4. Señales de Trading: Puntos de compra/venta sobre el precio")
    print("   5. Análisis de Umbrales: VIX y RSI con niveles dinámicos")
    
    print("\n🎨 CARACTERÍSTICAS:")
    print("   • Estilo moderno con colores distintivos")
    print("   • Resolución alta (300 DPI) para impresión")
    print("   • Formato PNG con fondo blanco")
    print("   • Nombres automáticos con timestamp")
    print("   • Visualización inmediata + guardado automático")
    
    print("\n💡 INTERPRETACIÓN:")
    print("   📊 Líneas ascendentes = mejor rendimiento")
    print("   📉 Drawdown menor = menor riesgo")
    print("   🟢 Puntos verdes = señales de compra")
    print("   🔴 Puntos rojos = señales de venta")
    print("   📈 Umbrales dinámicos = adaptación al mercado")


def main_menu():
    """Menú principal mejorado"""
    
    runner = EnhancedAnalysisRunner()
    runner.print_header()
    
    while True:
        print("\n" + "="*70)
        print("🎛️  MENÚ PRINCIPAL MEJORADO")
        print("="*70)
        print("1. 🚀 Análisis rápido mejorado (SPYG - 3 años + RF)")
        print("2. 📊 Análisis comprehensivo mejorado (múltiples activos + RF)")
        print("3. 🪙 Análisis de criptomonedas mejorado (BTC/ETH + RF)")
        print("4. 🛠️  Configuración personalizada mejorada")
        print("5. 📋 Mostrar estrategias mejoradas disponibles")
        print("6. ❓ Información sobre tasa libre de riesgo")
        print("7. 📈 Información sobre gráficos y visualizaciones")
        print("0. ❌ Salir")
        
        try:
            opcion = input("\n👉 Selecciona una opción (0-7): ").strip()
            
            if opcion == '0':
                print("\n👋 ¡Hasta luego!")
                break
                
            elif opcion == '1':
                print("\n🚀 ANÁLISIS RÁPIDO MEJORADO")
                results = runner.run_enhanced_analysis(QUICK_ANALYSIS_ENHANCED)
                
            elif opcion == '2':
                print("\n📊 ANÁLISIS COMPREHENSIVO MEJORADO")
                results = runner.run_enhanced_analysis(COMPREHENSIVE_ANALYSIS_ENHANCED)
                
            elif opcion == '3':
                print("\n🪙 ANÁLISIS DE CRIPTOMONEDAS MEJORADO")
                results = runner.run_enhanced_analysis(CRYPTO_ANALYSIS_ENHANCED)
                
            elif opcion == '4':
                custom_request = create_enhanced_custom_request()
                print(f"\n🛠️  ANÁLISIS PERSONALIZADO MEJORADO")
                results = runner.run_enhanced_analysis(custom_request)
                
            elif opcion == '5':
                show_enhanced_strategy_comparison()
                
            elif opcion == '6':
                show_risk_free_rate_info()
                
            elif opcion == '7':
                show_visualization_info()
                
            else:
                print("❌ Opción inválida. Intenta de nuevo.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Programa interrumpido. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error inesperado: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Función principal del sistema mejorado"""
    try:
        main_menu()
    except Exception as e:
        print(f"\n❌ Error crítico: {str(e)}")
        print("🔧 Verifica la instalación y configuración del sistema")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()