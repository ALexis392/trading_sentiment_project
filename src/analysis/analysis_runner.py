"""
Módulo para ejecutar diferentes tipos de análisis
"""
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from src.config.config import TradingConfig
from src.core.market_sentiment_trader import MarketSentimentTrader

class AnalysisRunner:
    """Clase para ejecutar y comparar diferentes análisis"""
    
    @staticmethod
    def run_single_analysis(symbol: str = 'SPYG', investment_start_years_ago: int = 5,
                          config: Optional[TradingConfig] = None) -> tuple[MarketSentimentTrader, Dict[str, Any]]:
        """
        Ejecuta análisis para un período específico
        
        Args:
            symbol: Símbolo a analizar
            investment_start_years_ago: Años atrás desde los cuales empezar
            config: Configuración personalizada
            
        Returns:
            Tuple con trader y resultados
        """
        print(f"=== ANÁLISIS DE {symbol} - ÚLTIMOS {investment_start_years_ago} AÑOS ===\n")
        
        # Inicializar trader
        trader = MarketSentimentTrader(symbol, config)
        
        try:
            # Obtener datos completos
            full_data = trader.fetch_data(period='max')
            full_data_with_indicators, vix_percentiles = trader.calculate_indicators(full_data)
            full_labeled_data = trader.create_market_labels(full_data_with_indicators, vix_percentiles)
            
            # Entrenar modelo con todos los datos
            accuracy = trader.train_model(full_labeled_data)
            
            # Filtrar para período de inversión
            end_date = full_labeled_data.index[-1]
            start_investment_date = end_date - pd.DateOffset(years=investment_start_years_ago)
            investment_period_data = full_labeled_data[full_labeled_data.index >= start_investment_date].copy()
            
            print(f"Período de análisis: {investment_period_data.index[0].strftime('%Y-%m-%d')} a {investment_period_data.index[-1].strftime('%Y-%m-%d')}")
            print(f"Datos: {len(investment_period_data)} días ({len(investment_period_data)/252:.2f} años)")
            
            # Predicción actual
            current_prediction = trader.predict_market_state(full_labeled_data)
            
            # Backtesting solo en período de inversión
            backtest_data, performance_metrics = trader.backtest_strategy(investment_period_data)
            
            # Compilar resultados
            results = {
                'trader': trader,
                'full_data': full_labeled_data,
                'investment_data': investment_period_data,
                'backtest_data': backtest_data,
                'performance_metrics': performance_metrics,
                'current_prediction': current_prediction,
                'model_accuracy': accuracy,
                'feature_importance': trader.get_feature_importance()
            }
            
            return trader, results
            
        except Exception as e:
            print(f"❌ Error en análisis: {str(e)}")
            raise
    
    @staticmethod
    def compare_periods(symbol: str = 'SPYG', periods: List[int] = [1, 3, 5, 10], 
                       include_max: bool = True, config: Optional[TradingConfig] = None) -> List[Dict[str, Any]]:
        """
        Compara el performance en diferentes períodos
        
        Args:
            symbol: Símbolo a analizar
            periods: Lista de años a comparar
            include_max: Si incluir período completo
            config: Configuración personalizada
            
        Returns:
            Lista con resultados de comparación
        """
        print(f"=== COMPARACIÓN DE PERÍODOS PARA {symbol} ===\n")
        
        results_comparison = []
        
        # Analizar períodos específicos
        for years in periods:
            print(f"🔄 Analizando período de {years} año(s)...")
            try:
                trader, results = AnalysisRunner.run_single_analysis(symbol, years, config)
                metrics = results['performance_metrics']
                
                results_comparison.append({
                    'period': f"{years} año(s)",
                    'years': years,
                    'period_years': metrics['period']['years'],
                    'strategy_return': metrics['strategy']['total_return'],
                    'strategy_annual': metrics['strategy']['annual_return'],
                    'buyhold_return': metrics['buyhold']['total_return'],
                    'buyhold_annual': metrics['buyhold']['annual_return'],
                    'dca_return': metrics['dca']['total_return'],
                    'dca_annual': metrics['dca']['annual_return'],
                    'strategy_sharpe': metrics['strategy']['sharpe_ratio'],
                    'buyhold_sharpe': metrics['buyhold']['sharpe_ratio'],
                    'dca_sharpe': metrics['dca']['sharpe_ratio'],
                    'outperformance_bh': metrics['comparison']['outperformance_vs_buyhold'],
                    'outperformance_dca': metrics['comparison']['outperformance_vs_dca'],
                    'num_trades': metrics['strategy']['num_trades'],
                    'win_rate': metrics['strategy']['win_rate']
                })
                
            except Exception as e:
                print(f"❌ Error analizando período de {years} años: {str(e)}")
            
            print("-" * 80)
        
        # Analizar período completo
        if include_max:
            print("🔄 Analizando período COMPLETO...")
            try:
                trader = MarketSentimentTrader(symbol, config)
                full_results = trader.run_complete_analysis(period='max')
                metrics = full_results['performance_metrics']
                
                results_comparison.append({
                    'period': 'COMPLETO',
                    'years': 'MAX',
                    'period_years': metrics['period']['years'],
                    'strategy_return': metrics['strategy']['total_return'],
                    'strategy_annual': metrics['strategy']['annual_return'],
                    'buyhold_return': metrics['buyhold']['total_return'],
                    'buyhold_annual': metrics['buyhold']['annual_return'],
                    'dca_return': metrics['dca']['total_return'],
                    'dca_annual': metrics['dca']['annual_return'],
                    'strategy_sharpe': metrics['strategy']['sharpe_ratio'],
                    'buyhold_sharpe': metrics['buyhold']['sharpe_ratio'],
                    'dca_sharpe': metrics['dca']['sharpe_ratio'],
                    'outperformance_bh': metrics['comparison']['outperformance_vs_buyhold'],
                    'outperformance_dca': metrics['comparison']['outperformance_vs_dca'],
                    'num_trades': metrics['strategy']['num_trades'],
                    'win_rate': metrics['strategy']['win_rate']
                })
                
            except Exception as e:
                print(f"❌ Error analizando período completo: {str(e)}")
        
        # Mostrar tabla comparativa
        if results_comparison:
            AnalysisRunner.print_comparison_table(results_comparison)
        
        return results_comparison
    
    @staticmethod
    def print_comparison_table(results: List[Dict[str, Any]]):
        """Imprime tabla comparativa de resultados"""
        print(f"\n=== TABLA COMPARATIVA DE PERÍODOS ===")
        print(f"{'Período':<10} {'Años':<6} {'Estrategia':<12} {'Buy&Hold':<12} {'DCA':<12} {'vs B&H':<10} {'vs DCA':<10} {'Trades':<8} {'Win%':<6}")
        print("-" * 90)
        
        for result in results:
            period_str = result['period'][:9]
            years_str = f"{result['period_years']:.1f}" if isinstance(result['years'], (int, float)) else "MAX"
            print(f"{period_str:<10} {years_str:<6} {result['strategy_annual']:>9.1f}%  {result['buyhold_annual']:>9.1f}%  {result['dca_annual']:>9.1f}%  {result['outperformance_bh']:>+7.1f}%  {result['outperformance_dca']:>+7.1f}%  {result['num_trades']:>6}  {result['win_rate']:>4.1f}%")
        
        # Análisis de resultados
        best_strategy = max(results, key=lambda x: x['strategy_annual'])
        best_sharpe = max(results, key=lambda x: x['strategy_sharpe'])
        
        print(f"\n=== RESUMEN COMPARATIVO ===")
        print(f"🏆 Mejor retorno anualizado: {best_strategy['period']} ({best_strategy['strategy_annual']:.1f}%)")
        print(f"🎯 Mejor Sharpe ratio: {best_sharpe['period']} ({best_sharpe['strategy_sharpe']:.3f})")
        
        # Estadísticas de éxito
        wins_vs_bh = sum(1 for r in results if r['outperformance_bh'] > 0)
        wins_vs_dca = sum(1 for r in results if r['outperformance_dca'] > 0)
        total_periods = len(results)
        
        print(f"📊 Estrategia supera Buy & Hold: {wins_vs_bh}/{total_periods} períodos ({wins_vs_bh/total_periods*100:.1f}%)")
        print(f"📊 Estrategia supera DCA: {wins_vs_dca}/{total_periods} períodos ({wins_vs_dca/total_periods*100:.1f}%)")