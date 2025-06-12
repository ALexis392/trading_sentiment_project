#!/usr/bin/env python3
"""
main.py - Módulo Principal del Sistema de Trading

Este es el punto de entrada principal para ejecutar el sistema de trading
basado en sentimiento de mercado.

Autor: Tu Nombre
Fecha: 2024
"""

import sys
import os
from typing import Optional
from src.config.config import TradingConfig, DEFAULT_CONFIG
from src.core.market_sentiment_trader import MarketSentimentTrader
from src.analysis.analysis_runner import AnalysisRunner



def print_header():
    """Imprime el header del programa"""
    print("=" * 80)
    print("🚀 SISTEMA DE TRADING BASADO EN SENTIMIENTO DE MERCADO")
    print("=" * 80)
    print("Versión: 1.0.0")
    print("Descripción: Sistema modularizado para análisis y trading automático")
    print("=" * 80)


def ejemplo_analisis_basico():
    """
    Ejemplo 1: Análisis básico de un símbolo
    """
    print("\n" + "="*60)
    print("📊 EJEMPLO 1: ANÁLISIS BÁSICO")
    print("="*60)
    
    try:
        # Crear trader con configuración por defecto
        trader = MarketSentimentTrader('SPYG')
        
        # Ejecutar análisis completo
        print("Ejecutando análisis completo para SPYG...")
        results = trader.run_complete_analysis(period='3y')
        
        # Mostrar resultados principales
        print(f"\n✅ RESULTADOS PARA {results['symbol']}:")
        print(f"📈 Período analizado: {results['period']}")
        print(f"📊 Puntos de datos: {results['data_points']:,}")
        print(f"🎯 Precisión del modelo: {results['model_accuracy']:.1%}")
        
        # Predicción actual
        pred = results['current_prediction']
        print(f"\n🔮 ESTADO ACTUAL DEL MERCADO:")
        print(f"   Predicción: {pred['prediction']}")
        print(f"   Señal: {pred['signal']}")
        print(f"   Probabilidades:")
        for estado, prob in pred['probabilities'].items():
            print(f"     {estado}: {prob:.1%}")
        
        # Métricas de performance
        metrics = results['performance_metrics']
        print(f"\n💰 PERFORMANCE (3 AÑOS):")
        print(f"   Estrategia: {metrics['strategy']['total_return']:+.1f}% anual")
        print(f"   Buy & Hold: {metrics['buyhold']['total_return']:+.1f}% anual")
        print(f"   Outperformance: {metrics['comparison']['outperformance_vs_buyhold']:+.1f}%")
        print(f"   Sharpe Ratio: {metrics['strategy']['sharpe_ratio']:.2f}")
        print(f"   Operaciones: {metrics['strategy']['num_trades']}")
        
        # Features más importantes
        importance = results['feature_importance']
        if importance is not None:
            print(f"\n🔍 TOP 3 FEATURES MÁS IMPORTANTES:")
            for i, (_, row) in enumerate(importance.head(3).iterrows(), 1):
                print(f"   {i}. {row['Feature']}: {row['Importance']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en análisis básico: {str(e)}")
        return False


def ejemplo_configuracion_personalizada():
    """
    Ejemplo 2: Análisis con configuración personalizada
    """
    print("\n" + "="*60)
    print("⚙️ EJEMPLO 2: CONFIGURACIÓN PERSONALIZADA")
    print("="*60)
    
    try:
        # Crear configuración personalizada
        config_personalizada = TradingConfig(
            INITIAL_CAPITAL=50000,        # Capital inicial mayor
            TRANSACTION_COST=0.0005,      # Menor costo de transacción (0.05%)
            MIN_LOOKBACK_DAYS=60,         # Más días históricos requeridos
            DCA_FREQUENCY=21,             # DCA cada 3 semanas
            RF_N_ESTIMATORS=200,          # Más árboles en Random Forest
            RF_MAX_DEPTH=15               # Mayor profundidad
        )
        
        print("📋 Configuración personalizada:")
        print(f"   Capital inicial: ${config_personalizada.INITIAL_CAPITAL:,}")
        print(f"   Costo transacción: {config_personalizada.TRANSACTION_COST:.3%}")
        print(f"   Días históricos mín: {config_personalizada.MIN_LOOKBACK_DAYS}")
        print(f"   Frecuencia DCA: {config_personalizada.DCA_FREQUENCY} días")
        print(f"   RF estimadores: {config_personalizada.RF_N_ESTIMATORS}")
        
        # Ejecutar análisis con configuración personalizada
        print(f"\nAnalizando QQQ con configuración personalizada...")
        trader, results = AnalysisRunner.run_single_analysis(
            symbol='QQQ',
            investment_start_years_ago=5,
            config=config_personalizada
        )
        
        # Mostrar resultados
        metrics = results['performance_metrics']
        print(f"\n✅ RESULTADOS CON CONFIGURACIÓN PERSONALIZADA:")
        print(f"   Período: {metrics['period']['years']:.1f} años")
        print(f"   Valor final estrategia: ${metrics['strategy']['final_value']:,.0f}")
        print(f"   Valor final Buy&Hold: ${metrics['buyhold']['final_value']:,.0f}")
        print(f"   Diferencia: ${metrics['strategy']['final_value'] - metrics['buyhold']['final_value']:+,.0f}")
        
        # Mostrar algunas operaciones
        trades = results['performance_metrics']['trades']
        if len(trades) > 0:
            print(f"\n📈 ÚLTIMAS 3 OPERACIONES:")
            for trade in trades[-3:]:
                emoji = '🟢' if trade['action'] == 'BUY' else '🔴'
                print(f"   {emoji} {trade['date'].strftime('%Y-%m-%d')}: {trade['action']} a ${trade['price']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en configuración personalizada: {str(e)}")
        return False


def ejemplo_comparacion_periodos():
    """
    Ejemplo 3: Comparación de múltiples períodos
    """
    print("\n" + "="*60)
    print("📊 EJEMPLO 3: COMPARACIÓN DE PERÍODOS")
    print("="*60)
    
    try:
        print("Comparando performance en diferentes períodos para BTC-USD...")
        print("⏳ Esto puede tomar unos minutos...\n")
        
        # Ejecutar comparación
        comparison_results = AnalysisRunner.compare_periods(
            symbol='SPYG',
            periods=[1, 3, 5,10],
            include_max=True
        )
        
        if comparison_results:
            print(f"\n🏆 RESUMEN DE RESULTADOS:")
            
            # Encontrar mejores períodos
            best_return = max(comparison_results, key=lambda x: x['strategy_annual'])
            best_sharpe = max(comparison_results, key=lambda x: x['strategy_sharpe'])
            
            print(f"   🥇 Mejor retorno: {best_return['period']} ({best_return['strategy_annual']:+.1f}% anual)")
            print(f"   🎯 Mejor Sharpe: {best_sharpe['period']} (ratio: {best_sharpe['strategy_sharpe']:.2f})")
            
            # Contar victorias
            wins_bh = sum(1 for r in comparison_results if r['outperformance_bh'] > 0)
            wins_dca = sum(1 for r in comparison_results if r['outperformance_dca'] > 0)
            total = len(comparison_results)
            
            print(f"   📈 Supera Buy&Hold: {wins_bh}/{total} períodos ({wins_bh/total:.0%})")
            print(f"   📈 Supera DCA: {wins_dca}/{total} períodos ({wins_dca/total:.0%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en comparación de períodos: {str(e)}")
        return False


def ejemplo_analisis_rapido(symbol: str):
    """
    Ejemplo 4: Análisis rápido de cualquier símbolo
    """
    print(f"\n📊 ANÁLISIS RÁPIDO DE {symbol}")
    print("-" * 40)
    
    try:
        trader = MarketSentimentTrader(symbol)
        results = trader.run_complete_analysis(period='2y')
        
        # Mostrar solo lo esencial
        pred = results['current_prediction']
        metrics = results['performance_metrics']
        
        print(f"🎯 Predicción actual: {pred['prediction']} → {pred['signal']}")
        print(f"💰 Retorno 2 años: {metrics['strategy']['annual_return']:+.1f}% vs {metrics['buyhold']['annual_return']:+.1f}% (B&H)")
        print(f"📊 Sharpe Ratio: {metrics['strategy']['sharpe_ratio']:.2f}")
        print(f"🔄 Operaciones: {metrics['strategy']['num_trades']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analizando {symbol}: {str(e)}")
        return False


def menu_interactivo():
    """Menú interactivo para el usuario"""
    print("\n" + "="*60)
    print("🎛️ MENÚ INTERACTIVO")
    print("="*60)
    
    while True:
        print("\nSelecciona una opción:")
        print("1. 📊 Análisis básico (SPYG - 3 años)")
        print("2. ⚙️ Configuración personalizada (QQQ - 5 años)")
        print("3. 📈 Comparación períodos (BTC-USD)")
        print("4. ⚡ Análisis rápido (símbolo personalizado)")
        print("5. 🚀 Ejecutar todos los ejemplos")
        print("0. ❌ Salir")
        
        try:
            opcion = input("\n👉 Ingresa tu opción (0-5): ").strip()
            
            if opcion == '0':
                print("\n👋 ¡Hasta luego!")
                break
            elif opcion == '1':
                ejemplo_analisis_basico()
            elif opcion == '2':
                ejemplo_configuracion_personalizada()
            elif opcion == '3':
                ejemplo_comparacion_periodos()
            elif opcion == '4':
                symbol = input("📝 Ingresa el símbolo (ej: AAPL, TSLA, ETH-USD): ").strip().upper()
                if symbol:
                    ejemplo_analisis_rapido(symbol)
                else:
                    print("❌ Símbolo inválido")
            elif opcion == '5':
                print("\n🚀 Ejecutando todos los ejemplos...")
                ejemplo_analisis_basico()
                ejemplo_configuracion_personalizada()
                ejemplo_comparacion_periodos()
            else:
                print("❌ Opción inválida. Intenta de nuevo.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Programa interrumpido por el usuario. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error inesperado: {str(e)}")


def main():
    """
    Función principal del programa
    """
    print_header()
    
    # Verificar que se pueden importar los módulos
    try:
        print("🔍 Verificando módulos del sistema...")
        trader_test = MarketSentimentTrader('SPYG')
        print("✅ Todos los módulos cargados correctamente")
    except Exception as e:
        print(f"❌ Error al cargar módulos: {str(e)}")
        print("💡 Verifica que la estructura de carpetas sea correcta")
        return
    
    # Preguntar al usuario qué quiere hacer
    print("\n¿Cómo quieres usar el sistema?")
    print("1. 🎛️ Menú interactivo")
    print("2. 🚀 Ejecutar todos los ejemplos automáticamente")
    print("3. 📊 Solo análisis básico")
    
    try:
        modo = input("\n👉 Selecciona el modo (1-3): ").strip()
        
        if modo == '1':
            menu_interactivo()
        elif modo == '2':
            print("\n🚀 Ejecutando todos los ejemplos automáticamente...\n")
            ejemplo_analisis_basico()
            ejemplo_configuracion_personalizada()
            ejemplo_comparacion_periodos()
        elif modo == '3':
            ejemplo_analisis_basico()
        else:
            print("❌ Opción inválida. Ejecutando análisis básico por defecto...")
            ejemplo_analisis_basico()
            
    except KeyboardInterrupt:
        print("\n\n👋 Programa interrumpido. ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
    
    print("\n" + "="*80)
    print("✅ Programa finalizado")
    print("="*80)


if __name__ == "__main__":
    main()