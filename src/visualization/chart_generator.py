# src/visualization/chart_generator.py
"""
Generador de gráficos para análisis de trading
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

class ChartGenerator:
    """Clase para generar gráficos de análisis"""
    
    def __init__(self):
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_multi_strategy_comparison(self, data: pd.DataFrame, metrics: Dict[str, Any], 
                                       symbol: str, save_path: str = None) -> str:
        """Crea gráfico comparativo completo de múltiples estrategias"""
        
        # Configurar figura con subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Subplot 1: Evolución del portfolio (principal)
        ax1 = fig.add_subplot(gs[0, :])
        
        dates = data.index
        
        # Plotear todas las estrategias
        if 'Strategy1_Value' in data.columns:
            ax1.plot(dates, data['Strategy1_Value'], label='Estrategia 1 (Capital + RF)', 
                    linewidth=3, color='#1f77b4')
        
        if 'Strategy2_Value' in data.columns:
            ax1.plot(dates, data['Strategy2_Value'], label='Estrategia 2 (Ahorro + RF)', 
                    linewidth=3, color='#2ca02c')
        
        if 'BuyHold_Value' in data.columns:
            ax1.plot(dates, data['BuyHold_Value'], label='Buy & Hold', 
                    linewidth=3, color='#ff7f0e')
        
        if 'DCA_Value' in data.columns:
            ax1.plot(dates, data['DCA_Value'], label='DCA', 
                    linewidth=3, color='#d62728')
        
        ax1.set_title(f'Comparación de Estrategias de Inversión - {symbol}', 
                     fontsize=18, fontweight='bold', pad=20)
        ax1.set_ylabel('Valor del Portfolio ($)', fontsize=14)
        ax1.legend(fontsize=12, loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        
        # Formatear eje Y
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Subplot 2: Drawdown comparison
        ax2 = fig.add_subplot(gs[1, 0])
        
        def calculate_drawdown(values):
            if isinstance(values, pd.Series):
                values_series = values
            else:
                values_series = pd.Series(values)
            peak = values_series.expanding().max()
            return (values_series - peak) / peak * 100
        
        if 'Strategy1_Value' in data.columns:
            dd1 = calculate_drawdown(data['Strategy1_Value'])
            ax2.fill_between(dates, dd1, 0, alpha=0.6, color='#1f77b4', label='Estrategia 1')
        
        if 'BuyHold_Value' in data.columns:
            dd_bh = calculate_drawdown(data['BuyHold_Value'])
            ax2.fill_between(dates, dd_bh, 0, alpha=0.6, color='#ff7f0e', label='Buy & Hold')
        
        ax2.set_title('Drawdown Comparativo', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Subplot 3: Métricas en barras
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Preparar datos para gráfico de barras
        strategies = []
        annual_returns = []
        sharpe_ratios = []
        
        for strategy_name in ['strategy1', 'strategy2', 'buyhold', 'dca']:
            if strategy_name in metrics:
                if strategy_name == 'strategy1':
                    strategies.append('Estrategia 1')
                elif strategy_name == 'strategy2':
                    strategies.append('Estrategia 2')
                elif strategy_name == 'buyhold':
                    strategies.append('Buy & Hold')
                elif strategy_name == 'dca':
                    strategies.append('DCA')
                
                annual_returns.append(metrics[strategy_name]['annual_return'])
                sharpe_ratios.append(metrics[strategy_name]['sharpe_ratio'])
        
        x = np.arange(len(strategies))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, annual_returns, width, label='Retorno Anual (%)', 
                       color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'][:len(strategies)])
        
        ax3_twin = ax3.twinx()
        bars2 = ax3_twin.bar(x + width/2, sharpe_ratios, width, label='Sharpe Ratio', 
                           color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'][:len(strategies)], alpha=0.7)
        
        ax3.set_title('Métricas de Rendimiento', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Retorno Anual (%)', fontsize=12)
        ax3_twin.set_ylabel('Sharpe Ratio', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(strategies, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for i, (ret, sharpe) in enumerate(zip(annual_returns, sharpe_ratios)):
            ax3.text(i - width/2, ret + 0.5, f'{ret:.1f}%', ha='center', va='bottom', fontweight='bold')
            ax3_twin.text(i + width/2, sharpe + 0.05, f'{sharpe:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Subplot 4: Señales de trading
        ax4 = fig.add_subplot(gs[2, :])
        
        # Plotear precio con señales
        if 'Close' in data.columns:
            ax4.plot(dates, data['Close'], label=f'Precio {symbol}', color='black', linewidth=1)
            
            # Agregar señales de compra y venta
            if 'Signal' in data.columns:
                buy_signals = data[data['Signal'] == 0]
                sell_signals = data[data['Signal'] == 2]
                
                if not buy_signals.empty:
                    ax4.scatter(buy_signals.index, buy_signals['Close'], 
                              color='green', marker='^', s=100, label='Señales de Compra', zorder=5)
                
                if not sell_signals.empty:
                    ax4.scatter(sell_signals.index, sell_signals['Close'], 
                              color='red', marker='v', s=100, label='Señales de Venta', zorder=5)
        
        ax4.set_title('Señales de Trading', fontsize=14, fontweight='bold')
        ax4.set_ylabel(f'Precio {symbol} ($)', fontsize=12)
        ax4.set_xlabel('Fecha', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar gráfico
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'enhanced_backtest_{symbol}_{timestamp}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   📈 Gráfico completo guardado en: {save_path}")
        
        # Mostrar gráfico
        plt.show()
        
        return save_path
    
    def create_threshold_analysis_chart(self, data: pd.DataFrame, symbol: str, 
                                      thresholds_info: Dict[str, Any]) -> str:
        """Crea gráfico de análisis de umbrales"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        dates = data.index
        
        # VIX con umbrales
        if 'VIX' in data.columns:
            ax1.plot(dates, data['VIX'], label='VIX', color='purple', linewidth=2)
            
            if 'VIX_Q_Fear' in data.columns:
                ax1.plot(dates, data['VIX_Q_Fear'], label='Umbral Miedo', 
                        color='red', linestyle='--', alpha=0.7)
            
            if 'VIX_Q_Euphoria' in data.columns:
                ax1.plot(dates, data['VIX_Q_Euphoria'], label='Umbral Euforia', 
                        color='green', linestyle='--', alpha=0.7)
            
            ax1.set_title('VIX y Umbrales Dinámicos', fontweight='bold')
            ax1.set_ylabel('VIX')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # RSI con umbrales
        if 'RSI' in data.columns:
            ax2.plot(dates, data['RSI'], label='RSI', color='blue', linewidth=2)
            
            if 'RSI_Q_Oversold' in data.columns:
                ax2.plot(dates, data['RSI_Q_Oversold'], label='Umbral Sobreventa', 
                        color='green', linestyle='--', alpha=0.7)
            
            if 'RSI_Q_Overbought' in data.columns:
                ax2.plot(dates, data['RSI_Q_Overbought'], label='Umbral Sobrecompra', 
                        color='red', linestyle='--', alpha=0.7)
            
            ax2.set_title('RSI y Umbrales Dinámicos', fontweight='bold')
            ax2.set_ylabel('RSI')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Bollinger Bands Position
        if 'BB_Position' in data.columns:
            ax3.plot(dates, data['BB_Position'], label='Posición BB', color='orange', linewidth=2)
            ax3.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='Límite Inferior')
            ax3.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Límite Superior')
            ax3.fill_between(dates, 0.05, 0.95, alpha=0.1, color='gray', label='Zona Normal')
            
            ax3.set_title('Posición en Bollinger Bands', fontweight='bold')
            ax3.set_ylabel('Posición BB')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Volatilidad
        if 'Volatility_20' in data.columns:
            ax4.plot(dates, data['Volatility_20'], label='Volatilidad 20d', color='red', linewidth=2)
            
            if 'Volatility_Q_High' in data.columns:
                ax4.plot(dates, data['Volatility_Q_High'], label='Umbral Alta Volatilidad', 
                        color='darkred', linestyle='--', alpha=0.7)
            
            ax4.set_title('Volatilidad y Umbrales', fontweight='bold')
            ax4.set_ylabel('Volatilidad (%)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Ajustar formato de fechas
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
        
        plt.suptitle(f'Análisis de Umbrales Dinámicos - {symbol}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Guardar
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'threshold_analysis_{symbol}_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return save_path