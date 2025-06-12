# src/backtesting/backtester_enhanced.py
"""
Backtester mejorado con estrategias de tasa libre de riesgo y visualizaciones
Versión modularizada y corregida
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SignalGenerator:
    """Generador de señales de trading con prevención de look-ahead bias"""
    
    def __init__(self, config):
        self.config = config
        self.thresholds = config.backtesting_thresholds
        self.prevent_lookahead = config.PREVENT_LOOKAHEAD_BIAS
    
    def calculate_expanding_percentiles(self, series: pd.Series, min_lookback_days: int, 
                                      percentile: float, prevent_lookahead: bool = True) -> pd.Series:
        """Calcula percentiles expandidos con prevención de look-ahead bias"""
        if not prevent_lookahead:
            return series.quantile(percentile)
        
        expanding_percentiles = []
        for i in range(len(series)):
            historical_data = series.iloc[:i+1]
            if len(historical_data) >= min_lookback_days:
                expanding_percentiles.append(historical_data.quantile(percentile))
            else:
                expanding_percentiles.append(np.nan)
        
        return pd.Series(expanding_percentiles, index=series.index)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Genera señales de trading con información detallada de umbrales"""
        print("📊 Generando señales de trading mejoradas...")
        print(f"   🔒 Prevención look-ahead bias: {'✅ Activada' if self.prevent_lookahead else '❌ Desactivada'}")
        print(f"   📊 Estrategia: {self.thresholds.get_strategy_name()}")
        
        df = data.copy()
        
        # Verificar columnas requeridas
        required_columns = ['VIX', 'RSI', 'BB_Position', 'Price_Change_20', 'Volatility_20']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️  Columnas faltantes: {missing_columns}")
            df['Signal'] = 1
            return df
        
        try:
            # Calcular percentiles con información detallada
            print("   📊 Calculando percentiles dinámicos...")
            
            if self.prevent_lookahead:
                print("      🔒 Método: Ventana expandida (sin look-ahead bias)")
                df = self._calculate_expanding_thresholds(df)
            else:
                print("      ⚠️  Método: Percentiles fijos (CON look-ahead bias)")
                df = self._calculate_fixed_thresholds(df)
            
            # Mostrar valores actuales de umbrales
            self._print_current_thresholds(df)
            
            # Generar señales
            df = self._generate_trading_signals(df)
            
            # Estadísticas de señales
            self._print_signal_statistics(df)
            
            return df
            
        except Exception as e:
            print(f"⚠️  Error generando señales: {str(e)}")
            df['Signal'] = 1
            return df
    
    def _calculate_expanding_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula umbrales expandidos"""
        df['VIX_Q_Fear'] = self.calculate_expanding_percentiles(
            df['VIX'], self.config.MIN_LOOKBACK_DAYS, self.thresholds.vix_fear_threshold
        )
        df['VIX_Q_Euphoria'] = self.calculate_expanding_percentiles(
            df['VIX'], self.config.MIN_LOOKBACK_DAYS, self.thresholds.vix_euphoria_threshold
        )
        df['RSI_Q_Oversold'] = self.calculate_expanding_percentiles(
            df['RSI'], self.config.MIN_LOOKBACK_DAYS, self.thresholds.rsi_oversold
        )
        df['RSI_Q_Overbought'] = self.calculate_expanding_percentiles(
            df['RSI'], self.config.MIN_LOOKBACK_DAYS, self.thresholds.rsi_overbought
        )
        df['Price_Change_Q_Extreme'] = self.calculate_expanding_percentiles(
            df['Price_Change_20'], self.config.MIN_LOOKBACK_DAYS, self.thresholds.price_change_extreme
        )
        df['Volatility_Q_High'] = self.calculate_expanding_percentiles(
            df['Volatility_20'], self.config.MIN_LOOKBACK_DAYS, self.thresholds.volatility_high
        )
        return df
    
    def _calculate_fixed_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula umbrales fijos"""
        df['VIX_Q_Fear'] = df['VIX'].quantile(self.thresholds.vix_fear_threshold)
        df['VIX_Q_Euphoria'] = df['VIX'].quantile(self.thresholds.vix_euphoria_threshold)
        df['RSI_Q_Oversold'] = df['RSI'].quantile(self.thresholds.rsi_oversold)
        df['RSI_Q_Overbought'] = df['RSI'].quantile(self.thresholds.rsi_overbought)
        df['Price_Change_Q_Extreme'] = df['Price_Change_20'].quantile(self.thresholds.price_change_extreme)
        df['Volatility_Q_High'] = df['Volatility_20'].quantile(self.thresholds.volatility_high)
        return df
    
    def _print_current_thresholds(self, df: pd.DataFrame):
        """Imprime umbrales actuales"""
        if len(df) > self.config.MIN_LOOKBACK_DAYS:
            last_valid_idx = df.dropna().index[-1]
            current_vix = df.loc[last_valid_idx, 'VIX']
            current_rsi = df.loc[last_valid_idx, 'RSI']
            
            if self.prevent_lookahead:
                vix_fear_threshold = df.loc[last_valid_idx, 'VIX_Q_Fear']
                vix_euphoria_threshold = df.loc[last_valid_idx, 'VIX_Q_Euphoria']
                rsi_oversold_threshold = df.loc[last_valid_idx, 'RSI_Q_Oversold']
                rsi_overbought_threshold = df.loc[last_valid_idx, 'RSI_Q_Overbought']
            else:
                vix_fear_threshold = df['VIX_Q_Fear'].iloc[-1]
                vix_euphoria_threshold = df['VIX_Q_Euphoria'].iloc[-1]
                rsi_oversold_threshold = df['RSI_Q_Oversold'].iloc[-1]
                rsi_overbought_threshold = df['RSI_Q_Overbought'].iloc[-1]
            
            print(f"\n   📊 UMBRALES ESPECÍFICOS CALCULADOS (fecha actual):")
            print(f"      VIX actual: {current_vix:.1f}")
            print(f"      VIX Miedo (>{self.thresholds.vix_fear_threshold:.0%}): {vix_fear_threshold:.1f}")
            print(f"      VIX Euforia (<{self.thresholds.vix_euphoria_threshold:.0%}): {vix_euphoria_threshold:.1f}")
            print(f"      RSI actual: {current_rsi:.1f}")
            print(f"      RSI Sobreventa (<{self.thresholds.rsi_oversold:.0%}): {rsi_oversold_threshold:.1f}")
            print(f"      RSI Sobrecompra (>{self.thresholds.rsi_overbought:.0%}): {rsi_overbought_threshold:.1f}")
    
    def _generate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera las señales de trading"""
        # Inicializar señales
        df['Signal'] = 1  # Mantener por defecto
        
        # === CONDICIONES DE COMPRA (MIEDO EXTREMO) ===
        buy_vix_extreme = df['VIX'] > df['VIX_Q_Fear']
        buy_rsi_oversold = df['RSI'] < df['RSI_Q_Oversold']
        buy_bb_lower = df['BB_Position'] < self.thresholds.bb_lower_threshold
        buy_high_vol = df['Volatility_20'] > df['Volatility_Q_High']
        
        buy_conditions = buy_vix_extreme | buy_rsi_oversold | buy_bb_lower | buy_high_vol
        df.loc[buy_conditions, 'Signal'] = 0  # Comprar
        
        # === CONDICIONES DE VENTA (EUFORIA) ===
        sell_vix_low = df['VIX'] < df['VIX_Q_Euphoria']
        sell_rsi_overbought = df['RSI'] > df['RSI_Q_Overbought']
        sell_bb_upper = df['BB_Position'] > self.thresholds.bb_upper_threshold
        sell_price_extreme = df['Price_Change_20'] > df['Price_Change_Q_Extreme']
        
        sell_conditions = sell_vix_low & sell_rsi_overbought & sell_bb_upper & sell_price_extreme
        df.loc[sell_conditions, 'Signal'] = 2  # Vender
        
        return df
    
    def _print_signal_statistics(self, df: pd.DataFrame):
        """Imprime estadísticas de señales"""
        signal_counts = df['Signal'].value_counts()
        total_days = len(df)
        
        print(f"\n   ✅ Señales generadas:")
        print(f"      🟢 Comprar: {signal_counts.get(0, 0)} días ({signal_counts.get(0, 0)/total_days*100:.1f}%)")
        print(f"      🟡 Mantener: {signal_counts.get(1, 0)} días ({signal_counts.get(1, 0)/total_days*100:.1f}%)")
        print(f"      🔴 Vender: {signal_counts.get(2, 0)} días ({signal_counts.get(2, 0)/total_days*100:.1f}%)")


class StrategySimulator:
    """Simulador de estrategias de inversión"""
    
    def __init__(self, config):
        self.config = config
        self.risk_free_rate = getattr(config, 'RISK_FREE_RATE_ANNUAL', 0.05)
        self.daily_rf_rate = (1 + self.risk_free_rate) ** (1/252) - 1
    
    def simulate_strategy1_capital_with_rf(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Estrategia 1: Capital completo esperando señales con tasa libre de riesgo"""
        print("   📈 Simulando Estrategia 1: Capital completo esperando señales...")
        
        cash_in_rf = self.config.INITIAL_CAPITAL
        cash_for_trading = 0
        shares = 0
        in_market = False
        trades = []
        values = []
        
        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            signal = data['Signal'].iloc[i]
            date = data.index[i]
            
            # Skip si no tenemos suficiente información histórica
            if i < self.config.MIN_LOOKBACK_DAYS:
                if cash_in_rf > 0:
                    cash_in_rf *= (1 + self.daily_rf_rate)
                
                total_value = cash_in_rf + cash_for_trading + (shares * current_price)
                values.append(total_value)
                continue
            
            # Aplicar tasa libre de riesgo al dinero no invertido
            if cash_in_rf > 0:
                cash_in_rf *= (1 + self.daily_rf_rate)
            
            # Ejecutar operaciones según señales
            if not pd.isna(signal):
                if signal == 0 and not in_market and cash_in_rf > 0:  # Comprar
                    cash_for_trading = cash_in_rf * (1 - self.config.TRANSACTION_COST)
                    shares = cash_for_trading / current_price
                    cash_in_rf = 0
                    cash_for_trading = 0
                    in_market = True
                    
                    trades.append({
                        'date': date, 'action': 'BUY', 'price': current_price,
                        'shares': shares, 'value': shares * current_price
                    })
                
                elif signal == 2 and in_market and shares > 0:  # Vender
                    sale_value = shares * current_price * (1 - self.config.TRANSACTION_COST)
                    cash_in_rf = sale_value
                    shares = 0
                    in_market = False
                    
                    trades.append({
                        'date': date, 'action': 'SELL', 'price': current_price,
                        'shares': 0, 'value': sale_value
                    })
            
            total_value = cash_in_rf + cash_for_trading + (shares * current_price)
            values.append(total_value)
        
        return {
            'values': values,
            'trades': trades,
            'final_cash_rf': cash_in_rf,
            'final_shares': shares,
            'in_market_final': in_market
        }
    
    def simulate_strategy2_savings_with_rf(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Estrategia 2: Ahorro mensual con tasa libre de riesgo hasta señales"""
        print("   💰 Simulando Estrategia 2: Ahorro mensual esperando señales...")
        
        # Calcular ahorro mensual para llegar al capital objetivo
        total_months = len(data) / 21  # Aproximadamente 21 días hábiles por mes
        monthly_savings = self.config.INITIAL_CAPITAL / total_months
        
        cash_in_rf = 0
        shares = 0
        in_market = False
        trades = []
        values = []
        days_since_last_savings = 0
        
        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            signal = data['Signal'].iloc[i]
            date = data.index[i]
            
            # Ahorrar mensualmente (cada 21 días aproximadamente)
            days_since_last_savings += 1
            if days_since_last_savings >= 21:  # Ahorro mensual
                cash_in_rf += monthly_savings
                days_since_last_savings = 0
            
            # Skip si no tenemos suficiente información histórica
            if i < self.config.MIN_LOOKBACK_DAYS:
                if cash_in_rf > 0:
                    cash_in_rf *= (1 + self.daily_rf_rate)
                
                total_value = cash_in_rf + (shares * current_price)
                values.append(total_value)
                continue
            
            # Aplicar tasa libre de riesgo al dinero ahorrado
            if cash_in_rf > 0:
                cash_in_rf *= (1 + self.daily_rf_rate)
            
            # Ejecutar operaciones según señales
            if not pd.isna(signal):
                if signal == 0 and not in_market and cash_in_rf > monthly_savings:  # Comprar
                    investment_amount = cash_in_rf * (1 - self.config.TRANSACTION_COST)
                    new_shares = investment_amount / current_price
                    shares += new_shares
                    cash_in_rf = 0
                    in_market = True
                    
                    trades.append({
                        'date': date, 'action': 'BUY', 'price': current_price,
                        'shares': new_shares, 'value': investment_amount
                    })
                
                elif signal == 2 and in_market and shares > 0:  # Vender
                    sale_value = shares * current_price * (1 - self.config.TRANSACTION_COST)
                    cash_in_rf += sale_value
                    shares = 0
                    in_market = False
                    
                    trades.append({
                        'date': date, 'action': 'SELL', 'price': current_price,
                        'shares': 0, 'value': sale_value
                    })
            
            total_value = cash_in_rf + (shares * current_price)
            values.append(total_value)
        
        return {
            'values': values,
            'trades': trades,
            'final_cash_rf': cash_in_rf,
            'final_shares': shares,
            'monthly_savings': monthly_savings,
            'total_saved': monthly_savings * (len(data) / 21)
        }
    
    def simulate_buyhold(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Simula estrategia Buy & Hold"""
        print("   📊 Calculando Buy & Hold...")
        initial_price = data['Close'].iloc[0]
        initial_shares = self.config.INITIAL_CAPITAL / initial_price
        buyhold_values = initial_shares * data['Close']
        
        return {
            'values': buyhold_values.tolist(),
            'initial_shares': initial_shares
        }
    
    def simulate_dca_corrected(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Simula Dollar Cost Averaging CORREGIDO - empezando desde cero como Strategy 2"""
        print("   💰 Simulando Dollar Cost Averaging corregido...")
        
        # Calcular inversión periódica basada en el total que queremos invertir
        total_periods = len(data) // self.config.DCA_FREQUENCY + 1
        investment_per_period = self.config.INITIAL_CAPITAL / total_periods
        
        # Inicializar variables - EMPEZANDO DESDE CERO
        accumulated_cash = 0  # Dinero acumulado para invertir
        dca_shares = 0
        dca_values = []
        dca_investments = []
        days_since_last_investment = 0
        
        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            date = data.index[i]
            
            # Acumular dinero cada día (simulando ahorro continuo)
            # Para que sea equivalente a la estrategia 2, acumulamos dinero gradualmente
            daily_savings = investment_per_period / self.config.DCA_FREQUENCY
            accumulated_cash += daily_savings
            
            # Aplicar tasa libre de riesgo al dinero acumulado
            if accumulated_cash > 0:
                accumulated_cash *= (1 + self.daily_rf_rate)
            
            # Invertir cada DCA_FREQUENCY días
            days_since_last_investment += 1
            if days_since_last_investment >= self.config.DCA_FREQUENCY and accumulated_cash >= investment_per_period:
                # Invertir el dinero acumulado hasta ahora
                investment_amount = min(investment_per_period, accumulated_cash)
                shares_bought = investment_amount / current_price
                dca_shares += shares_bought
                accumulated_cash -= investment_amount
                days_since_last_investment = 0
                
                dca_investments.append({
                    'date': date, 'price': current_price,
                    'amount': investment_amount, 'shares': shares_bought
                })
            
            # Valor total del portfolio
            dca_total_value = accumulated_cash + (dca_shares * current_price)
            dca_values.append(dca_total_value)
        
        return {
            'values': dca_values,
            'investments': dca_investments,
            'investment_per_period': investment_per_period,
            'final_cash': accumulated_cash,
            'final_shares': dca_shares
        }


class MetricsCalculator:
    """Calculador de métricas de rendimiento"""
    
    def __init__(self, config):
        self.config = config
        self.risk_free_rate = getattr(config, 'RISK_FREE_RATE_ANNUAL', 0.05)
    
    def calculate_strategy_metrics(self, values: List[float], strategy_name: str, 
                                 period_years: float) -> Dict[str, Any]:
        """Calcula métricas para una estrategia específica"""
        if not values or len(values) == 0:
            return {'error': f'Sin datos para {strategy_name}'}
        
        # Retornos
        total_return = (values[-1] - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL * 100
        annual_return = total_return / period_years if period_years > 0 else total_return
        
        # Retornos diarios
        returns = pd.Series(values).pct_change().dropna()
        
        # Sharpe ratio
        if len(returns) > 0 and returns.std() > 0:
            excess_return = returns.mean() - self.risk_free_rate/252
            sharpe_ratio = excess_return / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Volatilidad
        volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0.0
        
        # Max drawdown
        peak = pd.Series(values).expanding().max()
        drawdown = (pd.Series(values) - peak) / peak
        max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0.0
        
        return {
            'final_value': values[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def calculate_all_metrics(self, simulation_results: Dict[str, Any], 
                            period_years: float) -> Dict[str, Any]:
        """Calcula métricas para todas las estrategias"""
        print("📊 Calculando métricas mejoradas...")
        
        metrics = {}
        
        for strategy_name, strategy_data in simulation_results.items():
            if 'values' in strategy_data:
                metrics[strategy_name] = self.calculate_strategy_metrics(
                    strategy_data['values'], strategy_name, period_years
                )
        
        # Agregar comparaciones
        if 'strategy1' in metrics and 'buyhold' in metrics:
            metrics['comparisons'] = {
                'strategy1_vs_buyhold': metrics['strategy1']['total_return'] - metrics['buyhold']['total_return'],
                'strategy2_vs_buyhold': metrics['strategy2']['total_return'] - metrics['buyhold']['total_return'] if 'strategy2' in metrics else 0,
                'strategy1_vs_dca': metrics['strategy1']['total_return'] - metrics['dca']['total_return'] if 'dca' in metrics else 0,
                'strategy2_vs_dca': metrics['strategy2']['total_return'] - metrics['dca']['total_return'] if 'strategy2' in metrics else 0,
            }
        
        return metrics


class ChartGenerator:
    """Generador de gráficos de rendimiento"""
    
    def create_performance_chart(self, simulation_results: Dict[str, Any], data: pd.DataFrame, 
                               symbol: str, save_path: str = None) -> str:
        """Crea gráfico de rendimiento comparativo"""
        
        print("📈 Creando gráfico de rendimiento comparativo...")
        
        # Configurar el gráfico
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Preparar datos para gráfico
        dates = data.index
        
        # Gráfico 1: Evolución del valor del portfolio
        ax1.plot(dates, simulation_results['strategy1']['values'], 
                label='Estrategia 1 (Capital + RF)', linewidth=2, color='blue')
        
        if 'strategy2' in simulation_results:
            ax1.plot(dates, simulation_results['strategy2']['values'], 
                    label='Estrategia 2 (Ahorro + RF)', linewidth=2, color='green')
        
        ax1.plot(dates, simulation_results['buyhold']['values'], 
                label='Buy & Hold', linewidth=2, color='orange')
        ax1.plot(dates, simulation_results['dca']['values'], 
                label='DCA', linewidth=2, color='red')
        
        ax1.set_title(f'Evolución del Portfolio - {symbol}', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Valor del Portfolio ($)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        
        # Gráfico 2: Drawdown
        def calculate_drawdown(values):
            peak = pd.Series(values).expanding().max()
            return (pd.Series(values) - peak) / peak * 100
        
        dd1 = calculate_drawdown(simulation_results['strategy1']['values'])
        dd_bh = calculate_drawdown(simulation_results['buyhold']['values'])
        
        ax2.fill_between(dates, dd1, 0, alpha=0.3, color='blue', label='Estrategia 1')
        ax2.fill_between(dates, dd_bh, 0, alpha=0.3, color='orange', label='Buy & Hold')
        
        ax2.set_title('Drawdown Comparativo', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Fecha', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar gráfico
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'backtest_performance_{symbol}_{timestamp}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Gráfico guardado en: {save_path}")
        
        # Mostrar gráfico
        plt.show()
        
        return save_path


class ResultsPrinter:
    """Impresora de resultados del backtesting"""
    
    def __init__(self, config):
        self.config = config
        self.risk_free_rate = getattr(config, 'RISK_FREE_RATE_ANNUAL', 0.05)
    
    def print_enhanced_summary(self, metrics: Dict[str, Any]):
        """Imprime resumen detallado del backtesting mejorado"""
        
        print(f"\n📊 RESUMEN DETALLADO DEL BACKTESTING MEJORADO")
        print("=" * 90)
        
        # Información del período
        period = metrics['period']
        print(f"📅 Período: {period['start_date']} a {period['end_date']} ({period['years']:.1f} años)")
        print(f"💰 Capital inicial: ${self.config.INITIAL_CAPITAL:,.2f}")
        print(f"📈 Tasa libre de riesgo: {self.risk_free_rate:.1%} anual")
        
        # Rendimientos comparativos
        print(f"\n💰 RENDIMIENTOS ANUALIZADOS:")
        print(f"   🔵 Estrategia 1 (Capital + RF): {metrics['strategy1']['annual_return']:+.1f}%")
        if 'strategy2' in metrics:
            print(f"   🟢 Estrategia 2 (Ahorro + RF): {metrics['strategy2']['annual_return']:+.1f}%")
        print(f"   🟠 Buy & Hold: {metrics['buyhold']['annual_return']:+.1f}%")
        print(f"   🔴 DCA: {metrics['dca']['annual_return']:+.1f}%")
        
        # Outperformances
        comparisons = metrics['comparisons']
        print(f"\n🏆 OUTPERFORMANCES:")
        print(f"   Estrategia 1 vs Buy & Hold: {comparisons['strategy1_vs_buyhold']:+.1f}%")
        if 'strategy2_vs_buyhold' in comparisons:
            print(f"   Estrategia 2 vs Buy & Hold: {comparisons['strategy2_vs_buyhold']:+.1f}%")
        print(f"   Estrategia 1 vs DCA: {comparisons['strategy1_vs_dca']:+.1f}%")
        if 'strategy2_vs_dca' in comparisons:
            print(f"   Estrategia 2 vs DCA: {comparisons['strategy2_vs_dca']:+.1f}%")
        
        # Métricas de riesgo
        print(f"\n📊 MÉTRICAS DE RIESGO:")
        print(f"   📉 Max Drawdown Estrategia 1: {metrics['strategy1']['max_drawdown']:.1f}%")
        if 'strategy2' in metrics:
            print(f"   📉 Max Drawdown Estrategia 2: {metrics['strategy2']['max_drawdown']:.1f}%")
        print(f"   📉 Max Drawdown Buy & Hold: {metrics['buyhold']['max_drawdown']:.1f}%")
        print(f"   📉 Max Drawdown DCA: {metrics['dca']['max_drawdown']:.1f}%")
        print(f"   🎯 Sharpe Ratio Estrategia 1: {metrics['strategy1']['sharpe_ratio']:.2f}")
        if 'strategy2' in metrics:
            print(f"   🎯 Sharpe Ratio Estrategia 2: {metrics['strategy2']['sharpe_ratio']:.2f}")
        print(f"   🎯 Sharpe Ratio Buy & Hold: {metrics['buyhold']['sharpe_ratio']:.2f}")
        print(f"   🎯 Sharpe Ratio DCA: {metrics['dca']['sharpe_ratio']:.2f}")
        
        # Detalles de operaciones
        details = metrics['simulation_details']
        print(f"\n⚡ DETALLES DE OPERACIONES:")
        print(f"   Estrategia 1: {len(details['strategy1_trades'])} operaciones")
        if details['strategy2_trades']:
            print(f"   Estrategia 2: {len(details['strategy2_trades'])} operaciones")
            print(f"   Ahorro mensual: ${details['monthly_savings']:,.2f}")
        print(f"   DCA: {len(details['dca_investments'])} inversiones")
        
        # Valores finales
        print(f"\n💰 VALORES FINALES:")
        print(f"   🔵 Estrategia 1: ${metrics['strategy1']['final_value']:,.2f}")
        if 'strategy2' in metrics:
            print(f"   🟢 Estrategia 2: ${metrics['strategy2']['final_value']:,.2f}")
        print(f"   🟠 Buy & Hold: ${metrics['buyhold']['final_value']:,.2f}")
        print(f"   🔴 DCA: ${metrics['dca']['final_value']:,.2f}")
        
        # Ranking
        strategies_ranking = [
            ('Estrategia 1', metrics['strategy1']['annual_return']),
            ('Buy & Hold', metrics['buyhold']['annual_return']),
            ('DCA', metrics['dca']['annual_return'])
        ]
        
        if 'strategy2' in metrics:
            strategies_ranking.append(('Estrategia 2', metrics['strategy2']['annual_return']))
        
        strategies_ranking.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 RANKING POR RENDIMIENTO:")
        for i, (name, return_pct) in enumerate(strategies_ranking, 1):
            emoji = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '🏅'
            print(f"   {emoji} {i}. {name}: {return_pct:+.1f}%")
        
        # Información del gráfico
        print(f"\n📈 GRÁFICO GENERADO: {details['chart_path']}")
        
        # Metodología
        methodology = metrics['methodology']
        print(f"\n⚙️  METODOLOGÍA:")
        print(f"   🔒 Look-ahead bias: {'✅ Prevenido' if methodology['lookahead_prevention'] else '❌ NO prevenido'}")
        print(f"   📊 Tipo de estrategia: {methodology['strategy_type']}")
        print(f"   💰 Tasa libre de riesgo incluida: {'✅ Sí' if methodology['risk_free_rate_included'] else '❌ No'}")
        print(f"   🔧 DCA corregido: ✅ Ambas estrategias de ahorro inician desde cero")


class EnhancedBacktester:
    """Backtester mejorado con estrategias de tasa libre de riesgo - Versión modularizada"""
    
    def __init__(self, config):
        self.config = config
        self.thresholds = config.backtesting_thresholds
        self.prevent_lookahead = config.PREVENT_LOOKAHEAD_BIAS
        self.expanding_window = config.EXPANDING_WINDOW
        self.risk_free_rate = getattr(config, 'RISK_FREE_RATE_ANNUAL', 0.05)
        
        # Inicializar componentes modulares
        self.signal_generator = SignalGenerator(config)
        self.strategy_simulator = StrategySimulator(config)
        self.metrics_calculator = MetricsCalculator(config)
        self.chart_generator = ChartGenerator()
        self.results_printer = ResultsPrinter(config)
    
    def generate_trading_signals_enhanced(self, data: pd.DataFrame) -> pd.DataFrame:
        """Genera señales de trading con información detallada de umbrales"""
        return self.signal_generator.generate_signals(data)
    
    def simulate_strategy_with_risk_free_rate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Simula estrategias mejoradas con tasa libre de riesgo"""
        print("🎮 Simulando estrategias mejoradas con tasa libre de riesgo...")
        print(f"   💰 Tasa libre de riesgo anual: {self.risk_free_rate:.1%}")
        
        if 'Close' not in data.columns or 'Signal' not in data.columns:
            raise ValueError("Datos insuficientes para simulación")
        
        results = {}
        
        # === ESTRATEGIA 1: CAPITAL COMPLETO CON TASA LIBRE DE RIESGO ===
        results['strategy1'] = self.strategy_simulator.simulate_strategy1_capital_with_rf(data)
        print(f"   ✅ Estrategia 1: {len(results['strategy1']['trades'])} operaciones, valor final: ${results['strategy1']['values'][-1]:,.2f}")
        
        # === ESTRATEGIA 2: AHORRO MENSUAL CON TASA LIBRE DE RIESGO ===
        results['strategy2'] = self.strategy_simulator.simulate_strategy2_savings_with_rf(data)
        print(f"   ✅ Estrategia 2: {len(results['strategy2']['trades'])} operaciones, ahorro mensual: ${results['strategy2']['monthly_savings']:,.2f}")
        print(f"      Total ahorrado: ${results['strategy2']['total_saved']:,.2f}, valor final: ${results['strategy2']['values'][-1]:,.2f}")
        
        # === BUY & HOLD ===
        results['buyhold'] = self.strategy_simulator.simulate_buyhold(data)
        print(f"   ✅ Buy & Hold: valor final: ${results['buyhold']['values'][-1]:,.2f}")
        
        # === DCA CORREGIDO ===
        results['dca'] = self.strategy_simulator.simulate_dca_corrected(data)
        print(f"   ✅ DCA: {len(results['dca']['investments'])} inversiones, valor final: ${results['dca']['values'][-1]:,.2f}")
        
        return results
    
    def calculate_enhanced_metrics(self, simulation_results: Dict[str, Any], 
                                 data: pd.DataFrame, period_years: float) -> Dict[str, Any]:
        """Calcula métricas mejoradas para todas las estrategias"""
        return self.metrics_calculator.calculate_all_metrics(simulation_results, period_years)
    
    def create_performance_chart(self, simulation_results: Dict[str, Any], data: pd.DataFrame, 
                               symbol: str, save_path: str = None) -> str:
        """Crea gráfico de rendimiento comparativo"""
        return self.chart_generator.create_performance_chart(simulation_results, data, symbol, save_path)
    
    def run_enhanced_backtest(self, data: pd.DataFrame, symbol: str = "ASSET") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Ejecuta backtesting mejorado completo"""
        
        print("🚀 Iniciando backtesting mejorado...")
        print(f"   📊 Configuración: {self.thresholds.get_strategy_name()}")
        print(f"   🔒 Look-ahead bias: {'Prevenido' if self.prevent_lookahead else 'NO prevenido'}")
        print(f"   💰 Tasa libre de riesgo: {self.risk_free_rate:.1%} anual")
        print(f"   🔧 DCA corregido: ✅ Ambas estrategias de ahorro inician desde cero")
        
        try:
            # Verificar datos mínimos
            if 'Close' not in data.columns:
                raise ValueError("Datos no contienen columna 'Close'")
            
            if len(data) < self.config.MIN_LOOKBACK_DAYS:
                raise ValueError(f"Datos insuficientes: {len(data)} días")
            
            # Generar señales mejoradas
            df_with_signals = self.generate_trading_signals_enhanced(data)
            
            # Simular estrategias mejoradas
            simulation_results = self.simulate_strategy_with_risk_free_rate(df_with_signals)
            
            # Agregar datos al DataFrame
            df_with_signals['Strategy1_Value'] = simulation_results['strategy1']['values']
            if 'strategy2' in simulation_results:
                df_with_signals['Strategy2_Value'] = simulation_results['strategy2']['values']
            df_with_signals['BuyHold_Value'] = simulation_results['buyhold']['values']
            df_with_signals['DCA_Value'] = simulation_results['dca']['values']
            
            # Calcular período
            period_days = len(df_with_signals)
            period_years = period_days / 252
            
            # Calcular métricas mejoradas
            metrics = self.calculate_enhanced_metrics(simulation_results, df_with_signals, period_years)
            
            # Crear gráfico
            chart_path = self.create_performance_chart(simulation_results, df_with_signals, symbol)
            
            # Estructura de resultados mejorada
            performance_metrics = {
                'period': {
                    'start_date': df_with_signals.index[0].strftime('%Y-%m-%d'),
                    'end_date': df_with_signals.index[-1].strftime('%Y-%m-%d'),
                    'days': period_days,
                    'years': period_years
                },
                'strategy1': metrics.get('strategy1', {}),
                'strategy2': metrics.get('strategy2', {}),
                'buyhold': metrics.get('buyhold', {}),
                'dca': metrics.get('dca', {}),
                'comparisons': metrics.get('comparisons', {}),
                'simulation_details': {
                    'strategy1_trades': simulation_results['strategy1']['trades'],
                    'strategy2_trades': simulation_results['strategy2']['trades'] if 'strategy2' in simulation_results else [],
                    'dca_investments': simulation_results['dca']['investments'],
                    'risk_free_rate': self.risk_free_rate,
                    'monthly_savings': simulation_results['strategy2']['monthly_savings'] if 'strategy2' in simulation_results else 0,
                    'chart_path': chart_path
                },
                'methodology': {
                    'lookahead_prevention': self.prevent_lookahead,
                    'expanding_window': self.expanding_window,
                    'strategy_type': self.thresholds.get_strategy_name(),
                    'risk_free_rate_included': True,
                    'dca_corrected': True
                }
            }
            
            # Mostrar resumen detallado
            self.print_enhanced_summary(performance_metrics)
            
            return df_with_signals, performance_metrics
            
        except Exception as e:
            print(f"❌ Error en backtesting mejorado: {str(e)}")
            raise RuntimeError(f"Error en backtesting: {str(e)}")
    
    def print_enhanced_summary(self, metrics: Dict[str, Any]):
        """Imprime resumen detallado del backtesting mejorado"""
        self.results_printer.print_enhanced_summary(metrics)