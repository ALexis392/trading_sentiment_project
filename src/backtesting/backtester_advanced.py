# backtesting/backtester_advanced.py
"""
Backtester avanzado con prevención de look-ahead bias y configuraciones personalizables
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


class AdvancedBacktester:
    """Backtester avanzado con configuraciones personalizables"""
    
    def __init__(self, config):
        self.config = config
        self.thresholds = config.backtesting_thresholds
        self.prevent_lookahead = config.PREVENT_LOOKAHEAD_BIAS
        self.expanding_window = config.EXPANDING_WINDOW
    
    def calculate_expanding_percentiles(self, series: pd.Series, min_lookback_days: int, 
                                      percentile: float, prevent_lookahead: bool = True) -> pd.Series:
        """
        Calcula percentiles expandidos con prevención de look-ahead bias
        
        Args:
            series: Serie temporal de datos
            min_lookback_days: Días mínimos antes de calcular percentiles
            percentile: Percentil a calcular (0.0 a 1.0)
            prevent_lookahead: Si prevenir look-ahead bias
        """
        if not prevent_lookahead:
            # Método tradicional (CON look-ahead bias)
            return series.quantile(percentile)
        
        # Método sin look-ahead bias (ventana expandida)
        expanding_percentiles = []
        
        for i in range(len(series)):
            # Usar solo datos históricos hasta la fecha actual (inclusive)
            historical_data = series.iloc[:i+1]
            
            if len(historical_data) >= min_lookback_days:
                expanding_percentiles.append(historical_data.quantile(percentile))
            else:
                # No suficientes datos históricos
                expanding_percentiles.append(np.nan)
        
        return pd.Series(expanding_percentiles, index=series.index)
    
    def generate_trading_signals_advanced(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Genera señales de trading usando configuración avanzada y sin look-ahead bias
        """
        print("📊 Generando señales de trading avanzadas...")
        print(f"   🔒 Prevención look-ahead bias: {'✅ Activada' if self.prevent_lookahead else '❌ Desactivada'}")
        print(f"   📊 Estrategia: {self.thresholds.get_strategy_name()}")
        
        df = data.copy()
        
        # Verificar columnas requeridas
        required_columns = ['VIX', 'RSI', 'BB_Position', 'Price_Change_20', 'Volatility_20']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️  Columnas faltantes: {missing_columns}")
            print("🔄 Usando señales básicas...")
            df['Signal'] = 1
            return df
        
        try:
            # Calcular percentiles con o sin look-ahead bias
            print("   📊 Calculando percentiles...")
            
            if self.prevent_lookahead:
                print("      🔒 Método: Ventana expandida (sin look-ahead bias)")
                # VIX percentiles
                df['VIX_Q_Fear'] = self.calculate_expanding_percentiles(
                    df['VIX'], self.config.MIN_LOOKBACK_DAYS, self.thresholds.vix_fear_threshold
                )
                df['VIX_Q_Euphoria'] = self.calculate_expanding_percentiles(
                    df['VIX'], self.config.MIN_LOOKBACK_DAYS, self.thresholds.vix_euphoria_threshold
                )
                df['VIX_Q_Moderate_Fear'] = self.calculate_expanding_percentiles(
                    df['VIX'], self.config.MIN_LOOKBACK_DAYS, self.thresholds.vix_moderate_fear
                )
                
                # RSI percentiles
                df['RSI_Q_Oversold'] = self.calculate_expanding_percentiles(
                    df['RSI'], self.config.MIN_LOOKBACK_DAYS, self.thresholds.rsi_oversold
                )
                df['RSI_Q_Overbought'] = self.calculate_expanding_percentiles(
                    df['RSI'], self.config.MIN_LOOKBACK_DAYS, self.thresholds.rsi_overbought
                )
                df['RSI_Q_Moderate_Oversold'] = self.calculate_expanding_percentiles(
                    df['RSI'], self.config.MIN_LOOKBACK_DAYS, self.thresholds.rsi_moderate_oversold
                )
                
                # Price change percentiles
                df['Price_Change_Q_Extreme'] = self.calculate_expanding_percentiles(
                    df['Price_Change_20'], self.config.MIN_LOOKBACK_DAYS, self.thresholds.price_change_extreme
                )
                
                # Volatility percentiles
                df['Volatility_Q_High'] = self.calculate_expanding_percentiles(
                    df['Volatility_20'], self.config.MIN_LOOKBACK_DAYS, self.thresholds.volatility_high
                )
                
            else:
                print("      ⚠️  Método: Percentiles fijos (CON look-ahead bias)")
                # Percentiles fijos (método tradicional con look-ahead bias)
                df['VIX_Q_Fear'] = df['VIX'].quantile(self.thresholds.vix_fear_threshold)
                df['VIX_Q_Euphoria'] = df['VIX'].quantile(self.thresholds.vix_euphoria_threshold)
                df['VIX_Q_Moderate_Fear'] = df['VIX'].quantile(self.thresholds.vix_moderate_fear)
                df['RSI_Q_Oversold'] = df['RSI'].quantile(self.thresholds.rsi_oversold)
                df['RSI_Q_Overbought'] = df['RSI'].quantile(self.thresholds.rsi_overbought)
                df['RSI_Q_Moderate_Oversold'] = df['RSI'].quantile(self.thresholds.rsi_moderate_oversold)
                df['Price_Change_Q_Extreme'] = df['Price_Change_20'].quantile(self.thresholds.price_change_extreme)
                df['Volatility_Q_High'] = df['Volatility_20'].quantile(self.thresholds.volatility_high)
            
            # Inicializar señales
            df['Signal'] = 1  # Mantener por defecto
            
            # === CONDICIONES DE COMPRA (MIEDO EXTREMO) ===
            print("   🔍 Evaluando condiciones de compra...")
            
            # Condición 1: VIX extremadamente alto
            buy_vix_extreme = df['VIX'] > df['VIX_Q_Fear']
            
            # Condición 2: RSI en sobreventa
            buy_rsi_oversold = df['RSI'] < df['RSI_Q_Oversold']
            
            # Condición 3: RSI moderadamente oversold + VIX moderadamente alto
            buy_moderate_combo = (
                (df['RSI'] < df['RSI_Q_Moderate_Oversold']) & 
                (df['VIX'] > df['VIX_Q_Moderate_Fear'])
            )
            
            # Condición 4: Precio cerca del límite inferior de Bollinger
            buy_bb_lower = df['BB_Position'] < self.thresholds.bb_lower_threshold
            
            # Condición 5: Alta volatilidad (oportunidad en pánico)
            buy_high_vol = df['Volatility_20'] > df['Volatility_Q_High']
            
            # Combinar condiciones de compra (OR lógico para flexibilidad)
            buy_conditions = (
                buy_vix_extreme | 
                buy_rsi_oversold | 
                buy_moderate_combo | 
                buy_bb_lower | 
                buy_high_vol
            )
            
            df.loc[buy_conditions, 'Signal'] = 0  # Comprar
            
            # === CONDICIONES DE VENTA (EUFORIA) ===
            print("   🔍 Evaluando condiciones de venta...")
            
            # Condición 1: VIX muy bajo (complacencia)
            sell_vix_low = df['VIX'] < df['VIX_Q_Euphoria']
            
            # Condición 2: RSI en sobrecompra
            sell_rsi_overbought = df['RSI'] > df['RSI_Q_Overbought']
            
            # Condición 3: Precio cerca del límite superior de Bollinger
            sell_bb_upper = df['BB_Position'] > self.thresholds.bb_upper_threshold
            
            # Condición 4: Cambio de precio extremo (momentum insostenible)
            sell_price_extreme = df['Price_Change_20'] > df['Price_Change_Q_Extreme']
            
            # Para venta, usar AND para ser más selectivo (evitar ventas prematuras)
            sell_conditions = (
                sell_vix_low & 
                sell_rsi_overbought & 
                sell_bb_upper & 
                sell_price_extreme
            )
            
            df.loc[sell_conditions, 'Signal'] = 2  # Vender
            
            # Estadísticas de señales
            signal_counts = df['Signal'].value_counts()
            total_days = len(df)
            
            print(f"   ✅ Señales generadas:")
            print(f"      🟢 Comprar: {signal_counts.get(0, 0)} días ({signal_counts.get(0, 0)/total_days*100:.1f}%)")
            print(f"      🟡 Mantener: {signal_counts.get(1, 0)} días ({signal_counts.get(1, 0)/total_days*100:.1f}%)")
            print(f"      🔴 Vender: {signal_counts.get(2, 0)} días ({signal_counts.get(2, 0)/total_days*100:.1f}%)")
            
            # Guardar información de umbrales utilizados
            df.attrs['thresholds_used'] = {
                'strategy_name': self.thresholds.get_strategy_name(),
                'vix_fear_threshold': self.thresholds.vix_fear_threshold,
                'vix_euphoria_threshold': self.thresholds.vix_euphoria_threshold,
                'rsi_oversold': self.thresholds.rsi_oversold,
                'rsi_overbought': self.thresholds.rsi_overbought,
                'bb_lower_threshold': self.thresholds.bb_lower_threshold,
                'bb_upper_threshold': self.thresholds.bb_upper_threshold,
                'prevent_lookahead': self.prevent_lookahead,
                'expanding_window': self.expanding_window
            }
            
            return df
            
        except Exception as e:
            print(f"⚠️  Error generando señales: {str(e)}")
            print("🔄 Usando señales de mantener por defecto...")
            df['Signal'] = 1
            return df
    
    def simulate_strategy_advanced(self, data: pd.DataFrame) -> Tuple[List[float], List[Dict], List[float], List[Dict]]:
        """
        Simula estrategias con configuraciones avanzadas
        """
        print("🎮 Simulando estrategias con configuración avanzada...")
        
        # Verificar datos
        if 'Close' not in data.columns or 'Signal' not in data.columns:
            raise ValueError("Datos insuficientes para simulación")
        
        # === ESTRATEGIA ACTIVA ===
        print("   📈 Simulando estrategia activa...")
        
        portfolio_value = 0
        cash = self.config.INITIAL_CAPITAL
        shares = 0
        trades = []
        strategy_values = []
        
        days_with_valid_signals = 0
        
        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            signal = data['Signal'].iloc[i]
            date = data.index[i]
            
            # Aplicar período mínimo de lookback para prevenir uso de datos insuficientes
            if i < self.config.MIN_LOOKBACK_DAYS:
                portfolio_total = cash + (shares * current_price)
                strategy_values.append(portfolio_total)
                continue
            
            days_with_valid_signals += 1
            
            # Ejecutar operaciones según señales
            if not pd.isna(signal):
                if signal == 0 and shares == 0 and cash > 0:  # Comprar
                    # Calcular valor después de costos de transacción
                    transaction_value = cash * (1 - self.config.TRANSACTION_COST)
                    shares = transaction_value / current_price
                    cash = 0
                    
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares,
                        'value': transaction_value,
                        'transaction_cost': cash * self.config.TRANSACTION_COST,
                        'day_index': i
                    })
                
                elif signal == 2 and shares > 0:  # Vender
                    # Calcular valor después de costos de transacción
                    gross_value = shares * current_price
                    transaction_value = gross_value * (1 - self.config.TRANSACTION_COST)
                    cash = transaction_value
                    
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': shares,
                        'value': transaction_value,
                        'gross_value': gross_value,
                        'transaction_cost': gross_value * self.config.TRANSACTION_COST,
                        'day_index': i
                    })
                    shares = 0
            
            # Calcular valor total del portfolio
            portfolio_total = cash + (shares * current_price)
            strategy_values.append(portfolio_total)
        
        print(f"   ✅ Estrategia activa simulada:")
        print(f"      📊 {len(trades)} operaciones ejecutadas")
        print(f"      ⏰ {days_with_valid_signals} días con señales válidas")
        print(f"      💰 Valor final: ${strategy_values[-1]:,.2f}")
        
        # === DCA (DOLLAR COST AVERAGING) ===
        print("   💰 Simulando Dollar Cost Averaging...")
        
        # Calcular inversión por período
        total_periods = len(data) // self.config.DCA_FREQUENCY + 1
        dca_investment_per_period = self.config.INITIAL_CAPITAL / total_periods
        
        dca_cash = self.config.INITIAL_CAPITAL
        dca_shares = 0
        dca_values = []
        dca_investments = []
        
        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            date = data.index[i]
            
            # Invertir cada DCA_FREQUENCY días
            if i % self.config.DCA_FREQUENCY == 0 and dca_cash >= dca_investment_per_period:
                investment_amount = min(dca_investment_per_period, dca_cash)
                shares_bought = investment_amount / current_price
                dca_shares += shares_bought
                dca_cash -= investment_amount
                
                dca_investments.append({
                    'date': date,
                    'price': current_price,
                    'amount': investment_amount,
                    'shares': shares_bought,
                    'day_index': i
                })
            
            dca_total_value = dca_cash + (dca_shares * current_price)
            dca_values.append(dca_total_value)
        
        print(f"   ✅ DCA simulado:")
        print(f"      💰 {len(dca_investments)} inversiones realizadas")
        print(f"      📊 Inversión promedio: ${dca_investment_per_period:,.2f}")
        print(f"      💰 Valor final: ${dca_values[-1]:,.2f}")
        
        return strategy_values, trades, dca_values, dca_investments
    
    def calculate_advanced_metrics(self, strategy_values: List[float], buyhold_values: pd.Series, 
                                 dca_values: List[float], trades: List[Dict], 
                                 period_years: float) -> Dict[str, Any]:
        """Calcula métricas avanzadas con análisis detallado"""
        
        print("📊 Calculando métricas avanzadas...")
        
        def safe_calculate_max_drawdown(values):
            """Calcula max drawdown de forma segura"""
            try:
                values_series = pd.Series(values) if not isinstance(values, pd.Series) else values
                if len(values_series) == 0:
                    return 0.0
                
                # Calcular drawdown
                peak = values_series.expanding().max()
                drawdown = (values_series - peak) / peak
                max_dd = float(drawdown.min() * 100)
                
                # Encontrar fechas del máximo drawdown
                max_dd_idx = drawdown.idxmin()
                peak_idx = peak[:max_dd_idx].idxmax()
                
                return {
                    'max_drawdown_pct': max_dd,
                    'peak_date': peak_idx if hasattr(peak_idx, 'strftime') else None,
                    'trough_date': max_dd_idx if hasattr(max_dd_idx, 'strftime') else None,
                    'peak_value': float(peak.loc[peak_idx]) if peak_idx is not None else None,
                    'trough_value': float(values_series.loc[max_dd_idx]) if max_dd_idx is not None else None
                }
            except:
                return {'max_drawdown_pct': 0.0}
        
        def safe_calculate_sharpe_ratio(returns, risk_free_rate):
            """Calcula Sharpe ratio con análisis detallado"""
            try:
                if len(returns) == 0 or returns.std() == 0:
                    return {'sharpe_ratio': 0.0, 'avg_return': 0.0, 'volatility': 0.0}
                
                daily_rf_rate = risk_free_rate / 252
                excess_returns = returns - daily_rf_rate
                avg_excess_return = excess_returns.mean()
                volatility = returns.std()
                
                sharpe = float(avg_excess_return / volatility * np.sqrt(252))
                
                return {
                    'sharpe_ratio': sharpe,
                    'avg_return': float(avg_excess_return * 252 * 100),
                    'volatility': float(volatility * np.sqrt(252) * 100),
                    'excess_return': float(avg_excess_return * 252 * 100)
                }
            except:
                return {'sharpe_ratio': 0.0, 'avg_return': 0.0, 'volatility': 0.0}
        
        # Verificar y limpiar datos
        if not strategy_values or len(strategy_values) == 0:
            strategy_values = [self.config.INITIAL_CAPITAL] * len(buyhold_values)
        
        if not dca_values or len(dca_values) == 0:
            dca_values = [self.config.INITIAL_CAPITAL] * len(buyhold_values)
        
        # === RETORNOS TOTALES Y ANUALIZADOS ===
        strategy_total_return = ((strategy_values[-1] - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL * 100)
        buyhold_total_return = ((buyhold_values.iloc[-1] - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL * 100)
        dca_total_return = ((dca_values[-1] - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL * 100)
        
        # Anualizados
        strategy_annual = (strategy_total_return / period_years) if period_years > 0 else strategy_total_return
        buyhold_annual = (buyhold_total_return / period_years) if period_years > 0 else buyhold_total_return
        dca_annual = (dca_total_return / period_years) if period_years > 0 else dca_total_return
        
        # === ANÁLISIS DE RETORNOS DIARIOS ===
        strategy_daily_returns = pd.Series(strategy_values).pct_change().dropna()
        buyhold_daily_returns = buyhold_values.pct_change().dropna()
        dca_daily_returns = pd.Series(dca_values).pct_change().dropna()
        
        # === MÉTRICAS DE RIESGO ===
        strategy_sharpe_info = safe_calculate_sharpe_ratio(strategy_daily_returns, self.config.RISK_FREE_RATE)
        buyhold_sharpe_info = safe_calculate_sharpe_ratio(buyhold_daily_returns, self.config.RISK_FREE_RATE)
        dca_sharpe_info = safe_calculate_sharpe_ratio(dca_daily_returns, self.config.RISK_FREE_RATE)
        
        # Max drawdowns detallados
        strategy_dd_info = safe_calculate_max_drawdown(strategy_values)
        buyhold_dd_info = safe_calculate_max_drawdown(buyhold_values)
        dca_dd_info = safe_calculate_max_drawdown(dca_values)
        
        # === ANÁLISIS DE TRADING ===
        trading_analysis = self.analyze_trades(trades)
        
        # === ANÁLISIS DE CONSISTENCIA ===
        consistency_analysis = self.analyze_consistency(strategy_daily_returns, buyhold_daily_returns)
        
        print(f"   ✅ Métricas calculadas:")
        print(f"      📊 Estrategia: {strategy_annual:+.1f}% anual")
        print(f"      📈 Buy & Hold: {buyhold_annual:+.1f}% anual")
        print(f"      💰 DCA: {dca_annual:+.1f}% anual")
        print(f"      🎯 Outperformance: {strategy_annual - buyhold_annual:+.1f}%")
        
        return {
            # Retornos básicos
            'strategy_return': float(strategy_total_return),
            'buyhold_return': float(buyhold_total_return),
            'dca_return': float(dca_total_return),
            'strategy_annual': float(strategy_annual),
            'buyhold_annual': float(buyhold_annual),
            'dca_annual': float(dca_annual),
            
            # Métricas de riesgo detalladas
            'strategy_sharpe_info': strategy_sharpe_info,
            'buyhold_sharpe_info': buyhold_sharpe_info,
            'dca_sharpe_info': dca_sharpe_info,
            
            # Drawdown detallado
            'strategy_dd_info': strategy_dd_info,
            'buyhold_dd_info': buyhold_dd_info,
            'dca_dd_info': dca_dd_info,
            
            # Análisis de trading
            'trading_analysis': trading_analysis,
            
            # Análisis de consistencia
            'consistency_analysis': consistency_analysis,
            
            # Compatibilidad con versión anterior
            'strategy_sharpe': strategy_sharpe_info['sharpe_ratio'],
            'buyhold_sharpe': buyhold_sharpe_info['sharpe_ratio'],
            'dca_sharpe': dca_sharpe_info['sharpe_ratio'],
            'strategy_volatility': strategy_sharpe_info['volatility'],
            'buyhold_volatility': buyhold_sharpe_info['volatility'],
            'dca_volatility': dca_sharpe_info['volatility'],
            'strategy_max_dd': strategy_dd_info['max_drawdown_pct'],
            'buyhold_max_dd': buyhold_dd_info['max_drawdown_pct'],
            'dca_max_dd': dca_dd_info['max_drawdown_pct'],
            'num_trades': trading_analysis['num_trades'],
            'win_rate': trading_analysis['win_rate'],
            'time_in_market': trading_analysis['time_in_market']
        }
    
    def analyze_trades(self, trades: List[Dict]) -> Dict[str, Any]:
        """Analiza las operaciones de trading en detalle"""
        
        if not trades:
            return {
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_trade_return': 0.0,
                'best_trade': None,
                'worst_trade': None,
                'time_in_market': 0.0,
                'avg_holding_period': 0,
                'profitable_trades': 0,
                'losing_trades': 0
            }
        
        # Analizar pares de operaciones (compra-venta)
        trade_returns = []
        holding_periods = []
        profitable_trades = 0
        losing_trades = 0
        
        for i in range(0, len(trades) - 1, 2):
            if i + 1 < len(trades):
                buy_trade = trades[i]
                sell_trade = trades[i + 1]
                
                if buy_trade['action'] == 'BUY' and sell_trade['action'] == 'SELL':
                    # Calcular retorno de la operación
                    trade_return = (sell_trade['value'] - buy_trade['value']) / buy_trade['value'] * 100
                    trade_returns.append(trade_return)
                    
                    # Período de tenencia
                    holding_period = sell_trade['day_index'] - buy_trade['day_index']
                    holding_periods.append(holding_period)
                    
                    if trade_return > 0:
                        profitable_trades += 1
                    else:
                        losing_trades += 1
        
        # Calcular métricas
        num_complete_trades = len(trade_returns)
        win_rate = (profitable_trades / num_complete_trades * 100) if num_complete_trades > 0 else 0
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        # Mejor y peor operación
        best_trade = max(trade_returns) if trade_returns else None
        worst_trade = min(trade_returns) if trade_returns else None
        
        # Tiempo en mercado (aproximado)
        total_holding_days = sum(holding_periods)
        # Asumir que el período total son 252 * años
        time_in_market = min(100.0, total_holding_days / 252 * 100) if holding_periods else 0
        
        return {
            'num_trades': len(trades),
            'num_complete_trades': num_complete_trades,
            'win_rate': float(win_rate),
            'avg_trade_return': float(avg_trade_return),
            'best_trade': float(best_trade) if best_trade is not None else None,
            'worst_trade': float(worst_trade) if worst_trade is not None else None,
            'time_in_market': float(time_in_market),
            'avg_holding_period': float(avg_holding_period),
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'trade_returns': trade_returns
        }
    
    def analyze_consistency(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, Any]:
        """Analiza la consistencia de la estrategia vs benchmark"""
        
        try:
            # Períodos positivos
            strategy_positive_periods = (strategy_returns > 0).sum() / len(strategy_returns) * 100
            benchmark_positive_periods = (benchmark_returns > 0).sum() / len(benchmark_returns) * 100
            
            # Correlación
            correlation = strategy_returns.corr(benchmark_returns)
            
            # Beta (sensibilidad al mercado)
            covariance = strategy_returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 1.0
            
            # Períodos de outperformance
            outperformance = strategy_returns - benchmark_returns
            outperformance_periods = (outperformance > 0).sum() / len(outperformance) * 100
            
            # Tracking error
            tracking_error = outperformance.std() * np.sqrt(252) * 100
            
            return {
                'strategy_positive_periods': float(strategy_positive_periods),
                'benchmark_positive_periods': float(benchmark_positive_periods),
                'correlation': float(correlation) if not pd.isna(correlation) else 0.0,
                'beta': float(beta),
                'outperformance_periods': float(outperformance_periods),
                'tracking_error': float(tracking_error),
                'avg_outperformance': float(outperformance.mean() * 252 * 100)
            }
            
        except Exception as e:
            return {
                'strategy_positive_periods': 0.0,
                'benchmark_positive_periods': 0.0,
                'correlation': 0.0,
                'beta': 1.0,
                'outperformance_periods': 0.0,
                'tracking_error': 0.0,
                'avg_outperformance': 0.0,
                'error': str(e)
            }
    
    def run_backtest(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Ejecuta backtesting avanzado completo
        """
        print("🚀 Iniciando backtesting avanzado...")
        print(f"   📊 Configuración: {self.thresholds.get_strategy_name()}")
        print(f"   🔒 Look-ahead bias: {'Prevenido' if self.prevent_lookahead else 'NO prevenido'}")
        
        try:
            # Verificar datos mínimos
            if 'Close' not in data.columns:
                raise ValueError("Datos no contienen columna 'Close'")
            
            if len(data) < self.config.MIN_LOOKBACK_DAYS:
                raise ValueError(f"Datos insuficientes: {len(data)} días (mínimo: {self.config.MIN_LOOKBACK_DAYS})")
            
            # Generar señales avanzadas
            df_with_signals = self.generate_trading_signals_advanced(data)
            
            # Simular estrategias
            strategy_values, trades, dca_values, dca_investments = self.simulate_strategy_advanced(df_with_signals)
            
            # === CALCULAR BUY & HOLD ===
            print("   📊 Calculando Buy & Hold...")
            try:
                initial_price = df_with_signals['Close'].iloc[0]
                initial_shares = self.config.INITIAL_CAPITAL / initial_price
                buyhold_values = initial_shares * df_with_signals['Close']
                
                print(f"   ✅ Buy & Hold:")
                print(f"      💰 Inversión inicial: ${self.config.INITIAL_CAPITAL:,.2f}")
                print(f"      📊 Precio inicial: ${initial_price:.2f}")
                print(f"      🔢 Acciones: {initial_shares:.2f}")
                print(f"      💰 Valor final: ${buyhold_values.iloc[-1]:,.2f}")
                
            except Exception as e:
                print(f"   ⚠️  Error en Buy & Hold: {str(e)}")
                buyhold_values = pd.Series([self.config.INITIAL_CAPITAL] * len(df_with_signals), 
                                         index=df_with_signals.index)
            
            # Agregar valores al DataFrame
            df_with_signals['Strategy_Value'] = strategy_values
            df_with_signals['BuyHold_Value'] = buyhold_values.values
            df_with_signals['DCA_Value'] = dca_values
            
            # Calcular período
            period_days = len(df_with_signals)
            period_years = period_days / 252
            
            print(f"\n📅 Información del período:")
            print(f"   📊 Inicio: {df_with_signals.index[0].strftime('%Y-%m-%d')}")
            print(f"   📊 Fin: {df_with_signals.index[-1].strftime('%Y-%m-%d')}")
            print(f"   📊 Duración: {period_days} días ({period_years:.2f} años)")
            
            # Calcular métricas avanzadas
            metrics = self.calculate_advanced_metrics(strategy_values, buyhold_values, dca_values, trades, period_years)
            
            # Construir estructura de performance_metrics extendida
            performance_metrics = {
                'period': {
                    'start_date': df_with_signals.index[0].strftime('%Y-%m-%d'),
                    'end_date': df_with_signals.index[-1].strftime('%Y-%m-%d'),
                    'days': period_days,
                    'years': period_years
                },
                'strategy': {
                    'final_value': strategy_values[-1] if strategy_values else self.config.INITIAL_CAPITAL,
                    'total_return': metrics['strategy_return'],
                    'annual_return': metrics['strategy_annual'],
                    'volatility': metrics['strategy_volatility'],
                    'sharpe_ratio': metrics['strategy_sharpe'],
                    'max_drawdown': metrics['strategy_max_dd'],
                    'num_trades': metrics['num_trades'],
                    'win_rate': metrics['win_rate'],
                    'time_in_market': metrics['time_in_market'],
                    # Métricas avanzadas
                    'sharpe_info': metrics['strategy_sharpe_info'],
                    'drawdown_info': metrics['strategy_dd_info']
                },
                'buyhold': {
                    'final_value': float(buyhold_values.iloc[-1]),
                    'total_return': metrics['buyhold_return'],
                    'annual_return': metrics['buyhold_annual'],
                    'volatility': metrics['buyhold_volatility'],
                    'sharpe_ratio': metrics['buyhold_sharpe'],
                    'max_drawdown': metrics['buyhold_max_dd'],
                    # Métricas avanzadas
                    'sharpe_info': metrics['buyhold_sharpe_info'],
                    'drawdown_info': metrics['buyhold_dd_info']
                },
                'dca': {
                    'final_value': dca_values[-1] if dca_values else self.config.INITIAL_CAPITAL,
                    'total_return': metrics['dca_return'],
                    'annual_return': metrics['dca_annual'],
                    'volatility': metrics['dca_volatility'],
                    'sharpe_ratio': metrics['dca_sharpe'],
                    'max_drawdown': metrics['dca_max_dd'],
                    'num_investments': len(dca_investments),
                    # Métricas avanzadas
                    'sharpe_info': metrics['dca_sharpe_info'],
                    'drawdown_info': metrics['dca_dd_info']
                },
                'comparison': {
                    'outperformance_vs_buyhold': metrics['strategy_return'] - metrics['buyhold_return'],
                    'outperformance_vs_dca': metrics['strategy_return'] - metrics['dca_return'],
                    'risk_adjusted_return_vs_buyhold': metrics['strategy_sharpe'] - metrics['buyhold_sharpe'],
                    'risk_adjusted_return_vs_dca': metrics['strategy_sharpe'] - metrics['dca_sharpe']
                },
                'advanced_analysis': {
                    'trading_analysis': metrics['trading_analysis'],
                    'consistency_analysis': metrics['consistency_analysis'],
                    'thresholds_used': df_with_signals.attrs.get('thresholds_used', {}),
                    'methodology': {
                        'lookahead_prevention': self.prevent_lookahead,
                        'expanding_window': self.expanding_window,
                        'strategy_type': self.thresholds.get_strategy_name()
                    }
                },
                'trades': trades,
                'dca_investments': dca_investments
            }
            
            # Mostrar resumen final
            self.print_backtest_summary(performance_metrics)
            
            return df_with_signals, performance_metrics
            
        except Exception as e:
            print(f"❌ Error en backtesting avanzado: {str(e)}")
            raise RuntimeError(f"Error en backtesting: {str(e)}")
    
    def print_backtest_summary(self, metrics: Dict[str, Any]):
        """Imprime resumen detallado del backtesting"""
        
        print(f"\n📊 RESUMEN DETALLADO DEL BACKTESTING:")
        print("=" * 80)
        
        # Rendimientos
        print(f"💰 RENDIMIENTOS ({metrics['period']['years']:.1f} años):")
        print(f"   🎯 Estrategia Activa: {metrics['strategy']['annual_return']:+.1f}% anual")
        print(f"   📈 Buy & Hold: {metrics['buyhold']['annual_return']:+.1f}% anual")
        print(f"   💰 DCA: {metrics['dca']['annual_return']:+.1f}% anual")
        print(f"   🏆 Outperformance vs B&H: {metrics['comparison']['outperformance_vs_buyhold']:+.1f}%")
        print(f"   🏆 Outperformance vs DCA: {metrics['comparison']['outperformance_vs_dca']:+.1f}%")
        
        # Riesgo
        print(f"\n📊 MÉTRICAS DE RIESGO:")
        print(f"   📉 Max Drawdown Estrategia: {metrics['strategy']['max_drawdown']:.1f}%")
        print(f"   📉 Max Drawdown B&H: {metrics['buyhold']['max_drawdown']:.1f}%")
        print(f"   🎯 Sharpe Ratio Estrategia: {metrics['strategy']['sharpe_ratio']:.2f}")
        print(f"   🎯 Sharpe Ratio B&H: {metrics['buyhold']['sharpe_ratio']:.2f}")
        print(f"   📊 Volatilidad Estrategia: {metrics['strategy']['volatility']:.1f}%")
        print(f"   📊 Volatilidad B&H: {metrics['buyhold']['volatility']:.1f}%")
        
        # Trading
        trading = metrics['advanced_analysis']['trading_analysis']
        print(f"\n⚡ ANÁLISIS DE TRADING:")
        print(f"   🔢 Total operaciones: {trading['num_trades']}")
        print(f"   ✅ Operaciones ganadoras: {trading['profitable_trades']}")
        print(f"   ❌ Operaciones perdedoras: {trading['losing_trades']}")
        print(f"   🎯 Tasa de éxito: {trading['win_rate']:.1f}%")
        print(f"   ⏰ Tiempo en mercado: {trading['time_in_market']:.1f}%")
        print(f"   📅 Período promedio: {trading['avg_holding_period']:.0f} días")
        if trading['best_trade']:
            print(f"   🏆 Mejor operación: {trading['best_trade']:+.1f}%")
        if trading['worst_trade']:
            print(f"   💸 Peor operación: {trading['worst_trade']:+.1f}%")
        
        # Metodología
        methodology = metrics['advanced_analysis']['methodology']
        print(f"\n⚙️  METODOLOGÍA:")
        print(f"   🔒 Look-ahead bias: {'✅ Prevenido' if methodology['lookahead_prevention'] else '❌ NO prevenido'}")
        print(f"   📊 Tipo de estrategia: {methodology['strategy_type']}")
        print(f"   📈 Ventana expandida: {'✅ Sí' if methodology['expanding_window'] else '❌ No'}")
        
        # Umbrales utilizados
        thresholds = metrics['advanced_analysis']['thresholds_used']
        if thresholds:
            print(f"\n🎯 UMBRALES UTILIZADOS:")
            print(f"   📊 VIX Miedo: >{thresholds['vix_fear_threshold']:.0%} percentil")
            print(f"   📊 VIX Euforia: <{thresholds['vix_euphoria_threshold']:.0%} percentil")
            print(f"   📈 RSI Sobreventa: <{thresholds['rsi_oversold']:.0%} percentil")
            print(f"   📈 RSI Sobrecompra: >{thresholds['rsi_overbought']:.0%} percentil")
            print(f"   📉 BB Límite Inf: <{thresholds['bb_lower_threshold']:.0%}")
            print(f"   📉 BB Límite Sup: >{thresholds['bb_upper_threshold']:.0%}")