# backtester.py - Versión Corregida
"""
Módulo para backtesting de estrategias con cálculos corregidos
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


class Backtester:
    """Clase para realizar backtesting de estrategias"""
    
    def __init__(self, config):
        self.config = config
    
    def calculate_expanding_percentiles(self, series: pd.Series, min_lookback_days: int, percentile: float) -> pd.Series:
        """
        Calcula percentiles expandidos usando toda la información histórica hasta cada fecha
        """
        expanding_percentiles = []
        for i in range(len(series)):
            historical_data = series.iloc[:i+1]
            
            if len(historical_data) >= min_lookback_days:
                expanding_percentiles.append(historical_data.quantile(percentile))
            else:
                expanding_percentiles.append(np.nan)
        
        return pd.Series(expanding_percentiles, index=series.index)
    
    def generate_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Genera señales de trading basadas en percentiles expandidos
        """
        print("📊 Generando señales de trading...")
        df = data.copy()
        
        # Verificar que tenemos las columnas necesarias
        required_columns = ['VIX', 'RSI', 'BB_Position', 'Price_Change_20']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️  Columnas faltantes: {missing_columns}")
            print("🔄 Usando señales básicas...")
            df['Signal'] = 1  # Todo mantener
            return df
        
        try:
            # Calcular percentiles expandidos
            print("   Calculando percentiles expandidos...")
            df['VIX_Q90'] = self.calculate_expanding_percentiles(df['VIX'], self.config.MIN_LOOKBACK_DAYS, 0.9)
            df['VIX_Q95'] = self.calculate_expanding_percentiles(df['VIX'], self.config.MIN_LOOKBACK_DAYS, 0.95)
            df['VIX_Q5'] = self.calculate_expanding_percentiles(df['VIX'], self.config.MIN_LOOKBACK_DAYS, 0.05)
            df['RSI_Q5'] = self.calculate_expanding_percentiles(df['RSI'], self.config.MIN_LOOKBACK_DAYS, 0.05)
            df['RSI_Q10'] = self.calculate_expanding_percentiles(df['RSI'], self.config.MIN_LOOKBACK_DAYS, 0.1)
            df['RSI_Q95'] = self.calculate_expanding_percentiles(df['RSI'], self.config.MIN_LOOKBACK_DAYS, 0.95)
            df['Price_Change_20_Q95'] = self.calculate_expanding_percentiles(df['Price_Change_20'], self.config.MIN_LOOKBACK_DAYS, 0.95)
            
            # Señales por defecto
            df['Signal'] = 1  # Mantener
            
            # Señales de compra (Miedo Extremo)
            buy_conditions = (
                (df['VIX'] > df['VIX_Q95']) |
                (df['RSI'] < df['RSI_Q5']) |
                (df['BB_Position'] < 0.05) |
                ((df['VIX'] > df['VIX_Q90']) & (df['RSI'] < df['RSI_Q10']))
            )
            df.loc[buy_conditions, 'Signal'] = 0  # Comprar
            
            # Señales de venta (Euforia)
            sell_conditions = (
                (df['VIX'] < df['VIX_Q5']) &
                (df['RSI'] > df['RSI_Q95']) &
                (df['BB_Position'] > 0.95) &
                (df['Price_Change_20'] > df['Price_Change_20_Q95'])
            )
            df.loc[sell_conditions, 'Signal'] = 2  # Vender
            
            # Contar señales generadas
            signal_counts = df['Signal'].value_counts()
            print(f"   ✅ Señales generadas:")
            print(f"      Comprar: {signal_counts.get(0, 0)} días")
            print(f"      Mantener: {signal_counts.get(1, 0)} días") 
            print(f"      Vender: {signal_counts.get(2, 0)} días")
            
            return df
            
        except Exception as e:
            print(f"⚠️  Error generando señales: {str(e)}")
            print("🔄 Usando señales de mantener por defecto...")
            df['Signal'] = 1
            return df
    
    def simulate_strategy(self, data: pd.DataFrame) -> Tuple[List[float], List[Dict], List[float], List[Dict]]:
        """
        Simula la estrategia de trading y estrategias de benchmark
        """
        print("🎮 Simulando estrategias...")
        
        # Verificar que tenemos datos válidos
        if 'Close' not in data.columns or 'Signal' not in data.columns:
            raise ValueError("Datos insuficientes para simulación")
        
        # === SIMULACIÓN DE ESTRATEGIA ACTIVA ===
        portfolio_value = 0
        cash = self.config.INITIAL_CAPITAL
        shares = 0
        trades = []
        strategy_values = []
        
        print("   📈 Simulando estrategia activa...")
        
        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            signal = data['Signal'].iloc[i]
            date = data.index[i]
            
            # Skip si no tenemos suficiente información histórica
            if i < self.config.MIN_LOOKBACK_DAYS:
                portfolio_total = cash + (shares * current_price)
                strategy_values.append(portfolio_total)
                continue
            
            # Ejecutar operaciones de estrategia
            if not pd.isna(signal):
                if signal == 0 and shares == 0 and cash > 0:  # Comprar
                    transaction_value = cash * (1 - self.config.TRANSACTION_COST)
                    shares = transaction_value / current_price
                    cash = 0
                    trades.append({
                        'date': date, 'action': 'BUY', 'price': current_price,
                        'shares': shares, 'value': transaction_value
                    })
                
                elif signal == 2 and shares > 0:  # Vender
                    transaction_value = shares * current_price * (1 - self.config.TRANSACTION_COST)
                    cash = transaction_value
                    trades.append({
                        'date': date, 'action': 'SELL', 'price': current_price,
                        'shares': shares, 'value': transaction_value
                    })
                    shares = 0
            
            # Calcular valor total del portfolio
            portfolio_total = cash + (shares * current_price)
            strategy_values.append(portfolio_total)
        
        print(f"   ✅ Estrategia activa: {len(trades)} operaciones ejecutadas")
        
        # === SIMULACIÓN DCA ===
        print("   💰 Simulando Dollar Cost Averaging...")
        
        dca_investment_per_period = self.config.INITIAL_CAPITAL / (len(data) // self.config.DCA_FREQUENCY + 1)
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
                    'shares': shares_bought
                })
            
            dca_total_value = dca_cash + (dca_shares * current_price)
            dca_values.append(dca_total_value)
        
        print(f"   ✅ DCA: {len(dca_investments)} inversiones realizadas")
        
        return strategy_values, trades, dca_values, dca_investments
    
    def calculate_metrics(self, strategy_values: List[float], buyhold_values: pd.Series, 
                         dca_values: List[float], trades: List[Dict], period_years: float) -> Dict[str, Any]:
        """Calcula métricas de rendimiento"""
        
        print("📊 Calculando métricas de rendimiento...")
        
        def safe_calculate_max_drawdown(values):
            """Calcula max drawdown de forma segura"""
            try:
                values_series = pd.Series(values) if not isinstance(values, pd.Series) else values
                if len(values_series) == 0:
                    return 0.0
                
                peak = values_series.expanding().max()
                drawdown = (values_series - peak) / peak
                return float(drawdown.min() * 100)
            except:
                return 0.0
        
        def safe_calculate_sharpe_ratio(returns, risk_free_rate):
            """Calcula Sharpe ratio de forma segura"""
            try:
                if len(returns) == 0 or returns.std() == 0:
                    return 0.0
                excess_return = returns.mean() - risk_free_rate/252
                return float(excess_return / returns.std() * np.sqrt(252))
            except:
                return 0.0
        
        def safe_calculate_volatility(returns):
            """Calcula volatilidad anualizada de forma segura"""
            try:
                if len(returns) == 0:
                    return 0.0
                return float(returns.std() * np.sqrt(252) * 100)
            except:
                return 0.0
        
        # Verificar que tenemos valores válidos
        if not strategy_values or len(strategy_values) == 0:
            strategy_values = [self.config.INITIAL_CAPITAL] * len(buyhold_values)
        
        if not dca_values or len(dca_values) == 0:
            dca_values = [self.config.INITIAL_CAPITAL] * len(buyhold_values)
        
        # Calcular retornos totales
        strategy_return = ((strategy_values[-1] - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL * 100) if strategy_values else 0.0
        buyhold_return = ((buyhold_values.iloc[-1] - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL * 100) if len(buyhold_values) > 0 else 0.0
        dca_return = ((dca_values[-1] - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL * 100) if dca_values else 0.0
        
        # Calcular retornos anualizados
        strategy_annual = (strategy_return / period_years) if period_years > 0 else strategy_return
        buyhold_annual = (buyhold_return / period_years) if period_years > 0 else buyhold_return  
        dca_annual = (dca_return / period_years) if period_years > 0 else dca_return
        
        # Calcular retornos diarios para volatilidad y Sharpe
        strategy_daily_returns = pd.Series(strategy_values).pct_change().dropna()
        buyhold_daily_returns = buyhold_values.pct_change().dropna()
        dca_daily_returns = pd.Series(dca_values).pct_change().dropna()
        
        # Calcular métricas de riesgo
        strategy_volatility = safe_calculate_volatility(strategy_daily_returns)
        buyhold_volatility = safe_calculate_volatility(buyhold_daily_returns)
        dca_volatility = safe_calculate_volatility(dca_daily_returns)
        
        # Sharpe ratios
        strategy_sharpe = safe_calculate_sharpe_ratio(strategy_daily_returns, self.config.RISK_FREE_RATE)
        buyhold_sharpe = safe_calculate_sharpe_ratio(buyhold_daily_returns, self.config.RISK_FREE_RATE)
        dca_sharpe = safe_calculate_sharpe_ratio(dca_daily_returns, self.config.RISK_FREE_RATE)
        
        # Max drawdowns
        strategy_max_dd = safe_calculate_max_drawdown(strategy_values)
        buyhold_max_dd = safe_calculate_max_drawdown(buyhold_values)
        dca_max_dd = safe_calculate_max_drawdown(dca_values)
        
        # Estadísticas de trading
        num_trades = len(trades)
        if num_trades >= 2:
            # Calcular trades ganadores (comparar ventas con compras)
            winning_trades = 0
            for i in range(1, len(trades), 2):
                if i < len(trades) and trades[i]['action'] == 'SELL' and trades[i-1]['action'] == 'BUY':
                    if trades[i]['value'] > trades[i-1]['value']:
                        winning_trades += 1
            win_rate = (winning_trades / (num_trades // 2) * 100) if num_trades > 1 else 0
        else:
            win_rate = 0.0
        
        # Tiempo en mercado (aproximado)
        time_in_market = 100.0  # Simplificado por ahora
        
        print(f"   ✅ Métricas calculadas exitosamente")
        
        return {
            'strategy_return': float(strategy_return),
            'buyhold_return': float(buyhold_return),
            'dca_return': float(dca_return),
            'strategy_annual': float(strategy_annual),
            'buyhold_annual': float(buyhold_annual),
            'dca_annual': float(dca_annual),
            'strategy_sharpe': float(strategy_sharpe),
            'buyhold_sharpe': float(buyhold_sharpe),
            'dca_sharpe': float(dca_sharpe),
            'strategy_volatility': float(strategy_volatility),
            'buyhold_volatility': float(buyhold_volatility),
            'dca_volatility': float(dca_volatility),
            'strategy_max_dd': float(strategy_max_dd),
            'buyhold_max_dd': float(buyhold_max_dd),
            'dca_max_dd': float(dca_max_dd),
            'num_trades': int(num_trades),
            'win_rate': float(win_rate),
            'time_in_market': float(time_in_market)
        }
    
    def run_backtest(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Ejecuta backtesting completo con manejo robusto de errores
        """
        print("🚀 Iniciando backtesting...")
        
        try:
            # Verificar datos mínimos
            if 'Close' not in data.columns:
                raise ValueError("Datos no contienen columna 'Close'")
            
            if len(data) < self.config.MIN_LOOKBACK_DAYS:
                raise ValueError(f"Datos insuficientes: {len(data)} días (mínimo: {self.config.MIN_LOOKBACK_DAYS})")
            
            # Generar señales
            df_with_signals = self.generate_trading_signals(data)
            
            # Simular estrategias
            strategy_values, trades, dca_values, dca_investments = self.simulate_strategy(df_with_signals)
            
            # === CALCULAR BUY & HOLD CORRECTAMENTE ===
            print("   📊 Calculando Buy & Hold...")
            try:
                # Buy & Hold: comprar al inicio y mantener
                initial_price = df_with_signals['Close'].iloc[0]
                initial_shares = self.config.INITIAL_CAPITAL / initial_price
                buyhold_values = initial_shares * df_with_signals['Close']
                
                print(f"   ✅ Buy & Hold calculado:")
                print(f"      Precio inicial: ${initial_price:.2f}")
                print(f"      Acciones compradas: {initial_shares:.2f}")
                print(f"      Valor inicial: ${buyhold_values.iloc[0]:,.2f}")
                print(f"      Valor final: ${buyhold_values.iloc[-1]:,.2f}")
                
            except Exception as e:
                print(f"   ⚠️  Error en Buy & Hold: {str(e)}")
                # Fallback: crear serie con valores constantes
                buyhold_values = pd.Series([self.config.INITIAL_CAPITAL] * len(df_with_signals), 
                                         index=df_with_signals.index)
            
            # Agregar valores al DataFrame
            df_with_signals['Strategy_Value'] = strategy_values
            df_with_signals['BuyHold_Value'] = buyhold_values.values
            df_with_signals['DCA_Value'] = dca_values
            
            # Calcular período
            period_days = len(df_with_signals)
            period_years = period_days / 252
            
            print(f"📅 Período de backtesting:")
            print(f"   Inicio: {df_with_signals.index[0].strftime('%Y-%m-%d')}")
            print(f"   Fin: {df_with_signals.index[-1].strftime('%Y-%m-%d')}")
            print(f"   Duración: {period_days} días ({period_years:.2f} años)")
            
            # Calcular métricas
            metrics = self.calculate_metrics(strategy_values, buyhold_values, dca_values, trades, period_years)
            
            # Construir estructura de performance_metrics
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
                    'time_in_market': metrics['time_in_market']
                },
                'buyhold': {
                    'final_value': float(buyhold_values.iloc[-1]),
                    'total_return': metrics['buyhold_return'],
                    'annual_return': metrics['buyhold_annual'],
                    'volatility': metrics['buyhold_volatility'],
                    'sharpe_ratio': metrics['buyhold_sharpe'],
                    'max_drawdown': metrics['buyhold_max_dd']
                },
                'dca': {
                    'final_value': dca_values[-1] if dca_values else self.config.INITIAL_CAPITAL,
                    'total_return': metrics['dca_return'],
                    'annual_return': metrics['dca_annual'],
                    'volatility': metrics['dca_volatility'],
                    'sharpe_ratio': metrics['dca_sharpe'],
                    'max_drawdown': metrics['dca_max_dd'],
                    'num_investments': len(dca_investments)
                },
                'comparison': {
                    'outperformance_vs_buyhold': metrics['strategy_return'] - metrics['buyhold_return'],
                    'outperformance_vs_dca': metrics['strategy_return'] - metrics['dca_return'],
                    'risk_adjusted_return_vs_buyhold': metrics['strategy_sharpe'] - metrics['buyhold_sharpe'],
                    'risk_adjusted_return_vs_dca': metrics['strategy_sharpe'] - metrics['dca_sharpe']
                },
                'trades': trades,
                'dca_investments': dca_investments
            }
            
            # Mostrar resumen
            print(f"\n📊 RESUMEN DEL BACKTESTING:")
            print(f"   🎯 Estrategia Activa: {metrics['strategy_annual']:+.1f}% anual")
            print(f"   📈 Buy & Hold: {metrics['buyhold_annual']:+.1f}% anual") 
            print(f"   💰 DCA: {metrics['dca_annual']:+.1f}% anual")
            print(f"   🏆 Outperformance vs B&H: {performance_metrics['comparison']['outperformance_vs_buyhold']:+.1f}%")
            print(f"   ⚡ Operaciones realizadas: {metrics['num_trades']}")
            
            return df_with_signals, performance_metrics
            
        except Exception as e:
            print(f"❌ Error en backtesting: {str(e)}")
            raise RuntimeError(f"Error en backtesting: {str(e)}")