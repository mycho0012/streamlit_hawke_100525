import streamlit as st
import pandas as pd
import numpy as np
import pyupbit
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# import pandas_ta as ta  # 제거
import time
import os
import datetime
from hawkes import hawkes_process, vol_signal
from custom_indicators import atr  # 직접 구현한 ATR 함수 가져오기

# 페이지 설정
st.set_page_config(
    page_title="Hawkes 변동성 전략 최적화 도구",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 캐시 데코레이터 - 데이터 로드 최적화
@st.cache_data(ttl=3600, show_spinner=False)
def get_tickers():
    """거래 가능한 모든 코인 목록 가져오기"""
    return pyupbit.get_tickers(fiat="KRW")

@st.cache_data(ttl=3600, show_spinner=False)
def get_top_coins(limit=10):
    """거래량 기준 상위 코인 목록 가져오기"""
    markets = pyupbit.get_tickers(fiat="KRW")
    
    # 상위 코인 정보 수집 - 단일 API 호출로 모든 코인 가격 정보 가져오기
    try:
        # get_ticker 대신 get_current_price를 사용
        tickers_price = pyupbit.get_current_price(markets)
        
        # 최근 24시간 OHLCV 데이터로 거래량 계산
        coin_info = []
        for market in markets[:30]:  # 처음 30개 코인만 처리하여 시간 절약
            try:
                # 최근 24시간 데이터 가져오기 (일봉 데이터 1개)
                df = pyupbit.get_ohlcv(market, interval="day", count=1)
                if df is not None and not df.empty:
                    volume_price = df['volume'].iloc[0] * df['close'].iloc[0]  # 거래량 * 종가 = 거래대금
                    coin_info.append({
                        'market': market,
                        'volume': volume_price
                    })
                time.sleep(0.1)  # API 호출 간격
            except Exception as e:
                st.error(f"오류 발생 ({market}): {str(e)}")
        
        # 거래량 기준 정렬
        sorted_coins = sorted(coin_info, key=lambda x: x['volume'], reverse=True)
        return sorted_coins[:limit]
    
    except Exception as e:
        st.error(f"코인 정보 로드 중 오류 발생: {str(e)}")
        # 오류 발생 시 기본 코인 리스트 반환
        return [{'market': 'KRW-BTC', 'volume': 0}]

@st.cache_data(ttl=3600)
def get_historical_data(ticker, interval, count):
    """여러 요청을 통해 더 많은 과거 데이터 가져오기"""
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Upbit API는 한 번에 최대 200개 캔들만 제공
    max_per_request = 200
    requests_needed = (count // max_per_request) + (1 if count % max_per_request > 0 else 0)
    
    all_data = []
    for i in range(requests_needed):
        progress_text.text(f"{count}개의 캔들 데이터 로드 중... ({i+1}/{requests_needed})")
        progress_bar.progress((i+1)/requests_needed)
        
        if i == 0:
            # 첫 요청은 가장 최근 데이터
            df = pyupbit.get_ohlcv(ticker, interval=interval, count=min(count, max_per_request))
        else:
            # 이후 요청은 이전 데이터의 마지막 날짜 이전 데이터
            df = pyupbit.get_ohlcv(ticker, interval=interval, count=min(count - i*max_per_request, max_per_request), to=all_data[-1].index[0])
        
        if df is not None and not df.empty:
            all_data.append(df)
        else:
            break
    
    # 모든 데이터 결합
    if all_data:
        combined_df = pd.concat(all_data[::-1])  # 역순으로 결합하여 날짜 오름차순으로
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]  # 중복 제거
        combined_df.sort_index(inplace=True)
        
        # 요청한 수만큼만 반환
        result_df = combined_df.iloc[-count:] if len(combined_df) > count else combined_df
        progress_text.empty()
        progress_bar.empty()
        return result_df
    
    progress_text.empty()
    progress_bar.empty()
    return pd.DataFrame()

def prepare_hawkes_data(data, kappa, lookback):
    """ATR 계산 및 호크스 프로세스 적용"""
    # 정규화된 범위 계산
    data['log_high'] = np.log(data['high'])
    data['log_low'] = np.log(data['low'])
    data['log_close'] = np.log(data['close'])
    
    # ATR 계산 (직접 구현한 함수 사용)
    norm_lookback = 336  # 14일 (시간 단위)
    data['atr'] = atr(
        data['log_high'], 
        data['log_low'], 
        data['log_close'], 
        norm_lookback
    )
    
    # 정규화된 범위
    data['norm_range'] = (
        data['log_high'] - data['log_low']
    ) / data['atr']
    
    # 호크스 프로세스 적용
    data['v_hawk'] = hawkes_process(data['norm_range'], kappa)
    
    # 거래 신호 생성
    data['signal'] = vol_signal(
        data['close'], 
        data['v_hawk'], 
        lookback
    )
    
    # 롱 온리 전략 적용 (매도 신호 -1을 무시)
    data['long_only_signal'] = data['signal'].apply(lambda x: 1 if x == 1 else 0)
    
    # 변동성 분위수 계산
    data['q05'] = data['v_hawk'].rolling(lookback).quantile(0.05)
    data['q95'] = data['v_hawk'].rolling(lookback).quantile(0.95)
    
    return data

def run_backtest(data, commission_rate=0.0005, initial_capital=10000000, trading_ratio=1.0):
    """롱 온리 전략 백테스팅 실행"""
    # 백테스팅을 위한 변수 초기화
    capital = initial_capital
    coin_amount = 0
    
    # 수익률 및 포트폴리오 가치 추적
    data['portfolio_value'] = float(initial_capital)  # 초기값
    data['returns'] = 0.0
    data['trade'] = 0  # 1: 매수, -1: 매도, 0: 홀드
    data['coin_position'] = 0.0  # 코인 보유량
    data['position_value'] = 0.0  # 포지션 가치
    data['cash'] = float(initial_capital)  # 현금 보유량
    data['position_status'] = "중립"  # 포지션 상태
    
    # 거래 기록 저장
    trades = []
    
    # 일별 수익률 계산
    data['daily_returns'] = data['close'].pct_change()
    
    # 전략 수익률 계산
    for i in range(1, len(data)):
        # 이전 시점의 신호 확인
        prev_signal = data['long_only_signal'].iloc[i-1]
        curr_signal = data['long_only_signal'].iloc[i]
        
        # 매수 신호 (0 -> 1)
        if prev_signal == 0 and curr_signal == 1:
            # 매수할 코인 수량 계산
            trading_amount = capital * trading_ratio
            commission = trading_amount * commission_rate
            coin_to_buy = (trading_amount - commission) / data['close'].iloc[i]
            
            # 매수 실행
            capital -= trading_amount
            coin_amount += coin_to_buy
            data.loc[data.index[i], 'trade'] = 1
            data.loc[data.index[i], 'coin_position'] = float(coin_amount)
            data.loc[data.index[i], 'position_value'] = float(coin_amount * data['close'].iloc[i])
            data.loc[data.index[i], 'cash'] = float(capital)
            data.loc[data.index[i], 'position_status'] = "롱"
            
            # 거래 기록 추가
            trades.append({
                'time': data.index[i],
                'type': '매수',
                'price': data['close'].iloc[i],
                'amount': coin_amount,
                'value': trading_amount,
                'commission': commission,
                'hawk_value': data['v_hawk'].iloc[i],
                'q95_value': data['q95'].iloc[i]
            })
            
        # 매도 신호 (1 -> 0)
        elif prev_signal == 1 and curr_signal == 0 and coin_amount > 0:
            # 매도 실행
            selling_amount = coin_amount * data['close'].iloc[i]
            commission = selling_amount * commission_rate
            
            # 수익률 계산
            if trades:
                entry_price = trades[-1]['price']  # 진입 가격
                exit_price = data['close'].iloc[i]  # 청산 가격
                profit_pct = (exit_price / entry_price - 1) * 100
            else:
                profit_pct = 0
            
            capital += selling_amount - commission
            coin_amount = 0
            data.loc[data.index[i], 'trade'] = -1
            data.loc[data.index[i], 'coin_position'] = float(coin_amount)
            data.loc[data.index[i], 'position_value'] = 0.0
            data.loc[data.index[i], 'cash'] = float(capital)
            data.loc[data.index[i], 'position_status'] = "중립"
            
            # 거래 기록 추가
            trades.append({
                'time': data.index[i],
                'type': '매도',
                'price': data['close'].iloc[i],
                'amount': trades[-1]['amount'] if trades else 0,
                'value': selling_amount,
                'commission': commission,
                'profit_pct': profit_pct,
                'hawk_value': data['v_hawk'].iloc[i],
                'signal_value': curr_signal
            })
        
        # 포지션 유지 중
        else:
            data.loc[data.index[i], 'coin_position'] = float(coin_amount)
            data.loc[data.index[i], 'position_value'] = float(coin_amount * data['close'].iloc[i])
            data.loc[data.index[i], 'cash'] = float(capital)
            data.loc[data.index[i], 'position_status'] = "롱" if coin_amount > 0 else "중립"
        
        # 포트폴리오 가치 갱신
        portfolio_value = capital + (coin_amount * data['close'].iloc[i])
        data.loc[data.index[i], 'portfolio_value'] = float(portfolio_value)
        
        # 전략 수익률 계산
        if i > 0 and data['portfolio_value'].iloc[i-1] > 0:
            data.loc[data.index[i], 'returns'] = (data['portfolio_value'].iloc[i] / data['portfolio_value'].iloc[i-1]) - 1
    
    # 누적 수익률 계산
    data['returns'] = data['returns'].fillna(0)
    data['cumulative_returns'] = (1 + data['returns']).cumprod() - 1
    
    # 매매 시점 식별
    buy_points = data[data['trade'] == 1]
    sell_points = data[data['trade'] == -1]
    
    return data, buy_points, sell_points, trades

def calculate_performance_metrics(data, trades):
    """전략 성능 지표 계산"""
    # 주요 성능 지표 계산
    returns = data['returns'].dropna()
    
    # 완료된 거래 필터링
    completed_trades = [trade for trade in trades if trade.get('type') == '매도']
    
    # 기본 성능 지표
    metrics = {
        'total_trades': len(completed_trades),
        'win_trades': 0,
        'loss_trades': 0,
        'win_rate': 0,
        'avg_win': 0,
        'avg_loss': 0,
        'total_profit': 0,
        'final_portfolio': data['portfolio_value'].iloc[-1] if not data.empty else 0,
        'return_pct': 0,
        'max_drawdown': 0,
        'annualized_return': 0,
        'volatility': 0,
        'sharpe_ratio': 0,
        'profit_factor': 0
    }
    
    if not completed_trades:
        return metrics
    
    # 승리/손실 거래 구분
    win_trades = [trade for trade in completed_trades if trade.get('profit_pct', 0) > 0]
    loss_trades = [trade for trade in completed_trades if trade.get('profit_pct', 0) <= 0]
    
    metrics['win_trades'] = len(win_trades)
    metrics['loss_trades'] = len(loss_trades)
    
    # 승률
    metrics['win_rate'] = (metrics['win_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
    
    # 평균 수익/손실
    if win_trades:
        metrics['avg_win'] = sum(trade['profit_pct'] for trade in win_trades) / len(win_trades)
    
    if loss_trades:
        metrics['avg_loss'] = sum(trade['profit_pct'] for trade in loss_trades) / len(loss_trades)
    
    # 총 수익
    metrics['total_profit'] = sum(trade['profit_pct'] for trade in completed_trades)
    
    # 수익률
    metrics['return_pct'] = (metrics['final_portfolio'] / 10000000 - 1) * 100
    
    # 연간 수익률 (연율화)
    days = max(1, (data.index[-1] - data.index[0]).days)
    metrics['annualized_return'] = ((1 + metrics['return_pct']/100) ** (365 / days) - 1) * 100
    
    # 변동성 (표준편차) - 시간 단위에 따라 조정
    if data.index.freq == 'D' or (data.index[1] - data.index[0]).days >= 1:
        # 일봉 데이터
        metrics['volatility'] = returns.std() * np.sqrt(252) * 100  # 연간 변동성
    else:
        # 시간봉 또는 분봉 데이터 (1시간 간격 가정)
        hours_in_day = 24
        metrics['volatility'] = returns.std() * np.sqrt(hours_in_day * 365) * 100
    
    # 샤프 비율
    risk_free_rate = 0.03  # 연 3% 안전 자산 수익률 가정
    daily_rf = (1 + risk_free_rate) ** (1/365) - 1
    if metrics['volatility'] > 0:
        metrics['sharpe_ratio'] = (metrics['annualized_return']/100 - risk_free_rate) / (metrics['volatility']/100)
    else:
        metrics['sharpe_ratio'] = 0
    
    # 최대 낙폭 (MDD)
    if len(data) > 0:
        cumulative = data['portfolio_value'] / 10000000
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        metrics['max_drawdown'] = abs(drawdown.min() * 100)
    
    # Profit Factor
    total_profit = sum(trade['profit_pct'] for trade in win_trades) if win_trades else 0
    total_loss = sum(abs(trade['profit_pct']) for trade in loss_trades) if loss_trades else 0
    metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
    
    return metrics

def optimize_parameters(data, kappa_values, lookback_values):
    """KAPPA와 LOOKBACK 값 그리드 서치"""
    # 프로그레스 바 생성
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    results = []
    total_combinations = len(kappa_values) * len(lookback_values)
    completed = 0
    
    start_time = time.time()
    
    for kappa in kappa_values:
        for lookback in lookback_values:
            # 현재 진행상황 표시
            progress_text.text(f"파라미터 최적화 중... ({completed+1}/{total_combinations}) - KAPPA={kappa}, LOOKBACK={lookback}")
            progress_bar.progress((completed+1)/total_combinations)
            
            # 현재 조합에 대한 데이터 준비
            current_data = prepare_hawkes_data(data.copy(), kappa, lookback)
            
            # 백테스팅 실행
            backtest_data, _, _, trades = run_backtest(current_data)
            
            # 성능 지표 계산
            metrics = calculate_performance_metrics(backtest_data, trades)
            
            # 결과 저장
            results.append({
                'kappa': kappa,
                'lookback': lookback,
                'total_trades': metrics['total_trades'],
                'win_rate': metrics['win_rate'],
                'return_pct': metrics['return_pct'],
                'annualized_return': metrics['annualized_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'profit_factor': metrics['profit_factor']
            })
            
            # 카운터 증가
            completed += 1
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results)
    
    # 프로그레스 바 제거
    progress_text.empty()
    progress_bar.empty()
    
    return results_df

def visualize_results(data, buy_points, sell_points, trades, ticker, kappa, lookback):
    """거래 실행 및 포지션 변화 시각화"""
    completed_trades = [trade for trade in trades if trade.get('type') == '매도']
    
    # 포지션 변화 시각화
    fig = make_subplots(
        rows=5, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f'{ticker} 가격 & 포지션', 
            'Hawkes 프로세스 변동성 & 임계값', 
            '호크스 원본 신호', 
            '포지션 상태',
            '누적 수익률'
        ),
        row_heights=[0.25, 0.2, 0.15, 0.15, 0.25]
    )
    
    # 1. 가격 차트 및 포지션 변화
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name=ticker
        ),
        row=1, col=1
    )
    
    # 포지션 구간 색상 표시 (롱 포지션일 때 배경색)
    for i in range(len(buy_points)):
        if i < len(sell_points):
            # 매수부터 매도까지 구간
            buy_time = buy_points.index[i]
            sell_time = sell_points.index[i]
            
            fig.add_vrect(
                x0=buy_time, 
                x1=sell_time,
                fillcolor="rgba(0, 255, 0, 0.1)", 
                opacity=0.5,
                layer="below", 
                line_width=0,
                annotation_text="롱 포지션",
                annotation_position="top right",
                row=1, col=1
            )
    
    # 매수/매도 지점 표시
    if not buy_points.empty:
        for idx, row in buy_points.iterrows():
            fig.add_annotation(
                x=idx, 
                y=row['low'] * 0.995,
                text="매수",
                showarrow=True,
                arrowhead=1,
                arrowcolor="green",
                arrowsize=1.5,
                arrowwidth=2,
                ax=0,
                ay=20,
                row=1, col=1
            )
    
    if not sell_points.empty:
        for idx, row in sell_points.iterrows():
            # 수익/손실 여부에 따라 색상 변경
            for trade in completed_trades:
                if trade['time'] == idx:
                    color = "blue" if trade.get('profit_pct', 0) > 0 else "red"
                    text = f"매도 ({trade.get('profit_pct', 0):.1f}%)"
                    
                    fig.add_annotation(
                        x=idx, 
                        y=row['high'] * 1.005,
                        text=text,
                        showarrow=True,
                        arrowhead=1,
                        arrowcolor=color,
                        arrowsize=1.5,
                        arrowwidth=2,
                        ax=0,
                        ay=-20,
                        row=1, col=1
                    )
    
    # 2. 호크스 프로세스 변동성과 임계값
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['v_hawk'],
            line=dict(width=1.5, color='yellow'),
            name='Hawkes 변동성'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['q05'],
            line=dict(width=1, color='green', dash='dash'),
            name='5% 분위수'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['q95'],
            line=dict(width=1, color='red', dash='dash'),
            name='95% 분위수'
        ),
        row=2, col=1
    )
    
    # 3. 호크스 원본 신호
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['signal'],
            mode='lines',
            name='호크스 신호',
            line=dict(width=2, color='white')
        ),
        row=3, col=1
    )
    
    # 4. 포지션 상태 (카테고리 표시)
    position_vals = data['position_status'].map({"중립": 0, "롱": 1}).fillna(0)
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=position_vals,
            mode='lines',
            name='포지션 상태',
            line=dict(width=3, color='cyan')
        ),
        row=4, col=1
    )
    
    # 5. 누적 수익률
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['cumulative_returns'] * 100,
            name='누적 수익률 (%)',
            line=dict(width=2, color='#00FFAA')
        ),
        row=5, col=1
    )
    
    # 차트 레이아웃 설정
    final_return = data['cumulative_returns'].iloc[-1] * 100 if len(data) > 0 else 0
    fig.update_layout(
        title=f'{ticker} 호크스 프로세스 전략 (KAPPA={kappa}, LOOKBACK={lookback}) - 수익률: {final_return:.2f}%',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=800,
        width=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Y축 설정
    fig.update_yaxes(title_text="가격", row=1, col=1)
    fig.update_yaxes(title_text="Hawkes 값", row=2, col=1)
    fig.update_yaxes(
        title_text="신호", 
        tickvals=[-1, 0, 1], 
        ticktext=["매도", "중립", "매수"],
        range=[-1.1, 1.1],
        row=3, col=1
    )
    fig.update_yaxes(
        title_text="포지션", 
        tickvals=[0, 1], 
        ticktext=["중립", "롱"],
        range=[-0.1, 1.1],
        row=4, col=1
    )
    fig.update_yaxes(title_text="수익률 (%)", row=5, col=1)
    
    return fig

def show_heatmaps(results_df, kappa_values, lookback_values):
    """그리드 서치 결과 히트맵 시각화"""
    # 피벗 테이블 생성
    heatmap_data_return = results_df.pivot(index='lookback', columns='kappa', values='return_pct')
    heatmap_data_sharpe = results_df.pivot(index='lookback', columns='kappa', values='sharpe_ratio')
    heatmap_data_profit_factor = results_df.pivot(index='lookback', columns='kappa', values='profit_factor')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 누적 수익률 히트맵
        fig_return = px.imshow(
            heatmap_data_return, 
            x=[str(k) for k in kappa_values],
            y=[str(l) for l in lookback_values],
            color_continuous_scale='Viridis',
            title='누적 수익률 히트맵 (%)',
            labels=dict(x='KAPPA', y='LOOKBACK', color='수익률 (%)'),
            text_auto='.1f'
        )
        fig_return.update_layout(height=400)
        st.plotly_chart(fig_return, use_container_width=True)
        
        # Profit Factor 히트맵
        fig_profit_factor = px.imshow(
            heatmap_data_profit_factor, 
            x=[str(k) for k in kappa_values],
            y=[str(l) for l in lookback_values],
            color_continuous_scale='Greens',
            title='Profit Factor 히트맵',
            labels=dict(x='KAPPA', y='LOOKBACK', color='Profit Factor'),
            text_auto='.2f'
        )
        fig_profit_factor.update_layout(height=400)
        st.plotly_chart(fig_profit_factor, use_container_width=True)
    
    with col2:
        # 샤프 비율 히트맵
        fig_sharpe = px.imshow(
            heatmap_data_sharpe, 
            x=[str(k) for k in kappa_values],
            y=[str(l) for l in lookback_values],
            color_continuous_scale='RdBu',
            title='샤프 비율 히트맵',
            labels=dict(x='KAPPA', y='LOOKBACK', color='샤프 비율'),
            text_auto='.2f'
        )
        fig_sharpe.update_layout(height=400)
        st.plotly_chart(fig_sharpe, use_container_width=True)
        
        # 최적 파라미터 조합 출력
        st.subheader("최적 파라미터 조합")
        
        # 각 기준별 최적값
        best_by_return = results_df.loc[results_df['return_pct'].idxmax()]
        best_by_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        best_by_profit_factor = results_df.loc[results_df['profit_factor'].idxmax()]
        
        st.markdown(f"""
        **수익률 기준 최적 조합:**
        - KAPPA = {best_by_return['kappa']}
        - LOOKBACK = {best_by_return['lookback']}
        - 수익률: {best_by_return['return_pct']:.2f}%
        - 거래 횟수: {best_by_return['total_trades']}
        - 승률: {best_by_return['win_rate']:.2f}%
        
        **샤프 비율 기준 최적 조합:**
        - KAPPA = {best_by_sharpe['kappa']}
        - LOOKBACK = {best_by_sharpe['lookback']}
        - 샤프 비율: {best_by_sharpe['sharpe_ratio']:.2f}
        - 수익률: {best_by_sharpe['return_pct']:.2f}%
        
        **Profit Factor 기준 최적 조합:**
        - KAPPA = {best_by_profit_factor['kappa']}
        - LOOKBACK = {best_by_profit_factor['lookback']}
        - Profit Factor: {best_by_profit_factor['profit_factor']:.2f}
        - 수익률: {best_by_profit_factor['return_pct']:.2f}%
        """)

# 메인 앱 로직
def main():
    # 사이드바 설정
    st.sidebar.title("Hawkes 변동성 전략")
    st.sidebar.markdown("---")
    
    # 데이터 및 파라미터 설정
    st.sidebar.subheader("데이터 설정")
    
    # 코인 선택 옵션
    coin_selection_mode = st.sidebar.radio("코인 선택 방식", ["상위 코인에서 선택", "직접 입력"])
    
    selected_coin = None
    
    if coin_selection_mode == "상위 코인에서 선택":
        # 스피너 대신 placeholder 사용
        loading_text = st.sidebar.empty()
        loading_text.text("상위 코인 로딩 중...")
        
        try:
            top_coins = get_top_coins(10)
            loading_text.empty()  # 로딩 메시지 제거
            
            if top_coins:
                coin_options = [f"{coin['market']}" for coin in top_coins]
                selected_coin = st.sidebar.selectbox("코인 선택", coin_options)
        except Exception as e:
            loading_text.empty()
            st.sidebar.error(f"코인 목록 로드 중 오류 발생: {str(e)}")
            # 직접 입력 모드로 전환
            coin_selection_mode = "직접 입력"
    
    if coin_selection_mode == "직접 입력" or selected_coin is None:
        all_tickers = pyupbit.get_tickers(fiat="KRW")
        selected_coin = st.sidebar.selectbox("코인 선택", all_tickers, index=0 if all_tickers else None)
    
    # 캔들 주기 선택
    timeframe_options = {
        "1분": "minute1",
        "3분": "minute3",
        "5분": "minute5",
        "10분": "minute10",
        "15분": "minute15",
        "30분": "minute30",
        "60분": "minute60",
        "240분": "minute240",
        "일봉": "day"
    }
    selected_timeframe_display = st.sidebar.selectbox("캔들 주기", list(timeframe_options.keys()))
    selected_timeframe = timeframe_options[selected_timeframe_display]
    
    # 데이터 로드 기간 설정
    if selected_timeframe == "day":
        default_count = 365
        max_count = 1000
    elif "minute" in selected_timeframe:
        minutes = int(selected_timeframe.replace("minute", ""))
        if minutes <= 10:
            default_count = 24 * 60 // minutes * 7  # 약 1주일
            max_count = 24 * 60 // minutes * 30  # 약 1개월
        else:
            default_count = 24 * 30 // (minutes // 60 if minutes >= 60 else minutes)  # 약 1개월
            max_count = 24 * 365 // (minutes // 60 if minutes >= 60 else minutes)  # 약 1년
    
    data_count = st.sidebar.slider("데이터 로드 기간 (캔들 수)", 
                                  min_value=100, 
                                  max_value=max_count, 
                                  value=default_count)
    
    # 최적화 파라미터 설정
    st.sidebar.subheader("최적화 설정")
    
    # 최적화 모드 선택
    optimization_mode = st.sidebar.radio("최적화 모드", ["빠른 최적화", "전체 최적화", "사용자 정의"])
    
    if optimization_mode == "빠른 최적화":
        kappa_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        lookback_values = [48, 72, 96, 120]
    elif optimization_mode == "전체 최적화":
        kappa_values = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
        lookback_values = [24, 48, 72, 96, 120, 168, 240, 336]
    else:  # 사용자 정의
        st.sidebar.markdown("KAPPA 값 (쉼표로 구분)")
        kappa_input = st.sidebar.text_input("KAPPA 값", "0.1, 0.2, 0.3, 0.4, 0.5")
        kappa_values = [float(k.strip()) for k in kappa_input.split(",") if k.strip()]
        
        st.sidebar.markdown("LOOKBACK 값 (쉼표로 구분)")
        lookback_input = st.sidebar.text_input("LOOKBACK 값", "48, 72, 96, 120")
        lookback_values = [int(l.strip()) for l in lookback_input.split(",") if l.strip()]
    
    # 초기 자본금 및 수수료 설정
    st.sidebar.subheader("백테스팅 설정")
    initial_capital = st.sidebar.number_input("초기 자본금 (KRW)", min_value=1000000, max_value=1000000000, value=10000000, step=1000000)
    commission_rate = st.sidebar.number_input("거래 수수료 (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01) / 100
    
    # 메인 콘텐츠
    st.title("Hawkes 변동성 전략 최적화 도구")
    st.markdown(f"**선택된 코인: {selected_coin} | 캔들 주기: {selected_timeframe_display} | 데이터 기간: {data_count} 캔들**")
    
    # 실행 버튼
    start_button = st.button("전략 최적화 시작", type="primary")
    
    if start_button:
        # 데이터 로드
        with st.spinner(f"{selected_coin} 데이터 로드 중..."):
            data = get_historical_data(selected_coin, selected_timeframe, data_count)
            
            if data.empty:
                st.error("데이터를 가져올 수 없습니다. 다른 코인이나 기간을 선택해 주세요.")
                return
            
            st.success(f"데이터 로드 완료: {len(data)} 캔들 ({data.index[0]} ~ {data.index[-1]})")
        
        # 데이터 기본 정보 표시
        st.subheader("데이터 미리보기")
        st.dataframe(data.head())
        
        # 파라미터 최적화
        st.subheader("파라미터 최적화 실행")
        st.markdown(f"KAPPA 값: {kappa_values}")
        st.markdown(f"LOOKBACK 값: {lookback_values}")
        
        with st.spinner("파라미터 최적화 중..."):
            results_df = optimize_parameters(data.copy(), kappa_values, lookback_values)
            
            if results_df.empty:
                st.error("최적화 과정에서 오류가 발생했습니다.")
                return
        
        # 히트맵 표시
        st.subheader("최적화 결과 히트맵")
        show_heatmaps(results_df, kappa_values, lookback_values)
        
        # 최적 파라미터 적용
        st.subheader("최적 파라미터로 전략 실행")
        
        best_option = st.radio(
            "최적화 기준 선택",
            ["수익률", "샤프 비율", "Profit Factor"],
            horizontal=True
        )
        
        if best_option == "수익률":
            best_params = results_df.loc[results_df['return_pct'].idxmax()]
        elif best_option == "샤프 비율":
            best_params = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        else:  # Profit Factor
            best_params = results_df.loc[results_df['profit_factor'].idxmax()]
        
        best_kappa = best_params['kappa']
        best_lookback = int(best_params['lookback'])
        
        st.markdown(f"**선택된 최적 파라미터: KAPPA={best_kappa}, LOOKBACK={best_lookback}**")
        
        # 최적 파라미터로 전략 실행
        with st.spinner("최적 파라미터로 전략 실행 중..."):
            optimized_data = prepare_hawkes_data(data.copy(), best_kappa, best_lookback)
            backtest_data, buy_points, sell_points, trades = run_backtest(
                optimized_data,
                commission_rate=commission_rate,
                initial_capital=initial_capital
            )
            
            metrics = calculate_performance_metrics(backtest_data, trades)
        
        # 성능 지표 표시
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("총 거래 횟수", f"{metrics['total_trades']}회")
        col2.metric("승률", f"{metrics['win_rate']:.2f}%")
        col3.metric("누적 수익률", f"{metrics['return_pct']:.2f}%")
        col4.metric("최대 낙폭 (MDD)", f"{metrics['max_drawdown']:.2f}%")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("연간 수익률", f"{metrics['annualized_return']:.2f}%")
        col2.metric("샤프 비율", f"{metrics['sharpe_ratio']:.2f}")
        col3.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        col4.metric("평균 승리 수익", f"{metrics['avg_win']:.2f}%")
        
        # 거래 내역 표시
        st.subheader("거래 내역")
        trade_df = pd.DataFrame([
            {
                '시간': trade['time'],
                '유형': trade['type'],
                '가격': f"{trade['price']:,.0f}",
                '수량': f"{trade['amount']:.6f}",
                '수익률': f"{trade.get('profit_pct', 0):.2f}%" if trade['type'] == '매도' else "-"
            }
            for trade in trades
        ])
        
        st.dataframe(trade_df, use_container_width=True)
        
        # 시각화
        st.subheader("전략 시각화")
        fig = visualize_results(backtest_data, buy_points, sell_points, trades, selected_coin, best_kappa, best_lookback)
        st.plotly_chart(fig, use_container_width=True)
        
        # 결과 저장 옵션
        st.subheader("결과 저장")
        save_results = st.button("결과 저장하기")
        
        if save_results:
            # 결과 디렉토리 생성
            os.makedirs('results', exist_ok=True)
            
            # 파일명 생성 (타임스탬프 포함)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"results/{selected_coin}_{selected_timeframe}_{timestamp}"
            
            # CSV 파일 저장
            results_df.to_csv(f"{filename_base}_optimization.csv", index=False)
            backtest_data.to_csv(f"{filename_base}_backtest.csv")
            trade_df.to_csv(f"{filename_base}_trades.csv", index=False)
            
            # 시각화 파일 저장
            fig.write_html(f"{filename_base}_visualization.html")
            
            st.success(f"결과가 results 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main() 
