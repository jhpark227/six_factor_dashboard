import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import backtrader as bt
import time
import warnings
from datetime import datetime

if not hasattr(np, 'Inf'):
    np.Inf = np.inf

# 기초 설정
warnings.filterwarnings(action='ignore')
st.set_page_config(page_title="Six-Factor Strategy Dashboard", layout="wide")

# --- 설정 및 데이터 로드 ---
color_map = ['#142F46','#2A9D8F','#E9AB0C','#D9410A','#E76F51']

@st.cache_data
def load_corp_data():
    corp_df = pd.read_csv('corp_data.csv', header=0, encoding='cp949')
    corp_df = corp_df.sort_values(by='Market_Cap', ascending=False, ignore_index=True)
    return corp_df

# --- [성능 분석 함수] ---
def calculate_performance_metrics(returns, risk_free_rate=0.03):
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    mdd = abs(drawdowns.min())

    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = returns - daily_rf
    sharpe_ratio = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() != 0 else 0.0
    return mdd, sharpe_ratio, drawdowns
  
# --- 전략 클래스 (원본 로직 보존 + 시각화 데이터 추출용 수정) ---
class SixFactorStrategy(bt.Strategy):
    params = (
        ('stop_loss', -0.08), ('threshold', 7), 
        ('w_rs', 3), ('w_psar', 1), ('w_wma', 2), ('w_vwma', 2), ('w_rsi', 1), ('w_bb', 1),
        ('rsi_period', 10), ('bb_period', 20), ('bb_dev', 2.0),
        ('wma_period', 50), ('psar_af', 0.015), ('psar_max', 0.2), ('rs_period', 60),('adx_period', 14),
    )
    
    def __init__(self):
        self.daily_stock = self.datas[0]
        self.weekly_stock = self.datas[1]
        self.benchmark = self.datas[2]
        self.highest_price = 0
        self.last_sell_date = None
        self.last_sell_reason = ""
        self.bar_executed = 0
        self.step_1_done = False
        self.order_history = [] # 시각화를 위해 (날짜, 가격, 유형, 사유) 저장
        self.first_buy_date = None
        self.last_buy_date = None
        self.score_history = []
        self.wins = []
        self.losses = []

        # 지표 설정
        self.rsi_w = bt.indicators.RSI(self.weekly_stock.close, period=self.params.rsi_period)
        self.psar_w = bt.indicators.ParabolicSAR(self.weekly_stock, af=self.params.psar_af, afmax=self.params.psar_max)
        self.bb = bt.indicators.BollingerBands(self.daily_stock.close, period=self.params.bb_period, devfactor=self.params.bb_dev)
        self.wma = bt.indicators.WeightedMovingAverage(self.daily_stock.close, period=self.params.wma_period)
        self.vwma = bt.indicators.SumN(self.daily_stock.close * self.daily_stock.volume, period=20) / \
                    bt.indicators.SumN(self.daily_stock.volume, period=20)
        self.stock_roc = bt.indicators.RateOfChange(self.daily_stock.close, period=self.params.rs_period)
        self.bench_roc = bt.indicators.RateOfChange(self.benchmark.close, period=self.params.rs_period)
        self.wma_20 = bt.indicators.WeightedMovingAverage(self.daily_stock.close, period=20)
        self.adx = bt.indicators.ADX(self.daily_stock, period=self.params.adx_period)
        self.atr = bt.indicators.ATR(self.daily_stock, period=20)
        self.avg_vol = bt.indicators.SMA(self.daily_stock.volume, period=10)

    def get_six_factor_score(self):
        c_psar = self.psar_w[-1] < self.weekly_stock.close[-1] and self.psar_w[-2] > self.weekly_stock.close[-2]
        c_rsi  = self.rsi_w[-1] > 45 
        c_bb   = self.daily_stock.close[-1] > self.bb.mid[-1]
        c_wma  = self.daily_stock.close[-1] > self.wma[-1]
        c_vwma = self.daily_stock.close[-1] > self.vwma[-1]
        c_rs   = self.stock_roc[-1] > self.bench_roc[-1]
        return (c_rs * self.p.w_rs + c_psar * self.p.w_psar + c_wma * self.p.w_wma + 
                c_vwma * self.p.w_vwma + c_rsi * self.p.w_rsi + c_bb * self.p.w_bb)

    def next(self):
        if len(self.weekly_stock) < 2 or len(self.daily_stock) < self.params.wma_period: return
        is_bull = self.daily_stock.close[-1] > self.wma_20[-1]
        is_trending = self.adx[-1] > 25
        score = self.get_six_factor_score()
        current_date = self.data.datetime.date(0)

        # 모드 결정
        if is_bull and is_trending: threshold, stop_loss = self.params.threshold - 1, self.params.stop_loss - 0.02
        elif not is_bull and is_trending: threshold, stop_loss = self.params.threshold + 1.5, -0.05
        else: threshold, stop_loss = self.params.threshold, self.params.stop_loss

        # 스코어 기록 (시각화용)
        self.score_history.append({'Date': pd.to_datetime(current_date), 'Score': score})

        if not self.position:
            self.check_buy_signal(score, current_date, threshold)
        else:
            self.check_pyramiding_and_sell(score, current_date, stop_loss)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            curr_date = self.data.datetime.date(0)
            reason = order.info.get('reason', 'N/A')
            if order.isbuy():
                if self.first_buy_date is None: self.first_buy_date = curr_date
                self.last_buy_date = curr_date
                self.bar_executed = len(self)
                self.highest_price = max(self.highest_price, order.executed.price)
                self.order_history.append((curr_date, order.executed.price, 'BUY', reason))
            elif order.issell():
                self.highest_price = 0
                self.step_1_done = False
                self.first_buy_date = self.last_buy_date = None
                self.order_history.append((curr_date, order.executed.price, 'SELL',reason))

    def notify_trade(self, trade):
        if not trade.isclosed: return
        pnl = trade.pnlcomm
        if pnl > 0: self.wins.append(pnl)
        elif pnl < 0: self.losses.append(abs(pnl))

    def check_buy_signal(self, score, current_date, threshold):
        is_high_vol = self.daily_stock.volume[-1] > self.avg_vol[-1] * 1.3
        is_cooldown = (current_date - self.last_sell_date).days >= 7 if self.last_sell_date else True
        if all([score >= threshold, self.wma[-1] > self.wma[-5], self.adx[-1] > 25, is_high_vol, is_cooldown]):
            self.buy_action("신규 진입(80%)", 0.8)

    def check_pyramiding_and_sell(self, score, current_date, dynamic_stop_loss):
        avg_price = self.position.price
        curr_price = self.daily_stock.close[0]
        current_return = (curr_price - avg_price) / avg_price
        self.highest_price = max(self.highest_price, curr_price)
        pullback = (curr_price - self.highest_price) / self.highest_price
        hold_days = len(self) - self.bar_executed
        atr_stop_pct = ((avg_price - (self.atr[-1] * 2)) - avg_price) / avg_price
        final_stop_threshold = min(dynamic_stop_loss, atr_stop_pct)

        # --- 매수(불타기) --- #
        if not self.step_1_done and current_return >= 0.15 and self.rsi_w[-1] > 60 and score >= 8:
            self.buy_action("불타기(+15%)", 0.15); self.step_1_done = True
        # --- 매도 --- #
        # [1] 수익률/변동성(ATR) 손절(Stop Loss)
        if current_return <= final_stop_threshold: self.sell_action(f"Stop Loss({final_stop_threshold:.1%})"); return
        if (hold_days <= 5 and score < 4) or (hold_days >= 120 and current_return <= 0.05):
            self.sell_action("휩소(Whipsaw)/시간 손절"); return
        if current_return >= 0.15:
            if current_return >= 0.50: exit_limit = -0.25 if score >= 6 else -0.15
            elif current_return >= 0.30: exit_limit = -0.15 if score >= 7 else -0.12
            else: exit_limit = -0.12 if score >= 8 else -0.10
            if pullback <= exit_limit: self.sell_action("수익 보존 익절"); return

    def buy_action(self, reason, stake):
        size = int((self.broker.get_value() * stake) / self.daily_stock.close[0])
        if size > 0: 
            order = self.buy(size=size)
            order.addinfo(reason=reason)

    def sell_action(self, reason):
        order = self.sell(size=self.position.size)
        order.addinfo(reason=reason)
        self.last_sell_date = self.data.datetime.date(0)
        self.last_sell_reason = reason
        

# --- UI 레이아웃 ---
st.title("📈 Six-Factor 퀀트 전략 대시보드")
tab1, tab2 = st.tabs(["🚀 전체 시장 요약", "🔍 종목별 정밀 분석"])

# 공통 사이드바
st.sidebar.header("Configuration")
num_stocks = st.sidebar.slider("분석 종목 수 (시총 순)", 5, 100, 20)
start_date = st.sidebar.date_input("분석 시작일", datetime(2021, 1, 1))
cash = st.sidebar.number_input("초기 자산", value=10000)

corp_df = load_corp_data()

# --- Tab 1: 전체 시장 요약 ---
with tab1:
    if st.button("전체 종목 백테스트 실행"):
        target_list = corp_df[['Ticker','Ticker_mkt']].head(num_stocks)
        results_list = []
        total_returns = []
        
        progress_bar = st.progress(0)
        for i, (target, target_market) in enumerate(target_list.values, start=1):
            df_stock = yf.download(target, start=start_date, auto_adjust=True, progress=False)
            df_bench = yf.download(target_market, start=start_date, auto_adjust=True, progress=False)
            
            # 2. [중요] 멀티인덱스 컬럼 해결 (튜플 오류 방지)
            if isinstance(df_stock.columns, pd.MultiIndex):
                df_stock.columns = df_stock.columns.droplevel(1)
            if isinstance(df_bench.columns, pd.MultiIndex):
                df_bench.columns = df_bench.columns.droplevel(1)
            
            if len(df_stock) < 50: continue
            
            cerebro = bt.Cerebro()
            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='all_returns')
            d0 = bt.feeds.PandasData(dataname=df_stock)
            cerebro.adddata(d0)
            cerebro.resampledata(d0, timeframe=bt.TimeFrame.Weeks, compression=1)
            cerebro.adddata(bt.feeds.PandasData(dataname=df_bench))
            cerebro.addstrategy(SixFactorStrategy)
            cerebro.broker.setcash(cash)
            
            results = cerebro.run()
            strat = results[0]
            
            # 성과 집계
            rets = pd.Series(strat.analyzers.all_returns.get_analysis())
            total_returns.append(rets)
            
            win_rate = (len(strat.wins) / (len(strat.wins) + len(strat.losses)) * 100) if (len(strat.wins) + len(strat.losses)) > 0 else 0
            
            results_list.append({
                'Ticker': target,
                'Return': f"{(rets+1).prod()-1:.2%}",
                'Win_Rate': f"{win_rate:.1f}%",
                'Trades': len(strat.wins) + len(strat.losses),
                'Status': '보유' if strat.position else '현금'
            })
            progress_bar.progress(i / num_stocks)
            time.sleep(1)

        # 결과 시각화
        st.subheader("📊 백테스트 통계")
        res_df = pd.DataFrame(results_list)
        st.dataframe(res_df, use_container_width=True)
        
        if total_returns:
            st.subheader("📈 포트폴리오 누적 수익률 (평균)")
            all_rets = pd.concat(total_returns, axis=1).mean(axis=1)
            st.line_chart((1 + all_rets).cumprod())

# --- Tab 2: 종목별 정밀 분석 ---
with tab2:
    st.subheader("종목별 상세 시각화")
    
    # 1. 입력 수단 배치 (1:1 비율)
    c1, c2, c3 = st.columns([1.5,1.5,7])
    
    with c1:
        ticker_options = ["선택하세요"] + corp_df['Ticker'].head(num_stocks).tolist()
        selected_from_list = st.selectbox("리스트에서 선택", options=ticker_options, index=0)
    
    with c2:
        manual_ticker = st.text_input("Ticker 입력", value="").strip().upper()

    # 2. 최종 티커 결정 (우선순위: 직접 입력 > 드롭다운)
    selected_ticker = None
    if manual_ticker:
        selected_ticker = manual_ticker
    elif selected_from_list != "선택하세요":
        selected_ticker = selected_from_list

# tab2 내부 분석 실행 버튼 클릭 시 로직
if selected_ticker:
    # 시장 데이터(Ticker_mkt) 찾기
    # 직접 입력한 경우 corp_df에 없을 수 있으므로 기본값 처리
    matching_row = corp_df[corp_df['Ticker'] == selected_ticker]
    
    if not matching_row.empty:
        selected_mkt = matching_row['Ticker_mkt'].values[0]
        selected_name = matching_row['Name'].values[0]
    else:
        # 직접 입력한 티커가 리스트에 없는 경우 (예: 해외주식 등)
        selected_mkt = "^IXIC"  # 기본 벤치마크 (나스닥) 혹은 사용자가 입력하게 할 수도 있음
        selected_name = selected_ticker
            
    with st.spinner(f"{selected_ticker} 상세 분석 및 차트 생성 중..."):
        # 1. 데이터 로드 및 전처리 (MultiIndex 방지)
        df_s = yf.download(selected_ticker, start=start_date, auto_adjust=True, progress=False)
        df_b = yf.download(selected_mkt, start=start_date, auto_adjust=True, progress=False)
        
        # 2. [중요] 멀티인덱스 컬럼 해결 (튜플 오류 방지)
        if isinstance(df_s.columns, pd.MultiIndex):
            df_s.columns = df_s.columns.droplevel(1)
        if isinstance(df_b.columns, pd.MultiIndex):
            df_b.columns = df_b.columns.droplevel(1)

        # 2. Backtrader 실행부
        cerebro = bt.Cerebro()
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='all_returns')
        
        d0 = bt.feeds.PandasData(dataname=df_s)
        cerebro.adddata(d0)
        cerebro.resampledata(d0, timeframe=bt.TimeFrame.Weeks, compression=1)
        cerebro.adddata(bt.feeds.PandasData(dataname=df_b))
        
        cerebro.addstrategy(SixFactorStrategy)
        cerebro.broker.setcash(cash)
        results = cerebro.run()
        strat = results[0]

        # --- 3. 데이터 가공 (요청하신 로직 적용) ---
        trades = strat.order_history
        score_df = pd.DataFrame(strat.score_history)
        if not score_df.empty:
            score_df['Date'] = pd.to_datetime(score_df['Date'])
            score_df.set_index('Date', inplace=True)

        strat_returns = pd.Series(strat.analyzers.all_returns.get_analysis())
        stock_returns = df_s['Close'].pct_change().reindex(strat_returns.index).fillna(0)

        # 분석 시작일 (첫 데이터 혹은 스코어 발생일)
        ready_date = score_df.index.min() if not score_df.empty else strat_returns.index[0]

        # 데이터 슬라이싱 및 누적 수익률
        strat_returns = strat_returns[ready_date:]
        stock_returns = stock_returns[ready_date:]
        strat_idx = (1 + strat_returns).cumprod()
        stock_idx = (1 + stock_returns).cumprod()
        relative_ratio = strat_idx / stock_idx

        # MDD 계산 (calculate_performance_metrics 함수 활용)
        _, _, s_dd_series = calculate_performance_metrics(strat_returns)
        _, _, b_dd_series = calculate_performance_metrics(stock_returns)

        # --- 4. 4단 차트 레이아웃 설정 ---
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 9), sharex=True, 
                                            gridspec_kw={'height_ratios': [2.5, 1, 1, 1]})

        # [ax1] 상단 차트: 누적 수익률 및 매매 구간 음영
        ax1.plot(strat_idx, label='Strategy (6-Factor)', color='#142F46', lw=2.5, zorder=3)
        ax1.plot(stock_idx, label='Stock (Buy & Hold)', color='#2A9D8F', alpha=0.5, lw=1.5, zorder=2)

        labels_added = set()
        in_position = False
        entry_date, entry_price = None, None
        buy_new, buy_pyramid, sell_pts = [], [], []

        for t in trades:
            date = pd.Timestamp(t[0])
            if date not in strat_idx.index: continue
            price, side = t[1], t[2]
            curr_ret = strat_idx[date]
            
            if side == 'BUY':
                if not in_position:
                    in_position, entry_date, entry_price = True, date, price
                    buy_new.append((date, curr_ret))
                else:
                    buy_pyramid.append((date, curr_ret))
            elif side == 'SELL' and in_position:
                # 수익/손실 음영 구분
                color, label = ('#d4edda', 'Profit Zone') if price > entry_price else ('#f8d7da', 'Loss Zone')
                ax1.axvspan(entry_date, date, color=color, alpha=0.6, label=label if label not in labels_added else "")
                labels_added.add(label)
                sell_pts.append((date, curr_ret))
                in_position = False

        if in_position:
            ax1.axvspan(entry_date, strat_idx.index[-1], color='#fff3cd', alpha=0.7, label='Current Position')

        # 마커 플롯 (신규진입, 불타기, 매도)
        for pts, m, c, l, s in [(buy_new, '^', '#007bff', 'Initial Entry', 150), 
                                (buy_pyramid, '^', '#007bff', 'Pyramiding', 100), 
                                (sell_pts, 'v', '#dc3545', 'Exit', 120)]:
            if pts:
                d_pts, r_pts = zip(*pts)
                ax1.scatter(d_pts, r_pts, marker=m, s=s, c=c, edgecolors='black', label=l, zorder=5)

        ax1.set_title(f'Comprehensive Strategy Analysis: {selected_ticker}', fontsize=16, fontweight='bold')
        ax1.legend(loc='upper left', ncol=2, fontsize=10)
        ax1.grid(True, alpha=0.2)

        # [ax2] 상대 강도 (Strategy / Stock)
        ax2.plot(relative_ratio, color=color_map[0], lw=1.5, label='Relative Strength')
        ax2.axhline(1, color='k', lw=0.8, ls='--')
        ax2.fill_between(relative_ratio.index, relative_ratio, 1, where=(relative_ratio >= 1), color=color_map[1], alpha=0.4)
        ax2.set_ylabel('Rel. Ratio')
        ax2.grid(True, alpha=0.2)

        # [ax3] Six-Factor Score 추이
        if not score_df.empty:
            ax3.plot(score_df.index, score_df['Score'], color='#D9410A', lw=1.2, label='Factor Score')
            ax3.axhline(7, color='red', lw=1, ls=':', label='Buy Threshold')
            ax3.fill_between(score_df.index, score_df['Score'], 7, where=(score_df['Score'] >= 7), color='#D9410A', alpha=0.2)
        ax3.set_ylabel('Score')
        ax3.set_ylim(-0.5, 10.5)
        ax3.grid(True, alpha=0.2)

        # [ax4] Drawdown (MDD) 비교
        ax4.fill_between(s_dd_series.index, s_dd_series, 0, color=color_map[3], alpha=0.3, label='Strat DD')
        ax4.fill_between(b_dd_series.index, b_dd_series, 0, color='gray', alpha=0.6, label='Stock DD')
        ax4.set_ylabel('Drawdown')
        ax4.grid(True, alpha=0.2)
        ax4.legend(loc='lower left', fontsize=8)

        # plt.tight_layout()
        st.pyplot(fig)

        # --- 성과 지표 요약 (m1~m4) ---
        st.markdown("---")
        st.subheader(f"🔍 {selected_ticker} 전략 성과 요약")
        
        # 데이터 계산
        final_value = cerebro.broker.getvalue()  # 현재 평가액
        total_strat_ret = strat_idx[-1] - 1      # 전략 최종 수익률
        total_stock_ret = stock_idx[-1] - 1      # 종목 단순보유 수익률
        alpha = total_strat_ret - total_stock_ret # 알파 수익률
        
        win_count = len(strat.wins)
        loss_count = len(strat.losses)
        total_trades = win_count + loss_count
        
        # 승률 및 손익비(Profit Factor) 계산
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        total_win_amt = sum(strat.wins)
        total_loss_amt = sum(strat.losses)
        profit_factor = total_win_amt / total_loss_amt if total_loss_amt > 0 else (99.9 if total_win_amt > 0 else 0.0)

        # 메트릭 출력
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.metric(
                label="최종 수익률 / 알파", 
                value=f"{total_strat_ret:.2%}", 
                delta=f"{alpha:.2%} vs Bench",
                delta_color="normal"
            )
            st.caption(f"벤치마크 수익률: {total_stock_ret:.2%}")

        with m2:
            st.metric(
                label="현재 평가액", 
                value=f"${final_value:,.0f}",
                delta=f"${final_value - cash:,.0f} (손익)",
                delta_color="normal"
            )
            st.caption(f"초기 자본: ${cash:,.0f}")

        with m3:
            st.metric(
                label="승률 / 매매 횟수", 
                value=f"{win_rate:.1f}%", 
                delta=f"{total_trades}회 매매"
            )
            st.caption(f"승: {win_count} / 패: {loss_count}")

        with m4:
            st.metric(
                label="손익비 (PnL) / MDD", 
                value=f"{profit_factor:.2f}", 
                delta=f"{s_dd_series.min():.2%} MDD",
                delta_color="inverse"
            )
            st.caption("Profit Factor 2.5 이상 권장")
            
        # --- 상세 매매 내역 (Trade Log) 개선 버전 ---
        st.markdown("---")
        st.subheader(f"📜 {selected_ticker} 매매 내역 상세")

        trade_log = []
        in_pos = False
        temp_entry_date = None
        temp_entry_price = 0

        for t in trades:
            date, price, side, reason = pd.Timestamp(t[0]), t[1], t[2], t[3]
            
            if side == 'BUY':
                if not in_pos: # [1] 신규 매수 발생
                    temp_entry_date = date
                    temp_entry_price = price
                    in_pos = True
                    trade_log.append({
                        '날짜': date.strftime('%Y-%m-%d'),
                        '구분': '🔵 신규매수',
                        '가격': f"${price:,.0f}",
                        '사유': reason,
                        '수익률': "-",
                        '보유기간': "-"
                    })
                else: # [2] 추가 매수(불타기) 발생
                    trade_log.append({
                        '날짜': date.strftime('%Y-%m-%d'),
                        '구분': '➕ 추가매수',
                        '가격': f"${price:,.0f}",
                        '사유': reason,
                        '수익률': "-",
                        '보유기간': "-"
                    })
                    
            elif side == 'SELL' and in_pos: # [3] 매도(청산) 발생
                profit_pct = (price / temp_entry_price) - 1
                holding_days = (date - temp_entry_date).days
                
                trade_log.append({
                    '날짜': date.strftime('%Y-%m-%d'),
                    '구분': '🔴 전량매도',
                    '가격': f"${price:,.0f}",
                    '사유': reason,
                    '수익률': f"{profit_pct:+.2%}",
                    '보유기간': f"{holding_days}일"
                })
                in_pos = False

        # [4] 현재 보유 중인 경우 마지막 평가 상태 추가
        if in_pos:
            last_price = df_s['Close'].iloc[-1]
            last_date = pd.Timestamp(df_s.index[-1])
            profit_pct = (last_price / temp_entry_price) - 1
            holding_days = (last_date - temp_entry_date).days
            
            trade_log.append({
                '날짜': last_date.strftime('%Y-%m-%d'),
                '구분': '🟡 보유중(평가)',
                '가격': f"${last_price:,.0f}",
                '사유': "-",
                '수익률': f"{profit_pct:+.2%}",
                '보유기간': f"{holding_days}일"
            })

        # 데이터프레임 변환 및 출력
        if trade_log:
            # 최신 날짜가 위로 오게 하려면 .iloc[::-1] 사용 가능
            log_df = pd.DataFrame(trade_log)
            
            def color_status(val):
                if '신규매수' in val: return 'background-color: #e7f3ff'
                if '전량매도' in val: return 'background-color: #fff0f0'
                return ''

            def color_returns(val):
                if isinstance(val, str) and '+' in val: return 'color: blue; font-weight: bold'
                if isinstance(val, str) and '-' in val: return 'color: red; font-weight: bold'
                return ''

            styled_log = log_df.style.applymap(color_status, subset=['구분'])\
                                    .applymap(color_returns, subset=['수익률'])
                                    
            # 컬럼 순서 조정
            log_df = log_df[['날짜', '구분', '가격', '사유', '수익률', '보유기간']]
            st.dataframe(styled_log, use_container_width=True, hide_index=True)
        else:
            st.info("해당 기간 내 발생한 매매 내역이 없습니다.")
