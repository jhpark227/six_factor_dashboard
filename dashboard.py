import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import backtrader as bt
import warnings
import os
import pickle
from datetime import datetime, timedelta

yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
today = datetime.now().strftime('%Y-%m-%d')

# 기초 설정
warnings.filterwarnings(action='ignore')
st.set_page_config(page_title="Alpha Mosaic", layout="wide")

# --- 설정 및 데이터 로드 ---
color_map = ['#142F46','#2A9D8F','#E9AB0C','#D9410A','#E76F51']

# 1. 시총 순위 등 기초 정보 로드
@st.cache_data
def load_corp_data():
    corp_df = pd.read_csv('corp_data.csv', header=0, encoding='cp949')
    corp_df = corp_df.sort_values(by='Market_Cap', ascending=False, ignore_index=True)
    return corp_df

def get_data_and_update_pkl(ticker, start_date='2020-01-01'):
    FILE_NAME = 'corp_price_data_v2.pkl'
    today = datetime.now().date()
    
    # 1. 기존 pkl 파일 로드

    if os.path.exists(FILE_NAME):
        try:
            with open(FILE_NAME, 'rb') as f:
                market_db = pickle.load(f)
        except Exception as e:
            print(f"⚠️ 데이터 로드 오류 발생 (이전 버전 호환성 등): {e}")
            print(" -> 기존 데이터를 무시하고 새로 수집합니다.")
            market_db = {}
    else:
        market_db = {}

    # 2. 데이터 업데이트 및 다운로드 로직
    if ticker in market_db:
        df_old = market_db[ticker]
        last_date = df_old.index.max().date()
        
        # 마지막 날짜가 어제 이전인 경우에만 업데이트 (오늘 데이터는 보통 장 마감 후 생성되므로 안전하게 처리)
        if last_date < today - timedelta(days=1):
            fetch_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"🔄 {ticker}: 기존 데이터 연장 중... ({fetch_start} ~)")
            
            df_new = yf.download(ticker, start=fetch_start, auto_adjust=True, progress=False)
            
            if not df_new.empty:
                if isinstance(df_new.columns, pd.MultiIndex):
                    df_new.columns = df_new.columns.droplevel(1)
                
                # 기존 데이터 아래에 새 데이터 결합 및 중복 제거
                df_updated = pd.concat([df_old, df_new])
                df_updated = df_updated[~df_updated.index.duplicated(keep='last')]
                market_db[ticker] = df_updated
                
                # 파일 저장
                with open(FILE_NAME, 'wb') as f:
                    pickle.dump(market_db, f)
                print(f"💾 {ticker}: 업데이트 완료.")
        else:
            print(f"✅ {ticker}: 이미 최신 상태입니다.")
    
    else:
        # [신규 종목] 전체 데이터 다운로드
        print(f"🆕 {ticker}: 신규 종목 감지! 전체 수집 중...")
        df_full = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
        
        if not df_full.empty:
            if isinstance(df_full.columns, pd.MultiIndex):
                df_full.columns = df_full.columns.droplevel(1)
            
            market_db[ticker] = df_full
            with open(FILE_NAME, 'wb') as f:
                pickle.dump(market_db, f)
            print(f"💾 {ticker}: 신규 저장 완료.")

    return market_db.get(ticker, pd.DataFrame())

corp_df = load_corp_data()

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
st.title("📈 Alpha Mosaic")
tab1, tab2 = st.tabs(["🚀 전체 시장 요약", "🔍 종목별 정밀 분석"])

# 공통 사이드바
st.sidebar.header("Configuration")
num_stocks = st.sidebar.slider("분석 종목 수 (시총 순)", 1, 100, 3)
start_date = st.sidebar.date_input("분석 시작일", datetime(2020, 1, 1))
cash = st.sidebar.number_input("초기 자산", value=10000)

corp_df = load_corp_data()

# --- Tab 1: 전체 시장 요약 ---
with tab1:
    # 1. session_state 초기화 (결과 리스트가 없으면 생성)
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    if 'total_returns_data' not in st.session_state:
        st.session_state.total_returns_data = None
        
    if st.button("전체 종목 백테스트 실행"):
        target_list = corp_df[['Ticker','Ticker_mkt', 'Name']].head(num_stocks)
        results_list = []
        total_returns = []
        
        progress_bar = st.progress(0)
        status_text = st.empty() # 현재 진행 종목 표시용
        warning_placeholder = st.empty() # 경고 메시지 표시용(한 줄 갱신)

        for i, (target, target_market, name) in enumerate(target_list.values, start=1):
            status_text.text(f"🚀 분석 중: {name} ({target})")
            
            df_stock = get_data_and_update_pkl(target)
            df_bench = get_data_and_update_pkl(target_market)
            
            if df_stock.empty:
                continue
            
            # --- [최신일자 비교 로직] ---
            last_data_date = df_stock.index[-1].strftime('%Y-%m-%d')
            
            # 마지막 날짜가 오늘이나 어제가 아니라면 (즉, 데이터 업데이트가 늦어졌다면) 출력
            if last_data_date not in [today, yesterday]:
                warning_placeholder.warning(f"⚠️ **{name}({target})** 데이터가 최신이 아닙니다. (마지막 데이터: `{last_data_date}`)")
            
            if len(df_stock) < 50: 
                warning_placeholder.warning(f"⚠️ {target} 데이터 부족으로 건너뜁니다.")
                continue
            
            # --- [Backtrader 엔진 실행] ---
            cerebro = bt.Cerebro()
            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='all_returns')
            
            d0 = bt.feeds.PandasData(dataname=df_stock)
            cerebro.adddata(d0)
            cerebro.resampledata(d0, timeframe=bt.TimeFrame.Weeks, compression=1)
            cerebro.adddata(bt.feeds.PandasData(dataname=df_bench))
            
            cerebro.addstrategy(SixFactorStrategy)
            cerebro.broker.setcash(cash)
            cerebro.broker.setcommission(commission=0.001)
            cerebro.broker.set_coc(True)

            try:
                results = cerebro.run()
                strat = results[0]
                
                rets = pd.Series(strat.analyzers.all_returns.get_analysis())
                total_returns.append(rets)
                
                current_profit = 0
                if strat.position:
                    current_profit = (strat.daily_stock.close[0] - strat.position.price) / strat.position.price
                    unrealized_pnl = (strat.daily_stock.close[0] - strat.position.price) * strat.position.size
                    unrealized_pnl -= (strat.daily_stock.close[0] * strat.position.size * 0.001) 
                    if unrealized_pnl > 0:
                        strat.wins.append(unrealized_pnl)
                    elif unrealized_pnl < 0:
                        strat.losses.append(abs(unrealized_pnl))

                trades_count = len(strat.wins) + len(strat.losses)
                win_rate = (len(strat.wins) / trades_count * 100) if trades_count > 0 else 0
                pf = sum(strat.wins) / sum(strat.losses) if sum(strat.losses) > 0 else 9.9
                
                results_list.append({
                    'First Buy Date': strat.first_buy_date,
                    'Second Buy Date': strat.last_buy_date,
                    'Ticker': target,
                    'Name': name,
                    'Cum Ret': f"{(rets+1).prod()-1:.2%}",
                    'Current Profit': f"{current_profit:.2%}",
                    'Win_Rate': f"{win_rate:.1f}%",
                    'PL Ratio':f"{pf:.2f}",
                    'Trades': trades_count,
                    'Status': '보유' if strat.position else '-'
                })
            except Exception as e:
                st.error(f"{target} 백테스트 에러: {e}")

            progress_bar.progress(i / num_stocks)
            
        # 🔥 중요: 결과를 session_state에 저장
        st.session_state.backtest_results = pd.DataFrame(results_list)
        st.session_state.total_returns_data = total_returns
        status_text.success("✅ 모든 백테스트가 완료되었습니다.")

        # 2. 버튼 클릭과 상관없이 데이터가 있으면 화면에 출력
    if st.session_state.backtest_results is not None:
        st.subheader("📊 백테스트 통계 요약")
        
        res_df = st.session_state.backtest_results
        
        col1, col2 = st.columns([9,1])
        col2.download_button('CSV 다운로드', 
                             data=res_df.to_csv().encode('utf-8'), 
                             file_name='backtest_results.csv')
        
        st.dataframe(res_df, use_container_width=True)
        
        if st.session_state.total_returns_data:
            st.subheader("📈 포트폴리오 통합 수익률")
            all_rets = pd.concat(st.session_state.total_returns_data, axis=1).mean(axis=1)
            st.line_chart((1 + all_rets).cumprod())

# --- Tab 2: 종목별 정밀 분석 ---
with tab2:
    st.subheader("📊 전략 상세 분석 리포트")
    
    c1, _ = st.columns([1.5, 8.5]) 
    with c1:
        manual_ticker = st.text_input("Ticker 직접 입력", value="").strip().upper()
    selected_ticker = manual_ticker if manual_ticker else None

    if selected_ticker:
        matching_row = corp_df[corp_df['Ticker'] == selected_ticker]
        selected_mkt = matching_row['Ticker_mkt'].values[0] if not matching_row.empty else "^IXIC"
        selected_name = matching_row['Name'].values[0] if not matching_row.empty else selected_ticker

        with st.spinner(f"[{selected_name}] 전문 분석 데이터 생성 중..."):
            df_s = get_data_and_update_pkl(selected_ticker, start_date=start_date)
            df_b = get_data_and_update_pkl(selected_mkt, start_date=start_date)

            if not df_s.empty and not df_b.empty:
                last_data_date = df_s.index[-1].strftime('%Y-%m-%d')
                st.markdown(f"🗓️ **데이터 분석 기준일:** `{last_data_date}`")
                
                cerebro = bt.Cerebro()
                cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='all_returns')
                d0 = bt.feeds.PandasData(dataname=df_s); cerebro.adddata(d0)
                cerebro.resampledata(d0, timeframe=bt.TimeFrame.Weeks, compression=1)
                cerebro.adddata(bt.feeds.PandasData(dataname=df_b))
                cerebro.addstrategy(SixFactorStrategy)
                
                cerebro.broker.setcash(cash)
                cerebro.broker.setcommission(commission=0.001) # Tab 1과 일치
                cerebro.broker.set_coc(True)
                
                results = cerebro.run(); 
                strat = results[0]
                
                # Tab 1과 동일하게 미실현 손익 반영
                if strat.position:
                    current_price = df_s['Close'].iloc[-1]
                    unrealized_pnl = (current_price - strat.position.price) * strat.position.size
                    unrealized_pnl -= (current_price * strat.position.size * 0.001) 
                    if unrealized_pnl > 0:
                        strat.wins.append(unrealized_pnl)
                    elif unrealized_pnl < 0:
                        strat.losses.append(abs(unrealized_pnl))

                # --- [데이터 정제] ---
                trades = strat.order_history
                score_df = pd.DataFrame(strat.score_history)
                if not score_df.empty:
                    score_df['Date'] = pd.to_datetime(score_df['Date'])
                    score_df.set_index('Date', inplace=True)

                strat_returns = pd.Series(strat.analyzers.all_returns.get_analysis())
                stock_returns = df_s['Close'].pct_change().reindex(strat_returns.index).fillna(0)
                
                # 수치 계산은 Tab 1과 동일하게 전체 기간 기준
                win_c, loss_c = len(strat.wins), len(strat.losses)
                win_r = (win_c / (win_c + loss_c) * 100) if (win_c + loss_c) > 0 else 0
                pf = sum(strat.wins) / sum(strat.losses) if sum(strat.losses) > 0 else 9.9
                
                # --- [차트용 데이터: ready_date 기준 슬라이싱] ---
                ready_date = score_df.index.min() if not score_df.empty else strat_returns.index[0]
                
                strat_returns_clipped = strat_returns[ready_date:]
                stock_returns_clipped = stock_returns[ready_date:]
                
                strat_idx = (1 + strat_returns_clipped).cumprod()
                stock_idx = (1 + stock_returns_clipped).cumprod()
                relative_ratio = strat_idx / stock_idx
                
                _, _, s_dd = calculate_performance_metrics(strat_returns_clipped)
                _, _, b_dd = calculate_performance_metrics(stock_returns_clipped)

                # --- [1. 성과 요약 메트릭] ---
                st.markdown("---")
                final_val = cerebro.broker.getvalue()
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("최종 수익률", f"{(strat_idx[-1]-1):.1%}", f"{(strat_idx[-1]-stock_idx[-1]):+.1%} vs Stock")
                m2.metric("현재 자산", f"${final_val:,.0f}", f"${(final_val-cash):+,.0f}")
                m3.metric("승률 / 매매", f"{win_r:.1f}%", f"{win_c+loss_c}회")
                m4.metric("손익비", f"{pf:.2f}", f"MDD:{s_dd.min():.1%}", delta_color="inverse")

                # --- [2. 상세 매매 내역] ---
                st.markdown("---")
                st.subheader("📜 Transaction Log")
                trade_log = []
                in_pos = False
                temp_entry_date = None
                temp_entry_price = 0

                for t in trades:
                    date, price, side, reason = pd.Timestamp(t[0]), t[1], t[2], t[3]
                    if side == 'BUY':
                        if not in_pos:
                            temp_entry_date = date
                            temp_entry_price = price
                            in_pos = True
                            trade_log.append({'날짜': date.strftime('%Y-%m-%d'), '구분': '🔵 Buy', '가격': f"${price:,.1f}", '사유': reason, '수익률': "-", '보유기간': "-"})
                        else:
                            trade_log.append({'날짜': date.strftime('%Y-%m-%d'), '구분': '➕ Pyramiding', '가격': f"${price:,.1f}", '사유': reason, '수익률': "-", '보유기간': "-"})
                    elif side == 'SELL' and in_pos:
                        profit_pct = (price / temp_entry_price) - 1
                        holding_days = (date - temp_entry_date).days
                        trade_log.append({'날짜': date.strftime('%Y-%m-%d'), '구분': '🔴 Sell', '가격': f"${price:,.1f}", '사유': reason, '수익률': f"{profit_pct:+.2%}", '보유기간': f"{holding_days}일"})
                        in_pos = False

                if in_pos:
                    last_price = df_s['Close'].iloc[-1]
                    last_date = pd.Timestamp(df_s.index[-1])
                    current_profit_pct = (last_price / temp_entry_price) - 1
                    current_holding_days = (last_date - temp_entry_date).days
                    trade_log.append({'날짜': last_date.strftime('%Y-%m-%d'), '구분': '🟡 보유 중', '가격': f"${last_price:,.1f}", '사유': "현재 보유 중인 포지션", '수익률': f"{current_profit_pct:+.2%}", '보유기간': f"{current_holding_days}일"})

                if trade_log:
                    log_df = pd.DataFrame(trade_log).iloc[::-1] 
                    def highlight_returns(val):
                        if isinstance(val, str) and '+' in val: return 'color: blue; font-weight: bold'
                        if isinstance(val, str) and '-' in val: return 'color: red; font-weight: bold'
                        return ''
                    styled_log = log_df.style.applymap(highlight_returns, subset=['수익률'])
                    st.dataframe(styled_log, use_container_width=True, hide_index=True)
                else:
                    st.info("기록된 매매 내역이 없습니다.")

                # --- [3. 프로페셔널 차트 (Matplotlib)] ---
                st.markdown("---")
                st.subheader(f"📈 Performance Analysis - {selected_ticker}")
                
                plt.style.use('seaborn-v0_8-whitegrid')
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8), sharex=True, 
                                                    gridspec_kw={'height_ratios': [3, 1, 1, 1]})
                plt.subplots_adjust(hspace=0.15)

                # [Ax1] 메인 수익률 차트 (Clipped)
                ax1.plot(strat_idx, label='Strategy (6-Factor)', color='#1B263B', lw=2, zorder=2)
                ax1.plot(stock_idx, label='Benchmark (Buy&Hold)', color='#BDC3C7', lw=1)
                
                labels = set()
                in_pos, ent_d, ent_p = False, None, 0
                for t in trades:
                    d = pd.Timestamp(t[0])
                    if d < ready_date: continue
                    p, s = t[1], t[2]
                    
                    if s == 'BUY' and not in_pos: 
                        ent_d, ent_p, in_pos = d, p, True
                    elif s == 'SELL' and in_pos:
                        c = '#9BC1BC' if p > ent_p else '#F39C91' 
                        label_name = 'Profit Zone' if p > ent_p else 'Loss Zone'
                        ax1.axvspan(ent_d, d, color=c, alpha=0.5, zorder=1, label=label_name if label_name not in labels else "")
                        labels.add(label_name)
                        ax1.scatter(d, strat_idx[d], marker='v', color='#D90429', s=100, zorder=6, label='Exit' if 'Exit' not in labels else "")
                        labels.add('Exit')
                        in_pos = False
                    
                    if s == 'BUY': 
                        ax1.scatter(d, strat_idx[d], marker='^', color='#0077B6', s=100, zorder=6, label='Entry' if 'Entry' not in labels else "")
                        labels.add('Entry')

                if in_pos:
                    ax1.axvspan(ent_d, strat_idx.index[-1], color='#FEF3C7', alpha=0.8, zorder=1, label='Current Position' if 'Current Position' not in labels else "")
                    labels.add('Current Position')

                ax1.set_ylabel('Cumulative Return', fontweight='bold')
                ax1.legend(
                            loc='upper left', 
                            bbox_to_anchor=(0.0, 1.07),
                            ncol=7, 
                            borderaxespad=0,
                            frameon=False, 
                            facecolor='white', 
                            fontsize=8,        # 범례 글자 크기 조절
                            markerscale=0.5    # 🔥 범례 내 마커 크기만 0.6배로 축소
                        )

                # [Ax2] 상대 강도
                ax2.plot(relative_ratio, color='#023E8A', lw=1.2)
                ax2.axhline(1, color='black', lw=0.8, ls='--')
                ax2.set_ylabel('Rel. Strength', fontweight='bold')

                # [Ax3] Factor Score (Clipped)
                if not score_df.empty:
                    s_plot = score_df['Score'][ready_date:]
                    ax3.fill_between(s_plot.index, s_plot, 7, where=(s_plot>=7), color='#2D6A4F', alpha=0.2)
                    ax3.step(s_plot.index, s_plot, where='post', color='#2D6A4F', lw=1)
                    ax3.axhline(7, color='#D90429', lw=1, ls=':', alpha=0.6)
                ax3.set_ylabel('Factor Score', fontweight='bold')
                ax3.set_ylim(-0.5, 10.5)

                # [Ax4] Drawdown (Clipped)
                ax4.fill_between(s_dd.index, s_dd, 0, color='#D90429', alpha=0.3, label='Strategy')
                ax4.plot(s_dd, color='#D90429', lw=1)
                ax4.fill_between(b_dd.index, b_dd, 0, color='#6C757D', alpha=0.3, label='Benchmark')
                ax4.set_ylabel('Drawdown', fontweight='bold')
                ax4.legend(loc='lower left', fontsize=9)

                for ax in [ax1, ax2, ax3, ax4]:
                    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                st.pyplot(fig)
            else:
                st.error("데이터를 불러올 수 없습니다.")
        
