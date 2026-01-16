import yfinance as yf
import pandas as pd
import time
import pickle
from datetime import datetime


corp_df = pd.read_csv('corp_data.csv', header=0, encoding='cp949')
corp_df = corp_df.sort_values(by='Market_Cap', ascending=False, ignore_index=True).head(30)

tickers = corp_df['Ticker'].tolist()

start_date = '2020-01-01'
FILE_NAME = 'corp_price_data.pkl'

def initialize_stock_db(ticker_list, start):
    """
    N개 종목의 데이터를 받아 딕셔너리로 관리하고 하나의 CSV로 저장하는 함수
    """
    stock_dict = {}
    all_data_frames = []
    
    print(f"🚀 {len(ticker_list)}개 종목 초기 데이터 수집 시작...")

    for ticker in ticker_list:
        try:
            print(f"# {ticker} 다운로드 중...")
            # auto_adjust=True: 배정/분할이 반영된 수정주가 사용
            df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
            
            if df.empty:
                print(f"# {ticker} 데이터를 찾을 수 없습니다.")
                continue

            # MultiIndex 정리 (Ticker명이 컬럼에 남는 현상 방지)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # 딕셔너리에 저장 (Key: Ticker, Value: DataFrame)
            # 나중에 꺼내 쓰기 편하도록 Ticker 컬럼 삽입
            # df['Ticker'] = ticker
            stock_dict[ticker] = df
            all_data_frames.append(df)
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ {ticker} 처리 중 오류 발생: {e}")

    # 2. Pickle 파일로 저장
    with open(FILE_NAME, 'wb') as f:
        pickle.dump(stock_dict, f)
    
    print(f"✅ 저장 완료: {FILE_NAME}")
    return stock_dict
    

# 실행
master_dict = initialize_stock_db(tickers, start_date)
