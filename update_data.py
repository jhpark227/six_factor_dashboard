import pickle
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

FILE_NAME = 'corp_price_data_v2.pkl'
START_DATE = '2020-01-01'

def update_or_add_ticker(input_tickers=None):
    """
    input_tickers: 리스트 형태 (예: ['NVDA', 'AAPL'])
    None이면 기존 파일에 있는 모든 티커를 업데이트함.
    """
    # 1. 기존 Pickle 로드
    if os.path.exists(FILE_NAME):
        try:
            with open(FILE_NAME, 'rb') as f:
                stock_dict = pickle.load(f)
            print(f"📖 기존 데이터 로드 완료: {len(stock_dict)}개 종목")
        except Exception as e:
            print(f"⚠️ 데이터 로드 오류: {e}")
            stock_dict = {}
    else:
        print("⚠️ 기존 데이터 파일이 없습니다. 새로 생성을 시도합니다.")
        stock_dict = {}

    # 입력이 없으면 기존 키(티커) 전체 업데이트
    if input_tickers is None:
        input_tickers = list(stock_dict.keys())
        print(f"🔄 전체 종목 자동 업데이트 모드: {len(input_tickers)}개 코드를 확인합니다.")

    updated_count = 0
    today = datetime.now().date()
    
    for ticker in input_tickers:
        ticker = ticker.upper().strip()
        
        if ticker in stock_dict:
            # [기존 티커]
            df_old = stock_dict[ticker]
            if df_old.empty:
                last_date_obj = datetime.strptime(START_DATE, '%Y-%m-%d').date()
            else:
                last_date_obj = df_old.index.max().date()
            
            # 마지막 데이터가 어제보다 이전이면 업데이트 (오늘 날짜 데이터는 장중일 수 있으므로 상황에 따라 다름, 여기선 안전하게 +1일 부터 조회)
            # yfinance는 start date가 inclusive이므로, 마지막 날짜 다음날부터 요청해야 함
            if last_date_obj < today - timedelta(days=1):
                fetch_start = (last_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
                print(f"🔄 {ticker}: 업데이트 필요 ({fetch_start} ~)")
                
                try:
                    new_data = yf.download(ticker, start=fetch_start, auto_adjust=True, progress=False)
                    
                    if not new_data.empty:
                        if isinstance(new_data.columns, pd.MultiIndex):
                            new_data.columns = new_data.columns.droplevel(1)
                        new_data['Ticker'] = ticker
                        
                        df_updated = pd.concat([df_old, new_data])
                        df_updated = df_updated[~df_updated.index.duplicated(keep='last')]
                        stock_dict[ticker] = df_updated
                        updated_count += 1
                        print(f"   -> 추가 완료 ({len(new_data)}일 데이터)")
                    else:
                        print(f"   -> 새로운 데이터 없음")
                except Exception as e:
                    print(f"   ❌ {ticker} 수집 중 에러: {e}")
            else:
                # print(f"✅ {ticker}: 이미 최신 상태")
                pass
                
        else:
            # [신규 티커]
            print(f"🆕 {ticker}: 신규 수집 시작 ({START_DATE} ~)")
            try:
                new_df = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)
                
                if not new_df.empty:
                    if isinstance(new_df.columns, pd.MultiIndex):
                        new_df.columns = new_df.columns.droplevel(1)
                    new_df['Ticker'] = ticker
                    stock_dict[ticker] = new_df
                    updated_count += 1
                    print(f"   -> 신규 저장 완료")
                else:
                    print(f"   ❌ 데이터를 찾을 수 없음")
            except Exception as e:
                print(f"   ❌ {ticker} 수집 중 에러: {e}")

    # 2. 결과 저장 (업데이트된 건이 있을 때만 저장하여 불필요한 IO/커밋 방지 가능하지만, 안전을 위해 저장)
    if updated_count > 0:
        with open(FILE_NAME, 'wb') as f:
            pickle.dump(stock_dict, f)
        print(f"\n💾 {updated_count}개 종목 업데이트 완료 및 저장됨: {FILE_NAME}")
    else:
        print("\n✅ 변경된 데이터가 없어 저장하지 않았습니다.")

    return stock_dict

if __name__ == "__main__":
    # GitHub Action이나 로컬 실행 시 인자 없이 실행하면 전체 업데이트
    update_or_add_ticker()