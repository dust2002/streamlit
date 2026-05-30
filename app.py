
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ----------------------------------
# 페이지 설정
# ----------------------------------

st.set_page_config(
    page_title="AI 주식 분석기",
    page_icon="📈"
)

st.header("📈 AI 주식 분석기")
st.markdown("종목명 또는 티커를 입력하면 최근 1주일 주가 데이터와 AI 투자 의견을 제공합니다.")

# ----------------------------------
# GPT 모델
# ----------------------------------

def get_llm(temp=0):
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temp
    )

# ----------------------------------
# 종목명 -> 티커 변환 프롬프트
# ----------------------------------

TICKER_PROMPT = """
당신은 Yahoo Finance 티커 전문가입니다.

사용자가 입력한 기업명 또는 티커를 Yahoo Finance 티커로 변환하세요.

규칙:
1. 티커만 출력
2. 설명 금지
3. 삼성전자 -> 005930.KS
4. SK하이닉스 -> 000660.KS
5. 애플 -> AAPL
6. 엔비디아 -> NVDA
7. 테슬라 -> TSLA

입력:
{company}
"""

# ----------------------------------
# 투자 분석 프롬프트
# ----------------------------------

ANALYSIS_PROMPT = """
당신은 금융 데이터 분석 전문가입니다.

주식명:
{stock_name}

최근 1주일 데이터:
{data}

다음 형식으로 답변하세요.

### 최근 주가 흐름
최근 가격 변동 요약

### 거래량 분석
거래량 특징 설명

### 긍정적 요소
상승 가능 요인

### 부정적 요소
하락 위험 요인

### AI 투자 참고 의견
매수 / 보유 / 매도 중 하나 선택 + 이유 설명

마지막에 반드시:
"본 의견은 투자 권유가 아닌 참고용 분석입니다."
"""

# ----------------------------------
# 티커 변환 함수
# ----------------------------------

def company_to_ticker(company):

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(TICKER_PROMPT)

    chain = prompt | llm | StrOutputParser()

    ticker = chain.invoke({"company": company})

    return ticker.strip()

# ----------------------------------
# 분석 체인
# ----------------------------------

def get_analysis_chain():

    llm = get_llm(0.2)

    prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)

    chain = prompt | llm | StrOutputParser()

    return chain

# ----------------------------------
# 주가 데이터 가져오기
# ----------------------------------

def get_stock_data(ticker):

    stock = yf.Ticker(ticker)
    hist = stock.history(period="7d")

    if hist.empty:
        return None, None

    try:
        stock_name = stock.info.get("longName", ticker)
    except:
        stock_name = ticker

    return stock_name, hist

# ----------------------------------
# 메인
# ----------------------------------

def main():

    user_input = st.text_input(
        "종목명 또는 티커 입력",
        placeholder="예: 삼성전자, 애플, 엔비디아, Tesla, AAPL"
    )

    if st.button("분석하기"):

        if not user_input:
            st.warning("종목명을 입력하세요.")
            return

        # 1. 티커 변환
        with st.spinner("종목 검색 중..."):
            ticker = company_to_ticker(user_input)

        st.success(f"조회 티커: {ticker}")

        # 2. 주가 데이터
        with st.spinner("주가 데이터 수집 중..."):
            stock_name, hist = get_stock_data(ticker)

        if hist is None:
            st.error("주가 데이터를 찾을 수 없습니다.")
            return

        st.subheader("최근 1주일 거래 데이터")
        st.dataframe(hist[["Open", "High", "Low", "Close", "Volume"]])

        st.subheader("종가 추이")
        st.line_chart(hist["Close"])

        data_text = hist[["Open", "High", "Low", "Close", "Volume"]].to_string()

        # 3. AI 분석
        chain = get_analysis_chain()

        with st.spinner("AI 분석 중..."):
            result = chain.invoke({
                "stock_name": stock_name,
                "data": data_text
            })

        st.subheader("🤖 AI 투자 의견")
        st.write(result)

        st.info("본 서비스는 교육용 프로젝트이며 투자 권유가 아닙니다.")

# 실행
if __name__ == "__main__":
    main()
