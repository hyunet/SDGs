import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
import datetime
import networkx as nx
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title=" 경기도 음식물 쓰레기 대시보드", layout="wide")

# 파일 경로를 GitHub 기준 상대경로로 수정
file_path = '../data/2022~2024 경기도 일별 지자체 음식물 쓰레기 배출내역.csv'
encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr', 'ISO-8859-1', 'mac_roman']
df = None
for enc in encodings:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        break
    except Exception:
        continue
if df is None:
    st.error("CSV 파일을 읽을 수 없습니다. 파일 인코딩/구분자를 확인해 주세요.")
    st.stop()

# 컬럼명 맞추기
city_col = '기초지자체'
amount_col = '배출량(g)'

# 날짜 컬럼 만들기
df['날짜'] = pd.to_datetime(df['배출연도'].astype(str) + '-' + df['배출월'].astype(str).str.zfill(2) + '-' + df['배출일'].astype(str).str.zfill(2), errors='coerce')
df = df.dropna(subset=['날짜'])
df['연도'] = df['배출연도'].astype(int)
df['월'] = df['배출월']
df['일'] = df['배출일']

# 요일 한글로 변환
요일_한글 = ['월', '화', '수', '목', '금', '토', '일']
df['요일번호'] = df['날짜'].dt.weekday  # 0=월
df['요일'] = df['요일번호'].map({i:요일_한글[i] for i in range(7)})

# g → 톤 변환
df['배출량(톤)'] = df[amount_col] / 1000

# ---------------------- 시각화 ----------------------
st.title('♻️ 경기도 음식물 쓰레기 대시보드')
st.markdown('''#### 📅 요일/월/연도별, 공간별로 음식물 쓰레기 배출 패턴을 한눈에!  
데이터 과학적 시각으로 인사이트를 발굴해보자!''')

# 탭 생성
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    '1️⃣ 연도별 요일별 배출량비율',
    '2️⃣ 월별 배출량',
    '3️⃣ 연도별 추이',
    '4️⃣ 지자체 워드클라우드',
    '5️⃣ 지도+RFID',
    '6️⃣ 전년대비 요일별 배출량비율 차이'
])

with tab1:
    st.subheader('1️⃣ 연도별 요일별 음식물 쓰레기 배출량 비율')
    weekday_year_df = df.groupby(['연도', '요일'])['배출량(톤)'].sum().reset_index()
    weekday_year_df['배출량비율'] = weekday_year_df.groupby('연도')['배출량(톤)'].transform(lambda x: x / x.sum() * 100)
    weekday_year_df['연도'] = weekday_year_df['연도'].astype(str)
    fig = px.bar(
        weekday_year_df,
        x='요일', y='배출량비율', color='연도', barmode='group',
        text_auto='.1f',
        labels={'배출량비율': '배출량비율(%)'},
        title='연도별 요일별 음식물 쓰레기 배출량 비율',
        color_discrete_sequence=px.colors.qualitative.Set1  # 예쁜 색상
    )
    fig.update_traces(textposition='outside', width=0.18)
    fig.update_layout(bargap=0.25, height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader('2️⃣ 월별 음식물 쓰레기 배출량')
    monthly_df = df.groupby('월')['배출량(톤)'].sum().reindex(range(1,13)).reset_index()
    fig2 = px.bar(monthly_df, x='월', y='배출량(톤)', color='월',
                  text_auto='.1f',
                  title='월별 음식물 쓰레기 배출량', labels={'배출량(톤)': '배출량(톤)'})
    fig2.update_traces(textposition='outside', width=0.4, marker_line_width=1.5, marker_line_color='black')
    fig2.update_layout(bargap=0.3, height=400)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader('3️⃣ 연도별 음식물 쓰레기 배출 추이')
    yearly_df = df.groupby('연도')['배출량(톤)'].sum().reset_index()
    yearly_df['연도'] = yearly_df['연도'].astype(str)
    fig3 = px.line(yearly_df, x='연도', y='배출량(톤)', markers=True,
                   title="연도별 음식물 쓰레기 배출 추이", labels={'배출량(톤)': '배출량(톤)'})
    fig3.update_xaxes(type='category')
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader('4️⃣ 🧩 지자체별 워드 클라우드(배출량 비례)')
    wc_df = df.groupby(city_col)['배출량(톤)'].sum().reset_index()
    wc_df = wc_df.sort_values(by='배출량(톤)', ascending=False).head(30)
    np.random.seed(42)
    wc_df['x'] = np.random.rand(len(wc_df))
    wc_df['y'] = np.random.rand(len(wc_df))
    fig_wc = go.Figure()
    for _, row in wc_df.iterrows():
        fig_wc.add_trace(go.Scatter(
            x=[row['x']], y=[row['y']],
            text=[row[city_col]],
            mode='text',
            textfont=dict(
                size=10 + row['배출량(톤)'] / wc_df['배출량(톤)'].max() * 60,
                color='rgba(255,0,0,0.8)' if row['배출량(톤)']==wc_df['배출량(톤)'].max() else 'rgba(0,0,0,0.7)'
            ),
            hovertext=f"{row[city_col]}: {row['배출량(톤)']:.1f}톤",
            hoverinfo='text'
        ))
    fig_wc.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='white',
        title='지자체별 음식물 쓰레기 배출량 워드 클라우드',
        margin=dict(l=0, r=0, t=40, b=0),
        height=500
    )
    st.plotly_chart(fig_wc, use_container_width=True)

with tab5:
    st.subheader('5️⃣ 지도 히트맵 + RFID 추천')
    top10 = df.groupby(city_col)['배출량(톤)'].sum().sort_values(ascending=False).head(10).reset_index()
    np.random.seed(0)
    top10['lat'] = 37.2 + np.random.rand(len(top10)) * 0.5
    top10['lon'] = 127.0 + np.random.rand(len(top10)) * 0.5
    max_city = top10.loc[top10['배출량(톤)'].idxmax(), city_col]
    max_lat = top10.loc[top10['배출량(톤)'].idxmax(), 'lat']
    max_lon = top10.loc[top10['배출량(톤)'].idxmax(), 'lon']
    fig_map = px.scatter_mapbox(top10, lat='lat', lon='lon', size='배출량(톤)', color='배출량(톤)',
                                hover_name=city_col, zoom=8, mapbox_style='carto-positron')
    fig_map.add_trace(go.Scattermapbox(
        lat=[max_lat], lon=[max_lon], mode='markers+text',
        marker=dict(size=30, color='red'), text=[f"RFID 추천: {max_city}"], textposition='top right'
    ))
    st.plotly_chart(fig_map, use_container_width=True)
    st.info(f"🚩 **{max_city}**에 RFID 음식물쓰레기 배출기 설치를 추천합니다!")

with tab6:
    st.subheader('6️⃣ 전년대비 요일별 배출량 비율 차이')
    # 1. 연도별 요일별 배출량 집계
    weekday_year_df = df.groupby(['연도', '요일'])['배출량(톤)'].sum().reset_index()
    weekday_year_df['배출량비율'] = weekday_year_df.groupby('연도')['배출량(톤)'].transform(lambda x: x / x.sum())

    # 2. 전년대비 차이 계산
    weekday_year_df = weekday_year_df.sort_values(['요일', '연도'])
    weekday_year_df['전년대비비율차이'] = weekday_year_df.groupby('요일')['배출량비율'].diff()

    # 3. 최근 2개년(2023, 2024)만 추출 (또는 원하는 연도)
    latest_years = sorted(df['연도'].unique())[-2:]
    diff_df = weekday_year_df[weekday_year_df['연도'] == latest_years[1]][['요일', '전년대비비율차이']]

    # 4. Plotly 막대그래프
    fig = px.bar(
        diff_df,
        x='요일', y='전년대비비율차이',
        color='전년대비비율차이',
        color_continuous_scale='RdBu',
        text='전년대비비율차이',
        labels={'전년대비비율차이': '전년대비 배출량비율 차이'},
        title=f"{latest_years[0]}→{latest_years[1]} 전년대비 요일별 배출량 비율 차이"
    )
    fig.update_traces(
        texttemplate='%{text:.2%}',  # 퍼센트로 표시
        textposition='outside',
        width=0.4
    )
    fig.update_layout(
        bargap=0.3,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown('---')
st.markdown('''
### 🌱 데이터 과학 인사이트
- 네트워크/지도/AI 예측 등 다양한 시각화로 정책 설계와 미래 준비!
- 사용자가 직접 시나리오를 바꿔보며 데이터 과학적 사고를 키워보세요!
''')

with st.expander('💡 예상 답변/정책 제안 펼쳐보기'):
    st.markdown('''
**예상 답변 예시**
- 요일별로 배출량이 많은 요일(예: 금요일, 토요일)에 맞춰 음식물 쓰레기 감축 캠페인을 집중적으로 시행하면 효과적입니다.
- 배출량이 많은 지자체에는 RFID 음식물쓰레기 배출기, 스마트 모니터링 시스템을 우선 도입하는 것이 효율적입니다.
- AI 예측 결과, 2030년까지 음식물 쓰레기 배출량이 지속적으로 증가할 것으로 예상되므로, 조기 정책 개입이 필요합니다.
- 클러스터별 이상값(특이하게 많은 지자체)은 맞춤형 정책, 캠페인, 인프라 확충이 필요합니다.
- 데이터 과학적 분석을 통해 지역별, 시기별로 최적화된 정책을 설계할 수 있습니다.
    ''')
