import requests
import gradio as gr
from datetime import datetime, timedelta
import pandas as pd
from functools import lru_cache
import concurrent.futures
from typing import Dict, List, Tuple

fruit_count = {}
price_dict = {}
image_read = []

# 캐시를 위한 전역 변수
CACHE_TIMEOUT = 3600  # 1시간
last_cache_refresh = datetime.now()
price_cache = {}

# 날짜 계산 함수 
def get_date_range():
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    before_yesterday = (now - timedelta(days=2)).strftime("%Y-%m-%d")
    
    # 서버의 당일 데이터 업로드 시간 13:30 을 고려하여 14를 기점으로 날짜 지정
    # 서버 api가 하루만 데이터를 불러오기가 안되는 문제로 인하여 이틀(오늘-어제/어제-그제)씩 날짜 저장
    if now.hour > 14:
        return today, yesterday
    return yesterday, before_yesterday

# API 정보를 캐싱하여 사용 (효율화)
@lru_cache(maxsize=32)
def get_api_configs() -> Dict:
    
    return {
        'category_dict': {'apple': '400', 'banana': '400', 'carrot': '200', 'cucumber': '200', 
                         'mango': '400', 'bellpepper': '200', 'orange': '400', 'potato': '100', 
                         'strawberry': '200', 'tomato': '200'},
        'fruit_code_dict': {'apple': '411', 'banana': '418', 'carrot': '232', 'cucumber': '223',
                           'mango': '428', 'bellpepper': '256', 'orange': '421', 'potato': '152',
                           'strawberry': '226', 'tomato': '225'},
        'kind_code_dict': {'apple': '05', 'banana': '02', 'carrot': '01', 'cucumber': '02',
                          'mango': '00', 'bellpepper': '00', 'orange': '03', 'potato': '01',
                          'strawberry': '00', 'tomato': '00'},
        'unit_fruit_dict': {'apple': '10kg', 'banana': '13kg', 'carrot': '20kg', 'cucumber': '100개',
                           'mango': '5kg', 'bellpepper': '5kg', 'orange': '18kg', 'potato': '20kg',
                           'strawberry': '2kg', 'tomato': '5kg'}
    }

def fruits_status(prediction: str) -> Tuple[str, str]:
    """과일 상태 확인 함수 최적화"""
    name, status = prediction.split('_')
    return name, status

# kamis 데이터베이스에서 원하는 데이터 요청하는 함수  
# API 호출을 캐시하여 중복 요청 방지
@lru_cache(maxsize=100)
def get_fruit_price(fruits_name: str, start_date: str, end_date: str) -> requests.Response:
    
    api_configs = get_api_configs()
    
    api_url = "http://www.kamis.or.kr/service/price/xml.do?action=periodWholesaleProductList"
    api_key = "6d042380-dda3-4bd4-8044-49366c8b3ccc"
    
    params = {
        'p_cert_key': api_key,
        'p_cert_id': '5318',
        'p_returntype': 'json',
        'p_startday': start_date,
        'p_endday': end_date,
        'p_itemcategorycode': api_configs['category_dict'][fruits_name],
        'p_itemcode': api_configs['fruit_code_dict'][fruits_name],
        'p_kindcode': api_configs['kind_code_dict'][fruits_name],
        'p_productrankcode': '05'
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': api_key
    }
    
    return requests.post(api_url, headers=headers, params=params)

# 가격 계산 함수 
def calculate_prices(price_data: List, fruits_status: str, fruits_name: str) -> pd.DataFrame:

    api_configs = get_api_configs()
    columns = ['품목', '상태', '지역', '도매가격', '단위']
    df = pd.DataFrame(columns=columns)
    
    for i in range(5):
        price = float(price_data[i]['price'].replace(',', ''))
        if fruits_status == 'low':
            price = float(f"{price * 0.6}")

        formatted_price = f"{price:,.0f}"
        df.loc[i] = [
            price_data[i]['itemname'],
            fruits_status,
            price_data[i]['countyname'],
            formatted_price,
            api_configs['unit_fruit_dict'][fruits_name]
        ]
        
    return df

# 가격 정보 조회 함수 
def fruits_price(fruits_name: str, fruits_status: str, start_date: str, end_date: str) -> pd.DataFrame:
    
    global price_cache, last_cache_refresh
    
    # 캐시 만료 확인
    if (datetime.now() - last_cache_refresh).seconds > CACHE_TIMEOUT:
        price_cache.clear()
        last_cache_refresh = datetime.now()
    
    cache_key = f"{fruits_name}_{start_date}_{end_date}"
    if cache_key in price_cache:
        data = price_cache[cache_key]
    else:
        response = get_fruit_price(fruits_name, start_date, end_date)
        data = response.json()
        price_cache[cache_key] = data
    
    price_data = data['data']['item'][5:14:2]
    return calculate_prices(price_data, fruits_status, fruits_name)

## CustomVision으로 생성한 모델 API를 이용해 분류 함수 생성 
def predict_image(image_data) :
    
    endpoint = "https://sixth12teamcv-prediction.cognitiveservices.azure.com"
    prediction_key = "GGtD4gFvHfXvLxrb0jytHZ0MKEk5sqUNeTeO2lQd6MIl6UNOoBoFJQQJ99BBACYeBjFXJ3w3AAAIACOGYb3k"
    project_id = "cfd4ef85-739a-4db1-856e-4f593a35e58e"
    model_name = "Iteration2"
    
    url = f"{endpoint}/customvision/v3.0/Prediction/{project_id}/classify/iterations/{model_name}/image"
    
    headers = {
        'Prediction-Key': prediction_key,
        'Content-Type': 'application/octet-stream'
    }
    
    try:
        with requests.Session() as session:
            response = session.post(url, headers=headers, data=image_data)
            response.raise_for_status()
            predictions = response.json()['predictions']
            # 상위 1개 예측 결과 선택
            top_prediction = max(predictions, key=lambda x: x['probability'])
        
        return top_prediction['tagName']  # 태그 이름 그대로 반환
    
    except Exception as e:
        return f"Error: {str(e)}"

# 과일 및 채소의 상태별 개수 저장하기 위한 함수 
def fruit_detective(image: bytes) -> Tuple[Dict, pd.DataFrame]:
    
    prediction = predict_image(image)
    fruit_name, fruit_status = fruits_status(prediction)
    
    product_korean = {'apple': '사과', 'banana': '바나나', 'carrot': '당근', 'cucumber': '오이',
                           'mango': '망고', 'bellpepper': '파프리카', 'orange': '오렌지', 'potato': '감자',
                           'strawberry': '딸기', 'tomato': '토마토'}

    condition_korean = {'fr': '🟢', 'low': '🟠', 'rot': '🔴'}

    fruit_count[prediction] = fruit_count.get(prediction, 0) + 1
    count_data = [
    {
        "품목": product_korean[k.split('_')[0]],  # 영문 품목명을 한글로 변환
        "품질": condition_korean[k.split('_')[1]], # 상태 코드를 이모지로 변환
        "개수": v
    }
    for k, v in fruit_count.items()
]
    # 가격정보가 중복으로 누적되는걸 방지하기 위해 키값으로 dictionary에 저장 
    if fruit_status.lower() in ('fr', 'low'):
        price_dict[prediction] = True
    
    return price_dict, pd.DataFrame(count_data)

def upload_to_do(image: str, price_dataframes_state: List) -> Tuple:
    """업로드 처리 함수 최적화"""
    today_date, yesterday_date = get_date_range()
    
    with open(image, "rb") as img_file:
        image_bytes = img_file.read()
        price_dict, count_df = fruit_detective(image_bytes)
        image_read.append(image)
    
    # 병렬 처리를 위한 함수
    def process_fruit(key):
        fruit_name, fruit_status = key.split('_')
        return fruits_price(fruit_name, fruit_status, yesterday_date, today_date)
    
    # ThreadPoolExecutor를 사용한 병렬 처리
    all_dfs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_fruit, key) for key in price_dict]
        all_dfs = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    
    return image_read, all_dfs, count_df, combined_df

# Gradio 인터페이스 설정
with gr.Blocks() as demo_kr:
    gr.Markdown("# 🍎 건강한 과일, 편리한 관리")
    gr.Markdown("## 사진이 업로드 되면 자동으로 분류하여 가격 정보를 산출합니다.")
    
    price_dataframes = gr.State([])
    
    with gr.Row(height=350):
        image_upload = gr.Image(type="filepath", height=300)
        count_df = gr.Dataframe()
    
    with gr.Row(height=500):
        combined_price_df = gr.Dataframe()
    
    with gr.Row(height=1000):
        output_img_store = gr.Gallery(label='입력된 과일 및 채소 사진', columns=10)
    
    image_upload.upload(
        fn=upload_to_do,
        inputs=[image_upload, price_dataframes],
        outputs=[output_img_store, price_dataframes, count_df, combined_price_df]
    )

# 실행 코드 제거 -> FastAPI에서 import 해서 사용
#   if __name__ == \"__main__\":\n",
#       demo.launch()"