import os
import gc
import cv2
import json
import torch
import base64
import pickle
import pandas as pd
from PIL import Image
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()
login(token=os.getenv("HFTOKEN"))


device = 'cpu'

# Monkey patch torch.load to always map to CPU
torch_load_old = torch.load
def torch_load_cpu(*args, **kwargs):
    kwargs['map_location'] = torch.device('cpu')
    return torch_load_old(*args, **kwargs)

torch.load = torch_load_cpu

# ✅ 一次載入所有模型（避免 OOM）
@st.cache_resource
def load_models():
    return {
        'minilm': SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device),
        'LaBSE': SentenceTransformer('sentence-transformers/LaBSE', device=device),
        'bilingual': SentenceTransformer('am-azadi/bilingual-embedding-small_Fine_Tuned', trust_remote_code=True, device=device),
        'gemma': SentenceTransformer('google/embeddinggemma-300m', device=device),
        'qwen3': SentenceTransformer('tomaarsen/Qwen3-Embedding-0.6B-18-layers', device=device)
    }
model_map = load_models()

# 全域變數
if "search_history_dict" not in st.session_state:
    st.session_state.search_history_dict = {_tag: {} for _tag in model_map.keys()}

if "search_history_list" not in st.session_state:
    st.session_state.search_history_list = []

# st 外層設定
st.set_page_config(layout="wide")
st.title("🌮 Toby 多語言語意搜尋 Demo 系統")

st.markdown("""
<style>
.stApp {
    color: #0D47A1; /* 深藍色字體 */
}

h1, h2, h3 {
    color: #0D47A1; /* 深藍標題 */
}

.stSlider label {
    color: yellow; /* 設定標籤字體為黃色 */
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file, opacity):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(255,255,255,{opacity}), rgba(255,255,255,{opacity})),
                    url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

opacity = st.slider("選擇背景淡度 (0.0 = 完全透明, 1.0 = 完全白)", 0.0, 1.0, 0.5, 0.01)
set_png_as_page_bg('images/capybara01.png', opacity)
st.write(f"目前透明度：{opacity}")

# Sidebar UI
st.sidebar.title("🔍 多語言搜尋設定")
port_num = st.sidebar.number_input("請輸入 Port：", value=8040)
model_name = st.sidebar.selectbox("選擇模型", list(model_map.keys()))
query_mode = st.sidebar.selectbox("查詢類型", ['user', 'tag', 'content', '簡介', '聲音', '簡介smol'])
top_k = st.sidebar.number_input("輸出數量", min_value=1, max_value=100, value=10)
query_text = st.sidebar.text_input("輸入查詢字串")
search_button = st.sidebar.button("Search")

tab1, tab2, tab3 = st.tabs(
    ["🐹 語意查詢 Search", 
     "🐣 影片觀賞 View", 
     "🦖 歷史查詢 History"],
    width = "stretch"
)

# ✅ 載入 JSON 資料（避免每次查詢重讀）
@st.cache_data
def load_video_metadata():
    with open("vds/video_metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)
data_json = load_video_metadata()

# ✅ 快取 embeddings 和 DataFrame
@st.cache_data
def load_embeddings_and_df(query_mode, model_name):
    if query_mode == 'user':
        with open(f'embeddings/members_embeddings_{model_name}.pkl', 'rb') as f:
            ids, emb = pickle.load(f)
        df = pd.read_csv('datas/members.csv')
        df['text'] = df['account'].fillna('') + ' ' + df['nickname'].fillna('')
    elif query_mode == 'tag':
        with open(f'embeddings/tags_embeddings_{model_name}.pkl', 'rb') as f:
            ids, emb = pickle.load(f)
        df = pd.read_csv('datas/posts.csv')
        df['text'] = df['hash_tags'].fillna('')
    elif query_mode == 'content':
        with open(f'embeddings/content_embeddings_{model_name}.pkl', 'rb') as f:
            ids, emb = pickle.load(f)
        df = pd.read_csv('datas/posts.csv')
        df['text'] = df['content'].fillna('')
    else:
        ids, emb, df = [], torch.empty(0), pd.DataFrame()
    
    # ✅ embeddings 移到 CPU 一次完成
    emb = emb.cpu()
    return ids, emb, df

with tab1:
    st.title("🔎 查詢結果")
    if search_button and query_text.strip():
        # ✅ 檢查是否已有快取的 query_emb
        if query_text in st.session_state.search_history_dict[model_name]:
            query_emb = st.session_state.search_history_dict[model_name][query_text]
            st.info("使用快取的 Query Embedding ✅")
        else:
            model = model_map[model_name]
            query_emb = model.encode(query_text, convert_to_tensor=True).cpu()
            st.session_state.search_history_dict[model_name][query_text] = query_emb

        results = []
        if query_mode in ['簡介', '聲音', '簡介smol']:
            emb_field = f'{query_mode}_{model_name}'
            text_field = query_mode

            for item in data_json:
                emb_list = item.get(emb_field, [])
                if emb_list:
                    emb_tensor = torch.tensor(emb_list, dtype=torch.float32)
                    score = util.cos_sim(query_emb, emb_tensor)[0][0].item()
                    results.append({
                        '檔名': item['檔名'],
                        'score': round(score, 4),
                        'text': item[text_field]
                    })

            # 排序並顯示
            results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
            
            column_config={
                "檔名": st.column_config.TextColumn("檔名", width="medium"),
                "score": st.column_config.NumberColumn("相似度", format="%.4f"),
                "text": st.column_config.TextColumn("內容", width="large", help="點擊可展開完整內容"),
            }
            
                
        elif query_mode in ['user', 'tag', 'content']:
            ids, emb, df = load_embeddings_and_df(query_mode, model_name)
            scores = util.cos_sim(query_emb, emb)[0]
            top_indices = scores.argsort(descending=True)[:top_k]

            for i in top_indices:
                idx = int(i)
                results.append({
                    'id': ids[idx],
                    'score': round(float(scores[idx]), 4),
                    'text': df.iloc[idx]['text']
                })
            column_config={
                "id": st.column_config.TextColumn("ID", width="small"),
                "score": st.column_config.NumberColumn("相似度", format="%.4f"),
                "text": st.column_config.TextColumn("內容", width="large", help="點擊可展開完整內容"),
            }
        
        # ✅ 顯示結果表格
        if results:
            st.dataframe(
                pd.DataFrame(results),
                column_config = column_config
            )
            if query_mode in ['簡介', '聲音', '簡介smol']:
                for i, res in enumerate(results):
                    name = res['檔名']
                    url = f"http://localhost:{port_num}/{name}.mp4"
                    st.markdown(f'{i}. {url}', unsafe_allow_html=True)            
            

        else:
            st.warning("沒有找到相關結果。")

        # ✅ 更新歷史紀錄列表
        st.session_state.search_history_list.append({
            'query_text': query_text,
            'model_name': model_name,
            'query_mode': query_mode,
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # ✅ 釋放暫存張量
        del query_emb
        gc.collect()
        torch.cuda.empty_cache()

with tab2:
    st.header("💽 測試影片觀賞")

    # 取得影片清單
    video_dir = Path("vds/videos")
    video_files = list(video_dir.glob("*.mp4"))
    video_names = [video_f.name for video_f in video_files]

    if not video_files:
        st.warning("找不到影片檔案。請確認 vds/videos 資料夾中有 .mp4 檔案。")
    else:
        # 下拉選單選擇影片
        selected_video_name = st.selectbox("選擇影片", video_names)

        if selected_video_name:
            # 讀取影片資訊
            selected_video = f"vds/videos/{selected_video_name}"
            cap = cv2.VideoCapture(str(selected_video))
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                file_size = os.path.getsize(selected_video) / (1024 * 1024)  # MB
                cap.release()
                
            col1, col2, col3 = st.columns(3)
            with col1:
                # 顯示影片
                st.header("影片播放")
                st.video(str(selected_video))
                
            with col2:
                st.header("影片資訊")
                st.write(f"📏 解析度：{width} x {height}")
                st.write(f"🎞️ FPS：{fps:.2f}")
                st.write(f"⏱️ 時長：{duration:.2f} 秒")
                st.write(f"💾 檔案大小：{file_size:.2f} MB")
                
            with col3:
                st.header("隨手寫")
                st.text_area("這裡完全不會紀錄")       
            
with tab3:
    st.header("📜 歷史查詢紀錄")
    if st.session_state.search_history_list:
        st.dataframe(pd.DataFrame(st.session_state.search_history_list))
        if st.button("清除歷史紀錄"):
            st.session_state.search_history_list.clear()
            st.session_state.search_history_dict.clear()
            st.success("歷史紀錄已清除 ✅")
    else:
        st.info("目前沒有歷史紀錄。")

