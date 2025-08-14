
import os
import io
import time
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

st.set_page_config(page_title="Agent Phân tích dữ liệu điểm số", layout="wide")

# -------------------------
# Helpers
# -------------------------
EXPECTED_COLUMNS = [
    "student_id", "student_name", "class", "grade_level",
    "subject", "exam_date", "score", "max_score", "weight"
]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # map some common Vietnamese aliases if needed
    alias = {
        "ma_hs": "student_id",
        "ten_hs": "student_name",
        "lop": "class",
        "khoi": "grade_level",
        "mon": "subject",
        "ngay_kiem_tra": "exam_date",
        "diem": "score",
        "tong_diem": "max_score",
        "he_so": "weight",
    }
    df.rename(columns={k:v for k,v in alias.items() if k in df.columns}, inplace=True)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        st.warning(f"Cột còn thiếu: {missing}. Hãy dùng mẫu 'sample_data.csv' để tham khảo cấu trúc.")
    return df

def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        # Load bundled sample
        sample_path = os.path.join(os.path.dirname(__file__), "sample_data.csv")
        df = pd.read_csv(sample_path)
    else:
        df = pd.read_csv(uploaded_file)
    df = _normalize_columns(df)
    # typing and preprocessing
    df["exam_date"] = pd.to_datetime(df["exam_date"])
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["max_score"] = pd.to_numeric(df["max_score"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
    df["score_pct"] = (df["score"] / df["max_score"]) * 100.0
    df["grade_level"] = df["grade_level"].astype(str)
    df["class"] = df["class"].astype(str)
    df["subject"] = df["subject"].astype(str)
    return df.dropna(subset=["score_pct"])

@st.cache_data
def compute_overview(df: pd.DataFrame):
    overall = {
        "Số bài kiểm tra": len(df),
        "Số học sinh": df["student_id"].nunique(),
        "Số lớp": df["class"].nunique(),
        "Số môn": df["subject"].nunique(),
        "Điểm TB (%)": round(df["score_pct"].mean(), 2),
        "Tỉ lệ >= 50%": round((df["score_pct"] >= 50).mean() * 100, 2),
    }
    return overall

@st.cache_data
def agg_by_class_subject(df: pd.DataFrame):
    g = (df.groupby(["grade_level", "class", "subject"], as_index=False)
           .agg(so_bai=("score_pct", "count"),
                diem_tb=("score_pct", "mean"),
                ti_le_dat_50=("score_pct", lambda s: (s>=50).mean()*100)))
    g["diem_tb"] = g["diem_tb"].round(2)
    g["ti_le_dat_50"] = g["ti_le_dat_50"].round(2)
    return g

@st.cache_data
def latest_scores_by_student(df: pd.DataFrame):
    # Lấy bài kiểm tra mới nhất của mỗi học sinh theo từng môn
    df_sorted = df.sort_values(["student_id", "subject", "exam_date"])
    latest = df_sorted.groupby(["student_id", "student_name", "class", "grade_level", "subject"], as_index=False).tail(1)
    return latest

@st.cache_data
@st.cache_data
def at_risk_students(df: pd.DataFrame, threshold_pct=50.0, drop_threshold=10.0):
    # Sắp xếp dữ liệu theo học sinh, môn, ngày kiểm tra
    df_sorted = df.sort_values(["student_id", "subject", "exam_date"])
    
    # Tính điểm bài trước và độ tụt
    df_sorted["prev_score_pct"] = df_sorted.groupby(
        ["student_id", "subject"]
    )["score_pct"].shift(1)
    df_sorted["drop"] = df_sorted["prev_score_pct"] - df_sorted["score_pct"]
    
    # Lấy bài kiểm tra mới nhất của mỗi học sinh cho từng môn
    latest = df_sorted.groupby(
        ["student_id", "student_name", "class", "grade_level", "subject"],
        as_index=False
    ).tail(1)
    
    # Lọc học sinh có điểm thấp hoặc tụt mạnh
    at_risk = latest[
        (latest["score_pct"] < threshold_pct) |
        (latest["drop"].fillna(0) >= drop_threshold)
    ].copy()
    
    # Tính lại điều kiện trực tiếp trên at_risk để tránh lệch index
    cond_low = at_risk["score_pct"] < threshold_pct
    cond_drop = at_risk["drop"].fillna(0) >= drop_threshold
    
    # Ghép lý do
    at_risk["Lý do"] = np.where(cond_low, "Điểm dưới ngưỡng", "") + \
                       np.where(cond_drop, " | Tụt mạnh", "")
    
    # Sắp xếp kết quả
    at_risk = at_risk.sort_values(["class", "subject", "score_pct"])
    
    return at_risk


@st.cache_data
def forecast_trend(df: pd.DataFrame, horizon_days=30):
    # Dự báo trung bình theo (khối, lớp, môn) bằng hồi quy tuyến tính thời gian
    out_rows = []
    for keys, sub in df.groupby(["grade_level", "class", "subject"]):
        daily = (sub.groupby("exam_date", as_index=False)["score_pct"].mean()
                   .sort_values("exam_date"))
        if len(daily) < 3:
            continue
        # fit linear regression on date -> score
        X = (daily["exam_date"].map(pd.Timestamp.toordinal)).values.reshape(-1,1)
        y = daily["score_pct"].values
        model = LinearRegression()
        model.fit(X, y)
        future_date = daily["exam_date"].max() + timedelta(days=horizon_days)
        y_pred = float(model.predict([[future_date.toordinal()]]))
        out_rows.append({
            "grade_level": keys[0],
            "class": keys[1],
            "subject": keys[2],
            "last_date": daily["exam_date"].max().date(),
            "forecast_date": future_date.date(),
            "current_avg": round(float(y[-1]), 2),
            "forecast_avg": round(y_pred, 2)
        })
    return pd.DataFrame(out_rows)

def summarize_for_llm(df: pd.DataFrame, top_k=5) -> str:
    # Tạo tóm tắt gọn để gửi cho LLM
    g = agg_by_class_subject(df).sort_values("diem_tb")
    worst = g.head(top_k)
    best = g.tail(top_k)
    overall = compute_overview(df)
    payload = {
        "overall": overall,
        "lowest_groups": worst.to_dict(orient="records"),
        "highest_groups": best.to_dict(orient="records"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

def ask_gemini(prompt: str, model_name: str, api_key: str) -> str:
    try:
        import google.generativeai as genai
    except Exception as e:
        return "Chưa cài gói google-generativeai. Vui lòng `pip install google-generativeai`."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return getattr(resp, "text", str(resp))
    except Exception as e:
        return f"Lỗi gọi Gemini: {e}"

def ask_ollama(prompt: str, host: str, model_name: str) -> str:
    try:
        url = host.rstrip("/") + "/api/generate"
        r = requests.post(url, json={"model": model_name, "prompt": prompt, "stream": False}, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response", json.dumps(data))
    except Exception as e:
        return f"Lỗi gọi Ollama: {e}"

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("⚙️ Cấu hình")
provider = st.sidebar.selectbox("Chọn nhà cung cấp LLM", ["Không dùng LLM", "Gemini API", "Ollama (local)"])

gemini_model = st.sidebar.text_input("Gemini model", value=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
gemini_key = st.sidebar.text_input("GEMINI_API_KEY", value=os.getenv("GEMINI_API_KEY", ""), type="password")

ollama_host = st.sidebar.text_input("Ollama HOST", value=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
ollama_model = st.sidebar.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "mistral"))

risk_threshold = st.sidebar.slider("Ngưỡng điểm rủi ro (%)", min_value=0, max_value=100, value=50, step=1)
drop_threshold = st.sidebar.slider("Ngưỡng tụt điểm mạnh (điểm %)", min_value=0, max_value=50, value=10, step=1)
horizon_days = st.sidebar.slider("Số ngày dự báo", min_value=7, max_value=90, value=30, step=1)

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Tải CSV điểm (tùy chọn)", type=["csv"])
st.sidebar.caption("Nếu không tải, ứng dụng dùng `sample_data.csv`.")

# -------------------------
# Main
# -------------------------
st.title("📊 Agent Phân tích dữ liệu điểm số")
df = load_data(uploaded)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Tổng quan", "Theo lớp & môn", "Học sinh cần hỗ trợ", "Dự báo", "AI nhận xét"])

with tab1:
    st.subheader("Tổng quan nhanh")
    ov = compute_overview(df)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for i, (k,v) in enumerate(ov.items()):
        with (c1, c2, c3, c4, c5, c6)[i]:
            st.metric(k, v)

    st.markdown("#### Xu hướng điểm trung bình theo ngày")
    daily = df.groupby("exam_date", as_index=False)["score_pct"].mean()
    st.line_chart(daily.set_index("exam_date"))

with tab2:
    st.subheader("Phân tích theo khối / lớp / môn")
    agg = agg_by_class_subject(df)
    st.dataframe(agg, use_container_width=True)
    st.markdown("#### Top nhóm điểm thấp nhất")
    st.dataframe(agg.sort_values("diem_tb").head(10), use_container_width=True)
    st.markdown("#### Top nhóm điểm cao nhất")
    st.dataframe(agg.sort_values("diem_tb", ascending=False).head(10), use_container_width=True)

with tab3:
    st.subheader("Học sinh cần hỗ trợ")
    risk = at_risk_students(df, threshold_pct=risk_threshold, drop_threshold=drop_threshold)
    st.dataframe(risk[["student_id","student_name","class","grade_level","subject","exam_date","score_pct","prev_score_pct","drop","Lý do"]], use_container_width=True)
    st.info(f"Tổng số học sinh cần hỗ trợ (theo tiêu chí): {len(risk)}")

with tab4:
    st.subheader("Dự báo xu hướng")
    fc = forecast_trend(df, horizon_days=horizon_days)
    if len(fc) == 0:
        st.warning("Chưa đủ dữ liệu để dự báo (cần >= 3 mốc thời gian cho mỗi nhóm).")
    else:
        st.dataframe(fc, use_container_width=True)

with tab5:
    st.subheader("Nhận xét từ AI")
    summary_json = summarize_for_llm(df, top_k=5)
    st.code(summary_json, language="json")
    user_question = st.text_area("Bạn muốn AI nhận xét gì? (ví dụ: đề xuất biện pháp hỗ trợ nhóm rủi ro)", value="Hãy đưa ra 3 nhận xét chính và 3 khuyến nghị hành động.")
    if st.button("Phân tích bằng AI"):
        prompt = f"""
Bạn là chuyên gia học đường. Dựa vào JSON thống kê sau đây, hãy:
- Nêu 3 nhận xét chính (xu hướng, vấn đề nổi bật).
- Gợi ý 3 hành động hỗ trợ học sinh và kế hoạch theo dõi.
- Nói ngắn gọn, có bullet, không giải thích lan man.

DỮ LIỆU:
{summary_json}

YÊU CẦU THÊM CỦA NGƯỜI DÙNG:
{user_question}
"""
        with st.spinner("Đang gọi LLM..."):
            if provider == "Gemini API":
                if not gemini_key:
                    st.error("Thiếu GEMINI_API_KEY.")
                else:
                    answer = ask_gemini(prompt, gemini_model, gemini_key)
                    st.markdown(answer)
            elif provider == "Ollama (local)":
                answer = ask_ollama(prompt, ollama_host, ollama_model)
                st.markdown(answer)
            else:
                st.warning("Bạn đang ở chế độ 'Không dùng LLM'. Hãy chọn Gemini")
