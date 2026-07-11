import streamlit as st
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import StringIO

st.set_page_config(
    page_title="Sentimen Analisis — Pelemahan Rupiah",
    page_icon="assets/Analytics.svg",
    layout="centered",
)


# ── Helper SVG ────────────────────────────────────────────
@st.cache_resource
def svg_img(path, width=24):
    with open(path, "r", encoding="utf-8") as f:
        svg = f.read()
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


def svg_tag(path, width=24, css="vertical-align:middle;margin-right:6px"):
    src = svg_img(path, width)
    return f'<img src="{src}" width="{width}" style="{css}">'


@st.cache_resource
def load_model():
    return (
        joblib.load("model/model.pkl"),
        joblib.load("model/tfidf.pkl"),
    )


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"\\n|\\r", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_model():
    if "model" not in st.session_state:
        m, t = load_model()
        st.session_state["model"] = m
        st.session_state["tfidf"] = t
    return st.session_state["model"], st.session_state["tfidf"]


def predict(text):
    m, t = get_model()
    cleaned = clean_text(text)
    vec = t.transform([cleaned])
    pred = m.predict(vec)[0]
    proba = m.predict_proba(vec)[0]
    return pred, proba, cleaned


colors = {"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}
sentiment_svg = {
    "positive": "assets/Positive-sentiment.svg",
    "neutral": "assets/Neutral-sentiment.svg",
    "negative": "assets/Negative-sentiment.svg",
}
label_order = ["negative", "neutral", "positive"]

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<h2>{svg_tag('assets/Analytics.svg', 28)} Sentimen Analisis</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Klasifikasi sentimen komentar YouTube tentang pelemahan Rupiah.\n\n"
        "**Label:** positive / neutral / negative\n\n"
        "**Model:** Logistic Regression + TF-IDF + SMOTE"
    )
    st.markdown("---")
    st.markdown("**Contoh komentar:**")
    for ex in [
        "Pemerintah bebal dan korup",
        "Terimakasih kontennya sangat edukatif",
        "Dollar sedang menguat terhadap rupiah",
        "Makasih koh 😊 kuliah gratis",
        "Rupiah melemah karena sell off besar besaran",
    ]:
        st.markdown(f"- _{ex}_")
    st.markdown("---")
    st.markdown("Upload file CSV/XLSX untuk prediksi massal.")

# ── Main ─────────────────────────────────────────────────
st.markdown(
    f"<h1>{svg_tag('assets/Analytics.svg', 32)} Sentimen Analisis Komentar YouTube</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "Dua cara pakai: **(1)** tulis 1 komentar manual, atau **(2)** upload file CSV/XLSX untuk prediksi massal."
)

tab1, tab2 = st.tabs(["Input Manual", "Upload File"])

# ════════════════════════════════════════════════════════
# TAB 1 — Single prediction
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown(svg_tag("assets/Input.svg", 20) + " **Tulis komentar di bawah:**", unsafe_allow_html=True)
    user_input = st.text_area(
        "",
        placeholder="Contoh: Pemerintah harus segera turun tangan mengatasi pelemahan Rupiah...",
        height=120,
        label_visibility="collapsed",
    )

    if st.button("Prediksi", type="primary"):
        if user_input.strip():
            pred, proba, _ = predict(user_input)
            confidence = proba.max()
            icon_svg = svg_tag(sentiment_svg[pred], 48)

            st.markdown("---")
            st.markdown(
                f"<h3>{icon_svg} <b>{pred.upper()}</b></h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="padding:1rem;border-radius:0.5rem;background:{colors[pred]}22;'
                f'border-left:4px solid {colors[pred]}">'
                f"<b>Komentar:</b> {user_input}</div>",
                unsafe_allow_html=True,
            )

            st.markdown(f"**Confidence:** {confidence:.1%}")
            st.progress(float(confidence))

            st.markdown("**Distribusi Probabilitas:**")
            m, _ = get_model()
            proba_df = pd.DataFrame({
                "Sentimen": label_order,
                "Probabilitas": [proba[m.classes_.tolist().index(l)] for l in label_order],
            })
            fig, ax = plt.subplots(figsize=(6, 2.5))
            bars = ax.barh(
                proba_df["Sentimen"], proba_df["Probabilitas"],
                color=["#e74c3c", "#f39c12", "#2ecc71"],
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probabilitas")
            for bar, prob in zip(bars, proba_df["Probabilitas"]):
                ax.text(
                    bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{prob:.1%}", va="center", fontsize=11,
                )
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Silakan masukkan komentar terlebih dahulu.")

# ════════════════════════════════════════════════════════
# TAB 2 — Batch prediction from file
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown(svg_tag("assets/Upload.svg", 20) + " Format file: CSV (koma/titik-koma) atau XLSX dengan minimal 1 kolom berisi teks komentar.", unsafe_allow_html=True)
    uploaded = st.file_uploader("Pilih file:", type=["csv", "xlsx"], label_visibility="collapsed")

    if uploaded:
        if uploaded.name.endswith(".xlsx"):
            raw = pd.read_excel(uploaded)
        else:
            raw = pd.read_csv(uploaded, sep=None, engine="python")

        st.markdown(f"File terbaca: **{uploaded.name}** — {len(raw)} baris, {len(raw.columns)} kolom")
        st.dataframe(raw.head(5), use_container_width=True)

        text_col = st.selectbox("Pilih kolom yang berisi teks komentar:", raw.columns.tolist())

        if st.button("Prediksi Semua", type="primary"):
            with st.spinner(f"Memproses {len(raw)} komentar..."):
                m, _ = get_model()
                idx_neg = m.classes_.tolist().index("negative")
                idx_neu = m.classes_.tolist().index("neutral")
                idx_pos = m.classes_.tolist().index("positive")
                results = []
                for txt in raw[text_col].astype(str):
                    pred, proba, cleaned = predict(txt)
                    results.append({
                        "text_asli": txt[:200],
                        "sentimen": pred,
                        "confidence": proba.max(),
                        "prob_negative": proba[idx_neg],
                        "prob_neutral": proba[idx_neu],
                        "prob_positive": proba[idx_pos],
                    })

            out = pd.DataFrame(results)

            st.markdown("### Ringkasan")
            summary = out["sentimen"].value_counts()
            total = len(out)

            cols = st.columns(3)
            for i, label in enumerate(["negative", "neutral", "positive"]):
                cnt = summary.get(label, 0)
                svg_icon = svg_tag(sentiment_svg[label], 28)
                with cols[i]:
                    st.markdown(
                        f"<div style='text-align:center;padding:1rem;background:{colors[label]}22;"
                        f"border-radius:0.5rem;border-left:4px solid {colors[label]}'>"
                        f"<h3 style='margin:0'>{svg_icon} {label.title()}</h3>"
                        f"<h2 style='margin:0'>{cnt}</h2>"
                        f"<small>{cnt/total*100:.1f}%</small></div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("### Detail Hasil")
            st.dataframe(out, use_container_width=True, hide_index=True)

            csv_buffer = StringIO()
            out.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
            st.download_button(
                label="Download CSV Hasil",
                data=csv_buffer.getvalue(),
                file_name="hasil_sentimen.csv",
                mime="text/csv",
            )

# ── Footer ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<small>{svg_tag('assets/Analytics.svg', 14)} Dataset: 319 komentar YouTube | "
    "Model: Logistic Regression + TF-IDF + SMOTE | "
    "Dibuat untuk portofolio</small>",
    unsafe_allow_html=True,
)
