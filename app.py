# app.py – Streamlit Web App: "PDF → 問題集" 
"""
１ファイルで動く MVP。PDF をブラウザでアップロードすると、
GPT‑4o を呼び出して多肢選択問題を生成し、画面に表示＋JSON でダウンロード。
Streamlit Community Cloud に GitHub 連携だけでそのままデプロイできます。
"""

from __future__ import annotations

import json
import os
import re
from io import BytesIO
from typing import List, Dict

import streamlit as st
import pdfplumber
import tiktoken
import openai

# ------------------------- CONFIG -------------------------------------------
MODEL = "gpt-4o-mini"   # 速さ重視。必要なら gpt-4o へ
TOKENS_PER_CHUNK = 600
N_QUESTIONS_PER_CHUNK = 2

# OpenAI API キーを Streamlit の Secrets か環境変数に入れておく
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ------------------------- HELPER FUNCTIONS ---------------------------------

def extract_text_from_pdf(file: BytesIO) -> str:
    """Return concatenated plain text from every page of an in‑memory PDF."""
    text_pages: List[str] = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text_pages.append(page.extract_text() or "")
    return "\n".join(text_pages)

def split_into_chunks(text: str, max_tokens: int) -> List[str]:
    enc = tiktoken.encoding_for_model(MODEL)
    paragraphs = re.split(r"\n{2,}", text)
    chunks, current = [], []
    for para in paragraphs:
        if not para.strip():
            continue
        candidate = "\n".join(current + [para])
        if len(enc.encode(candidate)) > max_tokens:
            chunks.append("\n".join(current))
            current = [para]
        else:
            current.append(para)
    if current:
        chunks.append("\n".join(current))
    return chunks

def build_prompt(chunk: str, n_q: int) -> str:
    return (
        "あなたは厳格な試験委員です。以下の教材抜粋を参考に、"
        f"多肢選択問題を {n_q} 問作成してください。各問題について:\n"
        "1) 質問文（日本語 50 字以内）\n"
        "2) 選択肢 A〜D（A だけ正答）\n"
        "3) 各選択肢の正誤根拠（1 行）\n"
        "出力形式は JSON list とし、要素は {question, choices, explanations} で返す。\n\n"
        "教材抜粋:\n" + chunk
    )

def generate_questions(chunks: List[str]) -> List[Dict]:
    quiz_items: List[Dict] = []
    for chunk in chunks:
        prompt = build_prompt(chunk, N_QUESTIONS_PER_CHUNK)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        try:
            batch = json.loads(response.choices[0].message.content)
            quiz_items.extend(batch)
        except json.JSONDecodeError as e:
            st.warning("⚠️ JSON パースに失敗したので、このチャンクはスキップしました")
    return quiz_items

# ------------------------- STREAMLIT UI -------------------------------------

st.set_page_config(page_title="PDF → 問題集メーカー", page_icon="📝")
st.title("📝 PDF から自動で問題集を生成")

uploaded = st.file_uploader("PDF ファイルをアップロード", type=["pdf"])
if uploaded and OPENAI_API_KEY:
    with st.spinner("PDF を解析中 ..."):
        raw_text = extract_text_from_pdf(uploaded)
        chunks = split_into_chunks(raw_text, TOKENS_PER_CHUNK)
        questions = generate_questions(chunks)

    st.success(f"{len(questions)} 問を生成しました！")

    # 表示
    for idx, q in enumerate(questions, 1):
        st.subheader(f"Q{idx}. {q['question']}")
        for opt_key, opt_text in q["choices"].items():
            label = "✅" if opt_key == "A" else "❌"
            st.write(f"{label} **{opt_key}. {opt_text}**")
        with st.expander("根拠を見る"):
            st.json(q["explanations"], expanded=False)
        st.divider()

    # ダウンロード
    json_bytes = json.dumps(questions, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("📥 JSON をダウンロード", data=json_bytes, file_name="quiz.json", mime="application/json")
elif not OPENAI_API_KEY:
    st.error("OpenAI API キーが設定されていません。Streamlit Secrets または環境変数に OPENAI_API_KEY をセットしてください。")
else:
    st.info("左のサイドバーから PDF を選んでください。")
