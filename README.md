# 🛒 Retail AI Assistant — Function Calling App

A Streamlit-based AI retail assistant powered by Groq LLaMA-3, PDF-based RAG, and 5 intelligent tool-calling functions.

## 🛠️ Tools (All PDF-Powered)

| Tool | Purpose |
|---|---|
| 🔍 `tool_product_info` | Specs, features, warranty, materials |
| 📦 `tool_stock_level` | Stock availability & inventory |
| 📊 `tool_sales_summary` | Revenue, sales reports, best-sellers |
| 🔎 `tool_product_search` | Search by name, category, price |
| 🏷️ `tool_discount_eligibility` | Discounts, promos, coupons |

## 🚀 Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/retail-ai-langchain.git
cd retail-ai-langchain
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API key
```bash
export GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add `GROQ_API_KEY` in **Secrets** settings:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
5. Click **Deploy** — live in minutes!

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key from [console.groq.com](https://console.groq.com) |

## 📦 Tech Stack

- **Streamlit** — UI framework
- **Groq** — LLM inference with function calling (LLaMA 3.1 8B)
- **FAISS** — Vector similarity search
- **Sentence Transformers** — Text embeddings (all-MiniLM-L6-v2)
- **PyPDF2** — PDF text extraction
