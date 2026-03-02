# -*- coding: utf-8 -*-
import streamlit as st
import os, json
from groq import Groq
import PyPDF2
import faiss
import numpy as np

# ─────────────────────────────────────────
#  CACHE MODEL
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="⚡ Loading AI model (first time only)...")
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

EMBED_MODEL = load_embed_model()

@st.cache_resource
def get_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

client = get_groq_client()

# ─────────────────────────────────────────
#  PAGE CONFIG + CSS
# ─────────────────────────────────────────
st.set_page_config(page_title="Retail AI Assistant", page_icon="🛒", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0d0d0d; color: #f0ece4; }
.main { background-color: #0d0d0d; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }
.store-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #e94560; border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem; text-align: center;
}
.store-header h1 { font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800; color: #fff; margin: 0; }
.store-header p  { color: #a0aec0; margin: 0.4rem 0 0 0; font-size: 0.95rem; }
.accent { color: #e94560; }
.tool-badge {
    display: inline-block; background: #e9456022; border: 1px solid #e94560;
    color: #e94560; border-radius: 20px; padding: 2px 12px;
    font-size: 0.75rem; font-weight: 600; margin-bottom: 0.5rem;
}
.chat-user      { background: #1a1a2e; border-left: 3px solid #e94560; border-radius: 10px; padding: 0.8rem 1rem; margin: 0.5rem 0; }
.chat-assistant { background: #16213e; border-left: 3px solid #4ecdc4; border-radius: 10px; padding: 0.8rem 1rem; margin: 0.5rem 0; }
.status-ok   { color: #4ecdc4; font-weight: 600; }
.status-warn { color: #ffd93d; font-weight: 600; }
section[data-testid="stSidebar"] { background: #111111; border-right: 1px solid #222; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────
for key, default in {
    "rag_index":    None,
    "rag_chunks":   [],
    "pdf_loaded":   False,
    "pdf_name":     "",
    "chat_history": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────
#  PDF PROCESSING
# ─────────────────────────────────────────
def extract_and_chunk_pdf(pdf_file, chunk_size=600, overlap=60):
    reader     = PyPDF2.PdfReader(pdf_file)
    all_chunks = []
    buffer     = []
    for page in reader.pages:
        text = page.extract_text() or ""
        buffer.extend(text.split())
        while len(buffer) >= chunk_size:
            all_chunks.append(" ".join(buffer[:chunk_size]))
            buffer = buffer[chunk_size - overlap:]
    if buffer:
        all_chunks.append(" ".join(buffer))
    return all_chunks

def build_faiss_index(chunks, progress_bar):
    batch_size     = 32
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        emb   = EMBED_MODEL.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(emb)
        progress = min((i + batch_size) / len(chunks), 1.0)
        progress_bar.progress(progress, text=f"⚡ Indexing... {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")
    embeddings = np.vstack(all_embeddings).astype("float32")
    index      = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# ─────────────────────────────────────────
#  CORE RAG RETRIEVER
# ─────────────────────────────────────────
def retrieve(query: str, top_k: int = 5) -> str:
    """Search the uploaded PDF and return the most relevant chunks."""
    if not st.session_state.pdf_loaded:
        return "NO_PDF"
    q_emb = EMBED_MODEL.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = st.session_state.rag_index.search(q_emb, top_k)
    chunks = [
        st.session_state.rag_chunks[i]
        for i in indices[0]
        if i < len(st.session_state.rag_chunks)
    ]
    return "\n\n".join(chunks) if chunks else "NOT_FOUND"

# ─────────────────────────────────────────
#  5 TOOLS — ALL PDF-DRIVEN
# ─────────────────────────────────────────

def tool_product_info(product_name: str) -> str:
    """Specs, features, warranty, materials — all from PDF."""
    context = retrieve(f"{product_name} specifications features warranty materials details")
    if context == "NO_PDF":
        return json.dumps({"error": "No PDF uploaded. Please upload your product dataset first."})
    if context == "NOT_FOUND":
        return json.dumps({"error": f"No information found for '{product_name}' in the uploaded dataset."})
    return json.dumps({"product": product_name, "source": "PDF", "details": context})

def tool_stock_level(product_name: str) -> str:
    """Stock availability and inventory levels — from PDF."""
    context = retrieve(f"{product_name} stock inventory quantity available units")
    if context == "NO_PDF":
        return json.dumps({"error": "No PDF uploaded."})
    if context == "NOT_FOUND":
        return json.dumps({"error": f"No stock information found for '{product_name}' in the dataset."})
    return json.dumps({"product": product_name, "source": "PDF", "stock_info": context})

def tool_sales_summary(period: str) -> str:
    """Sales performance, revenue, best-sellers — from PDF."""
    context = retrieve(f"{period} sales revenue units sold best selling performance summary report")
    if context == "NO_PDF":
        return json.dumps({"error": "No PDF uploaded."})
    if context == "NOT_FOUND":
        return json.dumps({"error": f"No {period} sales data found in the dataset."})
    return json.dumps({"period": period, "source": "PDF", "sales_data": context})

def tool_product_search(query: str, category: str = "", price_range: str = "") -> str:
    """Search products by keyword, category, or price — from PDF."""
    search_query = f"{query}"
    if category:
        search_query += f" {category} category"
    if price_range:
        search_query += f" price {price_range}"
    context = retrieve(search_query, top_k=6)
    if context == "NO_PDF":
        return json.dumps({"error": "No PDF uploaded."})
    if context == "NOT_FOUND":
        return json.dumps({"error": f"No products found matching '{query}' in the dataset."})
    return json.dumps({"query": query, "category": category, "price_range": price_range, "source": "PDF", "results": context})

def tool_discount_eligibility(product_or_coupon: str) -> str:
    """Discount offers, promo codes, eligibility criteria — from PDF."""
    context = retrieve(f"{product_or_coupon} discount offer promo coupon eligibility price reduction sale")
    if context == "NO_PDF":
        return json.dumps({"error": "No PDF uploaded."})
    if context == "NOT_FOUND":
        return json.dumps({"error": f"No discount information found for '{product_or_coupon}' in the dataset."})
    return json.dumps({"query": product_or_coupon, "source": "PDF", "discount_info": context})

# ─────────────────────────────────────────
#  GROQ TOOL SCHEMAS
# ─────────────────────────────────────────
TOOLS = [
    {"type": "function", "function": {
        "name": "tool_product_info",
        "description": "Retrieve detailed product information including specifications, features, warranty, and materials from the uploaded PDF dataset.",
        "parameters": {"type": "object",
            "properties": {"product_name": {"type": "string", "description": "Name of the product to look up"}},
            "required": ["product_name"]}
    }},
    {"type": "function", "function": {
        "name": "tool_stock_level",
        "description": "Check stock availability, inventory levels, and quantity remaining for a product from the uploaded PDF dataset.",
        "parameters": {"type": "object",
            "properties": {"product_name": {"type": "string", "description": "Product name to check stock for"}},
            "required": ["product_name"]}
    }},
    {"type": "function", "function": {
        "name": "tool_sales_summary",
        "description": "Get sales performance data, revenue figures, best-selling products, and summary reports from the uploaded PDF dataset.",
        "parameters": {"type": "object",
            "properties": {"period": {"type": "string", "description": "Time period or report type: daily, weekly, monthly, best_sellers, or any specific period"}},
            "required": ["period"]}
    }},
    {"type": "function", "function": {
        "name": "tool_product_search",
        "description": "Search and filter products by keyword, name, category, or price range from the uploaded PDF dataset.",
        "parameters": {"type": "object",
            "properties": {
                "query":       {"type": "string",  "description": "Search keyword or product name"},
                "category":    {"type": "string",  "description": "Product category to filter by"},
                "price_range": {"type": "string",  "description": "Price range e.g. 'under 500' or '100 to 300'"}
            }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "tool_discount_eligibility",
        "description": "Find discount offers, promotional deals, coupon codes, and eligibility criteria from the uploaded PDF dataset.",
        "parameters": {"type": "object",
            "properties": {"product_or_coupon": {"type": "string", "description": "Product name or coupon code to check discounts for"}},
            "required": ["product_or_coupon"]}
    }},
]

TOOL_MAP = {
    "tool_product_info":         tool_product_info,
    "tool_stock_level":          tool_stock_level,
    "tool_sales_summary":        tool_sales_summary,
    "tool_product_search":       tool_product_search,
    "tool_discount_eligibility": tool_discount_eligibility,
}

# ─────────────────────────────────────────
#  MASTER SYSTEM PROMPT
# ─────────────────────────────────────────
MASTER_PROMPT = """You are an AI Powered Retail Store Assistant.
Your ONLY knowledge source is the uploaded PDF dataset.
You have 5 tools — ALL of them retrieve answers directly from the PDF.

TOOL ROUTING RULES:
- Product specs / features / warranty / materials  → tool_product_info
- Stock levels / inventory / availability          → tool_stock_level
- Sales / revenue / best-sellers / reports         → tool_sales_summary
- Search by name / category / price               → tool_product_search
- Discounts / coupons / offers / promotions        → tool_discount_eligibility

STRICT RULES:
1. ALWAYS call the appropriate tool — never answer from memory.
2. Use ONLY the tool result to generate the final answer.
3. DO NOT mention phrases like "based on the dataset",
   "according to the dataset", or "as per the tool output".
4. Present the answer naturally and directly.
5. If the tool returns an error or nothing relevant, say exactly:
   "This information is not available in the uploaded dataset."
6. Never fabricate prices, stock numbers, or product details.
7. Be professional, concise, and helpful.
"""

# ─────────────────────────────────────────
#  CHAT ENGINE
# ─────────────────────────────────────────
def chat_with_groq(user_message, history):
    messages = [{"role": "system", "content": MASTER_PROMPT}]
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_message})

    tool_used = None
    response  = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages, tools=TOOLS, tool_choice="auto", max_tokens=800
    )
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        })
        for tc in msg.tool_calls:
            tool_used = tc.function.name
            try:
                result = TOOL_MAP[tc.function.name](**json.loads(tc.function.arguments))
            except Exception as e:
                result = json.dumps({"error": str(e)})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        final = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages, max_tokens=800
        )
        return final.choices[0].message.content, tool_used

    return msg.content, tool_used

# ─────────────────────────────────────────
#  SIDEBAR — PDF UPLOAD
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Upload Product Dataset PDF")
    uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_pdf and uploaded_pdf.name != st.session_state.pdf_name:
        st.info(f"📄 `{uploaded_pdf.name}`")

        with st.spinner("📄 Reading PDF..."):
            chunks = extract_and_chunk_pdf(uploaded_pdf)
        st.success(f"✅ {len(chunks)} chunks extracted")

        progress_bar = st.progress(0, text="⚡ Indexing...")
        index = build_faiss_index(chunks, progress_bar)
        progress_bar.progress(1.0, text="✅ Done!")

        st.session_state.rag_index  = index
        st.session_state.rag_chunks = chunks
        st.session_state.pdf_loaded = True
        st.session_state.pdf_name   = uploaded_pdf.name
        st.success("🚀 PDF ready! Start asking questions.")

    st.markdown("---")
    st.markdown("### 🛠️ Tools (All PDF-Powered)")
    for icon, name in [
        ("🔍", "Product Info"),
        ("📦", "Stock Levels"),
        ("📊", "Sales Summary"),
        ("🔎", "Product Search"),
        ("🏷️", "Discounts & Offers")
    ]:
        st.markdown(f"{icon} {name}")

    st.markdown("---")
    st.markdown("### 💬 Example Questions")
    for q in [
        "What are the specs of [product]?",
        "Is [product] available in stock?",
        "Show monthly sales summary",
        "Find products under $500",
        "What discounts are available?",
    ]:
        st.markdown(f"• *{q}*")

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ─────────────────────────────────────────
#  MAIN UI
# ─────────────────────────────────────────
st.markdown("""
<div class='store-header'>
    <h1>🛒 Retail <span class='accent'>AI</span> Assistant</h1>
    <p>100% PDF-Powered · Groq LLM · 5 Intelligent Tools</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    if st.session_state.pdf_loaded:
        st.markdown(f"<p class='status-ok'>✅ {st.session_state.pdf_name}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='status-warn'>⚠️ Upload a PDF to begin</p>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<p class='status-ok'>💬 {len(st.session_state.chat_history)} messages</p>", unsafe_allow_html=True)
with col3:
    st.markdown("<p class='status-ok'>🤖 LLaMA3-8B (Groq)</p>", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────
#  CHAT DISPLAY
# ─────────────────────────────────────────
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-user'>👤 <b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        badge = f"<span class='tool-badge'>🔧 {msg.get('tool_used','')}</span><br>" if msg.get("tool_used") else ""
        st.markdown(f"<div class='chat-assistant'>{badge}🤖 <b>Assistant:</b><br>{msg['content']}</div>", unsafe_allow_html=True)

if not st.session_state.pdf_loaded:
    st.warning("⬆️ Please upload your product dataset PDF from the sidebar to start chatting.")
    st.stop()

# ─────────────────────────────────────────
#  CHAT INPUT
# ─────────────────────────────────────────
with st.form(key="chat_form", clear_on_submit=True):
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        user_input = st.text_input("", placeholder="Ask anything about your products...", label_visibility="collapsed")
    with col_btn:
        submitted = st.form_submit_button("Send 🚀")

if submitted and user_input.strip():
    with st.spinner("⚡ Searching your dataset..."):
        response, tool_used = chat_with_groq(user_input, st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "user",      "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": response, "tool_used": tool_used})
    st.rerun()
