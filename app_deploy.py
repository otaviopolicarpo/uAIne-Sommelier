import streamlit as st
import json
import uuid
from RAG import create_rag_chain

import os, streamlit as st

# =========================================================
# Streamlit Community Cloud / Env Vars
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else os.environ.get("OPENAI_API_KEY")
LANGCHAIN_KEY = st.secrets.get("LANGCHAIN_API_KEY") if "LANGCHAIN_API_KEY" in st.secrets else os.environ.get("LANGCHAIN_API_KEY")

# exporta para env (LangChain / LangSmith libs vão ler dessas envs)
if OPENAI_KEY:
    os.environ.setdefault("OPENAI_API_KEY", OPENAI_KEY)
if LANGCHAIN_KEY:
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGCHAIN_KEY)

# habilitar tracing V2
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.environ.get("LANGCHAIN_TRACING_V2", "true"))
os.environ.setdefault("LANGCHAIN_ENDPOINT", os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"))
os.environ.setdefault("LANGCHAIN_PROJECT", os.environ.get("LANGCHAIN_PROJECT","AdeguIA"))
os.environ.setdefault("LANGCHAIN_CALLBACKS_BACKGROUND", os.environ.get("LANGCHAIN_CALLBACKS_BACKGROUND", "true"))
# =========================================================


st.set_page_config(page_title="uAIne 🍷", page_icon="🍇")
st.title("🍷 uAIne – Concierge Enólogo Virtual")
st.write("Pergunte sobre vinhos, harmonizações, ocasiões especiais e encontre a escolha certa do catálogo Vila Vinho.")


# =========================================================
# INIT
# =========================================================
if "chain" not in st.session_state:
    st.session_state.chain = create_rag_chain()

if "chat_history" not in st.session_state:
    # Cada item = { role: "user"/"assistant", text: "...", html: "..." }
    st.session_state.chat_history = []

if "vinhos_recomendados" not in st.session_state:
    st.session_state.vinhos_recomendados = set()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())



# =========================================================
# HTML SANITIZATION
# =========================================================
def sanitize_html_for_markdown(html: str) -> str:
    """
    Remove indentação que o Markdown interpreta como bloco de código.
    Garante que <div> seja entendido como HTML novamente.
    """
    if not isinstance(html, str):
        return str(html)

    lines = [ln.lstrip() for ln in html.splitlines()]
    s = "\n".join(lines).strip()
    return s



# =========================================================
# HISTÓRICO PARA O MODELO (texto limpo)
# =========================================================
def chat_history_to_text_for_model(chat_history, max_turns=12) -> str:
    """
    Converte o histórico em texto simples (sem HTML), rotulado por role.
    Isso dá ao modelo o contexto REAL da conversa.
    """
    turns = []
    for msg in chat_history[-max_turns:]:
        role = msg.get("role", "user")

        if role == "user":
            text = msg.get("text", "")
        else:
            # Extrair texto limpo do HTML
            html = msg.get("html", "")
            plain = html.replace("<br>", "\n").replace("<br/>", "\n")
            import re
            plain = re.sub(r"<[^>]+>", "", plain)
            text = plain.strip()

        turns.append(f"{role.upper()}: {text}")

    return "\n".join(turns)



# =========================================================
# CARD RENDERER
# =========================================================
def render_response_html(data: dict) -> str:
    resposta = data.get("resposta", "")
    recs = data.get("recomendacoes", [])
    follow = data.get("pergunta_followup")

    html = f"""
        <div style='font-family:sans-serif; font-size:16px; color:#333; margin:0 0 8px 0;'>
            🍷 {resposta}
        </div>
    """

    if recs and len(recs) > 0:
        for item in recs:
            html += f"""
            <div style="border:1px solid #e6e6e6; padding:12px; border-radius:10px;
                        margin:8px 0; background:#fafafa;">
                
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-size:17px; font-weight:600;">{item.get("nome")}</div>
                    <div style="font-size:15px; font-weight:700; color:#8A0303;">
                        {item.get("preco")}
                    </div>
                </div>

                <div style="margin-top:8px; color:#444; line-height:1.4;">
                    {item.get("descricao")}
                </div>

                <div style="margin-top:10px;">
                    <a href="{item.get("link")}" target="_blank"
                       style="text-decoration:none; padding:7px 12px; border-radius:6px;
                              background:#8A0303; color:#fff; font-weight:600;">
                        🔗 Ver produto
                    </a>
                </div>

            </div>
            """

        if follow:
            html += f"""
            <div style='font-family:sans-serif; font-size:16px; color:#333; margin-top:10px;'>
                💬 {follow}
            </div>
            """

    return html



# =========================================================
# RENDER HISTÓRICO (HTML via markdown + sanitização)
# =========================================================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg.get("text", ""))
        else:
            safe_html = sanitize_html_for_markdown(msg.get("html", ""))
            st.markdown(safe_html, unsafe_allow_html=True)



# =========================================================
# USER INPUT
# =========================================================
user_input = st.chat_input("Me pergunte algo sobre vinhos... 🍷")

if user_input:

    # Salvar input no histórico
    st.session_state.chat_history.append({"role": "user", "text": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Placeholder/spinner
    placeholder = st.empty()
    with placeholder.container():
        with st.spinner("Analisando catálogo Vila Vinho... 🍇"):
            pass

    # Montar histórico limpo para o modelo
    hist_text = chat_history_to_text_for_model(st.session_state.chat_history)

    # INVOKE
    response = st.session_state.chain.invoke(
        {
            "pergunta": user_input,
            "vinhos_ja_mostrados": list(st.session_state.vinhos_recomendados),
            "chat_history": hist_text
        },
        config={
            "configurable": {"session_id": st.session_state.session_id}
        }
    )

    placeholder.empty()

    # Extrair JSON
    raw = response.content if hasattr(response, "content") else response
    data = json.loads(raw) if isinstance(raw, str) else raw

    # Atualizar lista de vinhos já mostrados
    for item in data.get("recomendacoes", []):
        nome = item.get("nome")
        if nome:
            st.session_state.vinhos_recomendados.add(nome)

    # Gerar HTML final
    html = render_response_html(data)

    # Render da resposta atual (mais bonito usando components)
    with st.chat_message("assistant"):
        st.components.v1.html(html, height=650, scrolling=True)



    # Salvar no histórico (sanitização apenas na exibição)
    st.session_state.chat_history.append({"role": "assistant", "html": html})
