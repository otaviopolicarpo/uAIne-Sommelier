# === IMPORTS NECESSÁRIOS ===
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import json


# === HISTÓRICO DE CONVERSA (por sessão) ===
session_store = {}

def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    # Normaliza mensagens já existentes
    history = session_store[session_id]
    history.messages = [
        AIMessage(content=normalize_message(m.content)) if isinstance(m, AIMessage)
        else AIMessage(content=normalize_message(m.content))
        for m in history.messages
    ]
    return history


def normalize_message(msg):
    """Garante que mensagens no histórico sejam sempre strings."""
    if hasattr(msg, "content"):
        msg = msg.content
    if not isinstance(msg, str):
        try:
            msg = json.dumps(msg, ensure_ascii=False)
        except:
            msg = str(msg)
    return msg


# === FUNÇÃO PRINCIPAL PARA CRIAR A RAG CHAIN ===
def create_rag_chain():

    # 1. CARREGAR JSON - mas convertendo para string depois
    loader = JSONLoader(
        "products_catalog.json",
        jq_schema=".",
        text_content=False
    )
    documentos = loader.load()

    # --- 🔧 Correção CRÍTICA: converte page_content (dict/list/etc) para string ---
    for d in documentos:
        if not isinstance(d.page_content, str):
            try:
                d.page_content = json.dumps(d.page_content, ensure_ascii=False)
            except Exception:
                d.page_content = str(d.page_content)


    # 2. SPLITTER
    recur_split = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["}, {"]
    )
    document_split = recur_split.split_documents(documentos)


    # 3. AJUSTAR METADADOS
    for i, doc in enumerate(document_split):
        if 'source' in doc.metadata and isinstance(doc.metadata['source'], str):
            doc.metadata['source'] = doc.metadata['source'].replace('arquivos/', '')
        doc.metadata['doc_id'] = i


    # 4. INDEXAÇÃO FAISS
    vectorstore = FAISS.from_documents(
        documents=document_split,
        embedding=OpenAIEmbeddings()
    )

    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 7, 'fetch_k': 35}
    )


    # 5. PROMPT TEMPLATE + HISTÓRICO
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Você é um sommelier do site Vila Vinhos. Seu nome é uAIne.
                Carismático porém sucinto, você deve fazer recomendações sobre vinhos, harmonizações, preço e perfil do cliente.
                Sempre responda em JSON com as chaves:
                - 'resposta' texto principal da sua resposta, em tom natural e acolhedor.  
                - "recomendacoes": lista de até 3 produtos do catálogo. Cada item deve ter:
                    - "nome"
                    - "preco"
                    - "estoque"
                    - "link"
                    - "descricao"
                - "pergunta_followup": uma pergunta curta (máx. 20 palavras) que direcione a conversa,
                EXCLUSIVAMENTE quando houver recomendações.
                
                - Se não houver produtos adequados, retorne uma resposta educada e SET apenas 'resposta'.
                - Você vai receber uma variável "chat_history" (texto) com os últimos turns da conversa.
                - Use "chat_history" para entender as interações passadas e manter a coerência da conversa.
                - Você vai receber uma lista "vinhos_ja_mostrados" com nomes já recomendados.
                - Caso o cliente peça novas recomendações NÃO repita um vinho que esteja em "vinhos_ja_mostrados".
                - Demonstre sua atenção ao pedido do usuário e andamento da conversa.
                """
            ),
            MessagesPlaceholder("chat_history"),
            (
                "human",
                "Catálogo:\n{catalog}\n\nPergunta: {pergunta}"
            ),
        ]
    )


    # 🔧 Junta documentos do retriever para o prompt, garantindo conversão em string
    def join_documents(input_dict):
        catalog_items = []
        for c in input_dict.get('catalog', []):
            content = getattr(c, "page_content", c)
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False)
                except:
                    content = str(content)
            catalog_items.append(content)

        input_dict['catalog'] = '\n\n'.join(catalog_items)
        return input_dict


    # 🔥 Força a query para o retriever ser sempre string simples
    def only_query(x):
        # "x" aqui é {"pergunta": "..."}
        q = x["pergunta"]
        if not isinstance(q, str):
            try:
                return json.dumps(q, ensure_ascii=False)
            except:
                return str(q)
        return q

    setup = RunnableParallel(
        {
            'pergunta': RunnablePassthrough(),
            'catalog': only_query | retriever  # <----- aqui está o segredo
        }
    ) | join_documents



    # 6. LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.15,
        response_format={"type": "json_object"}
    )


    # Injeta o histórico no input antes de ir para o prompt
    def inject_history(x):
        x["chat_history"] = x.get("chat_history", [])
        return x

    base_chain = setup | inject_history | prompt | llm


    # 🔧 Garante que a saída SEMPRE seja AIMessage com conteúdo string (evita crashes silenciosos)
    def force_aimessage(output):
        if isinstance(output, dict):
            return AIMessage(content=json.dumps(output, ensure_ascii=False))
        if hasattr(output, "content") and isinstance(output.content, dict):
            return AIMessage(content=json.dumps(output.content, ensure_ascii=False))
        if isinstance(output, AIMessage):
            return AIMessage(content=normalize_message(output.content))
        return AIMessage(content=normalize_message(output))


    final_chain = base_chain | force_aimessage


    # === ADICIONA HISTÓRICO À CADEIA ===
    chain_with_history = RunnableWithMessageHistory(
        final_chain,
        get_session_history,
        input_messages_key="pergunta",
        history_messages_key="chat_history",
    )

    return chain_with_history




