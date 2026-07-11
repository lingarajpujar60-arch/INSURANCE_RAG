import os
import streamlit as st
from dotenv import load_dotenv

from langchain import hub
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(
    page_title="Insurance Genie",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------
# CSS
# ---------------------------
st.markdown("""
<style>

.main{
    background:#f5f7fb;
}

.block-container{
    padding-top:2rem;
}

h1{
    color:#12355B;
    text-align:center;
}

.stChatMessage{
    border-radius:15px;
}

div[data-testid="stSidebar"]{
    background:#12355B;
}

div[data-testid="stSidebar"] *{
    color:white;
}

</style>
""", unsafe_allow_html=True)

DB_FAISS_PATH = "vectorstore/db_faiss"

# ---------------------------
# Load Vector Store
# ---------------------------
@st.cache_resource
def get_vectorstore():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding,
        allow_dangerous_deserialization=True
    )

    return db


# ---------------------------
# Load LLM
# ---------------------------
@st.cache_resource
def load_llm():

    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0.4,
        max_tokens=512
    )


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:

    st.title("🛡️ Insurance Genie")
    st.caption("Your AI Insurance Assistant")

    st.markdown("---")

    st.button("🏠 Home", use_container_width=True)
    st.button("📜 Chat History", use_container_width=True)
    st.button("📄 Upload Policy", use_container_width=True)

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.subheader("Suggested Questions")

    if st.button("🏥 What is Health Insurance?"):
        st.session_state.prompt = "What is Health Insurance?"

    if st.button("💰 How to file a claim?"):
        st.session_state.prompt = "How to file a claim?"

    if st.button("🛡 Difference between Term & Life Insurance?"):
        st.session_state.prompt = "Difference between Term & Life Insurance"

    if st.button("⏳ What is waiting period?"):
        st.session_state.prompt = "What is waiting period?"

# ---------------------------
# Main UI
# ---------------------------

st.title("🤖 Insurance Genie")

st.write("Ask anything about insurance policies, claims, premium, coverage and more.")

st.markdown("---")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome Message
if len(st.session_state.messages) == 0:
    st.chat_message("assistant").markdown(
        """
Hello 👋

I'm **Insurance Genie**.

I can help you understand:

- Insurance Policies
- Claims
- Premiums
- Coverage
- Benefits

Ask me anything!
"""
    )

# Display Chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Quick Buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🏥 Health Insurance"):
        st.session_state.prompt = "Explain Health Insurance"

with col2:
    if st.button("💰 Claim Process"):
        st.session_state.prompt = "Explain Claim Process"

with col3:
    if st.button("📄 Policy Coverage"):
        st.session_state.prompt = "Explain Policy Coverage"

# Chat Input
prompt = st.chat_input(
    "Ask your insurance question..."
)

if prompt is None:
    prompt = st.session_state.pop("prompt", None)

# ---------------------------
# RAG
# ---------------------------
if prompt:

    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    try:

        db = get_vectorstore()

        llm = load_llm()

        retrieval_prompt = hub.pull(
            "langchain-ai/retrieval-qa-chat"
        )

        combine_docs_chain = create_stuff_documents_chain(
            llm,
            retrieval_prompt
        )

        rag_chain = create_retrieval_chain(
            db.as_retriever(search_kwargs={"k": 3}),
            combine_docs_chain
        )

        response = rag_chain.invoke(
            {
                "input": prompt
            }
        )

        answer = response["answer"]

    except Exception as e:

        answer = f"❌ Error: {e}"

    st.chat_message("assistant").markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

# ---------------------------
# Footer
# ---------------------------

st.markdown("---")

st.markdown(
    "<center>🛡️ Insurance Genie | AI Insurance Assistant</center>",
    unsafe_allow_html=True
)