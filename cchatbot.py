import streamlit as st
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from textblob import TextBlob

# ----------------------------
# Load glossary
# ----------------------------
@st.cache_resource
def load_glossary():
    with open("construction_glossary.json", "r") as f:
        return json.load(f)

knowledge_base = load_glossary()

# ----------------------------
# Load AI model
# ----------------------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

nlp = load_model()

# ----------------------------
# Helper Functions
# ----------------------------
def correct_spelling(text):
    return str(TextBlob(text).correct())

def is_construction_related(text):
    return any(word in text.lower() for word in knowledge_base.keys())

def get_knowledge_answer(user_input):
    for key, value in knowledge_base.items():
        if key in user_input.lower():
            return value
    return None

def generate_response(user_input, history):
    kb_answer = get_knowledge_answer(user_input)
    if kb_answer:
        return kb_answer

    context = "\n".join([f"{u}: {m}" for u, m in history[-3:]])
    prompt = f"""
    You are a civil engineer expert.
    Answer clearly and only about construction materials, IS codes, and civil engineering.
    If irrelevant, reply: "Sorry, I only answer construction-related questions."

    Chat history:
    {context}

    User: {user_input}
    """
    response = nlp(prompt, max_length=250, num_return_sequences=1)[0]["generated_text"]
    return response

# ----------------------------
# Custom CSS for UI
# ----------------------------
st.set_page_config(page_title="Construction AI Assistant", page_icon="👷‍♂️", layout="centered")

st.markdown("""
    <style>
    body {background-color:#f4f6f7;}
    .chat-container {
        max-width: 700px;
        margin: auto;
        padding: 15px;
        border-radius: 12px;
        background: #ffffff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .user-bubble {
        text-align:right; background:#3498db; color:white;
        padding:10px; border-radius:15px; margin:5px; display:inline-block; max-width:70%;
    }
    .bot-bubble {
        text-align:left; background:#2ecc71; color:white;
        padding:10px; border-radius:15px; margin:5px; display:inline-block; max-width:70%;
    }
    .header { text-align:center; margin-bottom:15px; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Header with custom logo
# ----------------------------
col1, col2, col3 = st.columns([1,3,1])
with col2:
    st.image("https://img.icons8.com/color/96/construction-worker.png", width=120)  # <-- place a construction logo in same folder
    st.markdown("<h1 style='text-align:center;'>Construction AI Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Your smart civil engineering helper</p>", unsafe_allow_html=True)

# ----------------------------
# Chat History
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for sender, msg in st.session_state.history:
    if sender == "👤 You":
        st.markdown(f"<div class='user-bubble'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Input (Text only)
# ----------------------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("⌨️ Type your question")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    corrected = correct_spelling(user_input)
    if not is_construction_related(corrected):
        answer = "⚠️ Sorry, I only answer construction-related questions."
    else:
        if corrected.lower() != user_input.lower():
            st.info(f"🔧 Did you mean: **{corrected}** ?")
        answer = generate_response(corrected, st.session_state.history)

    st.session_state.history.append(("👤 You", user_input))
    st.session_state.history.append(("🤖 Bot", answer))
    st.rerun()
