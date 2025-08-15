import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Helper function to clean text ---
def safe_summary(text):
    banned_words = ["pornography", "explicit", "sexual", "adult"]
    words = text.split()
    return " ".join(w for w in words if w.lower() not in banned_words)

# --- Wrappers with filtered output ---
class SafeWikipediaWrapper(WikipediaAPIWrapper):
    def run(self, query: str) -> str:
        result = super().run(query)
        return safe_summary(result)

class SafeArxivWrapper(ArxivAPIWrapper):
    def run(self, query: str) -> str:
        result = super().run(query)
        return safe_summary(result)

# Tools
arxiv_wrapper = SafeArxivWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = SafeWikipediaWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# --- Streamlit UI ---
st.title(" LangChain - Chat with search")

st.sidebar.title("Settings")
api_key_input = st.sidebar.text_input("Enter your Groq API Key:", type="password")
api_key = api_key_input or os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("⚠️ Please provide a Groq API key in the sidebar")
    st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]

    #  Agent won't crash on parsing errors
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({'role': 'assistant', "content": response})
        st.write(response)
