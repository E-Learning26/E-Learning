# Dieser Code ist geschrieben Anlehung an
# https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
from typing import TypedDict, Annotated, List
import streamlit as st
import os
from langchain_core.messages import SystemMessage, AnyMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, END
import torch
from message_handler import MessageHandler
from search_tool import SearchTool
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from operator import add
from langchain_core.messages import AIMessage, AIMessageChunk

# Avatare (externe Dateien)
USER_AVATAR = "static/avatars/user.png"
ASSISTANT_AVATAR = "static/avatars/assistant.png"

load_dotenv()
FAQ = True
SYSTEM_PROMPT=("Du bist ein freundlicher Lern-Assistent mit dem Namen Rocky. Frage den Benutzer, ob ihm die Lerneinheiten gefallen haben. Gib als Beispielthemen Feststoffzerkleinern, der Energieaufwand für Mühlen an und Unklarheiten beim Erstellen des Protokolls, seiner Inhalte oder Verständnisprobleme und"
               "frage den Benutzer, wo bei ihm generell noch Unsicherheiten oder Unklarheiten bestehen."
                "Wenn du das Such-Tool verwendest, gib bei jeder Antwort die Quellenangabe deiner Antwort unbedingt an, außer bei Fragen zu den Protokollen, also Aufbau und Zusammenfassungen oder so. "
                "Gib die Benutzerfrage unverändert an das Tool weiter. Wenn du das Such-Tool verwendest, formatiere die Quellenangaben aus den Metadaten (Feld""*metadatas* im zurückgelieferten Objekt des SearchTools"
               "mit nummerierten Referenzen (z.B. [1], [2], [3]) im Text und der entsprechenden Quellenangabe am Ende des Textes (z.B.: [1] ZOGG_Einführung in die Mechanische VT.pdf, Kapitel 3. Kuchenfiltration, S. 23) [2] ZOGG_Einführung in die Mechanische VT.pdf, Kapitel 4 Druckfiltration, S. 58 [3] ZOGG_Einführung in die Mechanische VT.pdf"
               ".pdf, Kapitel 10 Vakuumfiltration, S. 230-260. Seitenzahlen kommen immer aus den Metadaten. Kapitelangaben aus dem Text."
                "Bei Fragen zu Beispielvideos von Brechern, Pressen und Walzen, frage welcher Art jeweils und gib den dazugehörigen Youtube-Link, formatiert als Link aus folgender Linkliste:"
                "Prallbrecher Metso Corps Youtube-Link: http://www.youtube.com/watch?v=nSiec3350OI&NR=1"
                "Prallmühle UK-Youtube-Link: http://www.youtube.com/watch?v=RnJps4Vnj3I&NR=1"
                "Kegelbrecher Metso Corp Youtube-Link: http://www.youtube.com/watch?v=tCAdq_AQzrI&NR=1"
                "Kegelbrecher UK: Youtube-Link: http://www.youtube.com/watch?v=c0UV0ArYMAg&feature=related, Youtube-Link: http://www.youtube.com/watch?v=716JzyX-ygc&NR=1"
                "Kugelmühle groß - Link: http://www.youtube.com/watch?v=4YKebs9zbPQ&feature=related"
                "Kugelmühle klein Youtube-Link: http://www.youtube.com/watch?v=blhEY-73qjo&feature=related"
                "Erzaufbereitung Youtube-Link: http://www.youtube.com/watch?v=rkDw1SAwksg&NR=1&feature=fvwp"
                "Kugelmühle IWF Youtube-Link: http://www.youtube.com/watch?v=O2-gg0C2Asw&feature=related"
                "Kugelmühle simulation Youtube-Link: http://www.youtube.com/watch?v=LYHzp6EcqL8&NR=1"
                "Backenbrecher Youtube-Link: http://www.youtube.com/watch?v=VqX47VNRDw4&feature=related"
                "Backenbrecher Retsch Youtube-Link: http://www.youtube.com/watch?v=Hv6UvLRr30Y&feature=related"
                "Backenbrecher Hartl Youtube-Link: http://www.youtube.com/watch?v=w7uTk_0zy0A&feature=related"
                "Backenbrecher kruszarka Doppelkniehebel Youtube-Link: http://www.youtube.com/watch?v=H4Qjk5W-whY&feature=related"
                "Backenbrecher kruszarka Einfachkniehebel http://www.youtube.com/watch?v=E5yW_h6Fec4&feature=related"
                "Backenbrecher UK Youtube-Link: http://www.youtube.com/watch?v=zvkEn6oytV8&feature=related"
                "Prallmühle UK Youtube-Link: http://www.youtube.com/user/AggNet#p/u/8/ncwe9e_FO8Y"
                "Screens Youtube-Link: http://www.youtube.com/user/AggNet#p/u/2/tkBC0V-C1Uk"
                "Sizer Youtube-Link: http://www.youtube.com/user/AggNet#p/u/6/NpW_uc_60Ws"
                "Kammerfilterpresse Youtube-Link: http://www.youtube.com/watch?v=btsXMiVtjcw&feature=related, Youtube-Link: http://www.youtube.com/watch?v=Eo8ce3_V6ic&feature=related, Youtube-Link: http://www.youtube.com/watch?v=03fIXtTDhpU&feature=related, Youtube-Link: http://www.youtube.com/watch?v=n6fTdvqCWk8&feature=related"
                "Anschwemmfilter, Youtube-Link: http://www.youtube.com/watch?v=9I2tfpGn8wo, Youtube-Link: http://www.fesfilter.de/vollautomatische-anschwemmfilter.html"
                "Filterkerzen – Tiefenfiltration, Youtube-Link: http://www.youtube.com/watch?v=sMpJZ3LiNvM")
MODEL_NAME = "openai/gpt-5-mini"
MAX_TOKEN = 24000

# Initialisiere Nachrichten
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialisiere das Basis-LLM
if "base_llm" not in st.session_state:
    st.session_state.base_llm = ChatOpenAI(
        #api_key=os.getenv("OPENROUTER_API_KEY"),
        api_key=st.secrets["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        model=MODEL_NAME,
        temperature=0.0,
        streaming=True
    )

# Initialisiere das Suchtool, falls im FAQ-Modus
if FAQ:
    if "tools_node" not in st.session_state:
        client = chromadb.PersistentClient(path="./chroma_neu")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        emb = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="jinaai/jina-embeddings-v2-base-de",
            device=device
        )
        print("Using device:", device)

        collection = client.get_or_create_collection(
            "verfahrenstechnik",
            embedding_function=emb)

        search_tool = SearchTool(collection)
        TOOLS = [search_tool]
        st.session_state.tools_node = ToolNode(TOOLS)
        st.session_state.llm = st.session_state.base_llm.bind_tools(TOOLS)
# Andernfalls ist das LLM das Base-LLM ohne Tools
else:
    st.session_state.llm = st.session_state.base_llm


# Track Nachrichten (messages) und speichere das LLM-Objekt, damit es
# beim Nachrichtenstreaming nicht verloren geht
class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add]
    llm: object


# Lese die Nachrichten und das LLM-Objekt aus dem Status des Graphs
# Nimm die nächste KI-Nachricht
def chat_node(state: GraphState) -> dict:
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    llm = state.get("llm")
    ai = llm.invoke(msgs)
    return {"messages": [ai]}


if "app_graph" not in st.session_state:
    graph = StateGraph(GraphState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    if FAQ:
        graph.add_node("tools", st.session_state.tools_node)
        graph.add_conditional_edges("chat", tools_condition, {"tools": "tools", "__end__": END})
        graph.add_edge("tools", "chat")
    else:
        graph.add_edge("chat", END)
    app_graph = graph.compile()
    st.session_state.app_graph = app_graph


# Zeige, die Chat-Historie an, falls es eine gibt.
for role, content in st.session_state.messages:
    if role == "user":
        avatar = USER_AVATAR
    else:
        avatar = ASSISTANT_AVATAR

    with st.chat_message(role, avatar=avatar):
        st.write(content)

# RAG-Chat auf Basis von Nutzereingaben
if prompt := st.chat_input("Frag, für mehr Informationen!"):
    st.session_state.messages.append(("user", prompt))
    content = st.session_state.messages[-1][1]
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(content)

    history_msgs = MessageHandler(model=MODEL_NAME.split("/")[-1],max_tokens=24000)
    for role, content in st.session_state.messages:
        history_msgs.add_message(HumanMessage(content=content) if role == "user" else AIMessage(content=content))

    # Nachrichten streamen
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        full_response = ""
        message_placeholder = st.empty()

        for event in st.session_state.app_graph.stream({"messages": history_msgs.get_conversation(), "llm": st.session_state.llm}, stream_mode="messages"):
            # Extract content from the event
            if isinstance(event[0], AIMessageChunk):
                chunk_content = event[0].content
                if chunk_content:
                    full_response += chunk_content
                    message_placeholder.markdown(full_response + " ")

        # Finale KI-Nachricht anzeigen
        message_placeholder.markdown(full_response)

    # Finale KI-Nachricht in der Historie speichern
    st.session_state.messages.append(("assistant", full_response))
    st.rerun()

