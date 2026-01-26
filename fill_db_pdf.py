import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from uuid import uuid4
from dotenv import load_dotenv
from streamlit.elements.widgets.selectbox import SelectboxSerde

load_dotenv()
documents = {
     "doc1" : {
         "title" : "Zogg, Martin: Einführung in die Verfahrenstechnik",
         "path" : "sources/ZOGG-noch-kürzer.pdf"
     },
    "doc2" : {
        "title" : "Beispiel 1 Protokoll Zerkleinern",
        "path" : "sources/Bsp_Protokolle_MVT_Zerkleinern.pdf"
    },
    "doc3" : {
        "title" : "Beispiel 2 Protokoll Zerkleinern",
        "path" : "sources/Bsp_Protokolle_Zerkleinerung.pdf"
    },
    "doc4" : {
        "title" : "Wulf, Alex: Scriptum Krümelkunde",
        "path" : "sources/Kruemelkunde-Skriptum.pdf"
    }
 }
# PDF laden
pages = []
for key, doc in documents.items():
    reader = PdfReader(doc["path"])
    if key == "doc1":
        start_page = 96
    elif key == "doc4":
        start_page = 101
    else: start_page = 1
    for page_number, page in enumerate(reader.pages, start=start_page):
        text = page.extract_text()
        if key=="doc2" and page_number <= 3:
                print(f"PAGE_NUMBER {page_number} TEXT {text}")
        if text:
            pages.append({
                 "page": page_number,
                 "text": text,
                 "title": doc["title"],
            })

print(f"Loaded {len(documents)} documents with {len(pages)} pages")

# Text in kleinere Bestandteile (Chunks) aufteilen
text_splitter = RecursiveCharacterTextSplitter(
     chunk_size=800,
     chunk_overlap=100,
 )
#
# # Vektordatenbank einrichten und Kollektion erstellen
client = chromadb.PersistentClient(path="./chroma_test")
emb = embedding_functions.SentenceTransformerEmbeddingFunction(
     model_name="jinaai/jina-embeddings-v2-base-de",
     device="cpu"
)
collection = client.get_or_create_collection(
     "verfahrenstechnik",
     embedding_function=emb
)
#
ids = []
metadatas = []
chunks = []
for page in pages:
    page_chunks = text_splitter.split_text(page["text"])

    for chunk in page_chunks:
         chunks.append(chunk)
         metadatas.append({
             "source": page["title"],
             "page": page["page"],
         })
         ids.append(str(uuid4()))

collection.add(
     documents=chunks,
     metadatas=metadatas,
     ids=ids
)

print("Stored PDF in Chroma.")

# Die Datenbank abfragen
query = "Welche anderen Filtrationsmethoden gibt es außer Druckfiltration?"
results = collection.query(
    query_texts=[query],
    n_results=10,
)

print(results)
