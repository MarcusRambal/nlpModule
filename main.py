from fastapi import FastAPI
import unicodedata
import string
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import re
import nltk
import chromadb
nltk.download('stopwords')  
from nltk.corpus import stopwords
from pydantic import BaseModel
from typing import Optional

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

app = FastAPI()


chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="comments_collection")

class BusquedaRequest(BaseModel):
    texto: str  
    categoria: Optional[str] = None 

class ProductCreate(BaseModel):
    title: str
    category: str
    conditions: str
    comment: str
    #ubicacion: str
    user_id: str


mis_documentos = [
    "El perro corre rápido por el parque.",
    "La inteligencia artificial está transformando el mundo.",
    "Receta para hacer un pastel de chocolate.",
    "Noticias sobre la economía global."
]
mis_ids = ["doc_perro", "doc_ia", "doc_pastel", "doc_economia"]

mis_metadatos = [
    {"categoria": "naturaleza", "fuente": "blog"},
    {"categoria": "tecnologia", "fuente": "articulo"},
    {"categoria": "cocina", "fuente": "recetario"},
    {"categoria": "finanzas", "fuente": "noticia"}
]

mis_embeddings = model.encode(mis_documentos)

embeddings_listos_para_chroma = mis_embeddings.tolist()

collection.add(
    embeddings=embeddings_listos_para_chroma, # <-- Tus vectores
    documents=mis_documentos,                  # <-- El texto original (para referencia)
    metadatas=mis_metadatos,               # <-- Metadatos asociados
    ids=mis_ids                                # <-- Los IDs
)


items = {"foo": "The Foo Wrestlers"}

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/items/{item_id}")
async def create_item(item_id: str, item: dict):
    items[item_id] = item
    return item

#Agregar cada item con embedding y metadatos a Chroma
@app.post("/add_item/create")
async def add_item(item: ProductCreate):
    text = item.comment
    cleanText = normalize_text(text)
    vector_embedding = model.encode(cleanText).tolist()
    
    # 3. ¡MUY IMPORTANTE! Preparamos los METADATOS para Chroma
    # Aquí es donde guardamos TODO lo demás que queremos usar para filtrar
    metadata_for_chroma = {
        "user_id": item.user_id,
        "category": item.category,
        "title": item.title,
        "conditions": item.conditions
    }

    # 5. Añadimos todo a la colección de Chroma
    try:
        collection.add(
            embeddings=[vector_embedding],      # El vector que creamos
            documents=[text],         # El texto original (para referencia)
            metadatas=[metadata_for_chroma],  # Los filtros
            ids= [f"item_{len(items)+1}"]                # Un ID único
        )
        
        return {
            "status": "Producto añadido y embebido exitosamente",
            "producto_guardado": item
        }
        
    except Exception as e:
        return {"status": "error", "detalle": str(e)}


@app.get("/search/")
async def busqueda_semantica(request: BusquedaRequest):
    
    text = request.texto
    cleanText = normalize_text(text)
    vector_query = model.encode(cleanText)

    filtros = {}
    if request.categoria:
        filtros["categoria"] = request.categoria

    if filtros:
        results = collection.query(
            query_embeddings=[vector_query.tolist()],
            n_results=1,
            where=filtros 
        )
    else:
        results = collection.query(
            query_embeddings=[vector_query.tolist()],
            n_results=1
        )
        
    response_data = {
        "query_texto_original": request.texto,
        "filtros_aplicados": filtros,
        "resultados_busqueda": results
    }
    
    return response_data



stopwords = set(stopwords.words('spanish'))

def normalize_text(text):
    text = text.lower()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    additional_punctuation = '¡¿´¬'
    all_punctuation = string.punctuation + additional_punctuation
    text = text.translate(str.maketrans('', '', all_punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    normalized = ' '.join(tokens)
    normalized = re.sub(r'[^\x00-\x7F]+','', normalized)
    return normalized


#def chromaSearch(vector_query, n_results=3):

#sentiment_analyzer = pipeline("sentiment-analysis", model="pysentimiento/robertuito-sentiment-analysis")

@app.get("/list_documents")
async def list_documents():
    """
    Endpoint para listar todos los documentos almacenados en Chroma
    """
    try:
        # Obtener todos los documentos de la colección
        result = collection.get()
        
        # Preparar la respuesta
        documents = []
        for i in range(len(result['documents'])):
            doc = {
                'id': result['ids'][i] if 'ids' in result else f'doc_{i}',
                'text': result['documents'][i],
                'metadata': result['metadatas'][i] if 'metadatas' in result else {}
            }
            documents.append(doc)
            
        return {
            "total_documents": len(documents),
            "documents": documents
        }
        
    except Exception as e:
        return {"status": "error", "detail": str(e)}
