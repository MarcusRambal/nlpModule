from fastapi import FastAPI
import unicodedata
import string
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import re
import nltk
import chromadb
import logging
import httpx

from nltk.corpus import stopwords
from pydantic import BaseModel
from typing import Optional

nltk.download("stopwords") # Descargar stopwords si no están ya descargadas (solo la primera vez )

model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
sentiment_analyzer = pipeline(
    "sentiment-analysis", model="pysentimiento/robertuito-sentiment-analysis"
)

app = FastAPI()

# Logging setup
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="comments_collection")


class BusquedaRequest(BaseModel):
    text: str
    category: Optional[str] = None
    sentiment_filter: Optional[str] = None


class ProductCreate(BaseModel):
    title: str
    category: str
    conditions: str
    comment: str
    # ubicacion: str
    user_id: str
    itemid: str
    itemStatus: bool  # Para no recomendar productos no disponibles


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/mock_api/users/{user_id}/history")
async def mock_user_history(user_id: str):
    # Respuesta de ejemplo para el usuario 'user123'
    return {
        "history": ["phone001", "bike001"],  # IDs vistos por el usuario
        "categories": ["electronica", "deportes"],  # Categorías de interés
    }


@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: str, category: str | None = None):

    MAIN_API_URL = "http://127.0.0.1:8000/mock_api"
    history_ids = []

    try:
        async with httpx.AsyncClient() as client:
            # Hacemos la petición GET al endpoint de historial de la API principal
            response = await client.get(f"{MAIN_API_URL}/users/{user_id}/history")
            response.raise_for_status()
            data = response.json()
            history_ids = data.get("history", [])
            history_categories = data.get("categories", [])

        if history_ids is None:
            logger.info(f"No se encontró historial para {user_id}. No se pueden generar recomendaciones personalizadas.")
            return {"user_id": user_id, "items_recomendados": []}

        interestVectors = collection.get(
            ids=history_ids, include=["embeddings"]  
        )

        vector_history = interestVectors.get("embeddings")


        if  vector_history is None:
            logger.warning(f"Historial {history_ids} para {user_id} no arrojó embeddings en ChromaDB.")
            return {"user_id": user_id, "items_recomendados": []}
        

        vector_array = np.array(vector_history)
        interestArray = np.mean(vector_array, axis=0)
        avg_vector = interestArray.tolist()

        filters_where = [{"itemStatus": True} , 
                         {"id": {"$nin": history_ids}}]  # Excluir items ya vistos

        if category:
            # Prioridad 1: El usuario está filtrando activamente.
            logger.info(f"Filtrando por categoría activa (query param): {category}")
            filters_where.append({"category": category})
        elif history_categories:
            # Prioridad 2: El usuario no filtra, usamos su historial.
            unique_categories = list(set(history_categories))
            logger.info(f"Filtrando por categorías del historial: {unique_categories}")
            filters_where.append({"category": {"$in": unique_categories}})

        logger.debug(f"Filtro 'where' final para ChromaDB: {filters_where}")

        filterForWhere = {"$and": filters_where}
        logger.debug(f"Filtro 'where' final para ChromaDB: {filterForWhere}")

        resultados_query = collection.query(
            query_embeddings=[avg_vector],
            n_results=10,
            where=filterForWhere,  
        )
        logger.debug(f"Resultados de la consulta: {resultados_query}")
        
        recommended_ids = resultados_query.get('ids', [[]])[0]
        #Aqui devolver los IDs recomendados a a la API principal

        return {"user_id": user_id, "items_recomendados": recommended_ids}
        

    except httpx.RequestError as exc:
        # Error al contactar la API principal
        logger.error(f"¡Error al llamar a la API Principal para {user_id}! {exc}")
        return {"user_id": user_id, "items_recomendados": []}




# Endpoint para reindexar un producto existente 
@app.post("/reindex/")
async def reindex_producto(item: ProductCreate):

    text = item.comment
    cleanText = normalize_text(text)
    new_vector = model.encode(cleanText).tolist()
    sentiment_result = sentiment_analyzer(cleanText)
    sentiment_label = sentiment_result[0]["label"]
    sentiment_score = sentiment_result[0]["score"]

    new_metadata = {
        "user_id": item.user_id,
        "titulo": item.title,
        "category": item.category,
        "conditions": item.conditions,
        "itemStatus": item.itemStatus,  
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
    }

    try:
        collection.update(
            ids=[item.itemid],
            embeddings=[new_vector],
            documents=[cleanText],
            metadatas=[new_metadata],
        )

        return {
            "status": "Producto actualizado exitosamente",
            "item_id": item.itemid,
            "nuevo_status_disponible": item.itemStatus,
        }

    except Exception as e:
        return {
            "status": "error",
            "detalle": f"No se pudo actualizar el item {item.itemid}: {str(e)}",
        }


# Agregar cada item con embedding y metadatos a Chroma
#Seria mejor hacerlo como un worker asíncrono
@app.post("/add_item/create")
async def add_item(item: ProductCreate):
    text = item.comment
    cleanText = normalize_text(text)
    vector_embedding = model.encode(cleanText).tolist()
    sentiment_result = sentiment_analyzer(cleanText)
    sentiment_label = sentiment_result[0]["label"]
    sentiment_score = sentiment_result[0]["score"]

    metadata_for_chroma = {
        "user_id": item.user_id,
        "title": item.title,
        "category": item.category,
        "conditions": item.conditions,
        "itemStatus": item.itemStatus,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
    }

    # 5. Añadimos todo a la colección de Chroma
    try:
        collection.add(
            embeddings=[vector_embedding],  
            documents=[cleanText],  
            metadatas=[metadata_for_chroma],  
            ids=[item.itemid],  
        )

        return {
            "status": "Producto añadido y embebido exitosamente",
            "producto_guardado": item,
        }

    except Exception as e:
        return {"status": "error", "detalle": str(e)}


# Busqueda semántica con filtros
@app.post("/search/")
async def busqueda_semantica(request: BusquedaRequest):

    text = request.text
    cleanText = normalize_text(text)
    vector_query = model.encode(cleanText)

    filtros = [{"itemStatus": True}]  

    if request.category:
        filtros.append({"category": request.category})

    if request.sentiment_filter:
        filtros.append({"sentiment_label": request.sentiment_filter})

    if len(filtros) > 1:
        filtros_where = {"$and": filtros}
    else:
        filtros_where = filtros[0]

    results = collection.query(
        query_embeddings=[vector_query.tolist()],
        n_results=3,
        where=filtros_where,  
    )

    response_data = {
        "query_texto_original": request.text,
        "filtros_aplicados": filtros,
        "resultados_busqueda": results,
    }

    return response_data


stopwords = set(stopwords.words("spanish"))


def normalize_text(text):
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    additional_punctuation = "¡¿´¬"
    all_punctuation = string.punctuation + additional_punctuation
    text = text.translate(str.maketrans("", "", all_punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    normalized = " ".join(tokens)
    normalized = re.sub(r"[^\x00-\x7F]+", "", normalized)
    return normalized


def getUserHistory(user_id: str):
    """
    Función para obtener el historial de productos comentados por un usuario específico.
    """
    try:
        results = collection.query(
            query_embeddings=[], n_results=10, where={"user_id": user_id}
        )
        return results
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# def chromaSearch(vector_query, n_results=3):

#


# ver chromaDB content
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
        for i in range(len(result["documents"])):
            doc = {
                "id": result["ids"][i] if "ids" in result else f"doc_{i}",
                "text": result["documents"][i],
                "metadata": result["metadatas"][i] if "metadatas" in result else {},
            }
            documents.append(doc)

        return {"total_documents": len(documents), "documents": documents}

    except Exception as e:
        return {"status": "error", "detail": str(e)}