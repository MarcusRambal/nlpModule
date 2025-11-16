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
import uuid

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
search_history_collection = chroma_client.get_or_create_collection(name="search_history")


class BusquedaRequest(BaseModel):
    text: str
    category: Optional[str] = None
    sentiment_filter: Optional[str] = None
    user_id: Optional[str] = None  

class ProductCreate(BaseModel):
    title: str
    category: str
    conditions: str
    comment: str
    # ubicacion: str
    user_id: str
    itemid: str
    itemStatus: bool  # Para no recomendar productos no disponibles


@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: str, category: str | None = None):

    try:
        user_searches = search_history_collection.get(
            where={"user_id": user_id},
            include=["embeddings", "metadatas"]
        )

        vector_history = user_searches.get("embeddings")

        if  vector_history is None or len(vector_history) == 0:
            logger.info(f"No se encontró historial de búsqueda para {user_id}.")
            return {"user_id": user_id, "items_recomendados": []}
        

        history_metadatas = user_searches.get("metadatas", [])
        history_categories = list(set(
            meta.get("category_filter") 
            for meta in history_metadatas 
            if meta.get("category_filter")
        ))
        
        vector_array = np.array(vector_history)
        interestArray = np.mean(vector_array, axis=0)
        avg_vector = interestArray.tolist()

        filters_where = [
            {"itemStatus": True},      
            {"user_id": {"$ne": user_id}} 
        ]

        if category:
            logger.info(f"Filtrando por categoría activa (query param): {category}")
            filters_where.append({"category": category})
        elif history_categories:
            logger.info(f"Filtrando por categorías del historial: {history_categories}")
            filters_where.append({"category": {"$in": history_categories}})

        filterForWhere = {"$and": filters_where}
        logger.debug(f"Filtro 'where' final para ChromaDB: {filterForWhere}")

        resultados_query = collection.query(
            query_embeddings=[avg_vector],
            n_results=10,
            where=filterForWhere, 
        )
        
        recommended_ids = resultados_query.get('ids', [[]])[0]
        return {"user_id": user_id, "items_recomendados": recommended_ids}

    except Exception as e:
        # Error general al procesar recomendaciones
        logger.error(f"Error al generar recomendaciones para {user_id}: {str(e)}")
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
        "title": item.title,
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

    if request.user_id:
        try:
            search_id = str(uuid.uuid4()) # ID único para este evento de búsqueda
            search_history_collection.add(
                embeddings=[vector_query.tolist()],
                documents=[cleanText],
                metadatas=[{"user_id": request.user_id, "category_filter": request.category}],
                ids=[search_id]
            )
            logger.info(f"Guardando búsqueda en historial para user_id: {request.user_id}")
        except Exception as e:
            logger.warning(f"No se pudo guardar el historial de búsqueda: {str(e)}")

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

# ver chromaDB content
@app.get("/list_documents")
async def list_documents():
    try:
        result = collection.get()

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



@app.get("/list_search_history")
async def list_search_history():
    try:
        result = search_history_collection.get()

        documents = []
        
        if not result or "documents" not in result or not result["documents"]:
             return {"total_documents": 0, "documents": []}

        for i in range(len(result["documents"])):
            doc = {
                "id": result["ids"][i] if "ids" in result and result["ids"] else f"doc_{i}",
                "text": result["documents"][i],
                "metadata": result["metadatas"][i] if "metadatas" in result and result["metadatas"] else {},
            }
            documents.append(doc)

        return {"total_documents": len(documents), "documents": documents}

    except Exception as e:
        return {"status": "error", "detail": str(e)}