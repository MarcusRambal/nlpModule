import httpx
import numpy as np
from sklearn.metrics import ndcg_score


class Colors:
    GREEN = '\033[92m' 
    YELLOW = '\033[93m'  
    RED = '\033[91m'   
    BOLD = '\033[1m'     
    ENDC = '\033[0m'    

def get_rr_color(rr):
    if rr == 1.0: return Colors.GREEN
    if rr > 0: return Colors.YELLOW
    return Colors.RED

def get_ndcg_color(ndcg):
    if ndcg == 1.0: return Colors.GREEN
    if ndcg > 0.5: return Colors.YELLOW
    return Colors.RED


ground_truth_set = {
    "telefono bueno": {
        "ids": ["item-001", "item-008", "item-004"],
        "relevance": [3, 0, 1] 
    },
    "bici para empezar": {
        "ids": ["item-002", "item-007"],
        "relevance": [3, 2] 
    },
    "libro de ciencia ficcion": {
        "ids": ["item-006"],
        "relevance": [3]
    },
    "ropa de mala calidad": {
        "ids": ["item-009", "item-004"],
        "relevance": [3, 1] 
    }
}

API_URL = "http://localhost:8000/search/"
K = 4

def get_search_results(query_text):
    try:
        response = httpx.post(API_URL, json={"text": query_text})
        data = response.json()
        
        results = data.get("resultados_busqueda", {})
        ids = results.get('ids', [[]])[0]
        return ids[:K] 
    except Exception as e:
        print(f"Error llamando a la API para query '{query_text}': {e}")
        return []

def calculate_metrics(ground_truth_set):
    reciprocal_ranks = []
    ndcg_scores = []
    
    print(f"\n{Colors.BOLD}Reporte Cualitativo de Búsqueda (Top {K}){Colors.ENDC}")
    print("-" * 80)
    header = f"{'Consulta':<30} | {'RR':<6} | {'NDCG@'+str(K):<6} | {'Resultados Obtenidos'}"
    print(f"{Colors.BOLD}{header}{Colors.ENDC}")
    print("-" * 80)

    for query, truth in ground_truth_set.items():
        system_results_ids = get_search_results(query)
        
        if not system_results_ids:
            print(f"{query:<30} | {Colors.RED}{'ERROR DE API':<6}{Colors.ENDC}")
            continue

        truth_ids = truth["ids"]
        truth_relevance_map = dict(zip(truth["ids"], truth["relevance"]))
        
        # --- Cálculo de MRR ---
        rr = 0.0
        for i, res_id in enumerate(system_results_ids):
            if res_id in truth_ids:
                rr = 1 / (i + 1)
                break
        reciprocal_ranks.append(rr)

        # --- Cálculo de NDCG ---
        true_relevance_scores = sorted(truth["relevance"], reverse=True)
        ideal_scores = np.zeros(K)
        ideal_scores[:len(true_relevance_scores)] = true_relevance_scores
        
        system_relevance_scores = np.zeros(K)
        for i, res_id in enumerate(system_results_ids):
            system_relevance_scores[i] = truth_relevance_map.get(res_id, 0)

        ndcg = ndcg_score([ideal_scores], [system_relevance_scores])
        ndcg_scores.append(ndcg)
        
        rr_color = get_rr_color(rr)
        ndcg_color = get_ndcg_color(ndcg)

        rr_str = f"{rr:.2f}"
        ndcg_str = f"{ndcg:.2f}"
        results_str = ", ".join(system_results_ids)
        
        print(f"{query:<30} | {rr_color}{rr_str:<6}{Colors.ENDC} | {ndcg_color}{ndcg_str:<6}{Colors.ENDC} | {results_str}")

    # --- Imprimir Resultados Finales ---
    avg_mrr = np.mean(reciprocal_ranks)
    avg_ndcg = np.mean(ndcg_scores)
    
    print("\n" + "="*30)
    print(f"{Colors.BOLD}Resultados Finales de la Evaluación{Colors.ENDC}")
    print("="*30)
    print(f"Total de consultas evaluadas: {Colors.BOLD}{len(reciprocal_ranks)}{Colors.ENDC}")
    print(f"MRR (Mean Reciprocal Rank): {get_rr_color(avg_mrr)}{Colors.BOLD}{avg_mrr:.4f}{Colors.ENDC}")
    print(f"NDCG@{K} (Promedio):          {get_ndcg_color(avg_ndcg)}{Colors.BOLD}{avg_ndcg:.4f}{Colors.ENDC}")
    print("="*30 + "\n")


if __name__ == "__main__":
    calculate_metrics(ground_truth_set)