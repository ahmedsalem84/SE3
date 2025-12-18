import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score

# --- MQ Calculation ---
def calculate_mq(graph, clusters):
    cluster_map = {}
    nodes = list(graph.nodes())
    
    # Map node index to cluster ID
    if isinstance(clusters, list) or isinstance(clusters, np.ndarray):
        for i, node in enumerate(nodes):
            c_id = clusters[i]
            if c_id not in cluster_map: cluster_map[c_id] = []
            cluster_map[c_id].append(node)
    else:
        for node, c_id in clusters.items():
            if c_id not in cluster_map: cluster_map[c_id] = []
            cluster_map[c_id].append(node)

    mq_sum = 0
    k = len(cluster_map)
    if k <= 1: return 0 

    for c_id, members in cluster_map.items():
        if not members: continue
        
        subgraph = graph.subgraph(members)
        mu_i = subgraph.number_of_edges()
        
        epsilon_i = 0
        for node in members:
            neighbors = list(graph.neighbors(node))
            for neighbor in neighbors:
                if neighbor not in members:
                    epsilon_i += 1
        
        if mu_i == 0 and epsilon_i == 0:
            cf_i = 0
        else:
            cf_i = (2 * mu_i) / ((2 * mu_i) + epsilon_i)
        mq_sum += cf_i

    return mq_sum / k

# --- Semantic Cohesion Calculation ---
def calculate_semantic_cohesion(clusters_list, embeddings):
    unique_clusters = set(clusters_list)
    scores = []
    
    for c in unique_clusters:
        indices = [i for i, x in enumerate(clusters_list) if x == c]
        if len(indices) < 2:
            scores.append(0.5) 
            continue
        
        cluster_vecs = embeddings[indices]
        sim_matrix = cosine_similarity(cluster_vecs)
        sum_sim = np.sum(sim_matrix) - len(indices)
        count = (len(indices) * len(indices)) - len(indices)
        
        if count > 0:
            avg_sim = sum_sim / count
            scores.append(avg_sim)
        else:
            scores.append(0)
            
    return np.mean(scores) if scores else 0

# --- Combined Fitness ---
def combined_fitness(solution, graph, embeddings, w1=0.6, w2=0.4):
    num_nodes = len(graph.nodes())
    # Dynamic K limit based on graph size
    max_clusters = max(2, int(num_nodes / 3)) 
    
    discrete_clusters = np.floor(np.array(solution) * max_clusters).astype(int)
    
    mq = calculate_mq(graph, discrete_clusters)
    sem = calculate_semantic_cohesion(discrete_clusters, embeddings)
    
    fitness = (w1 * mq) + (w2 * sem)
    
    return fitness, mq, sem, discrete_clusters

# --- NEW: Ground Truth Similarity (ARI) ---
def calculate_ground_truth_similarity(classes_list, predicted_clusters, repo_name, rules):
    """
    Compares the predicted clustering against the 'Human Expert' rules defined in config.
    Returns Adjusted Rand Index (ARI): 1.0 is perfect match, 0.0 is random.
    """
    if repo_name not in rules:
        return 0.0
    
    rule_map = rules[repo_name]
    true_labels = []
    
    # Assign 'True' labels based on keyword matching
    for cls_name in classes_list:
        label = "Misc" # Fallback
        cls_lower = cls_name.lower()
        
        # Check all keywords for this repo
        for keyword, service_name in rule_map.items():
            if keyword in cls_lower:
                label = service_name
                break
        true_labels.append(label)
        
    # Calculate ARI
    try:
        score = adjusted_rand_score(true_labels, predicted_clusters)
        return max(0, score) # Clip negative scores to 0 for display
    except:
        return 0.0