import os
import re
import networkx as nx

def get_class_name(content):
    """
    Extracts the public class name using Regex.
    """
    match = re.search(r'public\s+(?:abstract\s+)?class\s+(\w+)', content)
    return match.group(1) if match else None

def parse_java_project(repo_path):
    """
    Scans a Java project and builds a dependency graph.
    Returns: 
        G (NetworkX Graph): Nodes=Classes, Edges=Calls
        class_docs (dict): {ClassName: RawSourceCode} for RAG
    """
    print(f"Parsing project at: {repo_path}")
    
    file_map = {} # Map ClassName -> FilePath
    class_docs = {} # Map ClassName -> Source Code
    dependencies = [] # List of tuples (Source, Target)
    
    # 1. First Pass: Identify all Classes
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".java") and "test" not in root.lower(): # Exclude tests
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    class_name = get_class_name(content)
                    if class_name:
                        file_map[class_name] = full_path
                        # Clean content for LLM (Remove license headers/excessive whitespace)
                        clean_code = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
                        class_docs[class_name] = clean_code[:2000] # Limit to 2000 chars for Embedding speed
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    # 2. Second Pass: Build Edges based on usage
    # If Class A contains the string "ClassB", we assume a dependency.
    # This is a standard Lightweight Static Analysis technique.
    all_classes = list(file_map.keys())
    
    for class_name, code in class_docs.items():
        for potential_dependency in all_classes:
            if class_name == potential_dependency:
                continue
            
            # Simple check: does Class A mention Class B?
            # We look for "ClassB " (space) or "ClassB;" or "ClassB(" to avoid substring matches
            pattern = r'\b' + re.escape(potential_dependency) + r'\b'
            if re.search(pattern, code):
                dependencies.append((class_name, potential_dependency))

    # 3. Build Graph
    G = nx.Graph()
    for cls in all_classes:
        G.add_node(cls)
    
    for src, tgt in dependencies:
        G.add_edge(src, tgt)

    # Filter isolated nodes to keep the graph clean for optimization
    G.remove_nodes_from(list(nx.isolates(G)))
    
    filtered_classes = list(G.nodes())
    filtered_docs = {k: v for k, v in class_docs.items() if k in filtered_classes}

    print(f"Parsing Complete. Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
    return G, filtered_docs