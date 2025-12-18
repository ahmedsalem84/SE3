import requests
import numpy as np
import time
from src.config import OLLAMA_URL, OLLAMA_GEN_URL, OLLAMA_MODEL

def get_embedding(text, retries=3):
    """
    Gets vector embedding from Ollama with retry logic.
    """
    payload = {"model": OLLAMA_MODEL, "prompt": text}
    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_URL, json=payload)
            if response.status_code == 200:
                data = response.json()
                if 'embedding' in data: return data['embedding']
        except:
            pass
        time.sleep(0.5)
    return np.zeros(1024).tolist()

def generate_embeddings_for_classes(class_docs, progress_callback=None):
    """
    Batch processes class source code.
    progress_callback: A function taking (current_step, total_steps, text_status)
    """
    keys = list(class_docs.keys())
    embeddings = []
    total = len(keys)
    
    for i, cls in enumerate(keys):
        # Update UI if callback provided
        if progress_callback:
            progress_callback(i, total, f"Embedding {cls}...")

        text = class_docs[cls][:1500] 
        prompt = f"Source code for Java Class {cls}: {text}"
        emb = get_embedding(prompt)
        embeddings.append(emb)
    
    # Final 100% update
    if progress_callback:
        progress_callback(total, total, "Completed.")
            
    return keys, np.array(embeddings)

def generate_service_name(class_names):
    """
    Generates a name for the microservice.
    """
    classes_str = ", ".join(class_names[:15])
    prompt = (
        f"You are a Software Architect. I have a microservice containing these Java classes: [{classes_str}]. "
        "Suggest a single, professional, PascalCase name (e.g., InventoryService). "
        "Return ONLY the name."
    )
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_GEN_URL, json=payload)
        if response.status_code == 200:
            return response.json()['response'].strip().replace('"', '').replace("'", "").split('\n')[0]
    except:
        pass
    return "MicroService"