import os
import git
from src.config import DATASET_DIR, REPOS

def download_repo(repo_name):
    """
    Clones the requested repository into the datasets folder.
    """
    if repo_name not in REPOS:
        raise ValueError(f"Repository {repo_name} not defined in config.")
    
    repo_url = REPOS[repo_name]
    target_path = os.path.join(DATASET_DIR, repo_name)
    
    # Create datasets folder if not exists
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        
    # Check if already cloned
    if os.path.exists(target_path):
        print(f"[{repo_name}] already exists at {target_path}")
        return target_path
    
    print(f"Cloning {repo_name} from {repo_url}...")
    try:
        git.Repo.clone_from(repo_url, target_path)
        print("Clone successful.")
        return target_path
    except Exception as e:
        print(f"Error cloning {repo_name}: {e}")
        return None