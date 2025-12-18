import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

# Target Repositories
REPOS = {
    "Spring PetClinic": "https://github.com/spring-projects/spring-petclinic.git",
    "JHotDraw": "https://github.com/wumpz/jhotdraw.git", 
    "MyBatis 3": "https://github.com/mybatis/mybatis-3.git"
}

# Ollama Settings
OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GEN_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama2" 

# =========================================================
# GROUND TRUTH ORACLES
# =========================================================
# These rules map class name keywords to their expected "Module" or "Service".
# This simulates a Human Expert's decomposition.

GROUND_TRUTH_RULES = {
    "Spring PetClinic": {
        "vet": "VetService",
        "visit": "VisitService",
        "owner": "OwnerService",
        "pet": "PetService",
        "clinic": "ClinicService",
        "system": "SystemUtils",
        "model": "SharedModel"
    },
    "JHotDraw": {
        "figure": "FiguresModule",
        "tool": "ToolsModule",
        "handle": "HandlesModule",
        "connector": "ConnectorsModule",
        "view": "ViewLayer",
        "app": "ApplicationLayer",
        "geom": "GeometryUtils",
        "io": "InputOutput"
    },
    "MyBatis 3": {
        "cache": "CacheModule",
        "binding": "BindingModule",
        "builder": "BuilderModule",
        "executor": "ExecutorModule",
        "mapping": "MappingLayer",
        "session": "SessionLayer",
        "transaction": "TransactionModule",
        "parsing": "ParsingUtils",
        "logging": "LoggingUtils"
    }
}