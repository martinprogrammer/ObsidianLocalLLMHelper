import os

# Obsidian vault path - change this to your active vault
VAULT_PATH = os.environ.get("VAULT_PATH", "/Users/natwollny/Documents/Obsidian")

# Ollama model (must be pulled via: ollama pull qwen2)
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2:latest")

# Local embedding model (downloaded on first use, ~90MB)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Persistence
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
CHROMA_DIR = os.path.join(CACHE_DIR, "chroma")

# Limits to prevent context overflow with local LLMs
MAX_NOTE_CHARS_FOR_LLM = 3000   # truncate notes sent to LLM
MAX_NOTES_FOR_BATCH_ANALYSIS = 5  # notes shown together to LLM

# Ollama endpoint
OLLAMA_HOST = "http://localhost:11434"
