"""
Settings, env vars, and constants.
All configuration is read from .env via python-dotenv.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = "text-embedding-3-small"
LLM_GENERATION_MODEL: str = "gpt-4o"             # main generation model
LLM_INTENT_FALLBACK_MODEL: str = "gpt-4o-mini"   # cheap model for intent fallback
LLM_TEMPERATURE: float = 0.2
LLM_MAX_TOKENS: int = 800

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = os.path.join(
    os.path.dirname(__file__), ".chromadb"
)
CHROMA_COLLECTION_NAME: str = "uni_rag"

# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
BM25_INDEX_PATH: str = os.path.join(
    os.path.dirname(__file__), ".bm25_index.pkl"
)

# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------
DENSE_TOP_K: int = 10           # candidates from ChromaDB
SPARSE_TOP_K: int = 10          # candidates from BM25
RRF_K: int = 60                 # RRF constant (standard)
FINAL_TOP_K: int = 6            # chunks passed to context builder
# Threshold applies to final_score after RRF fusion + metadata boosts.
# RRF rank-1 base score = 1/(k+1) = ~0.016; boosts add up to ~0.51.
# A score below 0.10 means the query matched nothing in either dense or
# sparse retrieval — safe fallback threshold for this scale.
RETRIEVAL_CONFIDENCE_THRESHOLD: float = 0.10

# ---------------------------------------------------------------------------
# Context budget
# ---------------------------------------------------------------------------
MAX_CONTEXT_TOKENS: int = 2000  # approximate token budget for evidence

# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------
MAX_TURNS: int = 5              # turns stored per session
HISTORY_TURNS: int = 3          # turns included in prompt

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
FALLBACK_RESPONSE: str = (
    "I couldn't find enough reliable information in the available documents "
    "to answer that confidently. For accurate guidance, please contact "
    "ABC admissions directly."
)

SCOPE_RESPONSE: str = (
    "That question appears to be outside the topics I can help with. "
    "I cover ABC program eligibility, admissions, financing, scholarships, "
    "credit transfer, study formats, language requirements, visa, and application "
    "process. For anything else, please contact ABC directly."
)

