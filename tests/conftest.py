"""Force the deterministic offline provider and an isolated Chroma dir before `app` is imported."""

import atexit
import os
import shutil
import tempfile

# Set (not just unset) so app.llm's load_dotenv(override=False) cannot re-add values from a developer .env.
os.environ["LLM_PROVIDER"] = "offline"
os.environ["EMBEDDINGS_PROVIDER"] = "offline"
_CHROMA_TMP = tempfile.mkdtemp(prefix="regops-chroma-test-")
os.environ["CHROMA_DIR"] = _CHROMA_TMP
atexit.register(shutil.rmtree, _CHROMA_TMP, ignore_errors=True)  # best effort: Windows may still hold locks
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
