import logging
import warnings

from core.extractor import GraphRAGExtractor  # noqa: F401 — re-export
from core.store import GraphRAGStore  # noqa: F401 — re-export
from core.store import GraphRAGQueryEngine  # noqa: F401 — re-export
from models import GraphQueryResponse  # noqa: F401 — re-export

warnings.warn(
    "Importing from core_classes is deprecated. Use core.store and core.extractor.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)