
import cognee
from cognee.api.v1.search import SearchType

print("SearchType attributes:", dir(SearchType))
try:
    print("SearchType.SIMILARITY:", SearchType.SIMILARITY)
except AttributeError:
    print("SearchType.SIMILARITY not found")
