
import cognee
import inspect

def list_attributes(obj, name):
    print(f"--- Attributes of {name} ---")
    for attr in dir(obj):
        if not attr.startswith("_"):
            print(attr)

list_attributes(cognee, "cognee")

try:
    import cognee.api.v1 as api
    list_attributes(api, "cognee.api.v1")
    
    if hasattr(api, "history"):
        list_attributes(api.history, "cognee.api.v1.history")
except ImportError:
    print("Could not import cognee.api.v1")
    
try:
    from cognee.modules.users.methods import get_user
    print("Found get_user")
except ImportError:
    pass
