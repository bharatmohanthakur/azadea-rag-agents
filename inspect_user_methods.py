
import inspect
from cognee.modules.users.methods import get_user, create_user

print("get_user signature:", inspect.signature(get_user))
print("create_user signature:", inspect.signature(create_user))
