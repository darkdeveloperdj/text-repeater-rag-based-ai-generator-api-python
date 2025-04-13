# # app/main.py

# # Monkey-patch missing torch attributes for CPU-only installations
# import torch
# if not hasattr(torch, "compiler"):
#     class DummyCompiler:
#         @staticmethod
#         def disable(recursive=False):
#             def decorator(func):
#                 return func
#             return decorator
#     torch.compiler = DummyCompiler()




# if not hasattr(torch, "float8_e4m3fn"):
#     # Define a dummy value for float8_e4m3fn; torch.float16 is used as a fallback.
#     torch.float8_e4m3fn = torch.float16

# Continue with normal imports
# app/main.py
from fastapi import FastAPI
from app.api.routes import router

app = FastAPI()
app.include_router(router, prefix="/api")

# Remove the __main__ block completely
