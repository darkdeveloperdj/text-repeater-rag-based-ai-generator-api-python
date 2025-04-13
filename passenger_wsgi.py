# import os
# import sys


# sys.path.insert(0, os.path.dirname(__file__))


# def application(environ, start_response):
#     start_response('200 OK', [('Content-Type', 'text/plain')])
#     message = 'It works!\n'
#     version = 'Python %s\n' % sys.version.split()[0]
#     response = '\n'.join([message, version])
#     return [response.encode()]
# ===========================================================================
# import os
# import sys

# # Restrict thread usage for numerical libraries.
# os.environ["OPENBLAS_NUM_THREADS"] = "1"       # for OpenBLAS
# os.environ["OMP_NUM_THREADS"] = "1"              # for OpenMP (libgomp)
# os.environ["MKL_NUM_THREADS"] = "1"              # if using Intel MKL (optional)
# os.environ["TOKENIZERS_PARALLELISM"] = "false"   # disable Hugging Face tokenizer parallelism

    
    
# # passenger_asgi.py

# # Ensure the project directory is in sys.path.
# project_home = os.path.dirname(__file__)
# sys.path.insert(0, project_home)
# sys.path.insert(0, os.path.join(project_home, 'app'))

# # Import your FastAPI app
# from main import app as application

#=============================================================================

# passenger_wsgi.py
# import os
# import sys
# from a2wsgi import ASGIMiddleware  # Correct middleware

# # Restrict thread usage
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Add project to path
# project_home = os.path.dirname(__file__)
# sys.path.insert(0, project_home)
# sys.path.insert(0, os.path.join(project_home, 'app'))

# # Import and wrap FastAPI app
# from main import app as fastapi_app
# application = ASGIMiddleware(fastapi_app)  # Proper ASGI-to-WSGI conversion



#=============================================================================

import os
import sys
from a2wsgi import ASGIMiddleware

# Configure thread/process limits
os.environ.update({
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1"
})

# Path configuration
project_home = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_home)
sys.path.insert(0, os.path.join(project_home, 'app'))

# Import and wrap FastAPI app
from main import app as fastapi_app
application = ASGIMiddleware(fastapi_app)  # Remove extra parameters





























