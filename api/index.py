# -*- coding: utf-8 -*-
"""
Vercel Serverless Function Entry Point for ESTATEIQ
===================================================
Exports the WSGI Flask application object `app`.
"""

import sys
from pathlib import Path

# Add project root directory to Python module search path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app import app
