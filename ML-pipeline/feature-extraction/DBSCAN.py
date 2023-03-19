# Python code 3.9
import os
import sys

__file__ = "utilities_store.py"

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../fpt/'))
if not path in sys.path:
    sys.path.insert(1, path)

import utilities_store as us


def extractFrequencies():
    return 440