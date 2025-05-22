# train.py: Thin shim to call the real trainer in keisei.train

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import keisei.train

if __name__ == "__main__":
    keisei.train.main()
