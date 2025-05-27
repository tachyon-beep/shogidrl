# train.py: Thin shim to call the real trainer in keisei.train

import sys
import os
from dotenv import load_dotenv  # Add this import
from keisei.training.train import main  # Moved import to the top

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

load_dotenv()  # Load environment variables from .env file

if __name__ == "__main__":
    main()
