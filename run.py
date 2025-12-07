# run.py (Combined CLI and GUI)

import sys
import argparse
# Import CLI modules
from src import register, recognize 
# Import GUI module
from PySide6.QtWidgets import QApplication
from src.MainWindow import MainWindow 

# Make the src directory accessible for absolute imports
sys.path.append("./")

if __name__ == "__main__":
    
    # Check if a specific CLI mode argument is present (e.g., 'register' or 'recognize')
    if '--mode' in sys.argv:
        # --- CLI MODE EXECUTION (OLD FUNCTIONALITY) ---
        parser = argparse.ArgumentParser(description="Smart Office Face Recognition System")
        parser.add_argument('--mode', choices=['register', 'recognize'], required=True, help='Choose operation mode')
        parser.add_argument('--detector', choices=['cnn', 'classical'], default='cnn', help='Choose face detector (cnn or classical)')
        
        args = parser.parse_args()

        if args.mode == 'register':
            register.register_user(detector=args.detector)
        elif args.mode == 'recognize':
            # Note: This is CLI-only recognition, not the GUI feed
            recognize.recognize_user(detector=args.detector)
            
    else:
        # --- GUI MODE EXECUTION (NEW FUNCTIONALITY) ---
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())