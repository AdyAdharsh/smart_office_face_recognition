import argparse
from src import register, recognize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Office Face Recognition System")
    parser.add_argument('--mode', choices=['register', 'recognize'], required=True, help='Choose operation mode')
    parser.add_argument('--detector', choices=['cnn', 'classical'], default='cnn', help='Choose face detector (cnn or classical)')

    args = parser.parse_args()

    if args.mode == 'register':
        register.register_user(detector=args.detector)
    elif args.mode == 'recognize':
        recognize.recognize_user(detector=args.detector)