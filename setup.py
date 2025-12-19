import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def check_env_file():
    env_path = Path(".env")
    if not env_path.exists():
        print("!!! .env file not found!!!")
    print(".env file found")
    return True

def check_env_variables():

    load_dotenv()
    
    required_vars = [
        "ANTHROPIC_API_KEY",
        "PINECONE_API_KEY",
    ]
    
    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if not value or value.startswith("your_"):
            missing.append(var)
        else:
            print(f"{var} is set")
    
    if missing:
        print(f"\nMissing or invalid environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nPlease edit your .env file and add the required API keys.")
        return False
    
    return True

def install_dependencies():
    try:
        import gradio
        import anthropic
        import pinecone
        import trafilatura
        from sentence_transformers import SentenceTransformer
        print("✓ All dependencies installed")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e.name}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return False

def main():
    print("="*60)
    print("Personal Resarch Assistant - Setup")
    print("="*60)
    print()

    print("Checking dependencies...")
    deps_ok = install_dependencies()
    print()

    if not deps_ok:
        print("="*60)
        print("Setup incomplete. Please install dependencies first.")
        print("="*60)
        sys.exit(1)
    
    print("Checking environment configuration...")
    env_exists = check_env_file()
    print()

    if not env_exists:
        print("="*60)
        print("Setup incomplete. Please configure your .env file")
        print("="*60)
        sys.exit(1)

    env_ok = check_env_variables()
    print()
        
    if not env_ok:
        print("=" * 60)
        print("Setup incomplete. Please add your API keys to .env")
        print("=" * 60)
        sys.exit(1)

    print("=" * 60)
    print("✓ Setup complete! You're ready to go.")
    print("=" * 60)
    print()
    print("To start the application, run:")
    print("  python app.py")
    print()
    print("The app will be available at:")
    print("  http://localhost:7860")
    print()

if __name__ == "__main__":
    main()