"""
health_check.py - Health check script for API keys and services
Tests Hugging Face API token and Pinecone connection
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

def check_hf_api():
    """Check Hugging Face API token"""
    print("\n" + "="*70)
    print("HUGGING FACE API HEALTH CHECK")
    print("="*70)
    
    hf_token = os.getenv("HF_API_TOKEN", "")
    hf_model = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    
    if not hf_token:
        print("❌ HF_API_TOKEN not found in .env file")
        print("\nTo fix:")
        print("1. Get your token from: https://huggingface.co/settings/tokens")
        print("2. Add to .env: HF_API_TOKEN=hf_...")
        return False
    
    if not hf_token.startswith("hf_"):
        print(f"⚠️  HF_API_TOKEN format may be incorrect")
        print(f"   Expected to start with 'hf_', got: {hf_token[:10]}...")
        print("   Continuing anyway...")
    else:
        print(f"✓ HF_API_TOKEN found (length: {len(hf_token)})")
    
    print(f"✓ Model: {hf_model}")
    
    try:
        from huggingface_hub import InferenceClient
        
        print("\nTesting with InferenceClient...")
        client = InferenceClient(token=hf_token, model=hf_model)
        
        print("Testing simple inference...")
        try:
            response = client.text_generation(
                prompt="Hello",
                max_new_tokens=5,
                temperature=0.7
            )
            print("✓ InferenceClient works")
            print(f"✓ Test response: {response[:50]}...")
            return True
        except Exception as e:
            error_str = str(e).lower()
            if "401" in error_str or "authentication" in error_str:
                print("❌ Authentication failed - Invalid token")
                print("\nTo fix:")
                print("1. Check your token at: https://huggingface.co/settings/tokens")
                print("2. Make sure token has 'Read' access")
                print("3. Regenerate token if needed")
                return False
            elif "503" in error_str or "loading" in error_str:
                print("⚠️  Model is loading (this is normal on first request)")
                print("   Wait 30-60 seconds and try again")
                return True
            elif "404" in error_str or "not found" in error_str:
                print("⚠️  Model not found or not available")
                print(f"   Model: {hf_model}")
                print("   Try running: python test_llm_inference.py")
                return True
            else:
                print(f"⚠️  Inference test error: {str(e)[:100]}")
                return True
                
    except ImportError:
        print("⚠️  'huggingface_hub' not installed, testing with direct API...")
        try:
            import requests
            
            print("\nTesting API connection...")
            headers = {"Authorization": f"Bearer {hf_token}"}
            api_url = f"https://api-inference.huggingface.co/models/{hf_model}"
            
            response = requests.get(api_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                print("✓ API connection successful")
                return True
            elif response.status_code == 401:
                print("❌ Authentication failed - Invalid token")
                return False
            elif response.status_code == 404:
                print("⚠️  Model not found (may need to load on first request)")
                return True
            else:
                print(f"⚠️  API returned status: {response.status_code}")
                return True
        except ImportError:
            print("❌ 'requests' library not installed")
            print("   Run: pip install requests huggingface_hub")
            return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False


def check_pinecone():
    """Check Pinecone API key"""
    print("\n" + "="*70)
    print("PINECONE API HEALTH CHECK")
    print("="*70)
    
    pc_key = os.getenv("PINECONE_API_KEY", "")
    pc_env = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    pc_index = os.getenv("PINECONE_INDEX_NAME", "document-rag-index")
    
    if not pc_key:
        print("❌ PINECONE_API_KEY not found in .env file")
        print("\nTo fix:")
        print("1. Get your key from: https://app.pinecone.io/")
        print("2. Add to .env: PINECONE_API_KEY=...")
        return False
    
    print(f"✓ PINECONE_API_KEY found (length: {len(pc_key)})")
    print(f"✓ Environment: {pc_env}")
    print(f"✓ Index name: {pc_index}")
    
    try:
        from pinecone import Pinecone
        
        print("\nTesting Pinecone connection...")
        pc = Pinecone(api_key=pc_key)
        
        indexes = pc.list_indexes()
        print(f"✓ Connection successful")
        print(f"✓ Found {len(indexes)} index(es)")
        
        if pc_index in indexes:
            print(f"✓ Index '{pc_index}' exists")
        else:
            print(f"⚠️  Index '{pc_index}' not found (will be created on first use)")
        
        return True
        
    except ImportError:
        print("❌ 'pinecone' library not installed")
        print("   Run: pip install pinecone-client")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        return False


def check_env_file():
    """Check if .env file exists"""
    print("\n" + "="*70)
    print("ENVIRONMENT FILE CHECK")
    print("="*70)
    
    if os.path.exists(".env"):
        print("✓ .env file found")
        return True
    else:
        print("❌ .env file not found")
        print("\nTo fix:")
        print("1. Create a .env file in the project root")
        print("2. Add your API keys:")
        print("   HF_API_TOKEN=hf_...")
        print("   PINECONE_API_KEY=...")
        print("   PINECONE_ENVIRONMENT=us-east-1")
        return False


def main():
    """Run all health checks"""
    print("\n" + "="*70)
    print("DOC-RAG SERVICE HEALTH CHECK")
    print("="*70)
    
    results = []
    
    results.append(("Environment File", check_env_file()))
    results.append(("Hugging Face API", check_hf_api()))
    results.append(("Pinecone API", check_pinecone()))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✓ All checks passed! Your service is ready to use.")
        print("\n💡 Tip: Run 'python test_llm_inference.py' to test multiple models")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        print("\n💡 Tip: Run 'python test_llm_inference.py' to find working models")
        return 1


if __name__ == "__main__":
    sys.exit(main())

