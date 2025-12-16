"""
test_llm_inference.py - Test script for LLM inference and model availability
Tests multiple models to find working ones on Hugging Face Inference API
"""

import sys
import os
import time
from dotenv import load_dotenv

load_dotenv()

WORKING_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-2-7b-chat-hf",
    "google/flan-t5-large",
    "microsoft/phi-2",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "Qwen/Qwen2.5-7B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
]

def test_model_with_requests(model_name, api_token):
    """Test model using direct API requests"""
    import requests
    
    headers = {"Authorization": f"Bearer {api_token}"}
    endpoints = [
        f"https://api-inference.huggingface.co/models/{model_name}",
        f"https://router.huggingface.co/models/{model_name}"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, headers=headers, timeout=10)
            if response.status_code == 200:
                return True, endpoint, "Available"
            elif response.status_code == 401:
                return False, endpoint, "Authentication failed"
            elif response.status_code == 404:
                continue
            elif response.status_code == 410:
                continue
            else:
                return None, endpoint, f"Status {response.status_code}"
        except Exception as e:
            continue
    
    return None, "all endpoints", "Not found or unavailable"


def test_model_with_inference_client(model_name, api_token):
    """Test model using Hugging Face InferenceClient"""
    try:
        from huggingface_hub import InferenceClient
        
        client = InferenceClient(token=api_token, model=model_name)
        
        test_prompt = "Hello"
        
        try:
            response = client.text_generation(
                prompt=test_prompt,
                max_new_tokens=5,
                temperature=0.7,
                timeout=30
            )
            if response and len(response.strip()) > 0:
                return True, response[:50] if response else "Empty response"
            else:
                return None, "Empty response"
        except Exception as e:
            error_str = str(e).lower()
            error_full = str(e)
            
            if "401" in error_str or "authentication" in error_str or "unauthorized" in error_str:
                return False, "Authentication failed - check your API token"
            elif "404" in error_str or "not found" in error_str:
                return None, "Model not found on Inference API"
            elif "503" in error_str or "loading" in error_str or "service unavailable" in error_str:
                return "loading", "Model is loading (wait 30-60s, then try inference)"
            elif "429" in error_str or "rate limit" in error_str:
                return None, "Rate limit exceeded"
            elif "timeout" in error_str:
                return "loading", "Request timed out (model may be loading)"
            else:
                return None, f"Error: {error_full[:100]}"
                
    except ImportError:
        return None, "huggingface_hub not installed - run: pip install huggingface_hub"
    except Exception as e:
        return None, f"Exception: {str(e)[:100]}"


def test_model_inference(model_name, api_token):
    """Test actual inference on a model"""
    try:
        from huggingface_hub import InferenceClient
        
        client = InferenceClient(token=api_token, model=model_name)
        
        test_prompt = "What is artificial intelligence? Answer in one sentence."
        
        print(f"    Testing inference with prompt: '{test_prompt[:50]}...'")
        start_time = time.time()
        
        try:
            response = client.text_generation(
                prompt=test_prompt,
                max_new_tokens=50,
                temperature=0.7,
                return_full_text=False
            )
            elapsed = time.time() - start_time
            
            if response:
                return True, f"Success ({elapsed:.2f}s): {response[:100]}"
            else:
                return False, "Empty response"
                
        except Exception as e:
            error_str = str(e).lower()
            if "503" in error_str or "loading" in error_str:
                return None, "Model is loading (wait 30-60s and try again)"
            elif "429" in error_str or "rate limit" in error_str:
                return None, "Rate limit exceeded"
            else:
                return False, f"Inference failed: {str(e)[:100]}"
                
    except ImportError:
        return None, "huggingface_hub not installed"
    except Exception as e:
        return False, f"Exception: {str(e)[:100]}"


def test_single_model(model_name, api_token, test_inference=False):
    """Test a single model"""
    print(f"\n  Testing: {model_name}")
    print("  " + "-" * 68)
    
    results = {}
    
    print("  [1/2] Testing API availability...")
    available, endpoint, status = test_model_with_requests(model_name, api_token)
    results['api_available'] = available
    results['endpoint'] = endpoint
    results['api_status'] = status
    
    if available:
        print(f"    ✓ API available at: {endpoint}")
    elif available is False:
        print(f"    ❌ {status}")
        return results
    else:
        print(f"    ⚠️  {status}")
    
    print("  [2/2] Testing with InferenceClient...")
    client_result, client_status = test_model_with_inference_client(model_name, api_token)
    results['client_works'] = client_result
    results['client_status'] = client_status
    
    if client_result is True:
        print(f"    ✓ InferenceClient works: {client_status[:80]}")
    elif client_result == "loading":
        print(f"    ⏳ {client_status}")
        print(f"    💡 This model may work after warm-up (wait 30-60 seconds)")
    elif client_result is False:
        print(f"    ❌ {client_status}")
    else:
        print(f"    ⚠️  {client_status}")
    
    if test_inference and (available or client_works):
        print("  [3/3] Testing actual inference...")
        inference_works, inference_status = test_model_inference(model_name, api_token)
        results['inference_works'] = inference_works
        results['inference_status'] = inference_status
        
        if inference_works:
            print(f"    ✓ {inference_status}")
        elif inference_works is False:
            print(f"    ❌ {inference_status}")
        else:
            print(f"    ⚠️  {inference_status}")
    
    return results


def quick_test_single_model(model_name, api_token):
    """Quick test with wait for warm-up"""
    print(f"\n{'='*70}")
    print(f"QUICK TEST: {model_name}")
    print("="*70)
    
    try:
        from huggingface_hub import InferenceClient
        
        client = InferenceClient(token=api_token, model=model_name)
        
        print("\nTesting model (this may take 30-60 seconds on first request)...")
        
        for attempt in range(3):
            try:
                print(f"  Attempt {attempt + 1}/3...")
                response = client.text_generation(
                    prompt="Hello, respond with 'OK'",
                    max_new_tokens=10,
                    temperature=0.7,
                    timeout=60
                )
                
                if response:
                    print(f"\n✓ SUCCESS! Model is working.")
                    print(f"  Response: {response[:100]}")
                    return True, model_name
                else:
                    print("  ⚠️  Empty response, retrying...")
                    time.sleep(10)
                    
            except Exception as e:
                error_str = str(e).lower()
                if "503" in error_str or "loading" in error_str:
                    wait_time = 30 * (attempt + 1)
                    print(f"  ⏳ Model is loading, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                elif "401" in error_str or "authentication" in error_str:
                    print(f"  ❌ Authentication failed")
                    return False, "Invalid API token"
                else:
                    print(f"  ⚠️  Error: {str(e)[:100]}")
                    if attempt < 2:
                        time.sleep(10)
        
        return None, "Model did not respond after 3 attempts"
        
    except ImportError:
        print("  ❌ huggingface_hub not installed")
        print("  Run: pip install huggingface_hub")
        return False, "Missing dependency"
    except Exception as e:
        return False, f"Error: {str(e)[:100]}"


def main():
    """Main test function"""
    print("\n" + "="*70)
    print("HUGGING FACE LLM INFERENCE TEST")
    print("="*70)
    
    api_token = os.getenv("HF_API_TOKEN", "")
    
    if not api_token:
        print("\n❌ HF_API_TOKEN not found in .env file")
        print("\nTo fix:")
        print("1. Get your token from: https://huggingface.co/settings/tokens")
        print("2. Add to .env: HF_API_TOKEN=hf_...")
        return 1
    
    if not api_token.startswith("hf_"):
        print(f"\n⚠️  HF_API_TOKEN format may be incorrect")
        print(f"   Expected to start with 'hf_', got: {api_token[:10]}...")
        print("   Continuing anyway...")
    
    print(f"\n✓ Using API token: {api_token[:10]}...{api_token[-4:]}")
    
    if "--quick" in sys.argv or "-q" in sys.argv:
        model = os.getenv("HF_MODEL", WORKING_MODELS[0])
        print(f"\n🚀 Quick test mode - testing: {model}")
        success, result = quick_test_single_model(model, api_token)
        
        if success:
            print(f"\n{'='*70}")
            print("✓ Model is working! You can use it now.")
            print(f"  Model: {result}")
            return 0
        else:
            print(f"\n{'='*70}")
            print(f"⚠️  {result}")
            print("\nTry running full test: python test_llm_inference.py")
            return 1
    
    test_inference = "--test-inference" in sys.argv or "-i" in sys.argv
    
    if test_inference:
        print("\n⚠️  Inference testing enabled (will make API calls)")
        print("   This may take longer and use API quota")
    else:
        print("\n💡 Tips:")
        print("   • Use --test-inference to test actual text generation")
        print("   • Use --quick to test one model with warm-up wait")
    
    print(f"\n📋 Testing {len(WORKING_MODELS)} models...")
    print("="*70)
    
    working_models = []
    partially_working = []
    failed_models = []
    
    for model in WORKING_MODELS:
        results = test_single_model(model, api_token, test_inference)
        
        client_result = results.get('client_works')
        
        if client_result is True:
            if results.get('inference_works') or not test_inference:
                working_models.append((model, results))
            else:
                partially_working.append((model, results))
        elif client_result == "loading":
            partially_working.append((model, results))
        elif results.get('api_available'):
            partially_working.append((model, results))
        else:
            failed_models.append((model, results))
        
        time.sleep(0.5)
    
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    if working_models:
        print(f"\n✓ WORKING MODELS ({len(working_models)}):")
        for model, results in working_models:
            status_icons = []
            if results.get('api_available'):
                status_icons.append("API✓")
            if results.get('client_works'):
                status_icons.append("Client✓")
            if results.get('inference_works'):
                status_icons.append("Inference✓")
            
            print(f"  • {model}")
            if status_icons:
                print(f"    Status: {' | '.join(status_icons)}")
    
    if partially_working:
        print(f"\n⚠️  PARTIALLY WORKING ({len(partially_working)}):")
        for model, results in partially_working:
            print(f"  • {model}")
            print(f"    Status: {results.get('client_status', 'Unknown')}")
    
    if failed_models:
        print(f"\n❌ FAILED/UNAVAILABLE ({len(failed_models)}):")
        for model, results in failed_models:
            print(f"  • {model}")
            print(f"    Reason: {results.get('api_status', 'Unknown')}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if working_models:
        best_model = working_models[0][0]
        print(f"\n✓ Recommended model: {best_model}")
        print(f"\nAdd to your .env file:")
        print(f"  HF_MODEL={best_model}")
    elif partially_working:
        print("\n⚠️  Models found but need warm-up or have issues")
        print("\nModels that may work after warm-up:")
        for model, results in partially_working[:3]:
            if results.get('client_works') == "loading":
                print(f"  • {model}")
                print(f"    Status: {results.get('client_status', 'Unknown')}")
        
        if partially_working:
            best_model = partially_working[0][0]
            print(f"\n💡 Try this model (may need warm-up): {best_model}")
            print(f"\nAdd to your .env file:")
            print(f"  HF_MODEL={best_model}")
            print("\nThen wait 30-60 seconds before using the service")
    else:
        print("\n⚠️  No working models found")
        print("\nPossible issues:")
        print("  1. API token is invalid")
        print("     → Check: https://huggingface.co/settings/tokens")
        print("  2. Models need to warm up (503 errors are normal)")
        print("     → Wait 30-60 seconds and try again")
        print("     → Or run: python test_llm_inference.py --test-inference")
        print("  3. Rate limits exceeded")
        print("     → Free tier: 30,000 requests/month")
        print("  4. Network/firewall issues")
        print("     → Check internet connection")
        
        print("\n💡 Next steps:")
        print("  1. Verify your API token is correct")
        print("  2. Wait 1-2 minutes for models to warm up")
        print("  3. Try running with --test-inference flag")
        print("  4. Check Hugging Face status: https://status.huggingface.co/")
    
    print("\n" + "="*70)
    
    return 0 if working_models else 1


if __name__ == "__main__":
    sys.exit(main())

