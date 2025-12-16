# Hugging Face Setup Guide

## How to Get Your HF API Token

### Step 1: Create/Login to Hugging Face Account
1. Go to: https://huggingface.co/
2. Sign up for a free account (or login if you have one)
3. Verify your email if required

### Step 2: Generate API Token
1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "doc-rag-service")
4. Select **"Read"** access (sufficient for Inference API)
5. Click "Generate token"
6. **Copy the token immediately** (starts with `hf_...`)

### Step 3: Add Token to .env File
Add this line to your `.env` file:
```
HF_API_TOKEN=hf_your_token_here
```

## Recommended Models for Free Tier

### Option 1: Mistral-7B-Instruct (Default - Best Quality)
- **Model**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Size**: ~7B parameters
- **Quality**: Excellent for Q&A and text generation
- **Free Tier**: ✅ Works with Inference API free tier
- **Local**: Requires ~14GB RAM

### Option 2: Google Flan-T5 (Smaller, Faster)
- **Model**: `google/flan-t5-large`
- **Size**: ~780M parameters
- **Quality**: Good for classification and simple Q&A
- **Free Tier**: ✅ Works with Inference API free tier
- **Local**: Requires ~3GB RAM

### Option 3: Microsoft Phi-2 (Very Small)
- **Model**: `microsoft/phi-2`
- **Size**: ~2.7B parameters
- **Quality**: Good for simple tasks
- **Free Tier**: ✅ Works with Inference API free tier
- **Local**: Requires ~5GB RAM

## Configuration Options

### Use Inference API (Recommended for Free Tier)
```env
HF_API_TOKEN=hf_your_token_here
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.2
HF_USE_INFERENCE_API=true
```

**Benefits:**
- No local GPU/RAM required
- Free tier available
- Automatic model loading
- No installation of large models

**Limitations:**
- Rate limits on free tier
- Model may need to "warm up" (30-60 seconds first request)
- Requires internet connection

### Use Local Inference (No API Token Needed)
```env
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.2
HF_USE_INFERENCE_API=false
```

**Benefits:**
- No rate limits
- No internet needed after first download
- Full control

**Requirements:**
- Sufficient RAM (14GB+ for Mistral-7B)
- GPU recommended (but not required)
- First run downloads model (~14GB)

## Free Tier Limits

Hugging Face Inference API free tier includes:
- **30,000 requests/month** (approximately 1,000/day)
- **No credit card required**
- **Automatic model loading** (may take 30-60 seconds on first request)

## Troubleshooting

### "Model is loading" Error
- Wait 30-60 seconds and try again
- The model needs to warm up on first request
- Subsequent requests are faster

### "Authentication failed" Error
- Check your token starts with `hf_`
- Verify token has "Read" access
- Make sure token is in `.env` file

### "Out of memory" Error (Local)
- Use a smaller model (e.g., `google/flan-t5-large`)
- Set `HF_USE_INFERENCE_API=true` to use API instead
- Close other applications to free RAM

### Slow Performance (Local)
- Use Inference API instead (`HF_USE_INFERENCE_API=true`)
- Or use a smaller model
- GPU acceleration helps but isn't required

## Model Repository Links

- **Mistral-7B-Instruct**: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
- **Flan-T5-Large**: https://huggingface.co/google/flan-t5-large
- **Phi-2**: https://huggingface.co/microsoft/phi-2

## Next Steps

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add your token to `.env`:
   ```env
   HF_API_TOKEN=hf_your_token_here
   HF_MODEL=mistralai/Mistral-7B-Instruct-v0.2
   HF_USE_INFERENCE_API=true
   ```

3. Run the application:
   ```bash
   python main.py
   ```

The system will automatically use the Inference API if `HF_USE_INFERENCE_API=true`, or fall back to local inference if the API fails.

