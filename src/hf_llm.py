"""
hf_llm.py - Hugging Face LLM integration (API only)
Uses HF Inference API for text generation with comprehensive logging
"""

import logging
import traceback
from typing import Optional, List
import os
from dotenv import load_dotenv

load_dotenv()

from config import config

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
    logger.debug("huggingface_hub library imported successfully")
except ImportError as e:
    HF_HUB_AVAILABLE = False
    logger.error(f"huggingface_hub not available: {e}")
    logger.error("Please install: pip install huggingface_hub")
    raise


class HuggingFaceLLM:
    """
    Hugging Face LLM client using InferenceClient
    Matches the pattern from test-llm.py with comprehensive logging
    """
    
    def __init__(self):
        """Initialize HF LLM client with comprehensive logging"""
        logger.info("=" * 70)
        logger.info("Initializing Hugging Face LLM Client")
        logger.info("=" * 70)
        
        hf_token = os.getenv("HF_API_TOKEN") or os.getenv("HF_API_KEY")
        
        if not hf_token:
            error_msg = (
                "HF_API_TOKEN is required. Please set it in your .env file.\n"
                "Get your token from: https://huggingface.co/settings/tokens"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.api_token = hf_token.strip()
        
        if not self.api_token:
            error_msg = "HF_API_TOKEN is empty after stripping whitespace. Please check your .env file."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not self.api_token.startswith("hf_"):
            logger.warning(
                f"API token format may be incorrect. "
                f"Expected to start with 'hf_', got: {self.api_token[:10]}..."
            )
            logger.warning("This might cause authentication errors. Please verify your token.")
        
        self.model_name = (os.getenv("HF_MODEL") or config.HF_MODEL or "HuggingFaceH4/zephyr-7b-beta").strip()
        
        logger.debug(f"Model name: '{self.model_name}'")
        logger.debug(f"API token length: {len(self.api_token)} characters")
        logger.debug(f"API token prefix: {self.api_token[:10]}...")
        
        if not self.model_name:
            error_msg = "HF_MODEL is empty. Please set it in your .env file."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if "=" in self.model_name or "\n" in self.model_name:
            clean_model = self.model_name.split("=")[0].split("\n")[0].strip()
            logger.warning(
                f"Model name appears to have extra content. "
                f"Using: '{clean_model}' instead of '{self.model_name}'"
            )
            self.model_name = clean_model
        
        if not self.model_name or len(self.model_name) < 3:
            error_msg = f"Invalid HF_MODEL: '{self.model_name}'. Model name is too short or invalid."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not HF_HUB_AVAILABLE:
            error_msg = "huggingface_hub library is not available. Please install it."
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        try:
            logger.info(f"Creating InferenceClient with model: {self.model_name}")
            logger.debug(f"Token first 10 chars: {self.api_token[:10]}...")
            logger.debug(f"Token last 10 chars: ...{self.api_token[-10:]}")
            logger.debug(f"Token length: {len(self.api_token)}")
            logger.debug(f"Token starts with 'hf_': {self.api_token.startswith('hf_')}")
            
            self.client = InferenceClient(
                model=self.model_name,
                token=self.api_token
            )
            logger.info("✓ InferenceClient initialized successfully")
            logger.info(f"✓ Using model: {self.model_name}")
            logger.info("✓ Using chat_completion API (recommended)")
        except Exception as e:
            error_msg = f"Failed to initialize InferenceClient: {e}"
            logger.error(error_msg)
            logger.error(f"Token validation: length={len(self.api_token)}, starts_with_hf={self.api_token.startswith('hf_')}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e
        
        logger.info("=" * 70)
        logger.info("Hugging Face LLM Client initialized successfully")
        logger.info("=" * 70)
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        repetition_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate text using HF InferenceClient chat_completion API
        Matches the pattern from test-llm.py
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.1 to 2.0)
            system_prompt: Optional system prompt
            stream: Whether to stream the response (default: False)
            
        Returns:
            Generated text
        """
        logger.info("=" * 70)
        logger.info("Starting LLM Generation Request")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Max tokens: {max_tokens}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Streaming: {stream}")
        logger.info(f"System prompt provided: {system_prompt is not None}")
        
        if system_prompt:
            logger.debug(f"System prompt length: {len(system_prompt)} characters")
            logger.debug(f"System prompt preview: {system_prompt[:100]}...")
        
        logger.debug(f"User prompt preview: {prompt[:200]}...")
        
        try:
            return self._generate_with_client(
                prompt, max_tokens, temperature, system_prompt, stream,
                repetition_penalty, stop_sequences
            )
        except Exception as e:
            logger.error("=" * 70)
            logger.error("LLM Generation Failed")
            logger.error("=" * 70)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _generate_with_client(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
        stream: bool,
        repetition_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate using HuggingFace InferenceClient with chat_completion
        Matches the pattern from test-llm.py with comprehensive logging
        """
        logger.info("Preparing chat completion request...")
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            logger.debug("Added system prompt to messages")
        
        messages.append({"role": "user", "content": prompt})
        logger.debug("Added user prompt to messages")
        logger.debug(f"Total messages: {len(messages)}")
        
        max_tokens_clamped = min(max_tokens, 1024)
        if max_tokens != max_tokens_clamped:
            logger.warning(
                f"Max tokens clamped from {max_tokens} to {max_tokens_clamped} "
                "(HF API limit)"
            )
        
        temperature_clamped = max(0.1, min(temperature, 2.0))
        if temperature != temperature_clamped:
            logger.warning(
                f"Temperature clamped from {temperature} to {temperature_clamped} "
                "(valid range: 0.1-2.0)"
            )
        
        default_stop = ["\n\n\n", "---", "==="]
        final_stop = stop_sequences if stop_sequences else default_stop
        
        logger.info("Calling InferenceClient.chat_completion()...")
        logger.debug(
            f"Request parameters: max_tokens={max_tokens_clamped}, "
            f"temperature={temperature_clamped}, "
            f"repetition_penalty={repetition_penalty}, "
            f"stop_sequences={final_stop}, "
            f"stream={stream}"
        )
        
        try:
            if stream:
                logger.info("Using streaming mode")
                return self._generate_streaming(
                    messages, max_tokens_clamped, temperature_clamped,
                    repetition_penalty, final_stop
                )
            else:
                logger.info("Using non-streaming mode")
                
                generation_params = {
                    "messages": messages,
                    "max_tokens": max_tokens_clamped,
                    "temperature": temperature_clamped,
                    "stream": False
                }
                
                if final_stop:
                    try:
                        generation_params["stop"] = final_stop
                    except Exception as e:
                        logger.debug(f"stop_sequences not supported: {e}")
                
                try:
                    response = self.client.chat_completion(**generation_params)
                except TypeError as e:
                    if "repetition_penalty" in str(e):
                        logger.warning("repetition_penalty not supported by InferenceClient API - using alternative methods")
                        generation_params.pop("repetition_penalty", None)
                        response = self.client.chat_completion(**generation_params)
                    else:
                        raise
                
                logger.debug("Received response from InferenceClient")
                logger.debug(f"Response type: {type(response)}")
                logger.debug(f"Response attributes: {dir(response) if hasattr(response, '__dict__') else 'N/A'}")
                
                generated_text = None
                
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    if hasattr(response.choices[0], 'message'):
                        generated_text = response.choices[0].message.content
                        logger.debug("Extracted text from response.choices[0].message.content")
                    elif hasattr(response.choices[0], 'content'):
                        generated_text = response.choices[0].content
                        logger.debug("Extracted text from response.choices[0].content")
                elif isinstance(response, dict):
                    generated_text = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                    logger.debug("Extracted text from dict response")
                else:
                    generated_text = str(response)
                    logger.warning(f"Unexpected response format, converting to string: {type(response)}")
                
                if generated_text is None:
                    error_msg = f"Could not extract text from response: {response}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                logger.info("=" * 70)
                logger.info("LLM Generation Completed Successfully")
                logger.info("=" * 70)
                logger.info(f"Generated text length: {len(generated_text)} characters")
                logger.debug(f"Generated text preview: {generated_text[:200]}...")
                
                return generated_text.strip()
                
        except Exception as e:
            error_str = str(e).lower()
            error_msg_full = str(e)
            error_repr = repr(e)
            
            logger.error("=" * 70)
            logger.error("LLM Generation Error Detected")
            logger.error("=" * 70)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {error_msg_full}")
            logger.error(f"Error repr: {error_repr}")
            
            if hasattr(e, 'response'):
                logger.error(f"Response object: {e.response}")
            if hasattr(e, 'status_code'):
                logger.error(f"Status code: {e.status_code}")
            if hasattr(e, 'request'):
                logger.error(f"Request object: {e.request}")
            
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            if "401" in error_str or "authentication" in error_str or "unauthorized" in error_str or "invalid" in error_str:
                error_msg = (
                    "Authentication failed. Invalid HF_API_TOKEN.\n"
                    "Get your token from: https://huggingface.co/settings/tokens\n"
                    f"Error details: {error_msg_full}"
                )
                logger.error("AUTHENTICATION ERROR")
                logger.error(error_msg)
                raise ValueError(error_msg) from e
                
            elif "503" in error_str or "loading" in error_str or "model is currently loading" in error_str:
                error_msg = (
                    "Model is loading. Please wait 30-60 seconds and try again.\n"
                    f"Model: {self.model_name}\n"
                    "The model needs to warm up on the Inference API.\n"
                    f"Error details: {error_msg_full}"
                )
                logger.warning("MODEL LOADING ERROR")
                logger.warning(error_msg)
                raise RuntimeError(error_msg) from e
                
            elif "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                error_msg = (
                    "Rate limit exceeded. You've hit the free tier limit.\n"
                    "Free tier: 30,000 requests/month\n"
                    "Wait a bit or upgrade your plan.\n"
                    f"Error details: {error_msg_full}"
                )
                logger.error("RATE LIMIT ERROR")
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
                
            elif "timeout" in error_str:
                error_msg = (
                    "API request timed out. The model may be taking too long to respond.\n"
                    f"Model: {self.model_name}\n"
                    f"Error details: {error_msg_full}"
                )
                logger.error("TIMEOUT ERROR")
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
                
            elif "404" in error_str or "not found" in error_str:
                error_msg = (
                    f"Model not found: {self.model_name}\n"
                    "Please check if the model name is correct.\n"
                    f"Error details: {error_msg_full}"
                )
                logger.error("MODEL NOT FOUND ERROR")
                logger.error(error_msg)
                raise ValueError(error_msg) from e
                
            else:
                error_msg = (
                    f"Unexpected error in LLM generation: {error_msg_full}\n"
                    f"Model: {self.model_name}\n"
                    f"Error type: {type(e).__name__}"
                )
                logger.error("UNEXPECTED ERROR")
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
    
    def _generate_streaming(
        self,
        messages: list,
        max_tokens: int,
        temperature: float,
        repetition_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate with streaming support"""
        logger.info("Starting streaming generation...")
        
        try:
            params = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True
            }
            
            if stop_sequences:
                try:
                    params["stop"] = stop_sequences
                except Exception as e:
                    logger.debug(f"stop_sequences not supported in streaming: {e}")
            
            try:
                response = self.client.chat_completion(**params)
            except TypeError as e:
                if "repetition_penalty" in str(e):
                    logger.warning("repetition_penalty not supported in streaming - using alternative methods")
                    params.pop("repetition_penalty", None)
                    response = self.client.chat_completion(**params)
                else:
                    raise
            
            logger.debug("Streaming response started")
            generated_text = ""
            chunk_count = 0
            
            for chunk in response:
                chunk_count += 1
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    if hasattr(chunk.choices[0], 'delta'):
                        content = chunk.choices[0].delta.content
                        if content:
                            generated_text += content
                            logger.debug(f"Received chunk {chunk_count}: {len(content)} characters")
                elif isinstance(chunk, dict):
                    content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    if content:
                        generated_text += content
                        logger.debug(f"Received chunk {chunk_count}: {len(content)} characters")
            
            logger.info(f"Streaming completed: {chunk_count} chunks, {len(generated_text)} total characters")
            logger.info("=" * 70)
            logger.info("LLM Generation Completed Successfully (Streaming)")
            logger.info("=" * 70)
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            logger.debug(f"Streaming error traceback: {traceback.format_exc()}")
            raise
