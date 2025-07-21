import os
import json
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

import anthropic
import google.generativeai as genai
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

class AIClient(ABC):
    """Abstract base class for AI clients"""
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate response from AI model"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the AI client is available and configured"""
        pass

class ClaudeClient(AIClient):
    """Claude AI client using Anthropic API"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.api_key = api_key
    
    def is_available(self) -> bool:
        """Check if Claude API is available"""
        return bool(self.api_key)
    
    def generate_response(self, prompt: str, model: str = "claude-3-5-sonnet-20241022", 
                         max_tokens: int = 2000, temperature: float = 0.1) -> str:
        """Generate response using Claude API"""
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise

class GeminiClient(AIClient):
    """Gemini AI client using Google Generative AI"""
    
    def __init__(self, service_account_path: str):
        self.service_account_path = service_account_path
        self.model = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Gemini client with service account"""
        try:
            # Clear any conflicting Google API environment variables
            clear_google_api_conflicts()
            
            # Load service account credentials
            if not os.path.exists(self.service_account_path):
                raise FileNotFoundError(f"Service account file not found: {self.service_account_path}")
            
            # Clear any existing API key configuration to avoid conflicts
            if hasattr(genai, 'configure'):
                # Reset any existing configuration
                genai.configure(credentials=None, api_key=None)
            
            # Set up authentication with service account
            credentials = service_account.Credentials.from_service_account_file(
                self.service_account_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Configure Gemini with service account credentials only
            genai.configure(credentials=credentials)
            
            # Initialize model
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            
            logger.info("Gemini client initialized successfully with service account")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if Gemini API is available"""
        return self.model is not None
    
    def generate_response(self, prompt: str, model: str = "gemini-1.5-pro", 
                         max_tokens: int = 2000, temperature: float = 0.1) -> str:
        """Generate response using Gemini API"""
        try:
            if not self.model:
                raise ValueError("Gemini model not initialized")
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise

class AIClientFactory:
    """Factory for creating AI clients"""
    
    @staticmethod
    def create_client(provider: str, **kwargs) -> AIClient:
        """Create AI client based on provider"""
        provider = provider.lower()
        
        if provider == 'claude':
            api_key = kwargs.get('api_key')
            if not api_key:
                raise ValueError("Claude API key is required")
            return ClaudeClient(api_key)
        
        elif provider == 'gemini':
            service_account_path = kwargs.get('service_account_path')
            if not service_account_path:
                raise ValueError("Gemini service account path is required")
            return GeminiClient(service_account_path)
        
        else:
            raise ValueError(f"Unsupported AI provider: {provider}")

def create_service_account_file(service_account_json: str, file_path: str = "gemini-service-account.json"):
    """Create service account file from JSON string"""
    try:
        # Parse JSON to validate format
        service_account_data = json.loads(service_account_json)
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(service_account_data, f, indent=2)
        
        logger.info(f"Service account file created: {file_path}")
        return file_path
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to create service account file: {e}")
        raise

def clear_google_api_conflicts():
    """Clear conflicting Google API environment variables"""
    conflicting_vars = [
        'GOOGLE_API_KEY',
        'GOOGLE_APPLICATION_CREDENTIALS_API_KEY',
        'GEMINI_API_KEY'
    ]
    
    cleared_vars = []
    for var in conflicting_vars:
        if var in os.environ:
            del os.environ[var]
            cleared_vars.append(var)
            logger.info(f"Cleared conflicting environment variable: {var}")
    
    if cleared_vars:
        logger.info(f"Cleared {len(cleared_vars)} conflicting environment variables: {', '.join(cleared_vars)}")
    
    return cleared_vars 