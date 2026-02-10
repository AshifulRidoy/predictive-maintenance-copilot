"""
AWS Bedrock integration for Predictive Maintenance Copilot.
Provides alternative LLM backend using AWS Bedrock instead of OpenRouter.
"""
import json
import time
from typing import Optional, Dict, Any, List
import boto3
from botocore.exceptions import ClientError

from src.config import Config


class BedrockClient:
    """AWS Bedrock client for LLM inference."""
    
    # Available Bedrock models
    MODELS = {
        'claude-3-5-sonnet': 'anthropic.claude-3-5-sonnet-20241022-v2:0',
        'claude-3-5-haiku': 'anthropic.claude-3-5-haiku-20241022-v1:0',
        'claude-3-haiku': 'anthropic.claude-3-haiku-20240307-v1:0',
        'claude-3-sonnet': 'anthropic.claude-3-sonnet-20240229-v1:0',
        'llama-3-70b': 'meta.llama3-70b-instruct-v1:0',
        'llama-3-8b': 'meta.llama3-8b-instruct-v1:0',
        'mistral-7b': 'mistral.mistral-7b-instruct-v0:2',
        'mistral-large': 'mistral.mistral-large-2402-v1:0',
    }
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None
    ):
        """
        Initialize Bedrock client.
        
        Args:
            model_id: Bedrock model ID (e.g., 'claude-3-5-sonnet')
            region_name: AWS region
            aws_access_key_id: AWS access key
            aws_secret_access_key: AWS secret key
        """
        self.region_name = region_name or Config.AWS_REGION
        
        # Initialize boto3 client
        session_kwargs = {'region_name': self.region_name}
        
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs['aws_access_key_id'] = aws_access_key_id
            session_kwargs['aws_secret_access_key'] = aws_secret_access_key
        elif Config.AWS_ACCESS_KEY_ID and Config.AWS_SECRET_ACCESS_KEY:
            session_kwargs['aws_access_key_id'] = Config.AWS_ACCESS_KEY_ID
            session_kwargs['aws_secret_access_key'] = Config.AWS_SECRET_ACCESS_KEY
        
        self.bedrock_runtime = boto3.client('bedrock-runtime', **session_kwargs)
        
        # Set model
        self.model_id = model_id or 'claude-3-5-haiku'
        self.model_arn = self.MODELS.get(self.model_id, self.model_id)
        
        print(f"✓ Bedrock client initialized")
        print(f"  Region: {self.region_name}")
        print(f"  Model: {self.model_id} ({self.model_arn})")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1000,
        system: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a chat completion using Bedrock.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system: System prompt (optional)
            
        Returns:
            Response dictionary with usage stats
        """
        start_time = time.time()
        
        try:
            # Format for Claude models (Anthropic format)
            if 'claude' in self.model_arn.lower() or 'anthropic' in self.model_arn.lower():
                response = self._invoke_claude(messages, temperature, max_tokens, system)
            
            # Format for Llama models
            elif 'llama' in self.model_arn.lower():
                response = self._invoke_llama(messages, temperature, max_tokens, system)
            
            # Format for Mistral models
            elif 'mistral' in self.model_arn.lower():
                response = self._invoke_mistral(messages, temperature, max_tokens, system)
            
            else:
                raise ValueError(f"Unsupported model: {self.model_arn}")
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                'content': response['content'],
                'usage': response.get('usage', {}),
                'latency_ms': latency_ms,
                'model': self.model_arn
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            
            print(f"✗ Bedrock error [{error_code}]: {error_message}")
            
            raise Exception(f"Bedrock API error: {error_message}")
    
    def _invoke_claude(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        system: Optional[str] = None
    ) -> Dict[str, Any]:
        """Invoke Claude models via Bedrock."""
        # Convert messages format
        bedrock_messages = []
        for msg in messages:
            if msg['role'] != 'system':  # System handled separately
                bedrock_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        # Build request body
        body = {
            'messages': bedrock_messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'anthropic_version': 'bedrock-2023-05-31'
        }
        
        # Add system prompt if provided
        if system:
            body['system'] = system
        elif messages and messages[0]['role'] == 'system':
            body['system'] = messages[0]['content']
        
        # Invoke model
        response = self.bedrock_runtime.invoke_model(
            modelId=self.model_arn,
            body=json.dumps(body)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        # Extract content
        content = response_body['content'][0]['text']
        
        return {
            'content': content,
            'usage': {
                'prompt_tokens': response_body.get('usage', {}).get('input_tokens', 0),
                'completion_tokens': response_body.get('usage', {}).get('output_tokens', 0),
                'total_tokens': response_body.get('usage', {}).get('input_tokens', 0) + 
                               response_body.get('usage', {}).get('output_tokens', 0)
            }
        }
    
    def _invoke_llama(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        system: Optional[str] = None
    ) -> Dict[str, Any]:
        """Invoke Llama models via Bedrock."""
        # Llama uses prompt format
        prompt = self._format_llama_prompt(messages, system)
        
        body = {
            'prompt': prompt,
            'temperature': temperature,
            'max_gen_len': max_tokens,
        }
        
        response = self.bedrock_runtime.invoke_model(
            modelId=self.model_arn,
            body=json.dumps(body)
        )
        
        response_body = json.loads(response['body'].read())
        
        return {
            'content': response_body['generation'],
            'usage': {
                'prompt_tokens': response_body.get('prompt_token_count', 0),
                'completion_tokens': response_body.get('generation_token_count', 0),
                'total_tokens': response_body.get('prompt_token_count', 0) + 
                               response_body.get('generation_token_count', 0)
            }
        }
    
    def _invoke_mistral(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        system: Optional[str] = None
    ) -> Dict[str, Any]:
        """Invoke Mistral models via Bedrock."""
        # Mistral uses prompt format similar to Llama
        prompt = self._format_mistral_prompt(messages, system)
        
        body = {
            'prompt': prompt,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        
        response = self.bedrock_runtime.invoke_model(
            modelId=self.model_arn,
            body=json.dumps(body)
        )
        
        response_body = json.loads(response['body'].read())
        
        return {
            'content': response_body['outputs'][0]['text'],
            'usage': {
                'total_tokens': 0  # Mistral doesn't always provide token counts
            }
        }
    
    def _format_llama_prompt(self, messages: List[Dict[str, str]], system: Optional[str]) -> str:
        """Format messages for Llama prompt."""
        prompt_parts = []
        
        if system:
            prompt_parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>")
        
        for msg in messages:
            if msg['role'] == 'system' and not system:
                prompt_parts.append(f"<|start_header_id|>system<|end_header_id|>\n{msg['content']}<|eot_id|>")
            elif msg['role'] == 'user':
                prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n{msg['content']}<|eot_id|>")
            elif msg['role'] == 'assistant':
                prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n{msg['content']}<|eot_id|>")
        
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>")
        
        return "".join(prompt_parts)
    
    def _format_mistral_prompt(self, messages: List[Dict[str, str]], system: Optional[str]) -> str:
        """Format messages for Mistral prompt."""
        prompt_parts = []
        
        if system:
            prompt_parts.append(f"<s>[INST] {system}\n")
        else:
            prompt_parts.append("<s>")
        
        for i, msg in enumerate(messages):
            if msg['role'] == 'system' and not system:
                prompt_parts.append(f"[INST] {msg['content']}\n")
            elif msg['role'] == 'user':
                if i == 0 and not system:
                    prompt_parts.append(f"[INST] {msg['content']} [/INST]")
                else:
                    prompt_parts.append(f"[INST] {msg['content']} [/INST]")
            elif msg['role'] == 'assistant':
                prompt_parts.append(f"{msg['content']}</s>")
        
        return " ".join(prompt_parts)
    
    def list_available_models(self) -> List[str]:
        """List all available Bedrock models in the region."""
        try:
            bedrock_client = boto3.client('bedrock', region_name=self.region_name)
            response = bedrock_client.list_foundation_models()
            
            models = []
            for model in response.get('modelSummaries', []):
                models.append({
                    'modelId': model['modelId'],
                    'modelName': model['modelName'],
                    'provider': model['providerName']
                })
            
            return models
        except Exception as e:
            print(f"Error listing models: {e}")
            return []


class BedrockOpenAIAdapter:
    """
    Adapter to make Bedrock client compatible with OpenAI API interface.
    Allows drop-in replacement in existing code.
    """
    
    def __init__(self, bedrock_client: BedrockClient):
        """
        Initialize adapter.
        
        Args:
            bedrock_client: Initialized BedrockClient
        """
        self.bedrock = bedrock_client
        self.chat = self.Chat(bedrock_client)
    
    class Chat:
        """Chat completions namespace (OpenAI-compatible)."""
        
        def __init__(self, bedrock_client: BedrockClient):
            self.bedrock = bedrock_client
            self.completions = self.Completions(bedrock_client)
        
        class Completions:
            """Completions methods."""
            
            def __init__(self, bedrock_client: BedrockClient):
                self.bedrock = bedrock_client
            
            def create(
                self,
                model: str,
                messages: List[Dict[str, str]],
                temperature: float = 0.1,
                max_tokens: int = 1000,
                **kwargs
            ):
                """
                Create chat completion (OpenAI-compatible).
                
                Args:
                    model: Model name (ignored, uses Bedrock model)
                    messages: List of messages
                    temperature: Sampling temperature
                    max_tokens: Max tokens
                    
                Returns:
                    OpenAI-compatible response object
                """
                # Extract system message if present
                system = None
                filtered_messages = []
                
                for msg in messages:
                    if msg['role'] == 'system':
                        system = msg['content']
                    else:
                        filtered_messages.append(msg)
                
                # Call Bedrock
                response = self.bedrock.chat_completion(
                    messages=filtered_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system=system
                )
                
                # Format as OpenAI response
                class Choice:
                    def __init__(self, content):
                        self.message = type('Message', (), {'content': content})()
                        self.finish_reason = 'stop'
                        self.index = 0
                
                class Usage:
                    def __init__(self, usage_dict):
                        self.prompt_tokens = usage_dict.get('prompt_tokens', 0)
                        self.completion_tokens = usage_dict.get('completion_tokens', 0)
                        self.total_tokens = usage_dict.get('total_tokens', 0)
                
                class Response:
                    def __init__(self, content, usage_dict):
                        self.choices = [Choice(content)]
                        self.usage = Usage(usage_dict)
                        self.model = response['model']
                        self.created = int(time.time())
                
                return Response(response['content'], response['usage'])


def create_bedrock_client(model_id: str = 'claude-3-5-haiku') -> BedrockOpenAIAdapter:
    """
    Create Bedrock client with OpenAI-compatible interface.
    
    Args:
        model_id: Bedrock model to use
        
    Returns:
        OpenAI-compatible client
    """
    bedrock = BedrockClient(model_id=model_id)
    return BedrockOpenAIAdapter(bedrock)
