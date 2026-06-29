import os
import requests
import time
from dotenv import load_dotenv
import streamlit as st
import json

# Load environment variables
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
API_URL = "https://api.mistral.ai/v1/chat/completions"

class EnhancedQAEngine:
    """Enhanced Question-Answering engine with better prompting, streaming, and Ollama/Mistral support"""
    
    def __init__(self):
        self.api_key = API_KEY
        self.api_url = API_URL
        self.max_retries = 3
        self.retry_delay = 1
    
    def generate_answer_stream(self, context, question, settings=None, conversation_history=None):
        """
        Generate an answer stream using Mistral AI or Ollama with dynamic prompting and parameters.
        
        Args:
            context: Relevant text chunks
            question: User's question
            settings: Dictionary of LLM settings
            conversation_history: Previous conversation for context
            
        Yields:
            Tokens of the response
        """
        try:
            if not question.strip():
                yield "❌ Please provide a valid question."
                return
            
            # Create default settings if not provided
            if settings is None:
                settings = {
                    "provider": "Mistral",
                    "model": "mistral-small",
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": 1500,
                    "api_key": self.api_key,
                    "ollama_url": "http://localhost:11434"
                }
            
            is_chitchat = settings.get("is_chitchat", False)
            
            # Validate inputs
            if not is_chitchat and not context.strip():
                yield "❌ No relevant context found to answer your question."
                return
            
            provider = settings.get("provider", "Mistral")
            model = settings.get("model", "mistral-small")
            temperature = settings.get("temperature", 0.3)
            top_p = settings.get("top_p", 0.9)
            max_tokens = settings.get("max_tokens", 1500)
            api_key = settings.get("api_key") or self.api_key
            ollama_url = settings.get("ollama_url", "http://localhost:11434")
            
            # Create enhanced prompts
            if is_chitchat:
                system_prompt = "You are a helpful, polite AI document assistant. The user is offering a greeting, thanks, or general feedback. Respond briefly and politely in a natural conversational tone, and invite them to ask more questions about their documents if they need help."
                user_prompt = question
            else:
                is_comparison = settings.get("is_comparison", False) if settings else False
                system_prompt = self._create_system_prompt()
                user_prompt = self._create_user_prompt(context, question, conversation_history, is_comparison)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Add conversation history if available
            if conversation_history:
                messages = self._add_conversation_history(messages, conversation_history)
            
            # Route to appropriate provider stream
            if provider == "Mistral":
                yield from self._stream_mistral(messages, model, temperature, top_p, max_tokens, api_key)
            elif provider == "Ollama":
                yield from self._stream_ollama(messages, model, temperature, top_p, max_tokens, ollama_url)
            else:
                yield f"❌ Unsupported LLM provider: {provider}"
                
        except Exception as e:
            yield f"❌ Error generating stream: {str(e)}"
            
    def _stream_mistral(self, messages, model, temperature, top_p, max_tokens, api_key):
        """Stream from Mistral AI API"""
        if not api_key:
            yield "❌ Mistral API key is missing. Please add MISTRAL_API_KEY to your .env file or configure it in the Configuration settings."
            return
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": float(top_p),
            "stream": True
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    json=payload,
                    timeout=45,
                    stream=True
                )
                
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8').strip()
                            if decoded_line.startswith('data: '):
                                data_str = decoded_line[6:]
                                if data_str.strip() == '[DONE]':
                                    break
                                try:
                                    data = json.loads(data_str)
                                    delta = data['choices'][0]['delta'].get('content', '')
                                    if delta:
                                        yield delta
                                except Exception:
                                    pass
                    return  # Success
                    
                elif response.status_code == 429:  # Rate limit
                    wait_time = self.retry_delay * (2 ** attempt)
                    yield f"\n⚠️ Rate limit reached. Retrying in {wait_time}s...\n"
                    time.sleep(wait_time)
                    continue
                    
                elif response.status_code == 401:
                    yield "❌ Invalid Mistral API key. Please check your credentials."
                    return
                else:
                    error_detail = response.text
                    try:
                        error_detail = response.json().get("message", response.text)
                    except:
                        pass
                    yield f"❌ Mistral API Error {response.status_code}: {error_detail}"
                    return
                    
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    yield f"\n⚠️ Request timeout. Retrying...\n"
                    time.sleep(self.retry_delay)
                    continue
                else:
                    yield "❌ Connection timed out. Please try again later."
            except Exception as e:
                yield f"❌ Unexpected error in Mistral API connection: {str(e)}"
                return
                
    def _stream_ollama(self, messages, model, temperature, top_p, max_tokens, ollama_url):
        """Stream from local Ollama API"""
        endpoint = f"{ollama_url.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "num_predict": int(max_tokens)
            },
            "stream": True
        }
        
        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=45,
                stream=True
            )
            
            if response.status_code != 200:
                yield f"❌ Ollama returned status code {response.status_code}. Make sure Ollama is running and model '{model}' is installed."
                return
                
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        delta = data.get('message', {}).get('content', '')
                        if delta:
                            yield delta
                        if data.get('done', False):
                            break
                    except Exception:
                        pass
        except requests.exceptions.ConnectionError:
            yield f"❌ Connection Refused. Make sure Ollama is active at {ollama_url} (run 'ollama serve' or open the Ollama app) and that you have pulled the model '{model}'."
        except Exception as e:
            yield f"❌ Unexpected error in Ollama API connection: {str(e)}"
    
    def generate_answer(self, context, question, settings=None, conversation_history=None):
        """
        Generate a complete answer (non-streaming, backward compatible wrapper)
        """
        tokens = []
        for token in self.generate_answer_stream(context, question, settings, conversation_history):
            tokens.append(token)
        answer = "".join(tokens)
        return self._post_process_answer(answer)
        
    def _create_system_prompt(self):
        """Create an enhanced system prompt"""
        return """You are an intelligent AI assistant specializing in document analysis and question answering. Your role is to provide accurate, helpful, and well-structured answers based on the given context.

Guidelines for your responses:
1. **Accuracy**: Only use information from the provided context. If the answer isn't in the context, clearly state this.
2. **Clarity**: Provide clear, well-structured answers that are easy to understand.
3. **Completeness**: Give comprehensive answers that fully address the question.
4. **Citations**: When possible, reference specific parts of the context.
5. **Honesty**: If you're uncertain or if information is unclear, acknowledge this.
6. **Formatting**: Use proper formatting with bullet points, numbers, or paragraphs as appropriate.

Response Structure:
- Start with a direct answer to the question
- Provide supporting details from the context
- Include relevant examples or explanations
- End with any caveats or limitations if applicable

Remember: You are answering based on document content, so maintain an informative and professional tone."""
    
    def _create_user_prompt(self, context, question, conversation_history=None, is_comparison=False):
        """Create an enhanced user prompt with context and question"""
        # Truncate context if too long (keep within token limits)
        max_context_length = 4000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n[... content truncated ...]"
        
        if is_comparison:
            prompt = f"""Based on the provided sections from the two documents, perform a comprehensive comparison and answer the question.

**Document Comparison Context:**
{context}

**Question:** {question}

**Instructions:**
- Identify and compare findings, clauses, stats, or parameters between Document A and Document B.
- Use clean Markdown formatting. Highlight differences using a side-by-side comparison table where appropriate.
- Refer strictly to the content under the respective 'Source Document A' and 'Source Document B' headings. Do not invent differences that are not supported by the text.
- If the documents agree on a point, note it. If details are missing for one of the documents, explicitly state that."""
        else:
            prompt = f"""Based on the following document content, please answer the question comprehensively.

**Document Content:**
{context}

**Question:** {question}

**Instructions:**
- Use only the information provided in the document content above
- If the answer requires information not in the context, clearly state this limitation
- Provide specific details and examples from the text when possible
- Structure your answer clearly with appropriate formatting
- If multiple aspects of the question can be answered, address each one"""

        return prompt
    
    def _add_conversation_history(self, messages, history):
        """Add relevant conversation history for context"""
        if not history:
            return messages
        
        # Add last few exchanges for context
        recent_history = history[-3:] if len(history) > 3 else history
        
        for question, answer, _, _ in recent_history:
            messages.insert(-1, {"role": "user", "content": f"Previous Question: {question}"})
            messages.insert(-1, {"role": "assistant", "content": f"Previous Answer: {answer[:200]}..."})
        
        return messages
    
    def _post_process_answer(self, answer):
        """Post-process the generated answer"""
        if not answer:
            return "❌ No response generated."
        
        answer = answer.strip()
        answer = answer.replace('\n\n\n', '\n\n')
        
        if not answer.startswith(('❌', '✅', '📝', '💡', '⚠️')):
            if 'cannot' in answer.lower() or 'not found' in answer.lower():
                answer = f"⚠️ {answer}"
            else:
                answer = f"📝 {answer}"
        
        return answer
    
    def validate_api_key(self, api_key=None):
        """Validate the Mistral API key"""
        target_key = api_key or self.api_key
        if not target_key:
            return False, "No API key found. Please add MISTRAL_API_KEY to your .env file or input it in settings."
        
        headers = {
            "Authorization": f"Bearer {target_key}",
            "Content-Type": "application/json"
        }
        
        test_payload = {
            "model": "mistral-small",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5
        }
        
        try:
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=test_payload,
                timeout=8
            )
            
            if response.status_code == 200:
                return True, "API key is valid."
            elif response.status_code == 401:
                return False, "Invalid API key."
            else:
                return False, f"API test failed with status {response.status_code}"
                
        except Exception as e:
            return False, f"API test failed: {str(e)}"

# Global instance
qa_engine = EnhancedQAEngine()

def generate_answer_stream(context, question, settings=None, conversation_history=None):
    """
    Wrapper function for streaming answers
    """
    return qa_engine.generate_answer_stream(context, question, settings, conversation_history)

def generate_answer(context, question, settings=None, conversation_history=None):
    """
    Wrapper function for backward compatibility
    """
    return qa_engine.generate_answer(context, question, settings, conversation_history)

def validate_api_configuration(api_key=None):
    """Validate API configuration"""
    return qa_engine.validate_api_key(api_key)

def get_model_info():
    """Get information about the current models"""
    return {
        "providers": ["Mistral AI (Cloud)", "Ollama (Local)"],
        "mistral_models": [
            "mistral-small",
            "mistral-medium",
            "mistral-large-latest",
            "open-mixtral-8x7b"
        ],
        "ollama_default_models": [
            "llama3",
            "mistral",
            "gemma",
            "phi3"
        ],
        "features": [
            "Streaming responses",
            "Local Ollama support",
            "Context-aware parent chunks retrieval",
            "Multi-turn conversations",
            "Dynamic hyperparameter adjustments (temp, top_p, etc)"
        ]
    }