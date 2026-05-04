from flask import Blueprint, request, jsonify
from openai import OpenAI  # Using the new OpenAI client
import os
from dotenv import load_dotenv
import logging
from functools import wraps
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client with comprehensive checks
def initialize_openai():
    """Initialize OpenAI client with proper error handling"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    try:
        client = OpenAI(api_key=api_key)
        # Test the connection with a simple request
        client.models.list(timeout=5)  # 5 second timeout for initialization check
        return client
    except Exception as e:
        logger.error(f"OpenAI initialization failed: {str(e)}")
        raise

try:
    client = initialize_openai()
except Exception as e:
    logger.critical(f"Failed to initialize OpenAI: {str(e)}")
    # Create a dummy client that will fail gracefully
    client = None

chatbot_bp = Blueprint('chatbot', __name__)

# Rate limiting decorator
def rate_limit(max_calls=5, time_frame=60):
    def decorator(f):
        calls = []
        
        @wraps(f)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls_in_timeframe = [call for call in calls if call > now - time_frame]
            
            if len(calls_in_timeframe) >= max_calls:
                logger.warning("Rate limit exceeded")
                return jsonify({'error': 'Too many requests. Please wait a minute.'}), 429
                
            calls.append(now)
            return f(*args, **kwargs)
        return wrapper
    return decorator

@chatbot_bp.route('/ask_doctor', methods=['POST'])
@rate_limit(max_calls=10, time_frame=60)  # 10 requests per minute
def ask_doctor():
    """Handle chatbot requests with comprehensive error handling"""
    try:
        # Validate request
        if not request.is_json:
            logger.warning("Request is not JSON")
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        message = data.get('message', '').strip()
        symptoms = data.get('symptoms', '').strip()
        prediction = data.get('prediction', '').strip()

        if not message:
            logger.warning("Empty message received")
            return jsonify({'error': 'Message cannot be empty'}), 400

        # Check if OpenAI is available
        if not client:
            logger.error("OpenAI client not initialized")
            return jsonify({'error': 'AI service is currently unavailable'}), 503

        # Create safe prompt with input sanitization
        def sanitize_input(text):
            return text.replace('\n', ' ').replace('\r', '')[:500]  # Basic sanitization
            
        prompt = f"""You are Doctor AI, an intelligent medical assistant. Provide helpful information about:
        - Predicted condition: {sanitize_input(prediction) if prediction else 'Not specified'}
        - Reported symptoms: {sanitize_input(symptoms) if symptoms else 'Not specified'}
        - User question: {sanitize_input(message)}
        
        Guidelines:
        - Be professional and empathetic
        - Explain in simple terms
        - Don't diagnose or prescribe
        - Recommend consulting a doctor
        - Never provide dangerous advice"""
        
        # Make API call with timeout
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500,
                timeout=10  # 10 second timeout
            )
            
            reply = response.choices[0].message.content.strip()
            
            # Additional safety check
            if not reply or len(reply) > 2000:
                raise ValueError("Invalid response from AI")
                
            logger.info("Successfully generated response")
            return jsonify({'reply': reply})
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return jsonify({
                'error': 'Unable to get response from AI',
                'details': str(e)
            }), 502
            
    except Exception as e:
        logger.error(f"Unexpected server error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'request_id': request.headers.get('X-Request-ID', 'none')
        }), 500