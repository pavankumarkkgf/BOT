from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from chatbot import WebsiteChatbot
import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize chatbot with lazy loading
bot = None
initialization_lock = threading.Lock()
is_initializing = False

# Cache for responses
response_cache = {}
CACHE_TTL = 300  # 5 minutes

class BotInitializer:
    def __init__(self):
        self.bot = None
        self.initialized = False
        self.initialization_time = None
        self.error = None
        
    def initialize(self):
        global is_initializing
        
        with initialization_lock:
            if self.initialized or is_initializing:
                return self.bot
                
            is_initializing = True
            try:
                logger.info("🚀 Initializing NixVixa Chatbot...")
                
                # Website links to scrape
                links = [
                    "https://nixvixa-website.vercel.app/about",
                    "https://nixvixa-website.vercel.app/services", 
                    "https://nixvixa-website.vercel.app/projects",
                    "https://nixvixa-website.vercel.app/whychooseus",
                    "https://nixvixa-website.vercel.app/contact",
                    "https://nixvixa-website.vercel.app/website",
                    "https://nixvixa-website.vercel.app/digitalmarketing", 
                    "https://nixvixa-website.vercel.app/branbuilding",
                    "https://nixvixa-website.vercel.app/ai_automation",
                    "https://nixvixa-website.vercel.app/contentcreation",
                    "https://nixvixa-website.vercel.app/appdevlopment",
                    "https://nixvixa-website.vercel.app/privacy",
                    "https://nixvixa-website.vercel.app/terms"
                ]
                
                logger.info(f"📚 Loading {len(links)} website pages...")
                self.bot = WebsiteChatbot(links)
                
                # Train the bot
                logger.info("🧠 Training chatbot with website content...")
                self.bot.load_data()
                
                # Test bot initialization
                test_response = self.bot.generate_response("Hello")
                if test_response:
                    logger.info("✅ Chatbot initialized and tested successfully!")
                    logger.info(f"📊 Stats: {self.bot.get_stats()}")
                else:
                    raise Exception("Bot failed to generate test response")
                
                self.initialized = True
                self.initialization_time = datetime.now()
                self.error = None
                
            except Exception as e:
                logger.error(f"❌ Error initializing chatbot: {e}")
                self.error = str(e)
                self.bot = None
            finally:
                is_initializing = False
                
        return self.bot

bot_initializer = BotInitializer()

def get_bot():
    """Get or initialize the bot instance"""
    if not bot_initializer.initialized and not is_initializing:
        # Start initialization in background
        def init_in_background():
            bot_initializer.initialize()
        
        thread = threading.Thread(target=init_in_background)
        thread.daemon = True
        thread.start()
    
    return bot_initializer.bot

def get_cached_response(query: str) -> Optional[str]:
    """Get cached response if available and not expired"""
    if query in response_cache:
        cached_time, response = response_cache[query]
        if time.time() - cached_time < CACHE_TTL:
            return response
        else:
            del response_cache[query]
    return None

def cache_response(query: str, response: str):
    """Cache a response"""
    response_cache[query] = (time.time(), response)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    start_time = time.time()
    
    try:
        data = request.json
        if not data:
            return jsonify({
                'response': 'Invalid request. Please send JSON data.',
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            })
        
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({
                'response': 'Please enter a message.',
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
        
        # Check cache first
        cache_key = f"{session_id}:{user_message.lower()}"
        cached_response = get_cached_response(cache_key)
        if cached_response:
            logger.info(f"📦 Cache hit for: {user_message[:50]}...")
            return jsonify({
                'response': cached_response,
                'status': 'success',
                'cached': True,
                'timestamp': datetime.now().isoformat(),
                'response_time': round(time.time() - start_time, 3)
            })
        
        # Get bot instance
        bot_instance = get_bot()
        
        if not bot_instance:
            if is_initializing:
                return jsonify({
                    'response': '🤖 Chatbot is initializing. Please wait a moment and try again.',
                    'status': 'info',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'response': '⚠️ Chatbot is unavailable. Please try again later or contact support.',
                    'status': 'error',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Generate response
        bot_response = bot_instance.generate_response(user_message)
        
        # Cache the response
        cache_response(cache_key, bot_response)
        
        # Log the interaction
        logger.info(f"💬 User: {user_message[:100]}...")
        logger.info(f"🤖 Bot: {bot_response[:100]}...")
        
        return jsonify({
            'response': bot_response,
            'status': 'success',
            'cached': False,
            'timestamp': datetime.now().isoformat(),
            'response_time': round(time.time() - start_time, 3),
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'response': 'Sorry, I encountered an error while processing your request. Please try again.',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/health')
def health():
    """Health check endpoint"""
    bot_instance = get_bot()
    
    return jsonify({
        'status': 'healthy',
        'service': 'NixVixa Chatbot API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'bot': {
            'initialized': bot_initializer.initialized,
            'initializing': is_initializing,
            'initialization_time': bot_initializer.initialization_time.isoformat() if bot_initializer.initialization_time else None,
            'error': bot_initializer.error,
            'cache_size': len(response_cache)
        },
        'endpoints': {
            'chat': '/api/chat',
            'health': '/api/health',
            'stats': '/api/stats',
            'init': '/api/init'
        }
    })

@app.route('/api/stats')
def stats():
    """Get chatbot statistics"""
    bot_instance = get_bot()
    
    stats_data = {
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'bot_status': {
            'initialized': bot_initializer.initialized,
            'initializing': is_initializing,
            'uptime': str(datetime.now() - bot_initializer.initialization_time) if bot_initializer.initialization_time else None
        },
        'cache': {
            'size': len(response_cache),
            'ttl_seconds': CACHE_TTL
        }
    }
    
    if bot_instance and bot_initializer.initialized:
        try:
            bot_stats = bot_instance.get_stats()
            stats_data['content'] = bot_stats
        except:
            pass
    
    return jsonify(stats_data)

@app.route('/api/init', methods=['POST'])
def init_bot():
    """Manual initialization endpoint"""
    if is_initializing:
        return jsonify({
            'status': 'info',
            'message': 'Chatbot is already initializing.',
            'timestamp': datetime.now().isoformat()
        })
    
    try:
        # Initialize in background
        def init_and_notify():
            bot_initializer.initialize()
        
        thread = threading.Thread(target=init_and_notify)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Chatbot initialization started in background.',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Initialization error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    """Clear response cache"""
    global response_cache
    old_size = len(response_cache)
    response_cache = {}
    
    return jsonify({
        'status': 'success',
        'message': f'Cache cleared ({old_size} entries removed)',
        'timestamp': datetime.now().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

# Startup initialization
def startup():
    """Initialize bot on startup"""
    logger.info("🚀 Starting NixVixa Chatbot Server...")
    # Start initialization in background
    def init_in_background():
        time.sleep(2)  # Wait for server to fully start
        get_bot()
    
    thread = threading.Thread(target=init_in_background)
    thread.daemon = True
    thread.start()

# Call startup immediately when the module is loaded
startup()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🌐 Starting server on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)