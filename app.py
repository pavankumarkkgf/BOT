from flask import Flask, request, jsonify, render_template
from chatbot import WebsiteChatbot
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize chatbot (will load on first request)
bot = None

def initialize_bot():
    global bot
    if bot is None:
        try:
            logger.info("🚀 Initializing NixVixa Chatbot...")
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
            bot = WebsiteChatbot(links)
            bot.load_data()
            logger.info("✅ Chatbot initialized successfully!")
        except Exception as e:
            logger.error(f"❌ Error initializing chatbot: {e}")
            bot = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Initialize bot on first request
        if bot is None:
            initialize_bot()
        
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'response': 'Please enter a message.'})
        
        # Get response from chatbot
        if bot is None:
            return jsonify({
                'response': 'Chatbot is still initializing. Please try again in a moment.',
                'status': 'error'
            })
        
        bot_response = bot.generate_response(user_message)
        
        return jsonify({
            'response': bot_response,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'response': 'Sorry, I encountered an error. Please try again.',
            'status': 'error'
        })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'message': 'NixVixa Chatbot is running!',
        'bot_initialized': bot is not None
    })

@app.route('/init')
def init_bot():
    """Manual initialization endpoint"""
    try:
        initialize_bot()
        return jsonify({
            'status': 'success',
            'message': 'Chatbot initialized successfully!' if bot else 'Chatbot initialization failed'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Initialization error: {str(e)}'
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)