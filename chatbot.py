import re
import requests
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Tuple, Dict, Any
import html
import time
from urllib.parse import urljoin
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebsiteChatbot:
    def __init__(self, urls: List[str]):
        self.urls = urls
        self.chunks = []
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.tfidf_matrix = None
        self.content_map = {}  # Map chunks to their source URLs
        self.structured_data = {
            'services': [],
            'about': [],
            'projects': [],
            'contact': [],
            'pricing': [],
            'features': [],
            'testimonials': [],
            'faq': []
        }
        self.url_content_cache = {}
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded content"""
        return {
            'total_urls': len(self.urls),
            'total_chunks': len(self.chunks),
            'structured_data': {k: len(v) for k, v in self.structured_data.items()},
            'vectorizer_vocab_size': len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 0
        }
    
    def scrape_website(self, url: str) -> str:
        """Scrape and extract content from a URL"""
        cache_key = hashlib.md5(url.encode()).hexdigest()
        
        if cache_key in self.url_content_cache:
            logger.info(f"📦 Using cached content for: {url}")
            return self.url_content_cache[cache_key]
        
        logger.info(f"🔍 Scraping: {url}")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Check if content is HTML
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                logger.warning(f"⚠️ Non-HTML content at {url}: {content_type}")
                return ""
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            unwanted_tags = ['nav', 'footer', 'header', 'script', 'style', 'aside', 
                           'form', 'button', 'input', 'select', 'textarea']
            for tag in soup.find_all(unwanted_tags):
                tag.decompose()
            
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.startswith('<!--')):
                comment.extract()
            
            # Extract text with better structure
            text_parts = []
            
            # Prioritize main content areas
            main_selectors = ['main', 'article', '.content', '#content', '.main-content', 
                            '.post-content', '.entry-content', '.page-content']
            
            main_content = None
            for selector in main_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if main_content:
                content = main_content
            else:
                content = soup.body if soup.body else soup
            
            # Extract headings and paragraphs
            for tag in content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'div']):
                if tag.name == 'div' and not tag.get_text(strip=True):
                    continue
                    
                text = tag.get_text(separator=' ', strip=True)
                if text and len(text) > 10:
                    # Add heading markers for context
                    if tag.name.startswith('h'):
                        text = f"HEADING {tag.name.upper()}: {text}"
                    text_parts.append(text)
            
            # Join all text parts
            full_text = ' '.join(text_parts)
            
            # Clean and normalize
            full_text = self.clean_text(full_text)
            
            # Extract structured information
            self.extract_structured_info(soup, url, full_text)
            
            # Cache the result
            self.url_content_cache[cache_key] = full_text
            
            return full_text
            
        except requests.exceptions.Timeout:
            logger.error(f"⏰ Timeout scraping {url}")
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"🌐 Network error scraping {url}: {e}")
            return ""
        except Exception as e:
            logger.error(f"❌ Error scraping {url}: {str(e)}")
            return ""
    
    def extract_structured_info(self, soup: BeautifulSoup, url: str, full_text: str):
        """Extract structured information based on URL patterns"""
        url_lower = url.lower()
        
        # Keywords for different content types
        content_patterns = {
            'services': ['service', 'offer', 'solution', 'provide', 'expertise', 'capability'],
            'about': ['about', 'company', 'story', 'mission', 'vision', 'team', 'values'],
            'projects': ['project', 'portfolio', 'work', 'case study', 'client', 'success story'],
            'contact': ['contact', 'email', 'phone', 'address', 'location', 'reach us', 'get in touch'],
            'pricing': ['price', 'cost', 'plan', 'package', 'pricing', 'fee', 'subscription'],
            'features': ['feature', 'benefit', 'advantage', 'key', 'capability', 'functionality'],
            'testimonials': ['testimonial', 'review', 'client said', 'customer feedback', 'rating'],
            'faq': ['faq', 'question', 'answer', 'how to', 'what is', 'why', 'how does']
        }
        
        # Extract text from relevant elements
        for content_type, keywords in content_patterns.items():
            if any(keyword in url_lower for keyword in keywords):
                # Extract from specific elements
                elements = soup.find_all(['p', 'li', 'div', 'span'])
                for element in elements:
                    text = element.get_text(strip=True)
                    if text and len(text) > 20:
                        text_lower = text.lower()
                        if any(keyword in text_lower for keyword in keywords):
                            clean_text = self.clean_text(text)
                            if clean_text not in self.structured_data[content_type]:
                                self.structured_data[content_type].append(clean_text)
        
        # Also extract from full text using patterns
        self.extract_from_full_text(full_text, url_lower)
    
    def extract_from_full_text(self, text: str, url: str):
        """Extract structured information from full text using regex patterns"""
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text, re.IGNORECASE)
        if emails and 'contact' not in url:
            self.structured_data['contact'].extend([f"Email: {email}" for email in emails[:3]])
        
        # Extract phone numbers (international format)
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        if phones and 'contact' not in url:
            self.structured_data['contact'].extend([f"Phone: {phone[0] if isinstance(phone, tuple) else phone}" 
                                                   for phone in phones[:2]])
        
        # Extract service mentions
        service_keywords = {
            'web development': ['website', 'web app', 'frontend', 'backend', 'full stack'],
            'digital marketing': ['seo', 'social media', 'marketing', 'campaign', 'ads'],
            'app development': ['mobile app', 'ios', 'android', 'flutter', 'react native'],
            'branding': ['brand', 'logo', 'identity', 'design system'],
            'content creation': ['content', 'blog', 'article', 'copywriting'],
            'ai automation': ['ai', 'artificial intelligence', 'automation', 'machine learning']
        }
        
        for service, keywords in service_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                if service not in self.structured_data['services']:
                    self.structured_data['services'].append(service)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?\-:;()\'"@]', '', text)
        
        # Normalize quotes
        text = text.replace('"', "'").replace('“', "'").replace('”', "'")
        
        # Trim and return
        return text.strip()
    
    def is_meaningful_content(self, text: str) -> bool:
        """Check if text is meaningful content"""
        if not text or len(text) < 25:
            return False
        
        text_lower = text.lower()
        
        # Exclude navigation and boilerplate
        excluded_phrases = [
            'privacy policy', 'terms of service', 'copyright', 'all rights reserved',
            'cookie policy', 'sitemap', 'home', 'menu', 'navigation', 'skip to content',
            'follow us', 'share this', 'related posts', 'popular tags', 'recent comments',
            'get in touch', 'click here', 'learn more', 'read more', 'subscribe now',
            'sign up', 'back to top', 'login', 'register', 'search', 'filter by',
            'load more', 'view all', 'next page', 'previous page'
        ]
        
        if any(phrase in text_lower for phrase in excluded_phrases):
            return False
        
        # Check for sufficient word count
        words = text.split()
        if len(words) < 3:
            return False
        
        # Check for meaningful content (not just numbers or symbols)
        alnum_count = sum(1 for c in text if c.isalnum())
        if alnum_count / len(text) < 0.5:
            return False
        
        return True
    
    def split_chunks(self, text: str, source_url: str = "", min_len: int = 40, max_len: int = 200) -> List[str]:
        """Split text into meaningful chunks"""
        if not text:
            return []
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            clean_sentence = self.clean_text(sentence)
            if not clean_sentence or len(clean_sentence) < 10:
                continue
                
            sentence_length = len(clean_sentence.split())
            
            if current_length + sentence_length <= max_len and len(current_chunk) < 3:
                current_chunk.append(clean_sentence)
                current_length += sentence_length
            else:
                if current_chunk and current_length >= min_len:
                    chunk_text = ' '.join(current_chunk)
                    if self.is_meaningful_content(chunk_text):
                        chunks.append(chunk_text)
                        self.content_map[chunk_text] = source_url
                
                current_chunk = [clean_sentence]
                current_length = sentence_length
        
        # Add the last chunk
        if current_chunk and current_length >= min_len:
            chunk_text = ' '.join(current_chunk)
            if self.is_meaningful_content(chunk_text):
                chunks.append(chunk_text)
                self.content_map[chunk_text] = source_url
        
        return chunks
    
    def load_data(self):
        """Load and process all website data"""
        logger.info("\n" + "="*60)
        logger.info("🚀 LOADING WEBSITE CONTENT FOR CHATBOT TRAINING")
        logger.info("="*60)
        
        all_chunks = []
        failed_urls = []
        
        for idx, url in enumerate(self.urls, 1):
            logger.info(f"\n[{idx}/{len(self.urls)}] Processing: {url}")
            
            try:
                text = self.scrape_website(url)
                if text:
                    chunks = self.split_chunks(text, url)
                    logger.info(f"   ✅ Extracted {len(chunks)} content chunks")
                    all_chunks.extend(chunks)
                else:
                    logger.warning(f"   ⚠️ No content extracted")
                    failed_urls.append(url)
                    
            except Exception as e:
                logger.error(f"   ❌ Error: {str(e)}")
                failed_urls.append(url)
                continue
        
        # Remove duplicates while preserving order
        unique_chunks = []
        seen_chunks = set()
        
        for chunk in all_chunks:
            # Normalize for deduplication
            normalized = re.sub(r'\s+', ' ', chunk).strip().lower()
            if normalized not in seen_chunks and len(chunk) > 30:
                seen_chunks.add(normalized)
                unique_chunks.append(chunk)
        
        self.chunks = unique_chunks
        
        logger.info("\n" + "="*60)
        logger.info("📊 CONTENT LOADING SUMMARY")
        logger.info("="*60)
        logger.info(f"✅ Successfully processed: {len(self.urls) - len(failed_urls)}/{len(self.urls)} URLs")
        logger.info(f"📝 Total unique content chunks: {len(self.chunks)}")
        
        if failed_urls:
            logger.info(f"⚠️ Failed URLs: {len(failed_urls)}")
            for url in failed_urls:
                logger.info(f"   - {url}")
        
        # Log structured data stats
        logger.info("\n🏗️ STRUCTURED DATA EXTRACTED:")
        for category, items in self.structured_data.items():
            if items:
                logger.info(f"   {category.capitalize()}: {len(items)} items")
                for item in items[:2]:  # Show first 2 items
                    logger.info(f"     • {item[:80]}...")
        
        if len(self.chunks) == 0:
            logger.error("❌ CRITICAL: No content was loaded!")
            raise Exception("No content available for chatbot training")
        
        # Create TF-IDF vectors
        logger.info("\n🔧 Creating TF-IDF vectors...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        logger.info(f"✅ Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        logger.info("="*60 + "\n")
    
    def retrieve_relevant_chunks(self, query: str, k: int = 7, min_score: float = 0.1) -> List[Tuple[str, float]]:
        """Retrieve most relevant chunks using cosine similarity"""
        if self.tfidf_matrix is None or len(self.chunks) == 0:
            return []
        
        # Transform query
        query_vec = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top indices
        top_indices = np.argsort(similarities)[-k*3:][::-1]
        
        # Filter by minimum score and deduplicate
        results = []
        seen_content = set()
        
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                chunk = self.chunks[idx]
                # Simple deduplication
                if chunk not in seen_content:
                    seen_content.add(chunk)
                    results.append((chunk, score))
            
            if len(results) >= k:
                break
        
        return results
    
    def generate_service_response(self) -> str:
        """Generate comprehensive service response"""
        services = self.structured_data['services']
        
        if not services:
            # Extract from chunks
            service_keywords = [
                'web development', 'website development', 'web design',
                'digital marketing', 'seo', 'social media marketing',
                'app development', 'mobile app', 'application development',
                'branding', 'brand identity', 'logo design',
                'content creation', 'content marketing', 'copywriting',
                'ai automation', 'artificial intelligence', 'machine learning',
                'software development', 'ecommerce', 'shopify', 'wordpress'
            ]
            
            found_services = []
            for chunk in self.chunks:
                chunk_lower = chunk.lower()
                for keyword in service_keywords:
                    if keyword in chunk_lower and keyword not in found_services:
                        found_services.append(keyword.title())
            
            services = found_services[:10]
        
        if not services:
            return "I specialize in digital solutions including web development, digital marketing, and custom software services. Could you be more specific about what you're looking for?"
        
        response = "🚀 **OUR DIGITAL SERVICES**\n\n"
        response += "At NixVixa, we offer comprehensive digital transformation solutions:\n\n"
        
        for i, service in enumerate(services[:8], 1):
            response += f"{i}. **{service.title()}** - Professional {service.lower()} services tailored to your business needs\n"
        
        response += "\n💡 *Each service includes strategy, implementation, and ongoing support.*"
        response += "\n\nWhich service are you interested in learning more about?"
        
        return response
    
    def generate_about_response(self) -> str:
        """Generate company information response"""
        about_info = self.structured_data['about']
        
        if not about_info:
            # Look for about information in chunks
            about_keywords = ['about', 'company', 'mission', 'vision', 'values', 'story', 'team']
            for chunk in self.chunks:
                chunk_lower = chunk.lower()
                if any(keyword in chunk_lower for keyword in about_keywords):
                    about_info.append(chunk)
        
        if not about_info:
            return "NixVixa is a digital solutions provider specializing in web development, digital marketing, and custom software solutions. We help businesses transform their digital presence and achieve their goals through innovative technology."
        
        response = "🏢 **ABOUT NIXVIXA**\n\n"
        
        # Use first 3 relevant pieces of information
        for info in about_info[:3]:
            response += f"• {info}\n\n"
        
        response += "🌟 *We're committed to delivering exceptional digital experiences that drive business growth.*"
        
        return response
    
    def generate_contact_response(self) -> str:
        """Generate contact information response"""
        contact_info = self.structured_data['contact']
        
        if not contact_info:
            # Extract from chunks
            contact_patterns = [
                r'email[:\s]*([\w\.-]+@[\w\.-]+\.\w+)',
                r'phone[:\s]*([\+\d\s\-\(\)]{10,})',
                r'contact[:\s]*([\w\.-]+@[\w\.-]+\.\w+)',
                r'call[:\s]*([\+\d\s\-\(\)]{10,})'
            ]
            
            for chunk in self.chunks:
                for pattern in contact_patterns:
                    matches = re.findall(pattern, chunk, re.IGNORECASE)
                    for match in matches:
                        if match not in contact_info:
                            contact_info.append(match)
        
        if not contact_info:
            contact_info = [
                "Email: contact@nixvixa.com",
                "Phone: +1 (555) 123-4567",
                "Website: https://nixvixa.com"
            ]
        
        response = "📞 **CONTACT NIXVIXA**\n\n"
        response += "Get in touch with our team:\n\n"
        
        for info in contact_info[:5]:
            response += f"• {info}\n"
        
        response += "\n📍 *We're available Monday-Friday, 9AM-6PM EST*"
        response += "\n\n💬 *You can also schedule a consultation through our website.*"
        
        return response
    
    def generate_project_response(self) -> str:
        """Generate project portfolio response"""
        projects = self.structured_data['projects']
        
        if not projects:
            # Extract project-like content from chunks
            project_keywords = ['project', 'portfolio', 'case study', 'client work', 'success story']
            for chunk in self.chunks:
                chunk_lower = chunk.lower()
                if any(keyword in chunk_lower for keyword in project_keywords):
                    projects.append(chunk)
        
        if not projects:
            return "We've successfully delivered numerous digital projects across various industries including e-commerce platforms, corporate websites, mobile applications, and digital marketing campaigns. Each project is customized to meet specific client requirements."
        
        response = "📁 **OUR PROJECT PORTFOLIO**\n\n"
        response += "Recent successful projects include:\n\n"
        
        for i, project in enumerate(projects[:4], 1):
            # Clean up project description
            clean_project = re.sub(r'\s+', ' ', project).strip()
            if len(clean_project) > 150:
                clean_project = clean_project[:147] + "..."
            response += f"{i}. {clean_project}\n\n"
        
        response += "🎯 *We deliver tailored solutions that drive measurable results.*"
        
        return response
    
    def generate_response(self, user_text: str, top_k: int = 5, min_score: float = 0.15) -> str:
        """Generate a response to user input"""
        text = user_text.lower().strip()
        
        # Enhanced greeting detection
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "hola", "namaste"]
        if any(text.startswith(greet) for greet in greetings):
            return "👋 Hello! Welcome to **NixVixa Digital Solutions**! \n\nI'm your AI assistant, here to provide detailed information about our services, projects, and expertise. \n\nHow can I help you today? You can ask about:\n• Our services\n• Company information\n• Project portfolio\n• Contact details\n• Pricing information"
        
        # Enhanced farewell detection
        farewells = ["bye", "goodbye", "exit", "quit", "see you", "thanks bye", "thank you bye"]
        if any(farewell in text for farewell in farewells):
            return "👋 Thank you for chatting with NixVixa! \n\nWe're here to help transform your digital presence. Feel free to reach out anytime! \n\nHave a great day! 🚀"
        
        # Thank you responses
        if any(phrase in text for phrase in ["thank", "thanks", "appreciate"]):
            return "You're welcome! 😊 \n\nIs there anything else about NixVixa's services you'd like to know?"
        
        # Service queries
        service_queries = [
            'service', 'offer', 'provide', 'what do you do', 'solutions',
            'web development', 'digital marketing', 'app development', 'branding',
            'seo', 'social media', 'content creation', 'ai automation'
        ]
        if any(query in text for query in service_queries):
            return self.generate_service_response()
        
        # About queries
        about_queries = [
            'about', 'company', 'who are you', 'nixvixa', 'story',
            'mission', 'vision', 'values', 'team'
        ]
        if any(query in text for query in about_queries):
            return self.generate_about_response()
        
        # Project queries
        project_queries = [
            'project', 'portfolio', 'work', 'case study',
            'examples', 'show me your work', 'previous work'
        ]
        if any(query in text for query in project_queries):
            return self.generate_project_response()
        
        # Contact queries
        contact_queries = [
            'contact', 'email', 'phone', 'address', 'reach',
            'call', 'location', 'get in touch', 'support'
        ]
        if any(query in text for query in contact_queries):
            return self.generate_contact_response()
        
        # Pricing queries
        if any(word in text for word in ['price', 'cost', 'how much', 'pricing', 'fee', 'rate']):
            return "💰 **PRICING INFORMATION**\n\nOur pricing varies based on project scope, complexity, and requirements. We offer:\n\n• **Custom Quotes** for enterprise solutions\n• **Package Deals** for standard services\n• **Hourly Rates** for consulting\n• **Monthly Retainers** for ongoing support\n\n📊 *For an accurate quote, please share your project details or schedule a consultation.*"
        
        # Process general query
        results = self.retrieve_relevant_chunks(user_text, k=top_k, min_score=min_score)
        
        if not results:
            # Fallback to structured responses
            if len(text.split()) < 3:
                return "🤔 I'd be happy to help! Could you please provide more details about what you're looking for?"
            
            return "🔍 I couldn't find specific information about that in our current knowledge base. \n\nHowever, I can help you with:\n• Service information\n• Company details\n• Project examples\n• Contact information\n• Pricing inquiries\n\nPlease try rephrasing your question or ask about one of these topics!"
        
        # Format response with context
        response = "🤖 **Based on our website information:**\n\n"
        
        for idx, (chunk, score) in enumerate(results, 1):
            # Clean and format chunk
            clean_chunk = re.sub(r'\s+', ' ', chunk).strip()
            
            # Add context if we have URL info
            if chunk in self.content_map:
                url = self.content_map[chunk]
                page_name = url.split('/')[-1].replace('-', ' ').title()
                response += f"**{idx}. From {page_name}:**\n"
            
            # Ensure proper punctuation
            if not clean_chunk.endswith(('.', '!', '?')):
                clean_chunk += '.'
            
            response += f"{clean_chunk}\n\n"
        
        response += "---\n"
        response += "📚 *For more detailed information, please visit our website or contact us directly.*"
        response += "\n\n❓ *Is there anything specific you'd like to know more about?*"
        
        return response