import re
import requests
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebsiteChatbot:
    def __init__(self, urls):
        self.urls = urls if isinstance(urls, list) else [urls]
        self.chunks = []
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = None
        self.service_list = []
        self.company_info = []
        self.project_info = []
        self.contact_info = []
        
    def scrape_website(self, url):
        print(f"🔍 Scraping: {url}")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['nav', 'footer', 'header', 'script', 'style', 'aside']):
                element.decompose()
            
            # Extract ALL text content first
            all_text = soup.get_text(separator=' ', strip=True)
            
            # Then extract structured information
            self.extract_structured_info(soup, url)
            
            return all_text
            
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
            return ""
    
    def extract_structured_info(self, soup, url):
        """Extract specific information from different page types"""
        url_lower = url.lower()
        
        # Extract services
        if 'services' in url_lower:
            service_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'div'])
            for element in service_elements:
                text = element.get_text(strip=True)
                if self.is_service_content(text):
                    self.service_list.append(self.clean_text(text))
        
        # Extract about info
        elif 'about' in url_lower:
            about_elements = soup.find_all(['h1', 'h2', 'h3', 'p'])
            for element in about_elements:
                text = element.get_text(strip=True)
                if len(text) > 30 and self.is_about_content(text):
                    self.company_info.append(self.clean_text(text))
        
        # Extract project info
        elif 'projects' in url_lower or 'portfolio' in url_lower:
            project_elements = soup.find_all(['h1', 'h2', 'h3', 'p', 'div'])
            for element in project_elements:
                text = element.get_text(strip=True)
                if len(text) > 20 and self.is_project_content(text):
                    self.project_info.append(self.clean_text(text))
        
        # Extract contact info
        elif 'contact' in url_lower:
            contact_elements = soup.find_all(['p', 'div', 'span', 'li'])
            for element in contact_elements:
                text = element.get_text(strip=True)
                if self.is_contact_content(text):
                    self.contact_info.append(self.clean_text(text))
    
    def is_service_content(self, text):
        text_lower = text.lower()
        service_keywords = [
            'web development', 'digital marketing', 'app development', 'branding',
            'content creation', 'ai automation', 'website', 'mobile app',
            'seo', 'social media', 'graphic design', 'ui/ux', 'e-commerce'
        ]
        return any(keyword in text_lower for keyword in service_keywords) and len(text) > 10
    
    def is_about_content(self, text):
        text_lower = text.lower()
        about_keywords = ['about', 'company', 'story', 'mission', 'vision', 'team', 'expertise']
        return any(keyword in text_lower for keyword in about_keywords)
    
    def is_project_content(self, text):
        text_lower = text.lower()
        project_keywords = ['project', 'portfolio', 'work', 'case study', 'client', 'completed']
        return any(keyword in text_lower for keyword in project_keywords)
    
    def is_contact_content(self, text):
        text_lower = text.lower()
        contact_patterns = [
            r'\b[\w\.-]+@[\w\.-]+\.\w+\b',  # email
            r'\b\d{10}\b',  # phone number
            r'\bcontact\b', r'\bemail\b', r'\bphone\b', r'\baddress\b'
        ]
        return any(re.search(pattern, text_lower) for pattern in contact_patterns)
    
    def clean_text(self, text):
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s\.\!\?\-]', '', text)
        return text
    
    def is_meaningful_content(self, text):
        if not text or len(text) < 25:
            return False
        
        text_lower = text.lower()
        
        excluded_phrases = [
            'privacy policy', 'terms of service', 'copyright', 'all rights reserved',
            'home', 'menu', 'navigation', 'follow us', 'get in touch', 'click here',
            'learn more', 'read more', 'subscribe', 'sign up', 'back to top',
            'cookie', 'login', 'signin'
        ]
        
        if any(phrase in text_lower for phrase in excluded_phrases):
            return False
            
        return len(text.split()) >= 3
    
    def split_chunks(self, text, min_len=40):
        if not text:
            return []
            
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        
        for sentence in sentences:
            clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
            if len(clean_sentence) >= min_len and self.is_meaningful_content(clean_sentence):
                chunks.append(clean_sentence)
        
        return chunks
    
    def load_data(self):
        print("\n🔄 Loading website content...")
        all_chunks = []
        
        # Process all URLs
        for url in self.urls:
            try:
                text = self.scrape_website(url)
                if text:
                    chunks = self.split_chunks(text)
                    print(f"📄 From {url}: {len(chunks)} content pieces")
                    all_chunks.extend(chunks)
            except Exception as e:
                print(f"❌ Error processing {url}: {e}")
                continue
        
        # Remove duplicates
        unique_chunks = []
        seen = set()
        
        for chunk in all_chunks:
            if chunk and len(chunk) > 30:
                normalized = re.sub(r'\s+', ' ', chunk).strip().lower()
                if normalized not in seen:
                    seen.add(normalized)
                    unique_chunks.append(chunk)
        
        self.chunks = unique_chunks
        print(f"✅ Total content pieces: {len(self.chunks)}")
        
        # Remove duplicates from structured data
        self.service_list = list(set(self.service_list))[:15]
        self.company_info = list(set(self.company_info))[:10]
        self.project_info = list(set(self.project_info))[:10]
        self.contact_info = list(set(self.contact_info))[:8]
        
        print(f"🎯 Services found: {len(self.service_list)}")
        print(f"🏢 Company info: {len(self.company_info)}")
        print(f"📁 Project info: {len(self.project_info)}")
        print(f"📞 Contact info: {len(self.contact_info)}")
        
        if len(self.chunks) == 0:
            print("⚠️ No content found.")
            return
        
        print("🔧 Creating TF-IDF vectors...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        print("✅ Vectors created successfully\n")
    
    def retrieve_top_k(self, query, k=5, min_score=0.15):
        if self.tfidf_matrix is None or len(self.chunks) == 0:
            return []
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        top_indices = np.argsort(similarities)[-k*2:][::-1]
        results = []
        
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                results.append((self.chunks[idx], score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:k]
    
    def get_service_response(self):
        """Generate comprehensive service response"""
        if not self.service_list:
            # Fallback: extract from general content
            service_keywords = [
                'web development', 'digital marketing', 'app development', 
                'branding', 'content creation', 'ai automation'
            ]
            found_services = []
            
            for chunk in self.chunks:
                chunk_lower = chunk.lower()
                for service in service_keywords:
                    if service in chunk_lower and service not in found_services:
                        found_services.append(service.title())
            
            if found_services:
                self.service_list = found_services
        
        if not self.service_list:
            return None
        
        response = "**OUR SERVICES**\n\n"
        response += "NixVixa offers comprehensive digital solutions:\n\n"
        
        for service in self.service_list[:8]:
            response += f"• {service}\n"
        
        response += "\nWe provide end-to-end digital transformation services tailored to your business needs."
        return response
    
    def get_about_response(self):
        """Generate company information response"""
        if not self.company_info:
            return None
        
        response = "**ABOUT NIXVIXA**\n\n"
        
        for info in self.company_info[:3]:
            response += f"{info}\n\n"
        
        return response.strip()
    
    def get_project_response(self):
        """Generate project portfolio response"""
        if not self.project_info:
            return None
        
        response = "**OUR PROJECTS**\n\n"
        response += "Our project portfolio includes:\n\n"
        
        for project in self.project_info[:4]:
            response += f"• {project}\n"
        
        return response
    
    def get_contact_response(self):
        """Generate contact information response"""
        if not self.contact_info:
            return None
        
        response = "**CONTACT INFORMATION**\n\n"
        
        for contact in self.contact_info[:5]:
            response += f"• {contact}\n"
        
        return response
    
    def generate_response(self, user_text, top_k=5, min_score=0.15):
        text = user_text.lower().strip()
        
        # Enhanced greetings
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        if any(text.startswith(greet) for greet in greetings):
            return "Welcome to NixVixa Digital Solutions! I'm here to provide detailed information about our services, projects, and company expertise. How may I assist you today?"
        
        # Enhanced farewells
        farewells = ["bye", "goodbye", "exit", "quit", "see you"]
        if any(text.startswith(farewell) for farewell in farewells):
            return "Thank you for contacting NixVixa! We look forward to helping you achieve your digital goals."
        
        # Service queries
        service_queries = ['services', 'service', 'offer', 'provide', 'what do you do', 'solutions']
        if any(query in text for query in service_queries):
            service_response = self.get_service_response()
            if service_response:
                return service_response
        
        # About queries
        about_queries = ['about', 'company', 'who are you', 'nixvixa', 'story']
        if any(query in text for query in about_queries):
            about_response = self.get_about_response()
            if about_response:
                return about_response
        
        # Project queries
        project_queries = ['projects', 'portfolio', 'work', 'case studies']
        if any(query in text for query in project_queries):
            project_response = self.get_project_response()
            if project_response:
                return project_response
        
        # Contact queries
        contact_queries = ['contact', 'email', 'phone', 'address', 'reach', 'call']
        if any(query in text for query in contact_queries):
            contact_response = self.get_contact_response()
            if contact_response:
                return contact_response
        
        # General query processing
        results = self.retrieve_top_k(user_text, k=top_k, min_score=min_score)
        
        if not results:
            return "I specialize in providing information about NixVixa's digital services and solutions. For specific details about our offerings, please visit our website or contact our team directly."
        
        # Format professional response
        response = "Based on our website information:\n\n"
        
        for chunk, score in results:
            clean_chunk = re.sub(r'\s+', ' ', chunk).strip()
            if not clean_chunk.endswith(('.', '!', '?')):
                clean_chunk += '.'
            response += f"• {clean_chunk}\n"
        
        response += "\nFor comprehensive details, please visit our official website."
        return response