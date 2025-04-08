from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import numpy as np
import random
import requests
import re
from typing import Dict, List, Optional, Tuple

app = Flask(__name__)

# Initialize NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Models
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Hugging Face LLM Config
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HF_HEADERS = {"Authorization": "Bearer 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"}

# --- Quote Estimator ---
def estimate_quote(age: int, insurance_type: str, loan_amount: float) -> float:
    base_rate = {
        "life": 0.002,
        "disability": 0.003,
        "critical illness": 0.004,
        "job loss": 0.0025
    }
    rate = base_rate.get(insurance_type.lower(), 0.003)
    age_factor = 1 + (max(age - 25, 0) * 0.02)
    return round(loan_amount * rate * age_factor, 2)

def parse_quote_request(query: str) -> Optional[Dict[str, str]]:
    insurance_types = ["life", "disability", "critical illness", "job loss"]
    found_type = next((t for t in insurance_types if t in query.lower()), None)
    
    amount_match = re.search(r"\$?(\d{3,6})", query)
    age_match = re.search(r"\b(\d{2})\b", query)

    if found_type and amount_match and age_match:
        return {
            "insurance_type": found_type,
            "loan_amount": float(amount_match.group(1)),
            "age": int(age_match.group(1))
        }
    return None

# --- Knowledge Base ---
class RBCKnowledgeBase:
    def __init__(self):
        self.documents = []
        self.document_embeddings = None
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        self.documents = [
            {
                "title": "What is Creditor Insurance?",
                "content": "Creditor insurance is protection that helps cover your loan or credit card payments if you experience certain life events. RBC offers coverage for situations like job loss, disability, critical illness, or death.",
                "url": "https://www.rbcroyalbank.com/insurance/creditor-insurance.html#what-is"
            },
            {
                "title": "Types of Coverage",
                "content": "RBC provides four main types: 1. Life Insurance (covers outstanding balance if you die), 2. Disability Insurance (covers payments if you become disabled), 3. Critical Illness Insurance (covers payments if diagnosed with serious illness), 4. Job Loss Insurance (covers payments if you involuntarily lose your job).",
                "url": "https://www.rbcroyalbank.com/insurance/creditor-insurance.html#types"
            },
            {
                "title": "How to Apply",
                "content": "You can apply: 1. When opening a new loan/credit product, 2. By contacting RBC Insurance later. The process usually involves a short health questionnaire.",
                "url": "https://www.rbcroyalbank.com/insurance/creditor-insurance.html#apply"
            },
            {
                "title": "Cost Information",
                "content": "Costs vary based on: loan amount, repayment term, your age, health factors, and type of coverage selected. Contact RBC for a personalized quote.",
                "url": "https://www.rbcroyalbank.com/insurance/creditor-insurance.html#costs"
            },
            {
                "title": "Is this insurance mandatory?",
                "content": "No, creditor insurance is completely optional but recommended for financial protection.",
                "url": "https://www.rbcroyalbank.com/insurance/creditor-insurance.html#faq1"
            },
            {
                "title": "Can I cancel my coverage?",
                "content": "Yes, you can typically cancel with written notice, but terms vary by product. There may be a waiting period before cancellation takes effect.",
                "url": "https://www.rbcroyalbank.com/insurance/creditor-insurance.html#faq2"
            }
        ]
        
        texts = [f"{doc['title']} {doc['content']}" for doc in self.documents]
        if texts:
            self.document_embeddings = sentence_model.encode(texts)
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 2) -> List[Dict]:
        if self.document_embeddings is None or len(self.document_embeddings) == 0:
            return []
        query_embedding = sentence_model.encode(query)
        scores = np.dot(query_embedding, self.document_embeddings.T)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices if scores[i] > 0.5]

# --- LLM Wrapper ---
def query_llm(prompt: str, context: str = "") -> Optional[str]:
    try:
        if context:
            formatted_prompt = f"""<s>[INST] <<SYS>>
You are a helpful RBC financial expert assistant. Use the following context to answer the question:
{context}
<</SYS>>
{prompt} [/INST]"""
        else:
            formatted_prompt = f"""<s>[INST] <<SYS>>
You are a helpful RBC financial expert assistant. Answer the question professionally.
<</SYS>>
{prompt} [/INST]"""

        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }

        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '').strip()
        else:
            print(f"LLM API error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

# --- Chatbot ---
class RBCChatbot:
    def __init__(self):
        self.knowledge_base = RBCKnowledgeBase()
        self.chat_history = []
        self.greetings = [
            "Hello! I'm the RBC Financial Assistant. How can I help you today?",
            "Welcome to RBC's financial help. What would you like to know?",
            "Hi there! Ask me anything about RBC services or general finance."
        ]
        self.fallbacks = [
            "Based on RBC's typical offerings, I can tell you that...",
            "While I don't have the exact details, RBC generally provides...",
            "RBC usually handles similar cases by..."
        ]
    
    def generate_response(self, query: str) -> Dict:
        query = query.strip()
        if not query:
            return {"response": "Please type your question.", "sources": []}
        

        if any(word in query.lower() for word in ["hi", "hello", "hey"]):
            return {"response": random.choice(self.greetings), "sources": []}
        if any(word in query.lower() for word in ["thanks", "thank", "bye", "goodbye"]):
            return {"response": "You're welcome! Let me know if you have other questions.", "sources": []}
     

        # Check for repeated queries
        if len(self.chat_history) >= 2 and self.chat_history[-2]["role"] == "user" and \
            self.chat_history[-2]["content"].lower() == query.lower():
            return {
                "response": "I notice you've asked this question already. Let me try to provide more information or clarify my previous answer. " + 
                    "Please let me know if you have a specific aspect you'd like me to elaborate on.",
                "sources": []
    }
        
        # 👇 Check for quote request
        quote_data = parse_quote_request(query)
        if quote_data:
            quote = estimate_quote(quote_data["age"], quote_data["insurance_type"], quote_data["loan_amount"])
            return {
                "response": f"Sure! Based on a {quote_data['age']}-year-old applying for {quote_data['insurance_type']} insurance on a loan of ${quote_data['loan_amount']:.2f}, your estimated monthly premium is **${quote:.2f}**. Please contact RBC for exact pricing.",
                "sources": []
            }

        # 👇 Regular chat logic with RAG
        self.chat_history.append({"role": "user", "content": query})
        history_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.chat_history[-5:]])
        relevant_docs = self.knowledge_base.retrieve_relevant_docs(query)
        knowledge_context = "\n".join([f"{doc['title']}: {doc['content']}" for doc in relevant_docs])
        full_context = f"{history_context}\n\n{knowledge_context}".strip()

        response_text = query_llm(query, full_context)
        if not response_text:
            response_text = random.choice(self.fallbacks)
        self.chat_history.append({"role": "assistant", "content": response_text})
        sources = [{"title": doc["title"], "url": doc["url"]} for doc in relevant_docs]

        return {"response": response_text, "sources": sources}

# --- Flask Routes ---
chatbot = RBCChatbot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    data = request.json
    user_input = data.get('message', '')
    result = chatbot.generate_response(user_input)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
