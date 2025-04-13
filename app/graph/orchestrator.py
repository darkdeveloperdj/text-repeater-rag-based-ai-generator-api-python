from app.agents.quote_generator_agent import QuoteAgent
from app.agents.love_generator_agent import LoveAgent

class Orchestrator:
    def __init__(self, love_template_file: str = None, quote_template_file: str = None):
        self.quote_agent = QuoteAgent(love_templates_file=None, quote_templates_file=quote_template_file)
        self.love_agent = LoveAgent(love_templates_file=love_template_file, quote_templates_file=None)
    
    def process_request(self, query: str, placeholders: dict, req_type: str):
        if req_type == 'love':
            agent = self.love_agent
            response = agent.generate(query, placeholders, req_type)
            # Clean the string response for love messages
            cleaned_response = self.clean_response(response)
            return cleaned_response
        elif req_type == 'quote':
            agent = self.quote_agent
            response = agent.generate(query, {}, req_type)
            # For quotes, response is already a dict with 'quote' and 'author'
            return response
        else:
            raise ValueError(f"Invalid request type: {req_type}")

    
    
    def clean_response(self, text: str):
        print(f"Raw response: {text}")
        import re
        return re.sub(r'\s+', ' ', text).strip()

