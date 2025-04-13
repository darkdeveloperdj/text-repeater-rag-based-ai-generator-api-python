from .base_agent import BaseAgent
from app.models import FlanT5Generator
from app.models import OllamaClient

class QuoteAgent(BaseAgent):
    def __init__(self, love_templates_file: str = None, quote_templates_file: str = None):
        super().__init__(love_templates_file, quote_templates_file)
        #self.generator = OllamaClient(model_name="llama3.1:8b")
        self.generator = FlanT5Generator(model_name="google/flan-t5-small")
        self.prompt_template = (
            "Below is guidance on the style and tone for an inspirational quote:\n"
            "{theme_context}\n\n"
            "Please generate a completely original, motivational quote that does not copy any examples above. "
            "Output the quote in the exact format: \"quote\" - author.\n"
            "User Query: {query}\n"
            "Quote:"
        )

    def generate(self, query: str, placeholders: dict, req_type: str):
        # Retrieve thematic guidance based on stored samples.
        retrieved_templates = self.retrieve_templates(query, 2, req_type)
        
        # Build the theme context including direct quote examples.
        theme_context = self._build_theme_context(retrieved_templates)
        
        # Format the prompt with explicit instructions for originality.
        prompt = self.prompt_template.format(
            theme_context=theme_context,
            query=query
        )
        
        # Debugging: Check the constructed prompt.
        print(f"Prompt: {prompt}")
        
        # Generate text using parameters that encourage diversity.
        generated_text = self.generator.generate(
            prompt,
            max_length=120,
            temperature=0.75,
            top_p=0.9
        )
        
        # Ensure the output is a string.
        if not isinstance(generated_text, str):
            generated_text = str(generated_text)
        
        # Parse the output into quote and author using the expected format.
        if " - " in generated_text:
            quote, author = generated_text.rsplit(" - ", 1)
        else:
            quote = generated_text
            author = placeholders.get("author_name", "Unknown")
        
        return {
            "quote": quote.strip('" ').strip(),
            "author": author.strip()
        }

    def _build_theme_context(self, templates):
        if not templates:
            return ("Style Guidance: Your quote should be motivational, uplifting, and evoke a sense of resilience and hope.\n"
                    "Example: \"Live life to the fullest\" - Unknown")
        
        context_lines = []
        # For each retrieved template, include a direct example.
        for template in templates:
            # Format one direct example from the template.
            example = f" - \"{template.get('quote', 'No quote provided')}\" - {template.get('author', 'Unknown')}"
            context = (
                "Theme: Inspirational\n"
                "Style Guidance: Generate a creative and unique quote that is inspirational and thought provoking.\n"
                f"Example:\n{example}"
            )
            context_lines.append(context)
        
        return "\n\n".join(context_lines)
