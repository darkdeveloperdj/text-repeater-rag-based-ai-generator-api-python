from .base_agent import BaseAgent
from app.models import OllamaClient
from app.models import FlanT5Generator
import random

class LoveAgent(BaseAgent):
    def __init__(self, love_templates_file: str = None, quote_templates_file: str = None):
        # Pass template_file as required in BaseAgent
        super().__init__(love_templates_file, quote_templates_file)
        # Model initialization
        #self.generator = OllamaClient(model_name="llama3.1:8b")
        # Uncomment the line below to use alternative model if needed
        self.generator = FlanT5Generator(model_name="google/flan-t5-small")
        # Optimized prompt: directly ask for the love message with necessary details.
        self.prompt_template = (
            "Using the following guidance for tone and style:\n"
            "{theme_context}\n\n"
            "Recipient Details:\n"
            "- Name: {recipient_name}\n"
            "- Author: {author_name}\n\n"
            "User Query: {query}\n\n"
            "Now generate a completely original, heartfelt love message that does not repeat any of the examples. Message:"
        )

    def generate(self, query: str, placeholders: dict, req_type: str):
        # Retrieve templates with enhanced embeddings
        retrieved_templates = self.retrieve_templates(query, 2, req_type)
        
        # Build detailed theme context with direct examples from the templates.
        theme_context = self._build_theme_context(retrieved_templates)
        
        # Construct prompt with full context
        prompt = self.prompt_template.format(
            theme_context=theme_context,
            recipient_name=placeholders.get('recipient_name', 'Unknown'),
            author_name=placeholders.get('author_name', 'Anonymous'),
            query=query
        )

        # Debug: Print the constructed prompt for troubleshooting
        print(f"Prompt: {prompt}")
        
        # Generate with optimized settings.
        # The system instruction directs the model to output only the heartfelt message.
        return self.generator.generate(
            prompt,
            max_length=200,
            temperature=0.65,
            system=(
                "You are a romantic writer. Respond with only the heartfelt love message. "
                "Do not preface the message with any lead-in text or commentary."
            )
        )

    def _build_theme_context(self, templates):
        if not templates:
            return "Theme: general\nExamples: Simple heartfelt message"
            
        context_lines = []
        for template in templates:
            # Pick two direct examples from the list of templates for this theme.
            examples = "\n".join(
                [f" - {random.choice(template['templates'])}" for _ in range(2)]
            )
            context_lines.append(
                f"Theme: {template['emotion']}\n"
                f"Style Guidance: The message should be tender, original, and reflective of {template['emotion']} emotion.\n"
                f"Examples:\n{examples}"
            )
        
        return "\n\n".join(context_lines)
