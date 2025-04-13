from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.graph.orchestrator import Orchestrator

router = APIRouter()

orchestrator = Orchestrator(
    love_template_file="app/templates/love_generator_templates.json",
    quote_template_file="app/templates/quotes_generator_templates.json"
)

# Response Models
class QuoteGenerationResponse(BaseModel):
    quote: str
    author: str

class GenericGenerationResponse(BaseModel):
    result: str

# Request Models
class PlaceholderValues(BaseModel):
    recipient_name: str = Field(default="Kashish", example="Kashish")
    author_name: str = Field(default="Dhiraj", example="Dhiraj")

class GenerationRequest(BaseModel):
    query: str = Field(..., example="generate a love message")
    placeholders: PlaceholderValues = Field(default_factory=PlaceholderValues)
    type: str = Field(..., example="love")

class QuoteRequest(BaseModel):
    query: str = Field(..., example="generate a quote")
    type: str = Field(default="quote", example="quote")

# Endpoints
@router.post("/generate_love", response_model=GenericGenerationResponse)
async def generate_love(request: GenerationRequest):
    try:
        result = orchestrator.process_request(
            request.query,
            request.placeholders.model_dump(),
            request.type
        )
        return {"result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate_quote", response_model=QuoteGenerationResponse)
async def generate_quote(request: QuoteRequest):
    try:
        quote_data = orchestrator.process_request(
            request.query,
            {},  # No placeholders needed
            "quote"
        )

        if isinstance(quote_data, dict):
            return quote_data

        return {
            "quote": str(quote_data),
            "author": "Unknown"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
