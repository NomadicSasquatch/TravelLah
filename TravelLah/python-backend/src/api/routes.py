from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.agents.graph import ItineraryWorkflow
from api.schemas import ItineraryResponse, StreamOptions
from src.settings.logging import app_logger

app = FastAPI(
    title="Travel Agent API", description="API for generating travel itineraries"
)
origins = ["http://localhost:3000", "localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/itinerary", response_model=ItineraryResponse)
async def create_itinerary(options: StreamOptions):
    """
    Generate a travel itinerary based on the provided options
    
    Args:
        options: StreamOptions object with task and parameters
        
    Returns:
        Generated itinerary
    """
    app_logger.info(f"Received itinerary request: {options.task[:50]}...")
    
    try:
        # Convert Pydantic model to dict
        stream_options = options.model_dump()
        
        # Run the itinerary workflow
        itinerary = ItineraryWorkflow.run(stream_options)
        
        app_logger.info("Successfully generated itinerary")
        return itinerary
    except Exception as e:
        error_msg = f"Error generating itinerary: {str(e)}"
        app_logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

@app.get("/")
async def read_root():
    return {"message": "Travel Agent API is running"}
