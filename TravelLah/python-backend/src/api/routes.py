from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ..agents.graph import itinerary_workflow
from .schemas import ItineraryResponse, StreamOptions
from src.settings.logging import app_logger
from ..services.mongoDB import mongoDB
from ..utils.jsonify import transform_frontend_to_backend_format

app = FastAPI(
    title="Travel Agent API", description="API for generating travel itineraries"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

@app.post("/itinerary")
# async def create_itinerary(options: StreamOptions):
async def create_itinerary(payload: Dict[str, Any]):

    """
    Generate a travel itinerary based on the provided options
    
    Args:
        options: StreamOptions object with task and parameters
        
    Returns:
        Generated itinerary
    """
    # app_logger.info(f"Received itinerary request: {options.task[:50]}...")
    
    transformed_payload = transform_frontend_to_backend_format(payload)
    itinerary_params = transformed_payload.get("itinerary_params", {})

    options = StreamOptions(
        task="",  # Default value
        max_revisions=1,  # Example default
        revision_number=1,  # Example default
        itinerary_params=itinerary_params
    )

    app_logger.info(f"Received itinerary request with params: {itinerary_params}")   
    
    try:
        # Convert Pydantic model to dict
        stream_options = options.model_dump()
        
        # Run the itinerary workflow
        itinerary = itinerary_workflow.run(stream_options)
        
        app_logger.info("Successfully generated itinerary")
        if mongoDB.insert(itinerary):
            app_logger.info("Inserted itinerary into MongoDB")
        else:
            app_logger.error("Failed to insert itinerary into MongoDB")

        app_logger.info("Returning itinerary")
        return itinerary
    except Exception as e:
        error_msg = f"Error generating itinerary: {str(e)}"
        app_logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

@app.get("/")
async def read_root():
    return {"message": "Travel Agent API is running"}
