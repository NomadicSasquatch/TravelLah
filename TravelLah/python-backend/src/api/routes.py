from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ..agents.ItineraryGraph import itinerary_workflow
from ..agents.ItineraryUpdateGraph import ItineraryUpdateWorkflow

from .ItinerarySchemas import  StreamOptions


from .ItinerarySchemas import ItineraryResponse
from .ItineraryUpdateSchemas import StreamOptionsUpdate
from .ItinerarySchemas import StreamOptions


from src.settings.logging import app_logger
from ..services.mongoDB import mongoDB
from ..utils.jsonify import transform_frontend_to_backend_format_itinerary, transform_frontend_to_backend_format_updateActivity

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
    
    transformed_payload = transform_frontend_to_backend_format_itinerary(payload)
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

#  update(self, trip_id: str, activity_id: int, updated_activity: dict) -> bool:
@app.patch("/updateActivity")
async def update_activity(payload: Dict[str, Any]):
    """
    1) Transform the frontend data into 'activity_params'
    2) Run the itinerary update workflow to produce a refined activity
    3) Attach 'tripSerialNo' and 'activityId' so that the DB can find the correct slot
    4) Update the DB, and return the final updated activity
    """

    app_logger.info(f"Received activity update payload: {payload}...")

    # 1) Extract tripSerialNo and activityId from the incoming payload
    trip_serial_no = payload.get("tripSerialNo")
    activity_id = payload.get("activityId")
    if not trip_serial_no or activity_id is None:
        # If these are missing, we won't be able to update the correct slot
        raise HTTPException(
            status_code=400,
            detail="Payload must include 'tripSerialNo' and 'activityId'."
        )

    # 2) Transform the rest of the data into your internal structure
    transformed_payload = transform_frontend_to_backend_format_updateActivity(payload)
    activity_params = transformed_payload.get("activity_params", {})

    # 3) Build your Pydantic input
    options = StreamOptionsUpdate(
        task="",  # We can supply a custom prompt if desired
        max_revisions=1,
        revision_number=1,
        activity_params=activity_params
    )

    app_logger.info(f"Running update workflow with activity_params: {activity_params}")

    try:
        stream_options = options.model_dump()

        # 4) Run the update workflow, which produces the final "draft" activity JSON
        #    returned as a Python dictionary
        itineraryupdate = ItineraryUpdateWorkflow.run(stream_options)

        # 5) The returned 'itineraryupdate' usually won't contain tripSerialNo/activityId
        #    by default — so we attach them here, so the DB's array_filters can locate
        #    and update the correct activity
        itineraryupdate["tripSerialNo"] = trip_serial_no
        itineraryupdate["activityId"] = activity_id

        # 6) Update the DB
        app_logger.info("Successfully updated itinerary in LLM. Now updating MongoDB...")
        if mongoDB.update(itineraryupdate):
            app_logger.info("Successfully updated activity in MongoDB")
        else:
            app_logger.error("Failed to update activity in MongoDB")
            # Return a 404 if no document was matched:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No matching trip/activity found for tripSerialNo={trip_serial_no},"
                    f" activityId={activity_id}"
                )
            )

        # 7) Return the final updated activity
        return itineraryupdate

    except Exception as e:
        error_msg = f"Error updating activity: {str(e)}"
        app_logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)


@app.get("/")
async def read_root():
    return {"message": "Travel Agent API is running"}
