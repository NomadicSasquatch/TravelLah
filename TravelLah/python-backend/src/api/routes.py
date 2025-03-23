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

import logging
import sys

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

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True  # This forces the basic configuration, overriding any previous settings
)

@app.post("/itinerary")
async def create_itinerary(payload: Dict[str, Any]):
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
        stream_options = options.model_dump()
        # Run the itinerary workflow (LLM calls, etc)
        itinerary = itinerary_workflow.run(stream_options)

        ##########################################################################
        # FIX: Assign a unique day identifier and ensure each activity ID is unique #
        ##########################################################################
        for day_index, day in enumerate(itinerary["tripFlow"], start=1):  # <-- CHANGED: iterate with index
            day["dayId"] = f"DAY-{day_index}"  # <-- ADDED: unique day ID
            for act_index, activity in enumerate(day["activityContent"], start=1):
                # Compose activityId as {dayId}-ACT-{number}
                activity["activityId"] = f"{day['dayId']}-ACT-{act_index}"  # <-- CHANGED: unique activity ID

        app_logger.info("Successfully generated itinerary with unique day and activity IDs")

        if mongoDB.insert(itinerary):
            app_logger.info("Inserted itinerary into MongoDB")
        else:
            app_logger.error("Failed to insert itinerary into MongoDB")

        return itinerary

    except Exception as e:
        error_msg = f"Error generating itinerary: {str(e)}"
        app_logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)


@app.patch("/updateActivity")
async def update_activity(payload: Dict[str, Any]):
    app_logger.info("Received activity update payload: %s", payload)

    trip_serial_no = payload.get("tripSerialNo")
    activity_id = payload.get("activityId")
    activity_date = payload.get("date")
    if not trip_serial_no or not activity_id or not activity_date:
        raise HTTPException(
            status_code=400,
            detail="Payload must include 'tripSerialNo', 'activityId', and 'date'."
        )

    # Transform payload into internal structure
    transformed_payload = transform_frontend_to_backend_format_updateActivity(payload)
    activity_params = transformed_payload.get("activity_params", {})

    options = StreamOptionsUpdate(
        task="",
        max_revisions=1,
        revision_number=1,
        activity_params=activity_params
    )

    app_logger.info("Running update workflow with activity_params: %s", activity_params)

    try:
        stream_options = options.model_dump()
        itineraryupdate = ItineraryUpdateWorkflow.run(stream_options)
        itineraryupdate["tripSerialNo"] = trip_serial_no
        itineraryupdate["activityId"] = activity_id
        itineraryupdate["date"] = activity_date

        # Convert snake_case to camelCase if needed
        if "specific_location" in itineraryupdate:
            itineraryupdate["specificLocation"] = itineraryupdate.pop("specific_location")
        if "start_time" in itineraryupdate:
            itineraryupdate["startTime"] = itineraryupdate.pop("start_time")
        if "end_time" in itineraryupdate:
            itineraryupdate["endTime"] = itineraryupdate.pop("end_time")
        if "activity_type" in itineraryupdate:
            itineraryupdate["activityType"] = itineraryupdate.pop("activity_type")

        app_logger.info("Final update object before DB update: %s", itineraryupdate)

        if mongoDB.update(itineraryupdate):
            app_logger.info("Successfully updated activity in MongoDB")
        else:
            app_logger.error("Failed to update activity in MongoDB")
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No matching trip/activity found for "
                    f"tripSerialNo={trip_serial_no}, date={activity_date}, activityId={activity_id}"
                )
            )

    except Exception as e:
        error_msg = f"Error updating activity: {str(e)}"
        app_logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)


@app.get("/")
async def read_root():
    return {"message": "Travel Agent API is running"}
