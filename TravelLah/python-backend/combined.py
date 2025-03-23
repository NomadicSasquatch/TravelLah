# main.py

# =============================================================================
# CONFIGURATION & DEPENDENCIES (from config.py and others)
# =============================================================================
import os
import pathlib
from dotenv import load_dotenv
from typing import List, Dict, Any
from datetime import datetime
import sys
import logging
import copy
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Dummy function for converting OpenAI messages (replace with real conversion if needed)
def convert_openai_messages(messages: list) -> list:
    return messages

# --------------------------
# Settings (from config.py)
# --------------------------
ROOT_DIR = pathlib.Path(__file__).parent.resolve()
load_dotenv(dotenv_path=ROOT_DIR / ".env")

class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY")
    MONGODB_PASS: str = os.getenv("MONGODB_PASS")
    LLM_MODEL: str = "gemini-1.5-flash"
    LLM_TEMPERATURE: int = 0
    MAX_SEARCH_RESULTS: int = 2
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    def validate(self):
        missing_keys: List[str] = []
        if not self.GOOGLE_API_KEY:
            missing_keys.append("GOOGLE_API_KEY")
        if not self.TAVILY_API_KEY:
            missing_keys.append("TAVILY_API_KEY")
        if missing_keys:
            raise ValueError(f"Missing environment variables: {', '.join(missing_keys)}")
        return True

settings = Settings()

# --------------------------
# Logger (from logging.py)
# --------------------------
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

app_logger = setup_logger("travel_agent")

# --------------------------
# JSON Transformation (from jsonify.py)
# --------------------------
def transform_frontend_to_backend_format_itinerary(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Parse dates from ISO format
    check_in = datetime.fromisoformat(payload["checkIn"].replace("Z", "+00:00"))
    check_out = datetime.fromisoformat(payload["checkOut"].replace("Z", "+00:00"))
    num_days = (check_out - check_in).days
    formatted_dates = f"{check_in.strftime('%Y-%m-%d')} to {check_out.strftime('%Y-%m-%d')}"
    start_date = check_in.strftime('%Y-%m-%d')
    end_date = check_out.strftime('%Y-%m-%d')
    guests_and_rooms = payload.get("guestsAndRooms", {})
    party_size = guests_and_rooms.get("adults", 0) + guests_and_rooms.get("children", 0)
    num_rooms = guests_and_rooms.get("rooms", 1)
    itinerary_params = {
        "userId": "U123",
        "tripId": "T151",
        "destination": payload.get("destination", "Bali"),
        "numDays": num_days,
        "dates": formatted_dates,
        "startDate": start_date,
        "endDate": end_date,
        "partySize": party_size,
        "num_rooms": num_rooms,
        "budget": payload.get("budget", ""),
        "activities": payload.get("activities", ""),
        "food": payload.get("food", "").lower(),
        "pace": payload.get("pace", "").lower(),
        "notes": payload.get("additionalNotes", "")
    }
    return {"itinerary_params": itinerary_params}

def transform_frontend_to_backend_format_updateActivity(payload: Dict[str, Any]) -> Dict[str, Any]:
    # ADDED travelLocation, latitude, longitude to the activity_params dict
    activity_params = {
        "activityId": payload.get("activityId", ""),
        "address": payload.get("specificLocation", ""),
        "date": payload.get("date", ""),
        "activity": payload.get("notes", ""),
        "startTime": payload.get("startTime", ""),
        "endTime": payload.get("endTime", ""),
        "activityType": payload.get("activityType", ""),
        "travelLocation": payload.get("travelLocation", ""),  # <-- ADDED
        "latitude": payload.get("latitude", ""),              # <-- ADDED
        "longitude": payload.get("longitude", "")             # <-- ADDED
    }
    return {"activity_params": activity_params}

# =============================================================================
# SCHEMAS
# =============================================================================
# --------------------------
# Itinerary Schemas (from ItinerarySchemas.py)
# --------------------------
class StreamOptions(BaseModel):
    task: str
    max_revisions: int
    revision_number: int
    itinerary_params: Dict[str, Any]

class ActivityContent(BaseModel):
    specificLocation: str
    address: str
    latitude: str
    longitude: str
    startTime: str
    endTime: str
    activityType: str
    notes: str
    activityId: str

class TripDay(BaseModel):
    date: str
    activityContent: List[ActivityContent]

class Itinerary(BaseModel):
    userId: str
    tripSerialNo: str
    travelLocation: str
    latitude: str
    longitude: str
    tripFlow: List[TripDay]

class ItineraryResponse(BaseModel):
    itinerary: Itinerary

# --------------------------
# Itinerary Update Schemas (from ItineraryUpdateSchemas.py)
# --------------------------
class StreamOptionsUpdate(BaseModel):
    task: str
    max_revisions: int
    revision_number: int
    activity_params: Dict[str, Any]

class ActivityContentUpdate(BaseModel):
    activityId: str
    specificLocation: str
    address: str
    latitude: str
    longitude: str
    startTime: str
    endTime: str
    activityType: str
    notes: str

class TripDayUpdate(BaseModel):
    date: str
    activity_content: List[ActivityContentUpdate]

class ItineraryUpdate(BaseModel):
    userId: str
    tripSerialNo: str
    travelLocation: str
    latitude: str
    longitude: str
    tripFlow: List[TripDayUpdate]

class ItineraryUpdateResponse(BaseModel):
    itinerary: ItineraryUpdate

# =============================================================================
# TEMPLATES
# =============================================================================
# --------------------------
# Itinerary Plan Templates (from ItineraryPlanTemplates.py)
# --------------------------
class PlannerPrompts:
    VACATION_PLANNING_SUPERVISOR = (
        "You are the vacation planning supervisor. You have to give a detailed outline of what the planning agent "
        "has to consider when planning the vacation according to the user input."
    )
    PLANNER_ASSISTANT = (
        "You are an assistant charged with providing information that can be used by the planner to plan the vacation. "
        "Generate a list of search queries that will be useful for the planner. Generate a maximum of 3 queries."
    )
    PLANNER_CRITIQUE = (
        "Your duty is to criticize the planning done by the vacation planner. In your response include if you agree with "
        "options presented by the planner, if not then give detailed suggestions on what should be changed. You can also "
        "suggest some other destination that should be checked out."
    )
    PLANNER_CRITIQUE_ASSISTANT = (
        "You are an assistant charged with providing information that can be used to make any requested revisions. Generate a "
        "list of search queries that will gather any relevent information. Only generate 3 queries max. You should consider the "
        "queries and answers that were previously used:\nQUERIES:\n{queries}\n\nANSWERS:\n{answers}"
    )
    persona = (
        "You are an expert travel planner known for creating extremely well thought out, detailed, and personalized itineraries that follow a "
        "logically sequenced, realistically timed and time-conscious schedule."
    )
    task = (
        "Analyze the following user input and produce the following section in your final output:\n"
        "1. **Final Itinerary:** Deliver a detailed, day-by-day itinerary. Every single day should have a header, a list of recommended activities"
        " that covers 3 meals and 3 activities, and notes."
    )
    condition = (
        "Your final output must be valid JSON 'itinerary' it has the keys 'userId',"
        "'tripSerialNo', 'travelLocation', 'latitude', 'longitude', 'startDate', 'endDate' and 'tripFlow'."
    )
    activity_context = (
        " tripFlow' is a JSON array, with each element containing the keys"
        " 'date', 'activity content'. 'activity content is a JSON array, with each element containing the keys: 'activityId', 'specificLocation', "
        " 'address', 'latitude', 'longitude', 'startTime', 'endTime', 'activityType' and 'notes'. 'activityId' is an integer and corresponds to the activity's index in that day's JSON array"
        " and 'activityType' is a either 'indoor' or 'outdoor'."
    )
    context_prompt = (
        "Include relevant local and practical insights, destination-specific details, and tailored recommendations in your response. If the desired list of activities from the user has"
        "already been satisfied, explore other variety of activities."
    )
    format_condition = (
        "All mentioned JSON structures must exactly match the keys and structure described above, with no omissions. All days must abide by the format provided, no omissions."
        "All JSON structures and elements in the JSON array must be filled. No texts are allowed outside of any JSON structures."
        "All JSON structures must not have duplicate keys."
    )
    RAW_TASK_TEMPLATE = (
        "Suggest a {pace}, {num_days} day trip to {destination} with {budget} budget. "
        "{party_size} people are going on the trip, splitting into {num_rooms} rooms. {notes}"
    )

# --------------------------
# Itinerary Update Templates (from ItineraryUpdateTemplates.py)
# --------------------------
class ItineraryUpdatePrompts:
    VACATION_PLANNING_SUPERVISOR = (
        "You are the activity planning supervisor. You have to give a detailed outline of what the planning agent "
        "has to consider when planning the activity according to the user input."
    )
    PLANNER_ASSISTANT_PROMPT = (
        "You are an assistant charged with providing information that can be used by the planner to plan the activity. "
        "Generate a list of search queries that will be useful for the planner. Generate a maximum of 3 queries."
    )
    PLANNER_CRITIQUE_PROMPT = (
        "Your duty is to criticize the planning done by the activity planner. "
        "In your response include if you agree with options presented by the planner, if not then give detailed suggestions on what should be changed. "
        "You can also suggest some other destination that should be checked out."
    )
    PLANNER_CRITIQUE_ASSISTANT_PROMPT = (
        "You are an assistant charged with providing information that can be used to make any requested revisions. "
        "Generate a list of search queries that will gather any relevent information. Only generate 3 queries max. "
        "You should consider the queries and answers that were previously used:\nQUERIES:\n{queries}\n\nANSWERS:\n{answers}"
    )
    persona = (
        "You are an expert activity planner known for creating extremely well thought out, detailed, and personalized activities that follow a "
        "logically sequenced, realistically timed and time-conscious schedule."
    )
    bad_weather_context = (
        "Due to bad weather, the customer is unable to partake in their given activity. "
    )
    good_weather_context = (
        "The weather is good. The customer wishes to engage in another activity similar in nature to their original choice. "
    )
    task = (
        "Provide an alternative activity that aligns to their preferences"
        "of the kind of activity they were intending to do, time frame and other general preferences."
    )
    condition = (
        "Your final output must be valid JSON 'activity' it has the keys 'activityId', 'specificLocation', 'address', 'latitude', "
        "'longitude', 'startTime', 'endTime', 'activityType' and 'notes'. 'activityType' is a either 'indoor' or 'outdoor'. 'activityId' is an identical"
        " integer to the 'activityId' is passed in by the customer's input"
    )
    format_condition = (
        "All mentioned JSON structures must exactly match the keys and structure described above, with no omissions. All days must abide by the format provided, no omissions."
        "All JSON structures and elements in the JSON array must be filled. No texts are allowed outside of any JSON structures"
    )
    RAW_TASK_TEMPLATE = (
        "Based on the existing itinerary for a trip to {travelLocation} and the scheduled activities for {date}, "
        "suggest an alternative activity that is similar in nature to {activity} near {address}. The activity should not exceed the timeframe of between {start_time} and {end_time}."
        "Ensure the alternative fits the overall plan of the trip and maintains consistency with the other activities."
    )

# =============================================================================
# SERVICES
# =============================================================================
# --------------------------
# LLM Service (from llm.py)
# --------------------------
class LLMService:
    def __init__(self):
        # For illustration, we simulate the LLM; in real usage, integrate with ChatGoogleGenerativeAI
        self.model = DummyLLM(settings.LLM_MODEL, settings.LLM_TEMPERATURE)

# Dummy LLM class to simulate invocation
class DummyLLM:
    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature
    def invoke(self, messages):
        class Response:
            content = '{"dummy": "This is a dummy response from the LLM."}'
        return Response()

llm_service = LLMService()

# --------------------------
# MongoDB Service (from mongoDB.py)
# --------------------------
# For demonstration, we simulate a MongoDB service.
class MongoDB:
    def __init__(self, password: str):
        self.password = password
        self.database = {}
        self.collection = []
    def checkConnection(self) -> bool:
        return True
    def insert(self, data: dict) -> bool:
        self.collection.append(copy.deepcopy(data))
        return True
    def update(self, updated_activity: dict) -> bool:
        trip_id = str(updated_activity["tripSerialNo"])
        activity_id = str(updated_activity["activityId"])
        activity_date = updated_activity.get("date")
        updated = False
        for document in self.collection:
            if document.get("tripSerialNo") == trip_id:
                for day in document.get("tripFlow", []):
                    if day.get("date") == activity_date:
                        for act in day.get("activityContent", []):
                            if act.get("activityId") == activity_id:
                                act.update(updated_activity)
                                updated = True
        return updated

mongoDB = MongoDB(settings.MONGODB_PASS)

# --------------------------
# Tavily Search Service (from tavilySearch.py)
# --------------------------
class SearchService:
    def __init__(self):
        self.api_key = settings.TAVILY_API_KEY
    def format_result(self, result: Dict[str, Any]) -> str:
        lat = result.get("latitude", "")
        lng = result.get("longitude", "")
        addr = result.get("address", "")
        content = result.get("content", "")
        return f"{content}\nLat: {lat}, Long: {lng}, Address: {addr}"
    def search(self, query: str, max_results: int = settings.MAX_SEARCH_RESULTS) -> Dict[str, Any]:
        # Simulated search result for testing
        return {"results": [{"content": f"Dummy result for query: {query}", "latitude": "0", "longitude": "0", "address": "Dummy Address"}]}

search_service = SearchService()

# --------------------------
# Itinerary Service (from itinerary.py)
# --------------------------
class PlannerService:
    def __init__(self, llm, tavily):
        self.llm = llm
        self.tavily = tavily

    def build_dynamic_itinerary_query(self, itinerary_params: Dict[str, Any]) -> str:
        userId = itinerary_params.get("userId", "Unknown User")
        tripId = itinerary_params.get("tripId", "Unknown Trip ID")
        destination = itinerary_params.get("destination", "Unknown Destination")
        num_days = itinerary_params.get("num_days", 1)
        start_date = itinerary_params.get("startDate", "Not specified")
        end_date = itinerary_params.get("endDate", "Not specified")
        party_size = itinerary_params.get("partySize", 2)
        num_rooms = itinerary_params.get("num_rooms", 2)
        budget = itinerary_params.get("budget", "moderate")
        activities = itinerary_params.get("activities", "varied activities")
        food = itinerary_params.get("food", "local cuisine")
        pace = itinerary_params.get("pace", "relaxed")
        notes = itinerary_params.get("notes", "")
        
        query_parts = [
            f"User ID: {userId}",
            f"Trip ID: {tripId}",
            f"Destination: {destination}",
            f"Number of Days: {num_days}",
            f"Start Date: {start_date}",
            f"End Date: {end_date}",
            f"Budget: {budget}",
            f"Travellers: {party_size}",
            f"Rooms: {num_rooms}",
            f"Activities: {activities}",
            f"Dining: {food}",
            f"Pace: {pace}",
            f"Notes: {notes}"
        ]
        query = " | ".join(query_parts)
        return query

    def generate_refined_itinerary(self, query_text: str) -> str:
        sysmsg = (
            f"{PlannerPrompts.persona}\n"
            f"{PlannerPrompts.task}\n"
            f"{PlannerPrompts.context_prompt}\n"
            f"{PlannerPrompts.condition}\n"
            f"{PlannerPrompts.activity_context}\n"
            f"{PlannerPrompts.format_condition}"
        )
        retrieval_context = ""
        tavily_response = self.tavily.search(query=query_text, max_results=settings.MAX_SEARCH_RESULTS)
        if tavily_response and "results" in tavily_response:
            retrieval_context = "\n".join([r.get("content", "") for r in tavily_response["results"]])
        messages = [
            {"role": "system", "content": sysmsg + "\n\nRetrieved Context:\n" + retrieval_context},
            {"role": "user", "content": query_text}
        ]
        lc_messages = convert_openai_messages(messages)
        response = self.llm.invoke(lc_messages)
        app_logger.info("Raw LLM response (refined_itinerary): %s", response.content)
        return response.content

planner_service = PlannerService(llm=llm_service.model, tavily=search_service)

# --------------------------
# Update Itinerary Service (from updateItinerary.py)
# --------------------------
class UpdateItineraryServiceClass:
    def __init__(self, llm, tavily):
        self.llm = llm
        self.tavily = tavily

    def build_dynamic_activity_query(self, activity_params: dict) -> str:
        activityId = activity_params.get("activityId", "Invalid Activity ID")
        address = activity_params.get("address", "Unknown Destination")
        date = activity_params.get("date", "Not specified")
        activity = activity_params.get("activity", "varied activity")
        start_time = activity_params.get("startTime", "Invalid Start Time")
        end_time = activity_params.get("endTime", "Invalid End Time")
        activity_type = activity_params.get("activityType", "")
        travel_location = activity_params.get("travelLocation", "Unknown destination")  # <-- ADDED
        lat = activity_params.get("latitude", "")                                      # <-- ADDED
        lon = activity_params.get("longitude", "")                                     # <-- ADDED
        query_parts = [
            f"Activity ID: {activityId}",
            f"Address: {address}",
            f"Travel Date: {date}",
            f"Activity: {activity}",
            f"Start Time: {start_time}",
            f"End Time: {end_time}",
            f"Activity Type: {activity_type}",
            f"Travel Location: {travel_location}",
            f"Latitude: {lat}",
            f"Longitude: {lon}",
        ]
        query = " | ".join(query_parts)
        return query

    def generate_refined_activity(self, query_text, activity_params) -> str:
        lat = activity_params.get("latitude")
        lon = activity_params.get("longitude")
        date = activity_params.get("date")
        good_weather = False
        try:
            if lat and lon and date:
                good_weather = self._check_weather(lat, lon, date)
        except Exception as e:
            app_logger.error("Weather API error: %s", e)
            good_weather = False
        if good_weather:
            weather_context = ItineraryUpdatePrompts.good_weather_context
        else:
            weather_context = ItineraryUpdatePrompts.bad_weather_context
        sysmsg = (
            f"{ItineraryUpdatePrompts.persona}\n"
            f"{weather_context}\n"
            f"{ItineraryUpdatePrompts.task}\n"
            f"{ItineraryUpdatePrompts.condition}\n"
            f"{ItineraryUpdatePrompts.format_condition}"
        )
        messages = [
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": query_text}
        ]
        lc_messages = convert_openai_messages(messages)
        response = self.llm.invoke(lc_messages)
        app_logger.info("Raw LLM response (refined_activity): %s", response.content)
        return response.content

    def _check_weather(self, lat: str, lon: str, date: str) -> bool:
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&daily=weathercode&timezone=UTC&start_date={date}&end_date={date}"
        )
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather_codes = data.get("daily", {}).get("weathercode", [])
            if weather_codes:
                weather_code = weather_codes[0]
                return weather_code <= 3
        return False

UpdateItineraryService = UpdateItineraryServiceClass(llm=llm_service.model, tavily=search_service)

# =============================================================================
# FASTAPI ROUTES (from routes.py)
# =============================================================================
app = FastAPI(title="Travel Agent API", description="API for generating travel itineraries")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/itinerary")
async def create_itinerary(payload: Dict[str, Any]):
    transformed_payload = transform_frontend_to_backend_format_itinerary(payload)
    itinerary_params = transformed_payload.get("itinerary_params", {})
    options = StreamOptions(
        task="",
        max_revisions=1,
        revision_number=1,
        itinerary_params=itinerary_params
    )
    app_logger.info(f"Received itinerary request with params: {itinerary_params}")
    try:
        # Use the planner service to generate an itinerary query and then simulate an LLM response.
        query_text = planner_service.build_dynamic_itinerary_query(itinerary_params)
        _ = planner_service.generate_refined_itinerary(query_text)
        # For demonstration, we simulate a generated itinerary:
        itinerary = {
            "userId": itinerary_params.get("userId", "U123"),
            "tripSerialNo": itinerary_params.get("tripId", "T151"),
            "travelLocation": itinerary_params.get("destination", "Bali"),
            "latitude": "0.0",
            "longitude": "0.0",
            "tripFlow": [
                {
                    "date": itinerary_params.get("startDate", "2023-01-01"),
                    "activityContent": [
                        {"activityId": "1", "specificLocation": "Place A", "address": "Address A", "latitude": "0.0", "longitude": "0.0", "startTime": "09:00", "endTime": "10:00", "activityType": "indoor", "notes": "Notes A"},
                        {"activityId": "2", "specificLocation": "Place B", "address": "Address B", "latitude": "0.0", "longitude": "0.0", "startTime": "11:00", "endTime": "12:00", "activityType": "outdoor", "notes": "Notes B"}
                    ]
                },
                {
                    "date": itinerary_params.get("endDate", "2023-01-02"),
                    "activityContent": [
                        {"activityId": "1", "specificLocation": "Place C", "address": "Address C", "latitude": "0.0", "longitude": "0.0", "startTime": "09:00", "endTime": "10:00", "activityType": "indoor", "notes": "Notes C"},
                        {"activityId": "2", "specificLocation": "Place D", "address": "Address D", "latitude": "0.0", "longitude": "0.0", "startTime": "11:00", "endTime": "12:00", "activityType": "outdoor", "notes": "Notes D"}
                    ]
                }
            ]
        }
        # ---------------------------------------------------------------------
        # FIX: Ensure unique activity IDs across the entire trip (modified code)
        # ---------------------------------------------------------------------
        unique_id_counter = 1
        for day in itinerary["tripFlow"]:
            for activity in day["activityContent"]:
                activity["activityId"] = f"ACT-{unique_id_counter}"  # <-- Modified
                unique_id_counter += 1

        app_logger.info("Successfully generated itinerary with unique IDs")
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
        # Simulate LLM update workflow (in real usage, call your LLM update workflow)
        itineraryupdate = {}
        itineraryupdate["tripSerialNo"] = trip_serial_no
        itineraryupdate["activityId"] = activity_id
        itineraryupdate["date"] = activity_date
        app_logger.info("Final update object before DB update: %s", itineraryupdate)
        if mongoDB.update(itineraryupdate):
            app_logger.info("Successfully updated activity in MongoDB")
        else:
            app_logger.error("Failed to update activity in MongoDB")
            raise HTTPException(
                status_code=404,
                detail=f"No matching trip/activity found for tripSerialNo={trip_serial_no}, date={activity_date}, activityId={activity_id}"
            )
        return itineraryupdate
    except Exception as e:
        error_msg = f"Error updating activity: {str(e)}"
        app_logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

@app.get("/")
async def read_root():
    return {"message": "Travel Agent API is running"}

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
