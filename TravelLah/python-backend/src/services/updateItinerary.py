from typing import Dict, Any
from .llm import llm_service
from .tavilySearch import search_service
from ..settings.logging import app_logger as logger
from ..prompts.ItineraryUpdateTemplates import UpdatePrompts
from langchain_community.adapters.openai import convert_openai_messages


class UpdateItineraryService:

    def __init__(self, llm, tavily):
        self.llm = llm
        self.tavily = tavily
    """Service for generating travel itineraries"""
    
    def build_dynamic_activity_query(self, activity_params: dict) -> str:
        # userId = activity_params.get("userId", "Unknown User")
        # tripId = activity_params.get("tripId", "Unknown Trip ID")
        activityId = activity_params.get("activityId", "Invalid Activity ID")
        address = activity_params.get("address", "Unknown Destination")
        date = activity_params.get("date", "Not specified")
        #party_size = activity_params.get("party_size", 2) # not stored in json
        #budget = activity_params.get("budget", "moderate") # not stored in json
        activity = activity_params.get("activity", "varied activity") # activity notes
        start_time = activity_params.get("start_time", "Invalid Start Time")
        end_time = activity_params.get("end_time", "Invalid End Time")
        activity_type = activity_params.get("activity_type", "")
        #pace = activity_params.get("pace", "relaxed") # not stored in json
        #notes = activity_params.get("notes", "") # not stored in json, refers to personal notes and not activity notes
        
        query_parts = [
            # f"User ID: {userId}",
            # f"Trip ID: {tripId}",
            f"Activity ID: {activityId}",
            f"Address: {address}", # confirm with other parts of the code if we are using address or specific location or long lat
            f"Travel Date: {date}",
            #f"Travellers: {party_size}",
            #f"Budget: {budget}",
            f"Activity: {activity}",
            f"Start Time: {start_time}",
            f"End Time: {end_time}",
            f"Activity Type: {activity_type}",
            #f"Pace: {pace}",
            #f"Notes: {notes}"
        ]
        query = " | ".join(query_parts)

        return query
    
    def generate_refined_activity(self, query_text) -> str:
        sysmsg = (
            f"{UpdatePrompts.persona}\n"
            f"{UpdatePrompts.task}\n"
            f"{UpdatePrompts.condition}\n"
            f"{UpdatePrompts.format_condition}"
        )
        
        retrieval_context = ""
        tavily_response = self.tavily.search(query=query_text, max_results=2)
        destination_info = []
        if tavily_response and "results" in tavily_response:
            for r in tavily_response["results"]:
                lat = r.get("latitude", "")
                lng = r.get("longitude", "")
                addr = r.get("address", "")
                destination_info.append(f"Lat: {lat}, Long: {lng}, Address: {addr}")
            retrieval_context = "\n".join([r.get("content", "") for r in tavily_response["results"]])
        
        messages = [
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": query_text}
        ]
        if retrieval_context:
            messages.append({"role": "system", "content": f"Retrieved Context:\n{retrieval_context}"})
        
        lc_messages = convert_openai_messages(messages)
        response = self.llm.invoke(lc_messages)
        logger.info("Raw LLM response (refined_activity): %s", response.content)

        return response.content

# Initialize service
UpdateItineraryService = UpdateItineraryService(llm=llm_service.model, tavily=search_service.client)
