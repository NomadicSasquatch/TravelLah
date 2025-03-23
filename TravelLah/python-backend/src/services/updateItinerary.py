from typing import Dict, Any
from .llm import llm_service
from .tavilySearch import search_service
from ..settings.logging import app_logger as logger
from ..prompts.ItineraryUpdateTemplates import ItineraryUpdatePrompts as UpdatePrompts
from langchain_community.adapters.openai import convert_openai_messages
import requests


class UpdateItineraryService:

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

            # ADDED dayId and location context
            day_id = activity_params.get("dayId", "Unknown Day")          # <-- ADDED
            travel_location = activity_params.get("travelLocation", "Unknown destination")  # <-- ADDED
            lat = activity_params.get("latitude", "")                      # <-- ADDED
            lon = activity_params.get("longitude", "")                     # <-- ADDED

            query_parts = [
                f"Activity ID: {activityId}",
                f"Address: {address}",
                f"Travel Date: {date}",
                f"Day ID: {day_id}",                                      # <-- ADDED
                f"Activity: {activity}",
                f"Start Time: {start_time}",
                f"End Time: {end_time}",
                f"Activity Type: {activity_type}",
                f"Travel Location: {travel_location}",                    # <-- ADDED
                f"Latitude: {lat}",                                       # <-- ADDED
                f"Longitude: {lon}"                                       # <-- ADDED
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
            logger.error("Weather API error: %s", e)
            good_weather = False

        if good_weather:
            weather_context = UpdatePrompts.good_weather_context
        else:
            weather_context = UpdatePrompts.bad_weather_context

        sysmsg = (
            f"{UpdatePrompts.persona}\n"
            f"{weather_context}\n"
            f"{UpdatePrompts.task}\n"
            f"{UpdatePrompts.condition}\n"
            f"{UpdatePrompts.format_condition}"
        )

        messages = [
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": query_text}
        ]

        lc_messages = convert_openai_messages(messages)
        response = self.llm.invoke(lc_messages)
        logger.info("Raw LLM response (refined_activity): %s", response.content)
        return response.content


# Initialize service
UpdateItineraryService = UpdateItineraryService(llm=llm_service.model, tavily=search_service.client)
