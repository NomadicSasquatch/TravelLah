from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from src.settings.config import settings
from src.settings.logging import app_logger
import copy

class MongoDB:
    def __init__(self, password: str):
        self.client = MongoClient(f"mongodb+srv://GuruAI_Admin:{password}@guruaiproject20k.the6c.mongodb.net/?retryWrites=true&w=majority&appName=GuruAIProject20k", server_api=ServerApi("1"))
        self.database = self.client["TravelLah"]
        self.collection = self.database["UserItinerary"]

    def checkConnection(self) -> bool:
        try:
            self.client.admin.command("ping")
            return True
        except Exception as e:
            return False

    def insert(self, data: dict) -> bool:
        data_copy = copy.deepcopy(data)
        try:
            self.collection.insert_one(data_copy)
            return True
        except Exception as e:
            return False
        
    def update(self, updated_activity: dict) -> bool:
        try:
            trip_id = str(updated_activity["tripSerialNo"])
            activity_id = str(updated_activity["activityId"])
            activity_date = updated_activity.get("date")
            day_id = updated_activity.get("dayId")  # <-- ADDED: unique day identifier
            if not activity_date or not day_id:
                raise ValueError("Updated activity must include 'date' and 'dayId' fields.")

            app_logger.info("Update query parameters: tripSerialNo=%s, dayId=%s, activityId=%s, date=%s", trip_id, day_id, activity_id, activity_date)

            result = self.collection.update_one(
                {"tripSerialNo": trip_id},  # removed direct filtering on date
                {
                    "$set": {"tripFlow.$[day].activityContent.$[act]": updated_activity}
                },
                array_filters=[
                    {"day.dayId": day_id},  # <-- CHANGED: now filtering by unique day identifier
                    {"act.activityId": activity_id}
                ]
            )
            app_logger.info("Modified count: %s", result.modified_count)
            if result.modified_count > 0:
                app_logger.info("Successfully updated activity %s in trip %s on day %s", activity_id, trip_id, day_id)
                return True
            else:
                app_logger.info("No document updated for trip %s, day %s, and activity %s", trip_id, day_id, activity_id)
                return False
        except Exception as e:
            app_logger.error("Error updating document: %s", e)
            return False

            
mongoDB = MongoDB(settings.MONGODB_PASS)