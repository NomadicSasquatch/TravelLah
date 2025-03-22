from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from src.settings.config import settings
from src.settings.logging import app_logger as logger
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
        """
        Update the activity within a trip's itinerary.

        Parameters:
            updated_activity (dict): The new data for the activity. Must include
                "tripSerialNo" (the trip's identifier) and
                "activityId" (the ID of the activity to update),
                plus any other keys for the activity fields.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        try:
            # Extract the trip ID and activity ID from updated_activity
            trip_id = updated_activity["tripSerialNo"]
            activity_id = updated_activity["activityId"]

            # Use array filters to match the nested activity inside tripFlow.activityContent
            result = self.collection.update_one(
                {
                    "tripSerialNo": trip_id,
                    "tripFlow.activityContent.activityId": activity_id
                },
                {
                    "$set": {"tripFlow.$[outer].activityContent.$[inner]": updated_activity}
                },
                array_filters=[
                    {"outer.activityContent": {"$elemMatch": {"activityId": activity_id}}},
                    {"inner.activityId": activity_id}
                ]
            )

            if result.modified_count > 0:
                logger.info("Successfully updated activity %s in trip %s", activity_id, trip_id)
                return True
            else:
                logger.info("No document updated for trip %s and activity %s", trip_id, activity_id)
                return False

        except Exception as e:
            logger.error("Error updating document: %s", e)
            return False


    def close(self):
        self.client.close()
        logger.info("Closed connection to MongoDB")

mongoDB = MongoDB(settings.MONGODB_PASS)