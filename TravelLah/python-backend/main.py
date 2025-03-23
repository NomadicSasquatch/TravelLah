import uvicorn
from src.settings.config import settings
from src.settings.logging import app_logger, setup_logger
from src.api.routes import app
from src.services.mongoDB import mongoDB
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import logging
import sys


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True  # This forces the basic configuration, overriding any previous settings
)
logger = setup_logger("main")

if settings.validate():
    logger.info("Settings validated successfully")

if __name__ == "__main__":
    logger.info("Connecting to MongoDB")
    if  mongoDB.checkConnection():
        logger.info("Connected to MongoDB")
    else:
        logger.error("Failed to connect to MongoDB")
        exit(1)

    logger.info(f"Starting server at {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
    mongoDB.close()