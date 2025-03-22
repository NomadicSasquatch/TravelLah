from fastapi.testclient import TestClient
from src.api.routes import app

client = TestClient(app)

def test_create_itinerary():
    dummy_data = {
        "itinerary_params": {
            "userId": "U123",
            "tripId": "T123",
            "destination": "Vietnam",
            "num_days": 7,
            "dates": "2025-04-01 to 2025-04-07",
            "party_size": 4,
            "num_rooms": 2,
            "budget": "moderate",
            "activities": "cultural tours, historical sites, local markets, local cuisines",
            "food": "local-cuisine",
            "pace": "relaxed",
            "notes": "Include both indoor and outdoor activities; mention local festivals if applicable."
        },
        "task": "",
        "max_revisions": 1,
        "revision_number": 1
    }

    response = client.post("/itinerary", json=dummy_data)
    assert response.status_code == 200, response.text

    itinerary = response.json()

    assert "tripFlow" in itinerary
