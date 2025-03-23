def transform_frontend_to_backend_format_itinerary(payload):
    from datetime import datetime
    
    check_in = datetime.fromisoformat(payload["checkIn"].replace("Z", "+00:00"))
    check_out = datetime.fromisoformat(payload["checkOut"].replace("Z", "+00:00"))
    num_days = (check_out - check_in).days
    
    formatted_dates = f"{check_in.strftime('%Y-%m-%d')} to {check_out.strftime('%Y-%m-%d')}"
    start_date = str(check_in.strftime('%Y-%m-%d'))
    end_date = str(check_out.strftime('%Y-%m-%d'))

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


def transform_frontend_to_backend_format_updateActivity(payload):
    from datetime import datetime

    ##########################################################################
    # ADDED travelLocation, latitude, longitude, and dayId to the activity_params  #
    ##########################################################################
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
        "longitude": payload.get("longitude", ""),            # <-- ADDED
        "dayId": payload.get("dayId", "")                     # <-- ADDED: unique identifier for the day
    }

    return {"activity_params": activity_params}

