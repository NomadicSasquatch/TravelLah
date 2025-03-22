def transform_frontend_to_backend_format_itinerary(payload):
    # Calculate number of days between check-in and check-out
    from datetime import datetime
    
    # Parse dates from ISO format
    check_in = datetime.fromisoformat(payload["checkIn"].replace("Z", "+00:00"))
    check_out = datetime.fromisoformat(payload["checkOut"].replace("Z", "+00:00"))
    
    # Calculate trip duration in days
    num_days = (check_out - check_in).days
    
    # Format dates as YYYY-MM-DD to YYYY-MM-DD
    formatted_dates = f"{check_in.strftime('%Y-%m-%d')} to {check_out.strftime('%Y-%m-%d')}"
    print(formatted_dates)

    start_date=str(check_in.strftime('%Y-%m-%d'))
    end_date = str(check_out.strftime('%Y-%m-%d'))

    
    # Extract guest and room information
    guests_and_rooms = payload.get("guestsAndRooms", {})
    party_size = guests_and_rooms.get("adults", 0) + guests_and_rooms.get("children", 0)
    num_rooms = guests_and_rooms.get("rooms", 1)
    
    # Construct the transformed payload
    itinerary_params = {
        "userId": "U123",  # You might want to get this from authentication or user session
        "tripId": "T151",  # This could be generated or retrieved from elsewhere
        "destination": payload.get("destination", ""),
        "num_days": num_days,
        "dates": formatted_dates,
        "start_date": start_date,
        "end_date":end_date,
        "party_size": party_size,
        "num_rooms": num_rooms,
        "budget": payload.get("budget", ""),
        "activities": payload.get("activities", ""),
        "food": payload.get("food", "").lower(),
        "pace": payload.get("pace", "").lower(),
        "notes": payload.get("additionalNotes", "")
    }
    
    return {"itinerary_params": itinerary_params}


def transform_frontend_to_backend_format_updateActivity(payload):
    # Calculate number of days between check-in and check-out
    from datetime import datetime
    
    # Construct the transformed payload
    activity_params = {
        "activityId": payload.get("activityId", ""),
        "address": payload.get("specificLocation", ""),
        "date": payload.get("date", ""),
        "activity": payload.get("notes", ""),
        "start_time": payload.get("startTime", ""),
        "end_time": payload.get("endTime", ""),
        "activity_type": payload.get("activityType", ""),
    }
    
    return {"activity_params": activity_params}