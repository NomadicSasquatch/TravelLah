def transform_frontend_to_backend_format(payload):
    # Calculate number of days between check-in and check-out
    from datetime import datetime
    
    # Parse dates from ISO format
    check_in = datetime.fromisoformat(payload["checkIn"].replace("Z", "+00:00"))
    check_out = datetime.fromisoformat(payload["checkOut"].replace("Z", "+00:00"))
    
    # Calculate trip duration in days
    num_days = (check_out - check_in).days
    
    # Format dates as YYYY-MM-DD to YYYY-MM-DD
    formatted_dates = f"{check_in.strftime('%Y-%m-%d')} to {check_out.strftime('%Y-%m-%d')}"
    
    # Extract guest and room information
    guests_and_rooms = payload.get("guestsAndRooms", {})
    party_size = guests_and_rooms.get("adults", 0) + guests_and_rooms.get("children", 0)
    num_rooms = guests_and_rooms.get("rooms", 1)
    
    # Map budget value to budget category (you may need to adjust these thresholds)
    budget_value = int(payload.get("budget", 0))
    budget_category = "budget"
    if budget_value > 5000:
        budget_category = "luxury"
    elif budget_value > 2000:
        budget_category = "moderate"
    
    # Construct the transformed payload
    itinerary_params = {
        "userId": "U123",  # You might want to get this from authentication or user session
        "tripId": "T123",  # This could be generated or retrieved from elsewhere
        "destination": payload.get("destination", ""),
        "num_days": num_days,
        "dates": formatted_dates,
        "party_size": party_size,
        "num_rooms": num_rooms,
        "budget": budget_category,
        "activities": payload.get("activities", ""),
        "food": payload.get("food", "").lower(),
        "pace": payload.get("pace", "").lower(),
        "notes": payload.get("additionalNotes", "")
    }
    
    return {"itinerary_params": itinerary_params}