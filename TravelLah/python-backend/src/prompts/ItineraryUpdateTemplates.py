class ItineraryUpdatePrompts:
    """Collection of prompts used in the update itinerary plan"""

    VACATION_PLANNING_SUPERVISOR="""You are the activity planning supervisor. You have to give a detailed outline of what the planning agent \
    has to consider when planning the activity according to the user input."""

    PLANNER_ASSISTANT_PROMPT = """You are an assistant charged with providing information that can be used by the planner to plan the activity.
    Generate a list of search queries that will be useful for the planner. Generate a maximum of 3 queries."""

    PLANNER_CRITIQUE_PROMPT = """Your duty is to criticize the planning done by the activity planner.
    In your response include if you agree with options presented by the planner, if not then give detailed suggestions on what should be changed.
    You can also suggest some other destination that should be checked out.
    """

    PLANNER_CRITIQUE_ASSISTANT_PROMPT = """You are a assistant charged with providing information that can be used to make any requested revisions.
    Generate a list of search queries that will gather any relevent information. Only generate 3 queries max. 
    You should consider the queries and answers that were previously used:
    QUERIES:
    {queries}

    ANSWERS:
    {answers}
    """

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
        "Provide an alternative activity that aligns to their preferences" \
        "of the kind of activity they were intending to do, time frame and other general preferences." #maybe include budget and travel party size
    )
    condition = (
        "Your final output must be valid JSON 'activity' it has the keys 'activityId', 'specificLocation', 'address', 'latitude', "
        "'longitude', 'startTime', 'endTime', 'activityType' and 'notes'. 'activityType' is a either 'indoor' or 'outdoor'. 'activityId' is an identical" \
        " integer to the 'activityId' is passed in by the customer's input"
    )
    format_condition = (
        "All mentioned JSON structures must exactly match the keys and structure described above, with no omissions. All days must abide by the format provided, no omissions." \
        "All JSON structures and elements in the JSON array must be filled. No texts are allowed outside of any JSON structures"
    )

    RAW_TASK_TEMPLATE = (
        "Based on the existing itinerary for a trip to {travelLocation} and the scheduled activities for {date}, "
        "suggest an alternative activity that is similar in nature to {activity} near {address}. The activity should not exceed the timeframe of between {start_time} and {end_time}."
        "Ensure the alternative fits the overall plan of the trip and maintains consistency with the other activities."
    )


  