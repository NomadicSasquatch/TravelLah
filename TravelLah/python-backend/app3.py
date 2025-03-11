import os
import re
import json
import logging
from typing import TypedDict, List, Dict, Any
from langchain_community.adapters.openai import convert_openai_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from tavily import TavilyClient
# from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

load_dotenv(dotenv_path='.env')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    queries: List[str]
    answers: List[str]
    revision_number: int
    max_revisions: int
    itinerary_params: dict

class Queries(BaseModel):
    queries: List[str]

model = ChatGoogleGenerativeAI(
    temperature=0,
    model="gemini-1.5-flash",  # gemini-1.5-flash or gemini-1.5-pro
    convert_system_message_to_human=True,
    google_api_key=GOOGLE_API_KEY
)

tavily = TavilyClient(api_key=TAVILY_API_KEY)

VACATION_PLANNING_SUPERVISOR_PROMPT = (
    "You are the vacation planning supervisor. You have to give a detailed outline of what the planning agent "
    "has to consider when planning the vacation according to the user input."
)

PLANNER_ASSISTANT_PROMPT = (
    "You are an assistant charged with providing information that can be used by the planner to plan the vacation. "
    "Generate a list of search queries that will be useful for the planner. Generate a maximum of 3 queries."
)

PLANNER_CRITIQUE_PROMPT = (
    "Your duty is to criticize the planning done by the vacation planner. In your response include if you agree with "
    "options presented by the planner, if not then give detailed suggestions on what should be changed. You can also "
    "suggest some other destination that should be checked out."
)

PLANNER_CRITIQUE_ASSISTANT_PROMPT = (
    "You are a assistant charged with providing information that can be used to make any requested revisions. Generate a "
    "list of search queries that will gather any relevent information. Only generate 3 queries max. You should consider the "
    "queries and answers that were previously used:\nQUERIES:\n{queries}\n\nANSWERS:\n{answers}"
)

class TravelAgentPlanner:
    def __init__(self, llm, tavily):
        self.llm = llm
        self.tavily = tavily

    def build_dynamic_itinerary_query(self, itinerary_params: dict) -> str:
        userId = itinerary_params.get("userId", "Unknown User")
        tripId = itinerary_params.get("tripId", "Unknown Trip ID")
        destination = itinerary_params.get("destination", "Unknown Destination")
        num_days = itinerary_params.get("num_days", 1)
        # New keys for start and end date
        start_date = itinerary_params.get("start_date", "Not specified")
        end_date = itinerary_params.get("end_date", "Not specified")
        party_size = itinerary_params.get("party_size", 2)
        num_rooms = itinerary_params.get("num_rooms", 2)
        budget = itinerary_params.get("budget", "moderate")
        activities = itinerary_params.get("activities", "varied activities")
        food = itinerary_params.get("food", "local cuisine")
        pace = itinerary_params.get("pace", "relaxed")
        notes = itinerary_params.get("notes", "")
        
        query_parts = [
            f"User ID: {userId}",
            f"Trip ID: {tripId}",
            f"Destination: {destination}",
            f"Number of Days: {num_days}",
            f"Start Date: {start_date}",
            f"End Date: {end_date}",
            f"Budget: {budget}",
            f"Travellers: {party_size}",
            f"Rooms: {num_rooms}",
            f"Activities: {activities}",
            f"Dining: {food}",
            f"Pace: {pace}",
            f"Notes: {notes}"
        ]
        query = " | ".join(query_parts)
        return query
    
    def generate_refined_itinerary(self, query_text) -> str:
        persona = (
            "You are an expert travel planner known for creating extremely well thought out, thorough, and personalized itineraries that follow a "
            "logically sequenced, realistically timed and time-conscious schedule."
        )
        task = (
            "Analyze the following user input and produce a final itinerary. The itinerary must be detailed and organized day-by-day. "
            "Each day should include a header, and for that day provide a list of recommended activities that covers 3 meals and 3 activities. "
            "For each activity, include the start time, end time, and specify if it is an 'indoor' or 'outdoor' activity. Also include any relevant notes."
        )
        condition = (
            "Your final output must be valid JSON with exactly one key: 'itinerary'. The 'itinerary' object must have the keys 'userId', "
            "'tripSerialNo', 'TravelLocation', 'latitude', 'longitude', and 'tripFlow'. 'tripFlow' is a JSON array, with each element being an object containing the keys "
            "'date' and 'activity content'. 'activity content' is a JSON array where each element must include the keys: 'specific_location', "
            "'address', 'latitude', 'longitude', 'start_time', 'end_time', 'activity_type', and 'notes'. You must provide a concrete, non-placeholder value for every attribute based on available data or best estimates. Do not use placeholder text such as 'To be specified' or similar wording. 'activity_type' should only have the values 'indoor' or 'outdoor'."
        )
        context_prompt = (
            "Include relevant local insights, destination-specific details, and tailored recommendations in your response."
        )
        format_condition = (
            "All mentioned JSON structures must exactly match the keys and structure described above, with no omissions."
        )

        sysmsg = f"{persona}\n{task}\n{context_prompt}\n{condition}\n{format_condition}"
        
        retrieval_context = ""
        tavily_response = self.tavily.search(query=query_text, max_results=2)
        if tavily_response and "results" in tavily_response:
            retrieval_context = "\n".join([r.get("content", "") for r in tavily_response["results"]])
        
        messages = [
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": query_text}
        ]
        if retrieval_context:
            messages.append({"role": "system", "content": f"Retrieved Context:\n{retrieval_context}"})
        
        lc_messages = convert_openai_messages(messages)
        response = self.llm.invoke(lc_messages)
        logging.info("Raw LLM response (refined_itinerary): %s", response.content)
        return response.content

travel_agent_planner = TravelAgentPlanner(model, tavily)

def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=VACATION_PLANNING_SUPERVISOR_PROMPT), 
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    print("**********************************************************")
    print("Plan: ")
    print(response.content)
    print("**********************************************************")
    return {**state, "plan": response.content}

def research_plan_node(state: AgentState):
    pastQueries = state.get('queries', [])
    answers = state.get('answers', [])
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=PLANNER_ASSISTANT_PROMPT),
        HumanMessage(content=state['plan'])
    ])
    print("**********************************************************")
    print("Queries and Response: ")
    for q in queries.queries:
        print("Query: " + q)
        pastQueries.append(q)
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            lat = r.get("latitude", "")
            lng = r.get("longitude", "")
            addr = r.get("address", "")
            content = r.get("content", "")
            combined_info = f"{content}\nLat: {lat}, Long: {lng}, Address: {addr}"
            print("Tavily Response: " + combined_info)
            answers.append(combined_info)
    print("**********************************************************")
    return {**state, "queries": pastQueries, "answers": answers}

def generation_node(state: AgentState):
    itinerary_params = state.get("itinerary_params", {})
    dynamic_query = travel_agent_planner.build_dynamic_itinerary_query(itinerary_params)
    refined_itinerary = travel_agent_planner.generate_refined_itinerary(dynamic_query)
    print("**********************************************************")
    print("Dynamic Itinerary Query: ")
    print(dynamic_query)
    print("**********************************************************")
    print("Refined Itinerary: ")
    print(refined_itinerary)
    print("**********************************************************")
    return {**state, "draft": refined_itinerary, "revision_number": state.get("revision_number", 1) + 1}

def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=PLANNER_CRITIQUE_PROMPT), 
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    print("**********************************************************")
    print("Critique: ")
    print(response.content)
    print("**********************************************************")
    return {**state, "critique": response.content}

def research_critique_node(state: AgentState):
    pastQueries = state.get('queries', [])
    answers = state.get('answers', [])
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=PLANNER_CRITIQUE_ASSISTANT_PROMPT.format(queries=pastQueries, answers=answers)),
        HumanMessage(content=state['critique'])
    ])
    print("**********************************************************")
    print("Queries and Response:")
    for q in queries.queries:
        print("Query: " + q)
        pastQueries.append(q)
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            lat = r.get("latitude", "")
            lng = r.get("longitude", "")
            addr = r.get("address", "")
            content = r.get("content", "")
            combined_info = f"{content}\nLat: {lat}, Long: {lng}, Address: {addr}"
            print("Tavily Response: " + combined_info)
            answers.append(combined_info)
    print("**********************************************************")  
    return {**state, "queries": pastQueries, "answers": answers}

def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"

builder = StateGraph(AgentState)
builder.add_node("planner", plan_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_critique", research_critique_node)

builder.set_entry_point("planner")

builder.add_conditional_edges(
    "generate", 
    should_continue, 
    {END: END, "reflect": "reflect"}
)

builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

def run_itinerary_flow(stream_options):
    # with SqliteSaver.from_conn_string(":memory:") as memory:
    graph = builder.compile()
    thread = {"configurable": {"thread_id": "4"}}
    output_states = []
    final_draft_dict = None

    for state in graph.stream(stream_options, thread):
        if state.get('generate') and state.get('generate').get("draft"):
            draft_str = state.get('generate').get("draft")
            if draft_str:
                draft_str = draft_str.strip()
                draft_str = re.sub(r'^```json\s*', '', draft_str)
                draft_str = re.sub(r'```$', '', draft_str)
                draft_str = draft_str.strip()
                try:
                    draft_json = json.loads(draft_str)
                    final_draft_dict = draft_json
                    print("Parsed draft itinerary successfully!")
                except json.JSONDecodeError as e:
                    print("Error parsing the draft as JSON:", e)
        output_states.append(state)

    if not final_draft_dict:
        raise ValueError("Final draft itinerary not found in any state.")

    print("Formatted JSON:")
    print(json.dumps(final_draft_dict, indent=4))
    return final_draft_dict

from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
router = APIRouter() 
origins = [
    "http://localhost:3000",
    "localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StreamOptions(BaseModel):
    task: str
    max_revisions: int
    revision_number: int
    itinerary_params: Dict[str, Any]

@app.post("/itinerary")
async def create_itinerary(options: StreamOptions):
    stream_options = options.model_dump()
    try:
        itinerary = run_itinerary_flow(stream_options)
        return itinerary
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "FastAPI backend"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
