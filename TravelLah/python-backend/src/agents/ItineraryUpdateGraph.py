import json
import re
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from src.prompts.ItineraryUpdateTemplates import ItineraryUpdatePrompts
import uuid

from src.agents.state import AgentState
from src.agents.ItineraryUpdateNodes import (
    plan_node,
    research_plan_node,
    generation_node,
    reflection_node,
    research_critique_node,
    should_continue
)
from src.settings.logging import app_logger as logger

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True  # This forces the basic configuration, overriding any previous settings
)


class ItineraryUpdateWorkflow:
    """
    Manages the travel itinerary generation workflow
    """
    
    def __init__(self):
        # Build the workflow graph
        self.builder = StateGraph(AgentState)
        
        # Add all nodes
        self.builder.add_node("planner", plan_node)
        self.builder.add_node("research_plan", research_plan_node)
        self.builder.add_node("generate", generation_node)
        self.builder.add_node("reflect", reflection_node)
        self.builder.add_node("research_critique", research_critique_node)
        
        # Set entry point
        self.builder.set_entry_point("planner")
        
        # Add conditional edges
        self.builder.add_conditional_edges(
            "generate", 
            should_continue, 
            {END: END, "reflect": "reflect"}
        )
        
        # Add direct edges
        self.builder.add_edge("planner", "research_plan")
        self.builder.add_edge("research_plan", "generate")
        self.builder.add_edge("reflect", "research_critique")
        self.builder.add_edge("research_critique", "generate")
        
        # Compile the graph
        self.graph = self.builder.compile()
    
    def run(self, options) -> Dict[str, Any]:
        """
        Run the itinerary generation workflow
        
        Args:
            options: Dictionary with workflow options
            
        Returns:
            The generated itinerary as a dictionary
        """

        logger.info(f"Starting itinerary workflow with options: {options}")
        
        thread = {"configurable": {"thread_id": "4"}}
        output_states = []
        final_draft_dict = None

        activity_params = options.get("activity_params", {})

        # 1) Format the task from a known template (e.g. from PlannerPrompts)
        formatted_task = ItineraryUpdatePrompts.RAW_TASK_TEMPLATE.format(**activity_params)

        # 2) Overwrite the user-supplied (or empty) task in 'options' 
        options["task"] = formatted_task
                    
        # Execute the workflow graph
        for state in self.graph.stream(options, thread):
            if state.get('generate') and state.get('generate').get("draft"):
                draft_str = state.get("generate").get("draft")
                if draft_str:
                    draft_str = draft_str.strip()
                    draft_str = re.sub(r"^```json\s*", "", draft_str)
                    draft_str = re.sub(r"```$", "", draft_str)
                    draft_str = draft_str.strip()
                try:
                    draft_json = json.loads(draft_str)
                    final_draft_dict = draft_json
                    logger.info("Parsed draft update itinerary successfully!")
                except json.JSONDecodeError as e:
                    logger.info("Error parsing the update itinerary  draft as JSON:", e)
            
            output_states.append(state)
        
        # Ensure we have a final result
        if not final_draft_dict:
            error_msg = "Final draft update itinerary not found in any state"
            logger.error(error_msg)
            raise ValueError(error_msg)
        for day in final_draft_dict.get("tripFlow", []):
            for activity in day.get("activityContent", []):
                # Generate a new unique ID for each activity.
                # Optionally, you could check if an ID exists and only assign if missing.
                activity["activityId"] = str(uuid.uuid4())
        
        logger.info("Itinerary update workflow completed successfully")
        logger.info("HERE IS THE RETURN", final_draft_dict)
        return final_draft_dict

# Initialize workflow
ItineraryUpdateWorkflow = ItineraryUpdateWorkflow()