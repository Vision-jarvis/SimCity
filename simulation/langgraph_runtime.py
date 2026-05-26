from typing import Annotated, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from simulation.agents.personas import InfluencerAgent, BotAgent, SkepticAgent, CommunityAgent

# Define the state schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_turn: int
    max_turns: int

class LangGraphRuntime:
    """
    Constructs and runs the multi-agent simulation graph using LangGraph.
    """
    def __init__(self, llm=None):
        self.llm = llm
        self.agents = {
            "influencer": InfluencerAgent(llm=self.llm),
            "bot": BotAgent(llm=self.llm),
            "skeptic": SkepticAgent(llm=self.llm),
            "community": CommunityAgent(llm=self.llm)
        }
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Define agent nodes
        def influencer_node(state):
            msg = self.agents["influencer"].invoke(state["messages"])
            return {"messages": [msg]}
            
        def bot_node(state):
            msg = self.agents["bot"].invoke(state["messages"])
            return {"messages": [msg]}
            
        def skeptic_node(state):
            msg = self.agents["skeptic"].invoke(state["messages"])
            return {"messages": [msg]}
            
        def community_node(state):
            msg = self.agents["community"].invoke(state["messages"])
            return {"messages": [msg]}
            
        # The Platform Node acts as a broadcast / turn aggregator
        def platform_node(state):
            turn = state.get("current_turn", 0) + 1
            return {"current_turn": turn}

        # Add nodes
        workflow.add_node("platform", platform_node)
        workflow.add_node("influencer", influencer_node)
        workflow.add_node("bot", bot_node)
        workflow.add_node("skeptic", skeptic_node)
        workflow.add_node("community", community_node)

        # Build Broadcast Topology: Platform broadcasts to all agents
        workflow.add_edge("platform", "influencer")
        workflow.add_edge("platform", "bot")
        workflow.add_edge("platform", "skeptic")
        workflow.add_edge("platform", "community")

        # Agents send their responses back to the platform
        # We need a conditional edge from platform to determine if we continue or END
        def should_continue(state):
            if state["current_turn"] >= state["max_turns"]:
                return END
            return "platform"

        # After all agents respond, route back to platform
        workflow.add_edge("influencer", "platform_eval")
        workflow.add_edge("bot", "platform_eval")
        workflow.add_edge("skeptic", "platform_eval")
        workflow.add_edge("community", "platform_eval")
        
        # Evaluator node to check stopping condition
        def platform_eval(state):
            return {}
            
        workflow.add_node("platform_eval", platform_eval)
        workflow.add_conditional_edges("platform_eval", should_continue, {END: END, "platform": "platform"})

        workflow.set_entry_point("platform")
        return workflow.compile()

    def run(self, initial_message: str, max_turns: int = 2):
        print(f"--- Starting LangGraph Simulation: {initial_message} ---")
        initial_state = {
            "messages": [HumanMessage(content=initial_message, name="NewsEvent")],
            "current_turn": 0,
            "max_turns": max_turns
        }
        
        result = self.graph.invoke(initial_state)
        return result["messages"]

if __name__ == "__main__":
    # Test stub run
    runtime = LangGraphRuntime()
    out_msgs = runtime.run("BREAKING: New legislation passed regarding internet censorship.", max_turns=1)
    for m in out_msgs:
        print(f"{m.name}: {m.content}")
