import pytest
from simulation.langgraph_runtime import LangGraphRuntime
from simulation.agents.personas import InfluencerAgent, BotAgent

def test_agent_personas():
    bot = BotAgent()
    assert bot.radicalization == 1.0
    assert "spam" in bot.role.lower()
    
    influencer = InfluencerAgent()
    assert influencer.opinion_state == 0.5
    
def test_agent_stub_response():
    bot = BotAgent()
    # Invoke stub logic without real LLM
    from langchain_core.messages import HumanMessage
    msg = bot.invoke([HumanMessage(content="Test message")])
    assert msg.name == "BotNet_X"
    assert "fake news" in msg.content.lower()

def test_langgraph_broadcast_topology():
    runtime = LangGraphRuntime()
    assert runtime.graph is not None
    
    # Run a simple 1-turn simulation
    out_msgs = runtime.run("Breaking News Test Event", max_turns=1)
    
    # Platform injects 1 message, then 4 agents respond = 5 messages total in state
    assert len(out_msgs) == 5
    
    # Verify all agents responded
    names = [m.name for m in out_msgs]
    assert "BotNet_X" in names
    assert "SkepticUser" in names
    assert "EchoChamberUser" in names
    assert "Influencer_01" in names
