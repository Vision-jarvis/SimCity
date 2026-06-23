from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class BaseAgent:
    """
    Base class for a SimCity internet actor (agent).
    """
    def __init__(self, name: str, role: str, opinion_state: float, radicalization: float, llm=None):
        self.name = name
        self.role = role
        self.opinion_state = opinion_state
        self.radicalization = radicalization
        self.llm = llm # Langchain LLM instance, or None for mock
        self.memory = []

    def get_system_prompt(self) -> str:
        return f"""
        You are an internet user on a social platform.
        Your Persona: {self.role}
        Your current opinion bias is {self.opinion_state} (0=Extreme Left/A, 1=Extreme Right/B).
        Your radicalization level is {self.radicalization} (0=Open-minded, 1=Rigid/Echo-chamber).
        
        Read the incoming message and decide how to react. Keep your response under 2 sentences.
        If your radicalization is high, react aggressively to opposing opinions.
        """

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        # If we have a real LLM:
        if self.llm:
            sys_msg = HumanMessage(content=self.get_system_prompt())
            full_prompt = [sys_msg] + messages[-3:] # Only look at recent context
            response = self.llm.invoke(full_prompt)
            return response
            
        # Stub logic if no LLM provided
        if self.radicalization > 0.8:
            reply = f"[{self.name}] This is garbage. Fake news! Wake up!"
        elif self.opinion_state > 0.7:
            reply = f"[{self.name}] Totally agree! Finally someone says it."
        else:
            reply = f"[{self.name}] Interesting perspective, I'll have to read more about this."
            
        return AIMessage(content=reply, name=self.name)
