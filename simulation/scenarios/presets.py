import json
from simulation.langgraph_runtime import LangGraphRuntime
from simulation.engine import SimCityEngine

class ScenarioPresets:
    def __init__(self, llm=None):
        self.langgraph = LangGraphRuntime(llm=llm)
        
    def run_misinfo_outbreak(self):
        """
        Preset: A massive coordinated bot network injects disinformation.
        """
        print("--- Running Scenario: Misinfo Outbreak ---")
        
        # 1. LangGraph Phase: Inject the seed event
        seed_event = "ALERT: Secret documents reveal the election was rigged by the elites."
        
        # Simulate 2 turns of agent interaction
        agent_messages = self.langgraph.run(seed_event, max_turns=2)
        
        # Count the number of inflammatory bot/skeptic responses to compute the Hawkes shock
        # A real implementation would use NLP sentiment/stance detection here
        inflammatory_count = sum(
            1 for m in agent_messages 
            if m.name in ["BotNet_X", "SkepticUser"] and "fake" in m.content.lower()
        )
        
        # 2. Simulator Phase: Map LangGraph output to SEIR-Z-D parameters
        # High inflammatory activity creates a massive cross-platform Hawkes spike
        base_lambda = 1.0 + (inflammatory_count * 25.0) 
        
        engine = SimCityEngine(None, None, None, None, None)
        engine.initialize_simulation(
            initial_state=[90000, 1000, 0, 9000, 0, 0], # Mostly susceptible
            N=100000,
            theta=2.5,   # High algorithmic boost (bots manipulating the algo)
            sigma=0.7,   # Extremely fast activation (outrage spreads fast)
            gamma_I=0.05, # Slow decay (people stay angry longer)
            delta_D=0.005 # Almost never archived
        )
        
        print(f"\n--- SEIR-Z-D Dynamics ---")
        print(f"Computed Hawkes injected_lambda from Agent interactions: {base_lambda}")
        
        history = engine.run_scenario(
            steps=30,
            dt=1.0,
            base_beta_macro=0.9,
            base_lambda=base_lambda,
            baseline_lambda=1.0,
            decay_gamma=0.1, # Slow decay of the trend
            phi=0.08
        )
        
        zombie_peak = max([step['Z'] for step in history])
        print(f"Peak Algorithmic Zombies resurrected: {zombie_peak}")
        return history

if __name__ == "__main__":
    presets = ScenarioPresets()
    presets.run_misinfo_outbreak()
