import torch
import pandas as pd
from simulation.engine import SimCityEngine

class ScenarioBuilder:
    def __init__(self):
        # We can mock the required models for pure SEIR-Z-D simulation tests
        self.engine = SimCityEngine(
            tgn=None,
            influence_scorer=None,
            hmf_bridge=None,
            virality_head=None,
            hawkes_model=None
        )

    def run_influencer_tweet_scenario(self):
        """
        Simulates what happens when an influencer triggers a massive spike
        in the Hawkes cross-platform intensity.
        """
        print("Running Influencer Tweet Scenario (What-If)...")
        
        N = 100000
        # Start with a susceptible population, a few exposed, and some dormant recovered nodes
        initial_state = [80000, 1000, 0, 19000, 0, 0]
        
        # Initialize engine
        self.engine.initialize_simulation(
            initial_state=initial_state,
            N=N,
            theta=2.0,      # High algorithmic boost for zombies
            sigma=0.5,      # Fast activation E -> I
            gamma_I=0.1,    # Fast decay I -> R
            delta_D=0.01    # Slow archival Z -> D
        )
        
        # Suppose the TGN + HMF gave us a macro beta of 0.8
        base_beta_macro = 0.8
        
        # Baseline hawkes intensity is 1.0
        baseline_lambda = 1.0
        
        # INJECTION: Influencer tweets -> massive cross-platform Hawkes spike!
        injected_lambda = 50.0 
        decay_gamma = 0.2 # Fast exponential decay of the trend
        
        # Run simulation for 30 days, dt=1.0 days
        history = self.engine.run_scenario(
            steps=30,
            dt=1.0,
            base_beta_macro=base_beta_macro,
            base_lambda=injected_lambda,
            baseline_lambda=baseline_lambda,
            decay_gamma=decay_gamma,
            phi=0.05 # Conversion from lambda surge to zeta resurgence rate
        )
        
        df = pd.DataFrame(history)
        print("\nScenario Results (First 10 Days):")
        print(df.head(10).to_string())
        
        print("\nPeak Zombie Population:", df['Z'].max())
        print("Final Dead/Archived:", df['D'].iloc[-1])
        return df

if __name__ == "__main__":
    builder = ScenarioBuilder()
    builder.run_influencer_tweet_scenario()
