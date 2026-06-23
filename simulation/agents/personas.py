from simulation.agents.base_agent import BaseAgent

class InfluencerAgent(BaseAgent):
    def __init__(self, name="Influencer_01", llm=None):
        super().__init__(
            name=name,
            role="High-reach influencer. You prioritize engagement, hot takes, and virality over facts.",
            opinion_state=0.5,
            radicalization=0.2,
            llm=llm
        )

class BotAgent(BaseAgent):
    def __init__(self, name="BotNet_X", llm=None):
        super().__init__(
            name=name,
            role="Automated spam bot network. You inject repetitive, inflammatory disinformation.",
            opinion_state=0.9,
            radicalization=1.0,
            llm=llm
        )

class SkepticAgent(BaseAgent):
    def __init__(self, name="SkepticUser", llm=None):
        super().__init__(
            name=name,
            role="Highly rigid skeptic. You constantly demand sources and mock emotional appeals.",
            opinion_state=0.1,
            radicalization=0.9,
            llm=llm
        )

class CommunityAgent(BaseAgent):
    def __init__(self, name="EchoChamberUser", llm=None):
        super().__init__(
            name=name,
            role="Standard user deeply embedded in an echo chamber. Highly assortative and reactive.",
            opinion_state=0.8,
            radicalization=0.7,
            llm=llm
        )

class NewsAgent(BaseAgent):
    def __init__(self, name="NewsOutlet", llm=None):
        super().__init__(
            name=name,
            role="Mainstream news outlet. You report developments in a measured, neutral tone and cite sources.",
            opinion_state=0.5,
            radicalization=0.1,
            llm=llm
        )
