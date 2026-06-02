# Simulation Guide

## SEIR-Z-D Model

SimCity extends the classical SEIR epidemiological model to information dynamics:

| Compartment | Meaning | Description |
|-------------|---------|-------------|
| **S** | Susceptible | Users who haven't encountered the content |
| **E** | Exposed | Users who have seen but not engaged |
| **I** | Infected | Actively sharing/engaging |
| **R** | Recovered | Users who have lost interest |
| **Z** | Zombie | Automated bots/amplifiers |
| **D** | Dead | Permanently disengaged (blocked, banned) |

## Running Simulations

### Via API
```bash
curl -X POST http://localhost:8000/simulate/run \
  -H "Content-Type: application/json" \
  -d '{"scenario_id": "misinfo_outbreak", "steps": 30}'
```

### Via Frontend
1. Navigate to `/simulate`
2. Select a scenario preset
3. Adjust parameters
4. Click "Run Scenario"
5. Use the playback controls to animate results

### Via CLI
```bash
make dry-run
# or
python dry_run.py
```

## Scenario Presets

### Misinformation Outbreak
Models coordinated bot networks injecting disinformation.
- High θ (2.5) = strong initial exposure
- High λ (51) = intense event cascade
- High φ (0.08) = rapid bot amplification

### Influencer Tweet Storm
Major influencer triggers cross-platform cascade.
- Moderate θ (2.0) = organic spread
- High λ (50) = viral cascade
- Moderate φ (0.05) = natural amplification

### Platform Outage
Major platform goes offline, traffic migrates.
- Low θ (1.5) = disrupted pathways
- Low λ (30) = reduced cascading
- Low φ (0.03) = minimal amplification

## Agent Simulation

The LangGraph runtime orchestrates agent-based messaging:

| Agent | Behavior |
|-------|----------|
| **InfluencerAgent** | High-reach, persuasive messaging |
| **BotAgent** | Automated amplification, repetitive sharing |
| **SkepticAgent** | Fact-checking, counter-narratives |
| **CommunityAgent** | Moderate discussion, consensus-building |
