# Jurisdictionally Constrained Sharding with MAPPO

Code for an honours thesis: *A Multi-Agent Proximal Policy Optimisation Framework for Jurisdictionally Constrained Sharding in Decentralised Evidence Chains*.

Each shard cluster is an agent that decides where incoming writes should go. Legal rules (GDPR, APP 8) live inside the reward, so a write placed in a non-compliant jurisdiction costs far more than any throughput it could earn. The agents are trained with MAPPO in a discrete-event simulator that replays real Ethereum Sepolia transactions.

This README covers running the code. The method and results are in the thesis, and [`TECHNICAL_NOTES.md`](TECHNICAL_NOTES.md) explains the harder implementation decisions and what went wrong along the way.

## Layout

```
sim/       simulator, environment, networks, MAPPO trainer
scripts/   the things you actually run (table below)
```

The main pieces in `sim/`:

| file | what it does |
|---|---|
| `shardnode.py` | one shard: queue, CPU and disk, updated each tick (Section 3.3.3) |
| `simulator.py` | the tick loop, cross-shard reads, and the reward (Section 3.2.4) |
| `env.py` | Gym-style environment, one agent per node |
| `observations.py` | how node and transaction state become network inputs |
| `networks.py` | actor and centralised critic |
| `mappo.py` | the trainer |
| `data_sources.py` | loads saved Sepolia blocks and tags them with jurisdictions |

## Setup

Needs Python 3.10 or newer.

```
pip install -r requirements.txt
```

## Acquiring Data

Real transactions come from Sepolia via a free [Alchemy](https://www.alchemy.com) key. Set the ALCHEMY_URL environment variable to your endpoint, then:

```
python scripts/fetch_sepolia_trace.py
```

That saves `sepolia_trace.json` in the repo root. You can skip this and use `train_demo.py`, which runs on synthetic arrivals.

## Run Scripts  

Run these from the repo root. Each script finds the root itself, and anything it writes lands there.

| script | what it does |
|---|---|
| `train_demo.py` | quick check that training runs, on synthetic data (not a result) |
| `train_sepolia.py` | main training run on the real trace; writes `training_run/` (log, checkpoints, plot) |
| `finetune_routing.py <checkpoint>` | resumes from a checkpoint with the rejection-penalty, queue-preference and entropy-floor changes switched on; writes `training_run_finetune/` |
| `evaluate_baselines.py <checkpoint>` | compares the policy against random, round-robin and least-loaded routing on the same trace |
| `inspect_node_distribution.py <checkpoint>` | checks whether the policy spreads writes across a jurisdiction's nodes or leans on a favourite |
| `diagnose_load_awareness.py <checkpoint>` | checks how often it picks the least-loaded compliant node, against chance |
| `run_ablation.py <checkpoint>` | resumes from the original converged checkpoint under five configs (a control, each new training change on its own, and all three) across several seeds, scores them on the same held-out traces, and reports confidence intervals and Welch tests against the control; writes `ablation_run/` |
| `optuna_tune.py` | sweeps the reward weights w1/w2 |
| `main.py` | small simulator demo with no RL; the only place cross-shard reads get exercised |

Checkpoints, `training_run*/` and `ablation_run/` are git-ignored;

## Known limitations

- **Actions:** only Assign Write Target is implemented. Trigger Data Migration and Throttle Rate are in the thesis's formal action space but not built.
- **Latency is never exercised in training.** The Sepolia pipeline produces writes only, so the latency term of the reward is 0 in every training and evaluation run. The latency code (Eq. 5) works, but only `main.py` runs it.
- **Agents can't see other nodes' load.** Each agent observes its own node and the incoming transaction. That's what makes it a Dec-POMDP, but it also means the policy can learn an average-case preference between nodes, not live load balancing.
- **Hand-set parameters.** `mu_max`, `p` and `beta` were set by hand rather than calibrated against Hyperledger Fabric/Caliper, and `d_geo` (delay between jurisdictions) uses placeholder values.