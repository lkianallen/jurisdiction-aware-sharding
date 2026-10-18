"""Compare the trained policy against compliant-only routing baselines on the real trace.

Every baseline only ever picks compliant nodes, so their violation rate is 0 by construction. The comparison that matters is throughput and rejections.

  random_eligible        a random compliant node (Simulator.route's logic)
  round_robin_eligible   cycles through the compliant nodes in id order
  least_loaded_eligible  the compliant node with the shortest queue. This needs a global view of every node's queue, which no single agent has, so treat it as a ceiling rather than a fair competitor.

The policy runs twice: argmax (its single most confident choice) and sampled (how it chooses during training).

Each episode replays the same retagged trace against every strategy, which makes the comparison paired and cuts noise.

One caveat when citing results: the trace pipeline never generates reads, so latency is 0 for every strategy here. This is a throughput/rejection comparison, not a throughput-plus-latency one.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent

import random
import statistics

import torch

from sim.data_sources import RetaggableTrace
from sim.env import BlockchainShardingEnv
from sim.mappo import MAPPOTrainer
from sim.observations import build_local_state, build_tx_features
from sim.simulator import Simulator
from sim.shardnode import ShardNode

NUM_JURISDICTIONS = 3
NODES_PER_JURISDICTION = 3
NUM_NODES = NUM_JURISDICTIONS * NODES_PER_JURISDICTION
MAX_ARRIVALS_PER_NODE = 20
NODE_KWARGS = {"mu_max": 25.0, "p": 40}  # same capacity settings the policy was trained with

TRACE_PATH = REPO_ROOT / "sepolia_trace.json"
CHECKPOINT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "training_run" / "checkpoint_rollout200.pt"
NUM_EVAL_EPISODES = 10


def eligible_nodes(sim: Simulator, tx) -> list[ShardNode]:
    """Compliant and not full, the same definition Simulator.route() uses."""
    return [n for n in sim.nodes if sim.is_compliant_placement(tx, n) and not n.is_saturated()]


def _pick_heuristic_node(sim: Simulator, tx, strategy: str, rr_counters: dict) -> int:
    candidates = eligible_nodes(sim, tx)
    if not candidates:
        # No node is both compliant and free. Still send it to a compliant one so the
        # write gets attempted and rejected, rather than silently skipping it.
        compliant = [n for n in sim.nodes if sim.is_compliant_placement(tx, n)]
        return (compliant[0] if compliant else sim.nodes[0]).id

    if strategy == "random_eligible":
        return random.choice(candidates).id
    if strategy == "round_robin_eligible":
        key = tx.jurisdiction
        idx = rr_counters.get(key, 0) % len(candidates)
        rr_counters[key] = idx + 1
        return candidates[idx].id
    if strategy == "least_loaded_eligible":
        return min(candidates, key=lambda n: len(n.write_queue)).id
    raise ValueError(strategy)


def run_heuristic(trace_ticks, strategy: str) -> dict:
    """Run one heuristic through the same BlockchainShardingEnv the policy uses, so arrival caps (K), origin grouping and tick mechanics are identical across strategies."""
    env = BlockchainShardingEnv(
        num_jurisdictions=NUM_JURISDICTIONS,
        nodes_per_jurisdiction=NODES_PER_JURISDICTION,
        max_ticks=len(trace_ticks),
        max_arrivals_per_node=MAX_ARRIVALS_PER_NODE,
        node_kwargs=NODE_KWARGS,
        trace_ticks=trace_ticks,
        seed=1,
    )
    env.reset()
    rr_counters: dict[bytes, int] = {}

    total_reward = 0.0
    total_compliant, total_violation, total_rejected = 0, 0, 0

    for _tick in range(len(trace_ticks)):
        actions_dict = {}
        for agent_id, arrivals in env._pending_by_agent.items():
            actions_dict[agent_id] = [_pick_heuristic_node(env.simulator, tx, strategy, rr_counters) for tx in arrivals]

        _obs, rewards, dones, _infos = env.step(actions_dict)
        total_reward += next(iter(rewards.values()))
        stats = env.simulator.last_tick_stats
        total_compliant += stats.get("compliant", 0)
        total_violation += stats.get("violation", 0)
        total_rejected += stats.get("rejected", 0)

        if dones["__all__"]:
            break

    total = total_compliant + total_violation
    return {
        "strategy": strategy,
        "reward": total_reward,
        "violation_rate": (total_violation / total) if total else 0.0,
        "total_writes": total,
        "total_rejected": total_rejected,
    }


def run_policy(trace_ticks, trainer: MAPPOTrainer, deterministic: bool = True) -> dict:
    """Run the trained policy over a trace.

    deterministic=True takes the argmax (its most confident choice). False samples from its distribution, the way training does. Comparing the two shows whether deterministic play makes agents all pile onto whichever node looks best at the start of a tick, which sampling would naturally break up.
    """
    env = BlockchainShardingEnv(
        num_jurisdictions=NUM_JURISDICTIONS,
        nodes_per_jurisdiction=NODES_PER_JURISDICTION,
        max_ticks=len(trace_ticks),
        max_arrivals_per_node=MAX_ARRIVALS_PER_NODE,
        node_kwargs=NODE_KWARGS,
        trace_ticks=trace_ticks,
        seed=1,
    )
    env.reset()

    total_reward = 0.0
    total_compliant, total_violation, total_rejected = 0, 0, 0

    with torch.no_grad():
        for _tick in range(len(trace_ticks)):
            actions_dict = {}
            for agent_id, arrivals in env._pending_by_agent.items():
                node_id = env._node_id_from_agent(agent_id)
                local_state = torch.tensor(
                    build_local_state(env.simulator._node_by_id(node_id), NUM_JURISDICTIONS), dtype=torch.float32
                )
                chosen = []
                for tx in arrivals:
                    x = torch.cat([local_state, torch.tensor(build_tx_features(tx, NUM_JURISDICTIONS), dtype=torch.float32)])
                    if deterministic:
                        action = int(torch.argmax(trainer.actor.forward(x)).item()) + 1
                    else:
                        action, _log_prob, _entropy = trainer.actor.act(x)
                        action = int(action.item()) + 1
                    chosen.append(action)
                actions_dict[agent_id] = chosen

            _obs, rewards, dones, _infos = env.step(actions_dict)
            total_reward += next(iter(rewards.values()))
            stats = env.simulator.last_tick_stats
            total_compliant += stats.get("compliant", 0)
            total_violation += stats.get("violation", 0)
            total_rejected += stats.get("rejected", 0)

            if dones["__all__"]:
                break

    total = total_compliant + total_violation
    return {
        "strategy": "trained_policy_argmax" if deterministic else "trained_policy_sampled",
        "reward": total_reward,
        "violation_rate": (total_violation / total) if total else 0.0,
        "total_writes": total,
        "total_rejected": total_rejected,
    }


def main():
    trace = RetaggableTrace.from_files(TRACE_PATH)
    node_ids = list(range(1, NUM_NODES + 1))
    print(f"Loaded trace: {len(trace.blocks)} blocks")

    trainer = MAPPOTrainer(env=None, num_nodes=NUM_NODES, num_jurisdictions=NUM_JURISDICTIONS)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    trainer.actor.load_state_dict(checkpoint["actor_state_dict"])
    trainer.actor.eval()
    print(f"Loaded checkpoint from rollout {checkpoint['rollout']}")

    results: dict[str, list[dict]] = {
        "random_eligible": [], "round_robin_eligible": [], "least_loaded_eligible": [],
        "trained_policy_argmax": [], "trained_policy_sampled": [],
    }

    for episode in range(NUM_EVAL_EPISODES):
        # one retagged trace per episode, shared by every strategy (paired comparison)
        trace_ticks = trace.sample(node_ids=node_ids, num_jurisdictions=NUM_JURISDICTIONS,
                                    jurisdiction_strategy="realistic", seed=episode)

        results["random_eligible"].append(run_heuristic(trace_ticks, "random_eligible"))
        results["round_robin_eligible"].append(run_heuristic(trace_ticks, "round_robin_eligible"))
        results["least_loaded_eligible"].append(run_heuristic(trace_ticks, "least_loaded_eligible"))
        results["trained_policy_argmax"].append(run_policy(trace_ticks, trainer, deterministic=True))
        results["trained_policy_sampled"].append(run_policy(trace_ticks, trainer, deterministic=False))

    print()
    print(f"{'strategy':<22} | {'mean reward':>14} | {'std':>10} | {'viol%':>7} | {'writes/ep':>10} | {'rejected/ep':>12}")
    print("-" * 90)
    for strategy, runs in results.items():
        rewards = [r["reward"] for r in runs]
        viol = statistics.mean(r["violation_rate"] for r in runs)
        writes = statistics.mean(r["total_writes"] for r in runs)
        rejected = statistics.mean(r["total_rejected"] for r in runs)
        std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        print(f"{strategy:<22} | {statistics.mean(rewards):>14.2f} | {std:>10.2f} | {viol*100:>6.1f}% | {writes:>10.1f} | {rejected:>12.1f}")

    print()
    print("NOTE: Latency_i is 0.0 for every strategy above — no read transactions exist in this")
    print("trace pipeline. This compares throughput/rejections under a shared compliance constraint,")
    print("not the full TPS+latency+compliance trade-off Rt is formally defined over.")
    print("least_loaded_eligible has global cross-node visibility no single agent has access to —")
    print("treat it as an upper bound, not a fair peer comparison.")


if __name__ == "__main__":
    main()
