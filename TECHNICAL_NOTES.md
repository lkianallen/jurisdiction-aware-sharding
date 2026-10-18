# Technical Notes

Notes on the parts of the code that took the most thinking: why they look the way they do, and what went wrong along the way. Equation and section numbers point at the thesis methodology chapter. Where something is a measured result rather than a design choice, the note says which run it came from.

---

## 1. The tick equations (`shardnode.py`, Section 3.3.3)

Each node's state moves forward by five equations every tick. Most are simple on their own. The care is in how they fit together.

**The queue update (Eq. 1) never appears as a formula.** `receive_write()` appends arrivals to `write_queue` before the tick runs, and `process()` then pops up to μ items off the front. That append-then-pop order *is* `Qlen(t+1) = max(0, Qlen(t) + λ(t) − μ(t))`. Keeping a separate `Qlen` number updated by the formula would risk it drifting out of step with the real queue, so `Qlen` is always just `len(write_queue)`.

**Capacity (Eq. 2):**
```
μ(t) = min(Qlen(t) + λ(t), floor(μmax · (1 − exp(−β · (1 − Ucpu(t))))))
```
An idle node (`Ucpu → 0`) can process close to `μmax`. As CPU load climbs toward 1 the exponential term goes to 1 and capacity collapses toward zero, so a node is slowest exactly when it's busiest. The `min` with `Qlen + λ` stops a node "processing" work it doesn't have. The bracket placement was checked against the source equation image rather than trusted from a transcription, since that's an easy place for a silent error.

**CPU load (Eq. 4)** is a moving average:
```
Ucpu(t+1) = αcpu·Ucpu(t) + (1 − αcpu)·((λ(t) + Qlen(t)) / Cap_cpu)
```
The average stops one burst from slamming the load up, and dividing by `Cap_cpu` keeps `Ucpu` inside [0, 1], which Eq. 2 needs.

**Cross-shard reads (Eq. 5)** are the only part that needs another node's state: `L(A→B) = d_geo(A,B) + γ·(Qlen_B / μ_B) + δ`. A read is slow because of the node holding the data, not the one asking, so the latency uses the *destination's* queue and capacity. That means latency can only be worked out after every node has run `process()` for the tick (so μ_B is known), which is why `_process_reads()` runs after the main node loop in `Simulator.tick()`.

A read can also point at a block that hasn't been written yet. Those reads sit in a pending list and retry every tick. One whose block never arrives just waits forever, which is the right behaviour for a client asking for data that doesn't exist.

> **Caveat:** the Sepolia trace contains no reads, so none of this runs in training or evaluation, and the latency term of Rt is 0 everywhere. `main.py`'s synthetic demo is the only thing that exercises it.

---

## 2. Compliance and Mmeta (`simulator.py`, Section 3.2.4)

```
Rt = Σᵢ(w1·TPSᵢ − w2·Latencyᵢ) − Σᵢ C_jurisdiction(aᵢ, Mmeta)
```
The sum is straightforward. The interesting part is `is_compliant_placement()`.

It started as `transaction.jurisdiction == node.jurisdiction`, which mixes up two different things: where data came from, and where it's legally allowed to live. Real regimes often allow more than one place (EU data in any member state, or a country with an adequacy decision), and equality can't say that. So `Transaction.allowed_jurisdictions` (Mmeta) is its own field. It defaults to just the origin jurisdiction, so behaviour is unchanged until something sets a wider list:
```python
def is_compliant_placement(self, transaction, node) -> bool:
    return int(node.jurisdiction.decode()) in transaction.allowed_jurisdictions
```
Everything that asks "was that placement legal?" goes through this one function: the reward, the tick logging, and the training signal in §4.

That's deliberate. Duplicated logic drifted apart twice in this project: the observation encoding (`env.py` and `mappo.py` each built their own, now one `observations.py`), and the arrival capping in the baseline comparison (§10). The trainer also kept its own copy of the compliance check for a while, which was caught before it could diverge. The rule now is one function per concept, no copies.

---

## 3. One agent per node (`env.py`, Section 3.2)

The point of the Dec-POMDP framing is that each agent acts only on what it can see. That's harder than it sounds when the number of transactions per node changes every tick.

The first version took one flat list of target nodes, one per transaction in the tick, with no notion of who "owned" which decision. It looked multi-agent but was really one dispatcher choosing for everyone, which isn't what the formulation claims to test.

Now every transaction carries an `origin_node_id`, picked at random across all nodes and deliberately unrelated to its jurisdiction. That models a client submitting to whichever node is handy, not necessarily a compliant one, which is the situation the thesis is about. `env.step()` takes `{agent_id: [targets]}`, with each list sized to that agent's real arrivals for the tick (an agent with none gets an empty list). The action space changes size from tick to tick, but the dict absorbs that, so there's no fixed-size head, padding or masking at the boundary.

**An off-by-one bug in trace handling** is worth recording. The next tick's arrivals are generated during the `step()` that processes the current one (the usual Gym two-phase pattern). The first version decided the trace was finished by checking whether its cursor had reached the end, but the cursor reaches the end as a side effect of generating the final batch, so that final block was never acted on. Every episode silently dropped the last block of the trace. The fix checks whether the batch just generated is empty:
```python
trace_exhausted = self._trace_ticks is not None and all(
    len(v) == 0 for v in self._pending_by_agent.values()
)
```
A tiny 2-block test fixture caught it, because losing a block there is a 50% data loss and it produced a one-tick rollout that crashed training (§6). On a real 50+ block trace it would have cost about 2% of the data with no visible symptom.

---

## 4. Credit assignment (`mappo.py`, `update()`)

This is the fix that made learning work at all.

**The problem.** Rt is one shared number per tick (Section 3.2.4), but a tick can hold 100+ separate routing decisions. Standard GAE gives one advantage per tick, and broadcasting it to every decision means a compliant choice and a flagrant violation get exactly the same learning signal if they happened in the same tick. Gradient descent can't learn "that specific choice was wrong" from a signal that doesn't depend on which choice it was. In a real 200-rollout run the symptom was a violation rate stuck at the random baseline (~67%) while policy entropy crept *up* (about 2.17 to 2.19). That's noise, not slow learning.

**The fix.** Add a per-decision term, computed from that decision's own outcome:
```python
compliance_signal = decision_compliant * 2 - 1      # +1 compliant, -1 violation
per_decision_advantages = tick_level_advantages + local_signal_weight * compliance_signal
```
`decision_compliant` is recorded when each action is chosen, using the same `is_compliant_placement()` the environment uses. Rt itself is untouched, so logging and evaluation still use the true joint reward. Only the signal that trains the actor changes, which is the standard move in cooperative MARL when credit can't be assigned at the joint level. With it, violations fell from ~67% to near 0% over about 200 rollouts, and a repeat run reproduced that.

**A scaling mistake worth remembering.** The first attempt divided the ±1 signal by `reward_stats.std` to "normalise" it. That statistic is on the scale of raw rewards (thousands, because λ_violation = −1000), so the term shrank to almost nothing and the fix silently did nothing. The tick-level advantage is already normalised on its own, so the local term has to stay at unit scale to be comparable. A synthetic convergence test caught it: the violation rate stayed flat with the wrong scaling and fell to about 3% with the right one.

**Added later: rejection penalty, queue preference, entropy floor.** The same dilution applies to throughput, which had no per-decision signal at all. Two more terms follow the same pattern, computed from the simulator's true state at decision time, so the actor's own observation doesn't change:
```python
rejection_signal = -decision_rejected                                   # -1 if the write was bounced (queue full)
queue_signal = (decision_least_loaded * 2 - 1) * decision_had_choice    # +/-1, only where there was a real choice

per_decision_advantages = (
    tick_level_advantages
    + local_signal_weight * compliance_signal          # default 5.0
    + rejection_penalty_weight * rejection_signal      # default 2.0
    + queue_preference_weight * queue_signal           # default 1.5
)
```
The queue term has a built-in ceiling. The actor never sees other nodes' queues, so it can only learn an *average-case* preference for less-loaded nodes, not react to live load. That's a deliberate consequence of the Dec-POMDP setup, not an oversight.

The entropy floor is a small proportional controller. If mean policy entropy falls below `0.5 · ln(N)`, `entropy_coef` is raised (`base + 0.02 · gap`) to keep some randomness in the policy. The reasoning is a hypothesis, not a result: random routing spreads load well because its choices are independent, and a confident policy loses that.

These additions came after the baseline comparison in §10. `finetune_routing.py` applies them on top of a converged checkpoint, and the ablation in §11 isolates which of the three changes does the work.

---

## 5. Reward scaling (`mappo.py`, `RunningStats`)

With λ_violation = −1000, returns through GAE land in the thousands, and the critic's squared error then reaches around 10⁹, which swamps the actor's gradient. `RunningStats` keeps a running mean and variance (Welford's algorithm, which stays numerically stable in a single pass), and rewards are divided by the running standard deviation before GAE.

That standard deviation is unreliable for the first few samples. If the first couple of rewards happen to be nearly equal it's close to zero, and the division blows up. That actually happened: a critic loss around 10²² on the very first rollout. The normalised reward is now clipped to [−10, 10], the same guard OpenAI's `VecNormalize` uses.

---

## 6. GAE on short rollouts (`mappo.py`, `_compute_gae`)

GAE is the usual backward pass accumulating TD residuals. The edge case comes afterwards, when advantages are normalised. A one-tick rollout has zero degrees of freedom, so the sample standard deviation is NaN, and the NaN carried through to the policy logits and crashed training. Short rollouts happen at trace boundaries (see the bug in §3), so the normalisation uses the population standard deviation (`unbiased=False`), which is 0 rather than NaN for a single sample, with a small epsilon in the denominator.

---

## 7. Observation encoding (`observations.py`)

- **Jurisdiction is one-hot**, not 1/2/3. A raw integer implies an ordering the network shouldn't learn, and Chapter 3 specifies a one-hot `Jflag`. **Mmeta is multi-hot**, since a single allowed jurisdiction is just the one-element case of a set.
- **Payload goes through `log1p`.** Sepolia payloads run from a few bytes to over 2,000 within one trace. Fed in raw, next to state features that live in [0, 1], they dominate a small network. The test: an untrained actor, random state, payloads from 1 to 2500. 178 of 200 actions landed on a single node (89%). With payloads of 1 to 50 the choices spread across all nine. After `log1p` the large-payload case spread too, 9 to 36 per action. This is the likely cause of an early real-data run where one node soaked up nearly all the writes. The isolated test reproduces the collapse and the fix removes it, but that original run was never re-checked directly.

---

## 8. Actor design and weight sharing (`networks.py`)

Chapter 3 gives a 2×64 actor "per agent" but doesn't say how to handle an agent owning anywhere from 0 to K transactions in a tick. The options were a fixed `MultiDiscrete([N]*K)` head with masking, or one decision per transaction. The second won: the same small network runs once per transaction, nothing needs padding or masking between environment and network, and the network's size doesn't depend on K.

One actor's weights serve every agent. That's normal for homogeneous CTDE agents (Yu et al. 2021 do it) and doesn't centralise anything, because each forward pass only ever sees one agent's own observation. It keeps the parameter count flat as N grows. An early timing check ran 50 ticks across 30 agents in about 25 ms.

---

## 9. Re-taggable data (`data_sources.py`)

Real Ethereum transactions carry no legal jurisdiction, so the tags are necessarily synthetic. If a trace were tagged once and reused, every episode would see the same labelling and the policy could learn that instead of the general problem, a real risk with a single data pull. `RetaggableTrace` keeps the real parts fixed (payload, gas, block structure) and re-rolls jurisdiction and origin on every `.sample()`, which `env.py` calls on each `reset()`.

---

## 10. Comparing against baselines (`evaluate_baselines.py`)

Every baseline only picks compliant nodes, so their violation rate is 0 by construction. The comparison is throughput and rejections.

- `random_eligible` and `round_robin_eligible` use no more information than an agent has, so they're the fair opponents.
- `least_loaded_eligible` needs every node's queue length, which no agent can see. It's a ceiling to measure the gap against, not a competitor.
- The policy runs twice: argmax, and sampled (which is how it chooses during training).

Two design points. First, each episode replays one retagged trace through every strategy, which makes the comparison paired and cuts noise. Second, every strategy, heuristics included, runs through the same `BlockchainShardingEnv`. The first version ran the heuristics in a hand-rolled loop that skipped the environment's per-node arrival cap K, so they were handed about 9,635 write attempts per episode against about 8,037 for the policy, and the first throughput comparison was partly measuring workload. It's the same bug family as §2, and the fix is the same: one code path.

---

## 11. Where things stand

**Baseline (200-rollout checkpoint, before fine-tuning, 10 paired episodes):**

- **Compliance is solved.** The violation rate is about 0%, and it's been reproduced across two independent full runs.
- **Throughput was not.** On identical workloads the policy's writes were rejected for a full queue about 79% of the time with argmax and about 74% sampled, against 46–48% for random and round-robin.
- **The 56% load-awareness figure** (the policy picks the least-loaded compliant node 56% of the time against 33% chance) is statistically solid. But the actor can't see queues, so it seems to be a fixed preference that happens to correlate with load, not live balancing.

**After the ablation (`run_ablation.py`, 5 seeds, held-out traces, control trained for the same extra rollouts):**

- **The rejection penalty restored throughput to parity with random compliant routing.** The rejection rate fell from 72.3% (control) to 48.0%, and accepted writes per episode roughly doubled (4,180 vs 2,226, p < 0.001). This supports the explanation that throughput lacked a per-decision signal (§4).
- The queue-preference bonus reached a similar rejection rate (47.3%) but raised violations from about 2.1 to 3.6 per episode (p < 0.001). The entropy floor alone had no significant effect (p = 0.08), and all three together behaved like queue-preference alone.
- The favourite-node explanation (`inspect_node_distribution.py`) is confirmed for the baseline policy under argmax. Each jurisdiction sent 100% of its writes to a single node (node 3: 59.4% of all writes, node 6: 30.3%, node 7: 10.4%), and the other six nodes received none. The favourite saturates while its siblings sit idle, which accounts for the 79% rejection rate.

**Remaining limitations:**

- The sampled policy still lets through about 2–4 violations per episode, out of roughly 8,000 write attempts.
- **The w1/w2 sweep was flat** (12 trials, 80 rollouts each, two non-converged outliers). Latency is always 0, so w2 does nothing, and w1 only rescales throughput against the −1000 violation penalty, so the sweep says little about whether the weights matter.
- **Node capacity is hand-set.** `mu_max=25, p=40` replaced the defaults (10, 20), which rejected 63.6% of all writes for a full queue before jurisdiction was even checked. They're chosen so nodes don't choke, not calibrated to anything measured.
- **No real read workload.** Latency has never been exercised: public chain data has no read queries, so the Sepolia trace is writes only and the latency term is 0 everywhere. Only the synthetic demo in `main.py` runs it.

Not built: migration and throttle actions, Caliper calibration, real `d_geo` values, and a read workload.