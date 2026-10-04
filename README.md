# SafeLand: Shielded Reinforcement Learning with Temporal-Logic Guarantees for UAV Landing

A UAV must land on a pad under wind gusts and a finite battery. It must never enter a no-fly zone, hit a building, leave the geofence, touch down off a pad, or run out of battery, and once its battery is low it must land within a deadline. The agent learns by tabular Q-learning purely from its own simulator; **no dataset is used**. Safety comes from formal methods, **not** from control barrier functions:

1. **LTL → monitor.** The safety requirement is written in LTL and compiled into a deterministic monitor by formula progression.
2. **Shield synthesis.** A safety game on the product MDP (world × monitor) yields a shield that removes every action that could violate the specification under *any* gust sequence.
3. **Shielded RL.** Q-learning explores and acts only through the shield, so it never violates the specification, *not even during training*.
4. **Certification.** The learned policy is model-checked exactly: **CTL** on the closed-loop Kripke structure (worst case over gusts) and **PCTL** on the closed-loop Markov chain (gust probabilities). This is cross-checked by statistical model checking that shares no transition tables with the exact checker.

Everything is pure NumPy/SciPy, CPU-only, and fully reproducible (`scripts/reproduce.sh`).

---

## 1. Model

**World MDP** $\mathcal M = (S, A, P, L)$.
- Flying states are $s=(x,y,z,b)$ on a $W\times D$ grid, altitude $z\in\{1..Z\}$, battery $b\in\{1..B\}$.
- There are six absorbing sinks: *landed* (the goal) and *crash, collision, nfz, geofence, empty* (violations).
- Actions $A=\{N,S,E,W,\text{HOVER},\text{UP},\text{DOWN}\}$, each costing $\ge 1$ battery unit.
- Gusts: after the commanded move, if the new altitude is $\ge z_w$, a gust displaces the vehicle one cell along $\mathbf w$ with probability $p_w$.
- Labels: $L(s)\subseteq\{\textit{airborne, low, landed, crash, collision, nfz, geofence, empty, terminal, violation}\}$.

*Termination.* Battery strictly decreases on every flying step, so the flying part of $\mathcal M$ is a DAG and every run is absorbed within $B$ steps. All fixed points below are therefore exact after at most $B+1$ iterations (29 on the default world).

**Specification** (default, `configs/default.yaml`):

$$\varphi \;=\; \mathbf G(\lnot crash \wedge \lnot collision \wedge \lnot nfz \wedge \lnot geofence \wedge \lnot empty)\;\wedge\;\mathbf G\big(low \rightarrow \mathbf F_{\le 8}\, landed\big)$$

## 2. Method

**Monitor by progression** (Bacchus & Kabanza 2000):
$\text{prog}(\mathbf G\psi,\sigma)=\text{prog}(\psi,\sigma)\wedge\mathbf G\psi$, $\text{prog}(\mathbf F_{\le k}\psi,\sigma)=\text{prog}(\psi,\sigma)\vee\mathbf F_{\le k-1}\psi$, etc.
- Formulas are kept in negation normal form, and smart constructors simplify on the fly: complementary literals collapse, and bounded operators merge to the tightest deadline.
- Exploring progression over the world's label alphabet yields a deterministic monitor. For $\varphi$ above it has **11 states**, one of them the violation state `false`. Run `python -m safeland spec` to print it.

**Product** $\mathcal M\otimes\mathcal A_\varphi$:
- States $(s,q)$, with transitions $(s,q)\xrightarrow{a}(s', \delta(q, L(s')))$.
- *Bad* states are those with $q=\texttt{false}$, so $\varphi$ holds iff no bad state is visited.
- The default world has 84,546 product states.

**Sure shield (safety game; gusts adversarial).**

$$W_0=\lnot\textit{Bad},\qquad W_{i+1}=\{v\in W_i:\exists a\;\forall v'\in\operatorname{supp}P(\cdot\mid v,a):\,v'\in W_i\},\qquad W=\textstyle\bigcap_i W_i,$$

$$\text{allowed}(v)=\{a:\operatorname{supp}P(\cdot\mid v,a)\subseteq W\}.$$

> **Theorem 1 (sure safety, during training and deployment).** If the initial product state lies in $W$ and every executed action is in $\text{allowed}$, then every run satisfies $\varphi$. This holds for any learning algorithm, any exploration noise and any gust sequence.
> *Proof.* $W\cap\textit{Bad}=\emptyset$ and $W$ is closed under allowed actions; induction on time. ∎
>
> **Proposition 2 (maximal permissiveness).** $W$ is the complement of the adversary's attractor to *Bad*. Hence no shield with the sure guarantee can allow more actions in any state.

The guarantee does not rest on trusting the fixed-point code. `check_certificate` independently re-checks both proof obligations on the computed arrays: $W\cap Bad=\emptyset$, and every allowed action keeps every successor in $W$. The test suite also checks Proposition 2 against an independently coded attractor.

**Probabilistic shield** (Jansen et al. 2020): $V(v)=\max_\pi \Pr_v^\pi[\mathbf G\lnot Bad]$ (greatest fixed point), and $\text{allowed}_\lambda(v)=\{a: Q(v,a)\ge\lambda\max_{a'}Q(v,a')\}$ with $\lambda=0.95$. It is more permissive than the sure shield, but only bounds the risk.

**Shielded Q-learning** (preemptive shielding, Alshiekh et al. 2018):
- ε-greedy action choice over $\text{allowed}(v)$.
- Bootstrap target $r+\gamma\max_{a'\in\text{allowed}(v')}Q(v',a')$.
- Reward: +10 for a safe landing, −10 for a violation, −0.1 per step.

**Certification of the deployed greedy policy $\pi$.**
- *CTL* on the Kripke structure $K_\pi$ (gusts as nondeterminism): `AG !spec_violated`, `AF landed`, `EF landed`, `A[!spec_violated U landed]`. EU is computed as a least fixed point and EG as a greatest fixed point.
- *PCTL* on the Markov chain $D_\pi$: `P=?[!spec_violated U landed]`, `P=?[F spec_violated]`, `P=?[F landed]`, `R=?[F terminal]`. These use graph pre-computation of the probability-0 and probability-1 states, then a sparse linear solve (Baier & Katoen 2008, §10.1).
- *SMC cross-check:* 2,000 runs with a table-free world sampler and direct LTL progression, with a Clopper–Pearson interval.
- *Counterexample:* if `AG !spec_violated` fails, the shortest gust sequence that leads to a violation is returned.

## 3. Results

Default world (8×8, 4 altitudes, battery 30, two pads, a 2×2 no-fly zone, three buildings, gust probability 0.2 toward +x), 649 start states, 100,000 training episodes, **10 seeds**. Each value is mean ± 95% t-CI, and every probability is exact (PCTL), averaged over the uniform start distribution. Regenerate with `python -m safeland run --set experiment.name=main`.

| Method | Training violations | P(safe landing) | worst-start P(safe landing) | P(spec violated) | `AG safe` holds (fraction of starts) | E[steps] |
|---|---|---|---|---|---|---|
| Unshielded Q-learning | 48,588 ± 193 | 0.9995 ± 0.0003 | 0.849 ± 0.206 | 4.9e-4 ± 3.3e-4 | 0.170 ± 0.039 | 8.34 ± 0.10 |
| Probabilistic shield (λ = 0.95) | 701 ± 26 | 0.9999 ± 0.0000 | 0.9967 ± 0.0040 | 9.8e-5 ± 2.8e-5 | 0.154 ± 0.022 | 9.34 ± 0.16 |
| **Sure shield** | **0 ± 0** | **1.0000** | **1.0000** | **1.2e-9** | **0.847** (= share of starts in $W$) | 9.01 ± 0.09 |

**What the shield gives and what learning gives** (exact, the same shield with uniform-random actions instead of the learned policy):

| Shield | learned P(safe landing) | random P(safe landing) | learned E[steps] | random E[steps] |
|---|---|---|---|---|
| none | 0.9995 | 0.0026 | 8.34 | 6.07 |
| probabilistic | 0.9999 | 0.9826 | 9.34 | 21.27 |
| sure | 1.0000 | 1.0000 | 9.01 | 21.50 |

Findings:
1. **Safety during learning.** Unshielded Q-learning commits about 48,600 violations (crashes, NFZ incursions, geofence breaches) before converging. The sure shield commits none in all 10 seeds, as Theorem 1 predicts, and the probabilistic shield commits about 700 (`results/main/training.png`).
2. **Average-case vs. worst-case safety.** The unshielded policy looks excellent on average (P = 0.9995), yet it satisfies `AG !spec_violated` from only 17% of start states: some gust sequence breaks it almost everywhere. The probabilistic shield improves the probability but **not** the worst case (15%). Only the sure shield gives a worst-case guarantee, and it does so exactly on its winning region (84.7% of starts).
3. **The shield supplies safety; RL supplies efficiency.** Random actions inside the sure shield already land safely with probability 1, but take 21.5 steps. The learned policy needs 9.0, about 2.4× less flight time and battery.
4. **An interpretable by-product: the wind-aware battery reserve map** (`results/main/reserve_map.png`, `python -m safeland shield`). It gives, for each cell, the minimum battery from which safety is guaranteed. The shield discovers on its own that the cell just upwind of the no-fly zone is never safe, because a gust pushes into the NFZ and the building blocks the only escape. It also finds that the east boundary column is a trap under persistent gusts, and that a cell next to the NFZ needs a 13-unit reserve while its neighbours need 7.
5. **Price of the sure guarantee.** The sure shield treats gusts as adversarial, so 15% of start states (the east column, and cells that only escape by flying into the wind) have no guarantee at all. The probabilistic shield covers them, with worst-start success 0.9967. This is the classic sure-vs-probabilistic trade-off, quantified here.

**Robustness to gust probability** (`results/wind_sweep.md`; p = 0.2 uses 10 seeds, the others 5):

| gust p | method | training violations | P(safe landing) | worst-start P | `AG safe` (frac.) | E[steps] |
|---|---|---|---|---|---|---|
| 0.1 | unshielded | 46,173 ± 222 | 0.9988 | 0.888 | 0.116 | 8.09 |
| 0.1 | probabilistic | 400 ± 32 | 0.9999 | 0.988 | 0.126 | 9.27 |
| 0.1 | sure | **0** | 1.0000 | 1.000 | 0.847 | 8.74 |
| 0.2 | unshielded | 48,588 ± 192 | 0.9995 | 0.849 | 0.170 | 8.34 |
| 0.2 | probabilistic | 701 ± 26 | 0.9999 | 0.997 | 0.154 | 9.34 |
| 0.2 | sure | **0** | 1.0000 | 1.000 | 0.847 | 9.01 |
| 0.3 | unshielded | 50,974 ± 204 | 0.9993 | 0.809 | 0.248 | 8.62 |
| 0.3 | probabilistic | 549 ± 35 | 0.9998 | 0.997 | 0.215 | 9.93 |
| 0.3 | sure | **0** | 1.0000 | 1.000 | 0.847 | 8.92 |
| 0.4 | unshielded | 53,518 ± 394 | 0.9978 | 0.607 | 0.359 | 8.88 |
| 0.4 | probabilistic | 623 ± 42 | 0.9997 | 0.995 | 0.197 | 10.47 |
| 0.4 | sure | **0** | 1.0000 | 0.9997 | 0.847 | 8.86 |

As gusts get more likely, unshielded learning makes more violations and its worst-start performance collapses (0.888 → 0.607). The sure shield stays at zero violations, and its winning region is *identical* for every p (84.7% of starts), because sure safety depends only on whether a gust is possible, not on how likely it is. The probabilistic shield's region with V = 1 shrinks from 100% to 86% of starts.

## 4. Usage

```bash
pip install -e ".[dev]"
pytest                                               # 49 tests, ~30 s
python -m safeland shield                            # certificate + battery reserve map
python -m safeland spec                              # print the LTL monitor
python -m safeland run --set experiment.name=main    # full experiment (~12 min, CPU)
python -m safeland run --set world.wind_prob=0.4 --set rl.episodes=50000
python -m safeland summarize results/main            # rebuild tables from saved runs
python -m safeland plot results/main                 # redraw figures from logged data
```

Every run writes `config.yaml` (with a config digest), `environment.json` (versions and git commit), `shields.json` (synthesis statistics and certificate), `baselines.json`, and per seed `metrics.json`, `train_curve.csv` and `policy.npy`. Change the specification without touching code, for example `--set 'spec.safety_ltl=G(!crash & !nfz) & G(low -> F[<=5] landed)'`. Unknown atoms and malformed formulas are rejected.

## 5. Repository layout

```
src/safeland/
  config.py        frozen, validated dataclass configs (YAML + key=value overrides)
  world.py         UAV safe-landing MDP (wind, battery, NFZ, buildings, geofence)
  logic/ltl.py     LTL syntax, parser, progression, monitor construction
  logic/ctl.py     CTL parser + explicit-state model checker (EX, EU, EG fixed points)
  logic/pctl.py    PCTL on DTMCs: until, bounded until, globally, expected steps
  product.py       world × monitor product MDP
  shield.py        sure / probabilistic shield synthesis, certificate, reserve map
  rl.py            shielded tabular Q-learning (no dataset)
  verification.py  closed-loop DTMC/Kripke, CTL+PCTL certification, SMC, counterexamples
  experiment.py    seeds × methods runner, artifacts, 95% CIs, summary tables
  plots.py         figures from logged results only
tests/             49 tests (see below)
configs/default.yaml, scripts/reproduce.sh, .github/workflows/ci.yml
```

**What the tests establish** (not just "it runs"):
- The progression theorem $w\models\varphi \iff w^{1}\models\text{prog}(\varphi,w_0)$ on 3,000 random formulas × lasso words, checked against an independent LTL semantics.
- Negation correctness on random formulas.
- Every CTL operator (EX, AX, EU, AU, EG, AG, EF, AF) against brute-force path enumeration on random Kripke structures.
- PCTL until and expected steps against long-horizon iteration on random DTMCs.
- Shield maximality against an independent attractor computation.
- The certificate rejects a deliberately broken shield.
- Shielded random play never violates the specification.
- The CTL checker confirms `AG !spec_violated` on every winning start of a learned sure-shield policy, cross-checking Theorem 1 with a second algorithm.
- SMC agrees with exact PCTL.
- Transition tables match the direct dynamics, and training is deterministic.

## 6. Positioning and limitations (read before writing the paper)

- **The building blocks are established.** Shield synthesis (Bloem et al. 2015; Alshiekh et al. 2018), probabilistic shields (Jansen et al. 2020), LTL progression and CTL/PCTL model checking are all standard. The contribution of this repository is the *integrated, end-to-end-verified pipeline for UAV landing*: a battery- and deadline-aware temporal specification, shield synthesis that yields an interpretable wind-aware reserve map, zero-violation learning, and post-hoc exact CTL/PCTL certification with an independent SMC cross-check and a quantified sure-vs-probabilistic trade-off. As it stands, this is realistic for a workshop or application-track paper. A main-track paper needs one of the extensions below.
- **The guarantee is relative to the discrete model.** The grid MDP *is* the plant. Transferring the guarantee to real flight requires a *sound abstraction* of continuous dynamics, meaning a proof that every continuous behaviour is matched by the grid model, e.g. through over-approximated reachable sets per cell. That is the most valuable next research step.
- **The gust model is adversarial and unbounded.** Allowing a gust at every step is what makes the east column a trap. A bounded-gust (fairness) assumption, e.g. at most $k$ consecutive gusts, adds a counter to the product and would shrink the conservatism. It is a natural, quantifiable extension.
- **Scale.** Tabular Q-learning on 84k product states is the deliberate simple choice. The shield interface is policy-agnostic, so a shielded DQN or PPO with action masking is a drop-in replacement.

## References

- M. Alshiekh, R. Bloem, R. Ehlers, B. Könighofer, S. Niekum, U. Topcu. *Safe Reinforcement Learning via Shielding.* AAAI 2018.
- R. Bloem, B. Könighofer, R. Könighofer, C. Wang. *Shield Synthesis: Runtime Enforcement for Reactive Systems.* TACAS 2015.
- N. Jansen, B. Könighofer, S. Junges, A. Serban, R. Bloem. *Safe Reinforcement Learning Using Probabilistic Shields.* CONCUR 2020.
- F. Bacchus, F. Kabanza. *Using Temporal Logics to Express Search Control Knowledge for Planning.* Artificial Intelligence 116, 2000.
- C. Baier, J.-P. Katoen. *Principles of Model Checking.* MIT Press, 2008.
- O. Kupferman, M. Y. Vardi. *Model Checking of Safety Properties.* Formal Methods in System Design 19, 2001.
- C. J. C. H. Watkins, P. Dayan. *Q-learning.* Machine Learning 8, 1992.
- C. J. Clopper, E. S. Pearson. *The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial.* Biometrika 26, 1934.
