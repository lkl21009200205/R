
r"""Stackelberg.py

	Plug the composite score S_i from Model_Encourage.py, which already includes gate g_i and penalty P_i,
	into a quadratic-contract Stackelberg incentive game for single-round settlement and multi-round reputation extension.

	Core setting, aligned with the paper design:

	1) Scoring signal (on-chain/auditable): S_i \in [0,1]
		- Output by the scoring module: S_i = g_i * P_i * \hat S_i
		- The contract layer does not apply the gate again and settles directly on S_i.

	2) Leader (quadratic contract):
		R_i(S_i) = a_i + k_i S_i - (eta/2) S_i^2
		where a_i>=0, k_i>=0, eta>0

	3) Follower (closed-form best response, used to determine next-round training intensity):
		E[S_i] \approx theta_i * e_i,	cost c_i(e)= (c_i/2) e_i^2
	=> e_i^* = k_i*theta_i / (c_i + eta*theta_i^2)
	=> E[S_i] = k_i*theta_i^2 / (c_i + eta*theta_i^2)

	4) Budget constraint: sum_i R_i <= B
		- After choosing k_i/a_i, compute R_i from the realized S_i.
		- If the budget is exceeded, scale proportionally, or recompute with upper-level numerical optimization.

	Note: this file has no third-party dependency, making it easy to run directly and use in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import math
import random


ClientId = str


@dataclass(frozen=True)
class ContractConfig:
	"""Quadratic contract configuration."""

	eta: float = 1.0
	budget_B: float = 10.0
	# Whether to truncate negative payments to zero; real contracts usually do not allow negative payouts.
	floor_at_zero: bool = True
	# Whether to scale proportionally within budget when sum(R_i) > B.
	scale_to_budget: bool = True


@dataclass(frozen=True)
class LeaderKKTConfig:
	"""Parameters required for leader optimization using the closed-form KKT solution.

	The paper objective is max sum_i (v*E[S_i] - E[R_i]) subject to sum_i E[R_i] <= B.
	This provides a runnable closed-form version that optimizes only k_i; a_i is rule-based or set to 0.
	"""

	v: float = 1.0
	# Baseline subsidy a_i: can be set to 0 or to a0*x_i.
	a0: float = 0.0
	# Whether a_i and k_i depend on reputation x_i.
	use_reputation: bool = True


@dataclass(frozen=True)
class ReputationConfig:
	"""Multi-round reputation update configuration."""

	rho: float = 0.2
	mode: str = "smooth"  # "smooth" or "gating"


def follower_best_response_e(
	k_i: float,
	*,
	theta_i: float,
	c_i: float,
	eta: float,
) -> float:
	"""Follower best effort e_i^* in closed form."""
	den = c_i + eta * (theta_i ** 2)
	if den <= 0:
		return 0.0
	e = k_i * theta_i / den
	return float(max(0.0, e))


def expected_score_S(
	k_i: float,
	*,
	theta_i: float,
	c_i: float,
	eta: float,
) -> float:
	"""Compute expected score E[S_i] from contract strength."""
	den = c_i + eta * (theta_i ** 2)
	if den <= 0:
		return 0.0
	return float(max(0.0, k_i * (theta_i ** 2) / den))


def contract_payment_R(
	S_i: float,
	*,
	a_i: float,
	k_i: float,
	eta: float,
	floor_at_zero: bool = True,
) -> float:
	"""Quadratic contract payment R_i(S_i).

	R_i = a_i + k_i*S_i - eta/2 * S_i^2
	"""
	R = float(a_i + k_i * S_i - 0.5 * eta * (S_i ** 2))
	if floor_at_zero:
		return float(max(0.0, R))
	return R


def _safe_unit_interval(x: float) -> float:
	return float(max(0.0, min(1.0, x)))


def _normalize_keys(si_map: Mapping[ClientId, float]) -> Dict[ClientId, float]:
	# Model_Encourage returns string keys; normalize defensively here as well.
	return {str(k): float(v) for k, v in si_map.items()}


def _clamp_nonneg(x: float) -> float:
	return float(max(0.0, x))


def effort_update_towards_optimal(
	e_prev: float,
	*,
	e_star: float,
	step_size: float,
	noise_std: float,
	rng: random.Random,
	max_effort: float = 10.0,
) -> float:
	"""Update next-round training intensity based on the best response.

	e_{t+1} = (1-α) e_t + α e* + ξ_t
	where α=step_size and ξ~N(0, noise_std^2)
	"""
	alpha = float(max(0.0, min(1.0, step_size)))
	noise = rng.gauss(0.0, float(max(0.0, noise_std)))
	e_next = (1.0 - alpha) * float(e_prev) + alpha * float(e_star) + noise
	return float(max(0.0, min(float(max_effort), e_next)))


@dataclass
class PlayerParams:
	"""Client parameters.

	- theta: efficiency coefficient from effort to score; larger means the same effort more easily achieves a high score.
	- c_cost: effort cost coefficient; larger means lazier or more expensive.
	- effort: actual training intensity in the current round.
	"""

	theta: float
	c_cost: float
	effort: float = 0.0
	# Effort-intensity update step size and disturbance term.
	effort_step: float = 0.35
	effort_noise: float = 0.02


@dataclass
class LeaderParams:
	"""Strategy parameters for the leader, i.e. the settlement/evaluation node."""

	mode: str = "reputation"  # "reputation" or "kkt"
	# Rule-based version: a_i=a0*x_i, k_i=k0*x_i.
	a0_rule: float = 0.2
	k0_rule: float = 2.0
	# KKT version: v, a0, and whether to depend on reputation.
	kkt: LeaderKKTConfig = LeaderKKTConfig(v=1.0, a0=0.0, use_reputation=True)


class StackelbergLeader:
	"""Abstract leader: produce contract (a_i,k_i) from reputation and player parameters."""

	def __init__(self, *, params: LeaderParams, contract: ContractConfig):
		self.params = params
		self.contract = contract

	def choose_contract(
		self,
		client_ids: Iterable[ClientId],
		*,
		x_rep: Optional[Mapping[ClientId, float]] = None,
		players: Optional[Mapping[ClientId, PlayerParams]] = None,
	) -> Tuple[Dict[ClientId, float], Dict[ClientId, float]]:
		mode = str(self.params.mode).lower().strip()
		ids = [str(i) for i in client_ids]
		if mode == "reputation":
			return leader_choose_contract_reputation_rule(
				ids,
				x_rep=x_rep,
				a0=self.params.a0_rule,
				k0=self.params.k0_rule,
			)

		if players is None:
			raise ValueError("leader_mode='kkt' requires players parameters (theta, c_cost)")
		theta = {cid: float(players[cid].theta) for cid in ids if cid in players}
		cost_c = {cid: float(players[cid].c_cost) for cid in ids if cid in players}
		return leader_choose_contract_kkt(
			ids,
			theta=theta,
			cost_c=cost_c,
			x_rep=x_rep,
			contract=self.contract,
			leader=self.params.kkt,
		)


class StackelbergGame:
	"""Module that decouples the game-theoretic logic from the main training flow.

	In main.py, result["Si"] is already available each round.
	This module performs:
	- leader contract selection (a_i,k_i)
	- settlement R_i using realized Si
	- next-round training intensity update based on the best response
	- reputation update x_i(t)
	"""

	def __init__(
		self,
		*,
		client_ids: List[ClientId],
		players: Mapping[ClientId, PlayerParams],
		leader: StackelbergLeader,
		rep_config: ReputationConfig = ReputationConfig(),
		seed: int = 123,
	):
		self.client_ids = [str(x) for x in client_ids]
		self.players: Dict[ClientId, PlayerParams] = {str(k): v for k, v in players.items()}
		self.leader = leader
		self.rep_config = rep_config
		self.rng = random.Random(seed)
		self.t = 0
		# Default initial reputation: 1.0; set it to 0.5 for a more conservative initialization.
		self.x: Dict[ClientId, float] = {cid: 1.0 for cid in self.client_ids}
		self.history: List[Dict[str, object]] = []

	@staticmethod
	def create_default(
		*,
		client_ids: Iterable[ClientId],
		leader_mode: str = "reputation",
		contract: ContractConfig = ContractConfig(eta=1.2, budget_B=5.0, floor_at_zero=True, scale_to_budget=True),
		rep_config: ReputationConfig = ReputationConfig(rho=0.3, mode="smooth"),
		seed: int = 123,
	) -> "StackelbergGame":
		"""Initialize game state for the given client set."""

		ids = [str(x) for x in client_ids]
		rng = random.Random(seed)
		players: Dict[ClientId, PlayerParams] = {}
		for cid in ids:
			theta = rng.uniform(0.6, 1.4)
			c_cost = rng.uniform(0.8, 2.0)
			effort0 = rng.uniform(0.0, 0.5)
			players[cid] = PlayerParams(theta=float(theta), c_cost=float(c_cost), effort=float(effort0))

		leader_params = LeaderParams(mode=leader_mode)
		leader = StackelbergLeader(params=leader_params, contract=contract)
		return StackelbergGame(client_ids=ids, players=players, leader=leader, rep_config=rep_config, seed=seed)

	@staticmethod
	def create_default_4players(
		*,
		leader_mode: str = "reputation",
		contract: ContractConfig = ContractConfig(eta=1.2, budget_B=5.0, floor_at_zero=True, scale_to_budget=True),
		rep_config: ReputationConfig = ReputationConfig(rho=0.3, mode="smooth"),
		seed: int = 123,
	) -> "StackelbergGame":
		"""Initialize 4 participants."""

		return StackelbergGame.create_default(
			client_ids=["1", "2", "3", "4"],
			leader_mode=leader_mode,
			contract=contract,
			rep_config=rep_config,
			seed=seed,
		)

	def step(
		self,
		si_map: Mapping[ClientId, float],
		*,
		realized_efforts: Optional[Mapping[ClientId, float]] = None,
	) -> Dict[str, object]:
		"""Advance one round: input score Si and realized training intensity, output settlement and next-round intensity."""

		si = _normalize_keys(si_map)
		ids = list(si.keys())
		for cid in ids:
			if cid not in self.client_ids:
				self.client_ids.append(cid)
			if cid not in self.x:
				self.x[cid] = 1.0
			if cid not in self.players:
				self.players[cid] = PlayerParams(theta=1.0, c_cost=1.5, effort=0.0)

		if realized_efforts is not None:
			for cid, effort in _normalize_keys(realized_efforts).items():
				if cid in self.players:
					self.players[cid].effort = max(0.0, float(effort))

		# Issue this round's settlement contract based on current reputation.
		a_map, k_map = self.leader.choose_contract(ids, x_rep=self.x, players=self.players)

		# Settle rewards based on this round's score.
		settle = settle_payments_from_Si(si, a_map=a_map, k_map=k_map, contract=self.leader.contract)

		# Update reputation first, then determine next-round training intensity from the new reputation.
		self.x = update_reputation(self.x, si_map=si, config=self.rep_config)
		a_next, k_next = self.leader.choose_contract(ids, x_rep=self.x, players=self.players)

		e_prev: Dict[ClientId, float] = {cid: float(self.players[cid].effort) for cid in ids}
		e_star: Dict[ClientId, float] = {}
		e_next: Dict[ClientId, float] = {}
		S_expected: Dict[ClientId, float] = {}

		for cid in ids:
			p = self.players[cid]
			k_i = float(k_next.get(cid, 0.0))
			e_i_star = follower_best_response_e(k_i, theta_i=p.theta, c_i=p.c_cost, eta=self.leader.contract.eta)
			e_star[cid] = float(e_i_star)
			S_expected[cid] = expected_score_S(k_i, theta_i=p.theta, c_i=p.c_cost, eta=self.leader.contract.eta)
			e_i_next = effort_update_towards_optimal(
				p.effort,
				e_star=e_i_star,
				step_size=p.effort_step,
				noise_std=p.effort_noise,
				rng=self.rng,
			)
			p.effort = float(e_i_next)
			e_next[cid] = float(e_i_next)

		out = {
			"t": self.t,
			"Si": si,
			"a": a_map,
			"k": k_map,
			"a_next": a_next,
			"k_next": k_next,
			"R": settle["R"],
			"total_R": settle["total_R"],
			"scale": settle["scale"],
			"x_after": dict(self.x),
			"effort_prev": e_prev,
			"effort_star": e_star,
			"effort_next": e_next,
			"S_expected": S_expected,
		}
		self.history.append(out)
		self.t += 1
		return out


def configure_game_effort_dynamics(
	game: StackelbergGame,
	*,
	effort_step: float = 0.45,
	effort_noise: float = 0.0,
) -> None:
	"""Configure effort dynamics so they visibly converge toward the optimal e*.

	- Larger effort_step: faster approach to e*.
	- Smaller effort_noise: less fluctuation, making convergence easier to see.

	Note: if realistic fluctuation is desired, set effort_noise back to 0.02~0.05.
	"""

	step = float(max(0.0, min(1.0, effort_step)))
	noise = float(max(0.0, effort_noise))
	for p in game.players.values():
		p.effort_step = step
		p.effort_noise = noise


def _ensure_default_contract(contract: Optional[ContractConfig]) -> ContractConfig:
	if contract is None:
		return ContractConfig(eta=1.2, budget_B=5.0, floor_at_zero=True, scale_to_budget=True)
	return contract


def _ensure_default_rep(rep_config: Optional[ReputationConfig]) -> ReputationConfig:
	if rep_config is None:
		return ReputationConfig(rho=0.3, mode="smooth")
	return rep_config


def stackelberg_settle_from_model_encourage(
	result: Mapping[str, object],
	*,
	game: Optional[StackelbergGame] = None,
	realized_efforts: Optional[Mapping[ClientId, float]] = None,
) -> Tuple[Dict[str, object], StackelbergGame]:
	"""Match the main.py call style: input result from compute_incentive_Si_auto and output this round's game settlement.

	Usage inside the main for-loop:
		stack_out, game = stackelberg_settle_from_model_encourage(result, game=game)
		
	Key points:
	- game is cross-round state and stores each participant's effort e(t) and reputation x(t).
	- Passing the returned game back each round lets e approach e* round by round.
	"""

	si = result.get("Si")
	if not isinstance(si, Mapping):
		raise ValueError("result must contain the result['Si'] dictionary")
	if game is None:
		game = StackelbergGame.create_default(client_ids=list(si.keys()))
		configure_game_effort_dynamics(game, effort_step=0.45, effort_noise=0.0)
	out = game.step(si, realized_efforts=realized_efforts)  # type: ignore[arg-type]
	return out, game


def stackelberg_settle_Si_each_round(
	si_map: Mapping[ClientId, float],
	*,
	game: Optional[StackelbergGame] = None,
	seed: int = 20260121,
	leader_mode: str = "reputation",
	contract: Optional[ContractConfig] = None,
	rep_config: Optional[ReputationConfig] = None,
	effort_step: float = 0.45,
	effort_noise: float = 0.0,
) -> Tuple[Dict[str, object], StackelbergGame]:
	"""More direct interface to call when main.py already has Si each round.

	Desired behavior: as the for-loop advances, each participant's e(t) approaches that round's optimal e*(k_i).
	Implementation:
	- game persistently stores e(t).
	- Each step internally applies e_{t+1}=(1-α)e_t+α e*(t)+noise.
	"""

	if game is None:
		contract_ = _ensure_default_contract(contract)
		rep_ = _ensure_default_rep(rep_config)
		game = StackelbergGame.create_default(
			client_ids=list(si_map.keys()),
			leader_mode=leader_mode,
			contract=contract_,
			rep_config=rep_,
			seed=seed,
		)
		configure_game_effort_dynamics(game, effort_step=effort_step, effort_noise=effort_noise)
	out = game.step(si_map)
	return out, game


def leader_choose_contract_kkt(
	client_ids: Iterable[ClientId],
	*,
	theta: Mapping[ClientId, float],
	cost_c: Mapping[ClientId, float],
	x_rep: Optional[Mapping[ClientId, float]] = None,
	contract: ContractConfig = ContractConfig(),
	leader: LeaderKKTConfig = LeaderKKTConfig(),
) -> Tuple[Dict[ClientId, float], Dict[ClientId, float]]:
	"""Use the closed-form KKT structure to compute (a_i, k_i).

	Derivation points, consistent with the single-round static setup:
	- d_i = theta_i^2 / (c_i + eta*theta_i^2)
	- E[S_i] = k_i * d_i
	- E[R_i] = a_i + k_i*E[S_i] - eta/2*E[S_i]^2
	       = a_i + k_i^2 * q_i
	  where q_i = d_i - (eta/2)*d_i^2 > 0
	- Leader utility: U_i = v*E[S_i] - E[R_i] = v*d_i*k_i - a_i - q_i*k_i^2
	- Budget: sum_i E[R_i] <= B

	Given a_i, or with a_i set to 0, optimizing k_i is a separable quadratic problem.
	Closed form: k_i = v*d_i / (2*(1+lambda)*q_i), where lambda is determined by the budget.
	"""

	ids = [str(i) for i in client_ids]
	eta = float(contract.eta)
	B = float(contract.budget_B)
	v = float(leader.v)
	if B <= 0:
		return ({cid: 0.0 for cid in ids}, {cid: 0.0 for cid in ids})

	# a_i is first assigned by the reputation rule; it can also be all zero.
	a_map: Dict[ClientId, float] = {}
	for cid in ids:
		x = 1.0
		if leader.use_reputation and x_rep is not None:
			x = float(x_rep.get(cid, 1.0))
		a_map[cid] = float(max(0.0, leader.a0 * x))

	# Budget left for k in expectation.
	B_k = B - sum(a_map.values())
	if B_k <= 0:
		return (a_map, {cid: 0.0 for cid in ids})

	# Precompute d_i and q_i.
	d_map: Dict[ClientId, float] = {}
	q_map: Dict[ClientId, float] = {}
	for cid in ids:
		t = float(theta.get(cid, 0.0))
		c = float(cost_c.get(cid, 0.0))
		den = c + eta * (t ** 2)
		if den <= 0 or t <= 0:
			d = 0.0
		else:
			d = (t ** 2) / den
		# d < 1/eta  => q = d - eta/2*d^2 > 0
		q = d - 0.5 * eta * (d ** 2)
		d_map[cid] = float(max(0.0, d))
		q_map[cid] = float(max(0.0, q))

	# If all q_i=0 or d_i=0, incentives cannot be created.
	A = 0.0
	for cid in ids:
		d = d_map[cid]
		q = q_map[cid]
		if d <= 0 or q <= 0:
			continue
		# A = sum v^2*d^2/(4*q)
		A += (v ** 2) * (d ** 2) / (4.0 * q)

	if A <= 0:
		return (a_map, {cid: 0.0 for cid in ids})

	# From B_k = A/(1+lambda)^2, get 1+lambda = sqrt(A/B_k), with lambda>=0.
	one_plus_lambda = math.sqrt(A / B_k)
	one_plus_lambda = max(1.0, float(one_plus_lambda))

	k_map: Dict[ClientId, float] = {}
	for cid in ids:
		d = d_map[cid]
		q = q_map[cid]
		if d <= 0 or q <= 0:
			k = 0.0
		else:
			k = v * d / (2.0 * one_plus_lambda * q)
		# Optional: multiply k by reputation to encode that better actors deserve stronger rewards.
		if leader.use_reputation and x_rep is not None:
			k *= float(max(0.0, x_rep.get(cid, 1.0)))
		k_map[cid] = float(max(0.0, k))

	return (a_map, k_map)


def leader_choose_contract_reputation_rule(
	client_ids: Iterable[ClientId],
	*,
	x_rep: Optional[Mapping[ClientId, float]] = None,
	a0: float = 0.0,
	k0: float = 1.0,
) -> Tuple[Dict[ClientId, float], Dict[ClientId, float]]:
	"""Rule-based version closer to the paper's example strategy: a_i=a0*x_i, k_i=k0*x_i."""
	ids = [str(i) for i in client_ids]
	a_map: Dict[ClientId, float] = {}
	k_map: Dict[ClientId, float] = {}
	for cid in ids:
		x = 1.0 if x_rep is None else float(x_rep.get(cid, 1.0))
		x = float(max(0.0, x))
		a_map[cid] = float(max(0.0, a0 * x))
		k_map[cid] = float(max(0.0, k0 * x))
	return (a_map, k_map)


def settle_payments_from_Si(
	si_map: Mapping[ClientId, float],
	*,
	a_map: Mapping[ClientId, float],
	k_map: Mapping[ClientId, float],
	contract: ContractConfig = ContractConfig(),
) -> Dict[str, object]:
	"""Settle R_i from realized S_i and optionally scale by budget B."""

	si = _normalize_keys(si_map)
	eta = float(contract.eta)
	B = float(contract.budget_B)

	R_map: Dict[ClientId, float] = {}
	for cid, S in si.items():
		S_ = _safe_unit_interval(float(S))
		a_i = float(a_map.get(cid, 0.0))
		k_i = float(k_map.get(cid, 0.0))
		R_map[cid] = contract_payment_R(
			S_,
			a_i=a_i,
			k_i=k_i,
			eta=eta,
			floor_at_zero=contract.floor_at_zero,
		)

	total = float(sum(R_map.values()))
	scale = 1.0
	if contract.scale_to_budget and B > 0 and total > B:
		scale = B / total
		for cid in list(R_map.keys()):
			R_map[cid] = float(R_map[cid] * scale)
		total = float(sum(R_map.values()))

	return {
		"R": R_map,
		"total_R": total,
		"scale": scale,
	}


def update_reputation(
	x_rep: MutableMapping[ClientId, float],
	*,
	si_map: Mapping[ClientId, float],
	g_map: Optional[Mapping[ClientId, float]] = None,
	config: ReputationConfig = ReputationConfig(),
) -> Dict[ClientId, float]:
	"""Update reputation x_i(t+1).

	- mode="gating": x'=(1-rho)x + rho*1[g_i=1]
	- mode="smooth": x'=(1-rho)x + rho*S_i

	Note: if g_i is not provided separately, S_i>0 can approximate 1[g_i=1].
	"""

	rho = float(config.rho)
	rho = float(max(0.0, min(1.0, rho)))
	mode = str(config.mode).lower().strip()
	for cid, S in _normalize_keys(si_map).items():
		x = float(x_rep.get(cid, 0.0))
		if mode == "gating":
			if g_map is not None:
				g = 1.0 if float(g_map.get(cid, 0.0)) >= 0.5 else 0.0
			else:
				g = 1.0 if float(S) > 0.0 else 0.0
			x_new = (1.0 - rho) * x + rho * g
		else:
			S_ = _safe_unit_interval(float(S))
			x_new = (1.0 - rho) * x + rho * S_
		x_rep[cid] = _safe_unit_interval(float(x_new))
	return dict(x_rep)


def run_stackelberg_round(
	si_map: Mapping[ClientId, float],
	*,
	# Leader contract-selection mode.
	leader_mode: str = "kkt",  # "kkt" or "reputation"
	# Parameters required by KKT.
	theta: Optional[Mapping[ClientId, float]] = None,
	cost_c: Optional[Mapping[ClientId, float]] = None,
	# Reputation.
	x_rep: Optional[Mapping[ClientId, float]] = None,
	# Contract.
	contract: ContractConfig = ContractConfig(),
	leader_kkt: LeaderKKTConfig = LeaderKKTConfig(),
	# Rule-based contract parameters.
	a0_rule: float = 0.0,
	k0_rule: float = 1.0,
) -> Dict[str, object]:
	"""Single round: input S_i -> leader chooses (a_i,k_i) -> settle payment R_i."""

	si = _normalize_keys(si_map)
	ids = list(si.keys())
	mode = str(leader_mode).lower().strip()

	if mode == "reputation":
		a_map, k_map = leader_choose_contract_reputation_rule(
			ids,
			x_rep=x_rep,
			a0=a0_rule,
			k0=k0_rule,
		)
	else:
		if theta is None or cost_c is None:
			raise ValueError("leader_mode='kkt' requires theta and cost_c")
		a_map, k_map = leader_choose_contract_kkt(
			ids,
			theta=theta,
			cost_c=cost_c,
			x_rep=x_rep,
			contract=contract,
			leader=leader_kkt,
		)

	settle = settle_payments_from_Si(si, a_map=a_map, k_map=k_map, contract=contract)

	# Theoretical follower response, used for explanation only and not for realized settlement.
	e_star: Dict[ClientId, float] = {}
	S_exp: Dict[ClientId, float] = {}
	if theta is not None and cost_c is not None:
		for cid in ids:
			k_i = float(k_map.get(cid, 0.0))
			t = float(theta.get(cid, 0.0))
			c = float(cost_c.get(cid, 0.0))
			e_star[cid] = follower_best_response_e(k_i, theta_i=t, c_i=c, eta=contract.eta)
			S_exp[cid] = expected_score_S(k_i, theta_i=t, c_i=c, eta=contract.eta)

	return {
		"a": a_map,
		"k": k_map,
		"Si": si,
		"R": settle["R"],
		"total_R": settle["total_R"],
		"scale": settle["scale"],
		"e_star": e_star,
		"S_expected": S_exp,
	}


def run_multi_round_simulation(
	si_rounds: List[Mapping[ClientId, float]],
	*,
	leader_mode: str = "reputation",
	contract: ContractConfig = ContractConfig(),
	leader_kkt: LeaderKKTConfig = LeaderKKTConfig(),
	rep_config: ReputationConfig = ReputationConfig(),
	# Rule-based strategy parameters.
	a0_rule: float = 0.0,
	k0_rule: float = 1.0,
	# Required by KKT.
	theta: Optional[Mapping[ClientId, float]] = None,
	cost_c: Optional[Mapping[ClientId, float]] = None,
) -> Dict[str, object]:
	"""Multi-round run: input score S_i(t) each round, update reputation x_i(t), and output each settlement."""

	if len(si_rounds) == 0:
		return {"rounds": [], "x": {}}

	# Initialize reputation.
	all_ids: List[ClientId] = []
	for si in si_rounds:
		for cid in si.keys():
			if str(cid) not in all_ids:
				all_ids.append(str(cid))
	# Default initial reputation is 1; it can also be changed to 0.5.
	x: Dict[ClientId, float] = {cid: 1.0 for cid in all_ids}

	rounds_out: List[Dict[str, object]] = []
	for t, si in enumerate(si_rounds):
		res = run_stackelberg_round(
			si,
			leader_mode=leader_mode,
			theta=theta,
			cost_c=cost_c,
			x_rep=x,
			contract=contract,
			leader_kkt=leader_kkt,
			a0_rule=a0_rule,
			k0_rule=k0_rule,
		)
		x = update_reputation(x, si_map=res["Si"], config=rep_config)
		res["x_after"] = dict(x)
		res["t"] = t
		rounds_out.append(res)

	return {"rounds": rounds_out, "x": x}


def generate_sample_si_rounds_4players(
	*,
	n_rounds: int = 8,
	seed: int = 20260121,
	# Gating effect: set Si directly to zero when it is below the threshold.
	gating_threshold: float = 0.05,
) -> List[Dict[ClientId, float]]:
	"""Generate sample Si sequences for 4 participants across multiple rounds.

	When integrated with main.py, result['Si'] is provided by the scoring module.

	Generation logic:
	- Each participant has a latent contribution level q_i(t) that drifts slightly by round.
	- Observed S_i(t) = clamp(q_i(t) + noise).
	- If S_i < gating_threshold, set it to 0.
	"""

	rng = random.Random(seed)
	n_rounds = int(max(1, n_rounds))
	ids = ["1", "2", "3", "4"]

	# Initial latent contribution: 1/2 are better, 3 is average, and 4 is more lazy/malicious.
	q = {
		"1": rng.uniform(0.60, 0.80),
		"2": rng.uniform(0.45, 0.70),
		"3": rng.uniform(0.20, 0.55),
		"4": rng.uniform(0.00, 0.20),
	}

	si_rounds: List[Dict[ClientId, float]] = []
	for t in range(n_rounds):
		# As rounds progress, good participants rise slightly while weaker participants rise slowly or fluctuate.
		for cid in ids:
			if cid == "1":
				drift = rng.uniform(0.00, 0.03)
			elif cid == "2":
				drift = rng.uniform(-0.01, 0.03)
			elif cid == "3":
				drift = rng.uniform(-0.02, 0.02)
			else:
			# Participant 4 occasionally slacks off, causing this round's Si to be near 0.
				drift = rng.uniform(-0.03, 0.01)

			q[cid] = float(max(0.0, min(1.0, q[cid] + drift)))

		Si_t: Dict[ClientId, float] = {}
		for cid in ids:
			noise = rng.gauss(0.0, 0.04 if cid != "4" else 0.08)
			val = _safe_unit_interval(q[cid] + noise)
			if val < gating_threshold:
				val = 0.0
			# Keep 3 decimal places for readable output.
			Si_t[cid] = float(f"{val:.3f}")
		si_rounds.append(Si_t)

	return si_rounds


def _print_game_round(out: Mapping[str, object]) -> None:
	"""Format and print one round of output."""
	print_stackelberg_round(out)


def print_stackelberg_round(
	out: Mapping[str, object],
	*,
	round_title: Optional[str] = None,
	per_client: bool = True,
	show_maps: bool = False,
) -> None:
	"""Print each Stackelberg settlement, reputation, and next-round training intensity.

	Parameters:
	- per_client: when True, print one client at a time.
	- show_maps: when True, additionally print full maps for debugging.
	"""

	def _fmt_map(m: Mapping[str, object] | Mapping[ClientId, float], prec: int = 3) -> str:
		items: List[Tuple[str, float]] = []
		for k, v in m.items():
			try:
				items.append((str(k), float(v)))
			except Exception:
				continue
		items.sort(key=lambda x: x[0])
		return "{" + ", ".join(f"{k}:{val:.{prec}f}" for k, val in items) + "}"

	def _argmax(m: Mapping[ClientId, float]) -> Tuple[Optional[ClientId], float]:
		best_k: Optional[ClientId] = None
		best_v = -1.0
		for k, v in m.items():
			fv = float(v)
			if fv > best_v:
				best_k, best_v = str(k), fv
		return best_k, best_v

	t = out.get("t")
	si = out.get("Si") or {}
	a_map = out.get("a") or {}
	k_map = out.get("k") or {}
	R_map = out.get("R") or {}
	x_after = out.get("x_after") or {}
	S_expected = out.get("S_expected") or {}
	total_R = float(out.get("total_R") or 0.0)
	scale = float(out.get("scale") or 1.0)

	shown_round = int(t) + 1 if isinstance(t, int) else t
	if round_title is None:
		round_title = f"Round {shown_round} (t={t})"
	print("\n" + "=" * 92)
	print(round_title)
	print("=" * 92)

	print("[Settlement Summary]")
	if scale < 0.999:
		print(f"- Budget constraint triggered: total spending was scaled into budget with scale={scale:.3f}")
	print(f"- Total spending this round total_R = {total_R:.6f}")
	print("- Contract form: R_i = a_i + k_i*Si - (eta/2)*Si^2")

	prev = out.get("effort_prev") or {}
	star = out.get("effort_star") or {}
	next_ = out.get("effort_next") or {}

	if show_maps:
		print("\n[Debug] Full maps")
		if isinstance(si, Mapping):
			print("- Si:", _fmt_map(si, prec=3))
		if isinstance(a_map, Mapping):
			print("- a :", _fmt_map(a_map, prec=3))
		if isinstance(k_map, Mapping):
			print("- k :", _fmt_map(k_map, prec=3))
		if isinstance(R_map, Mapping):
			print("- R :", _fmt_map(R_map, prec=3))
		if isinstance(x_after, Mapping):
			print("- x :", _fmt_map(x_after, prec=3))

	if per_client:
		print("\n[Per-Client Details]")
		if not isinstance(si, Mapping):
			print("- Si is not a dictionary, so per-client output is unavailable:", si)
			return

		client_ids = sorted(si.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
		for cid in client_ids:
			si_i = float(si.get(cid, 0.0))
			a_i = float(a_map.get(cid, 0.0)) if isinstance(a_map, Mapping) else 0.0
			k_i = float(k_map.get(cid, 0.0)) if isinstance(k_map, Mapping) else 0.0
			R_i = float(R_map.get(cid, 0.0)) if isinstance(R_map, Mapping) else 0.0
			x_i = float(x_after.get(cid, 0.0)) if isinstance(x_after, Mapping) else 0.0
			Se_i = float(S_expected.get(cid, 0.0)) if isinstance(S_expected, Mapping) else 0.0

			print(f"- client {cid}")
			print(f"  Score signal Si = {si_i:.6f}  (from Model_Encourage, including gate/penalty)")
			print(f"  Contract parameters: a_i={a_i:.6f}, k_i={k_i:.6f}  (larger k_i means higher next-round training intensity)")
			print(f"  Payment settlement: R_i={R_i:.6f}  (if budget scaling was triggered, scale={scale:.3f} is included)")
			print(f"  Reputation update: x_i(t+1)={x_i:.6f}  (used for the next-round contract strategy)")
			if Se_i > 0:
				print(f"  Expected score: E[Si](next-round contract)={Se_i:.6f}")

			if isinstance(prev, Mapping) and isinstance(star, Mapping) and isinstance(next_, Mapping):
				ep = float(prev.get(cid, 0.0))
				es = float(star.get(cid, 0.0))
				en = float(next_.get(cid, 0.0))
				gap_prev = abs(ep - es)
				gap_next = abs(en - es)
				trend = "converging" if gap_next <= gap_prev + 1e-12 else "adjusting"
				print(f"  Training intensity: e(t)={ep:.6f} -> e*(t+1)={es:.6f} -> e(t+1)={en:.6f}")
				print(f"           gap: |e-e*| {gap_prev:.6f} -> {gap_next:.6f}  ({trend})")
			else:
				print("  (effort_prev/effort_star/effort_next fields were not provided)")
	else:
		# Compatibility path: print a concise overview when not printing per-client details.
		print("\n[Overview]")
		if isinstance(si, Mapping):
			print("- Si:", _fmt_map(si, prec=3))
		if isinstance(k_map, Mapping):
			print("- k :", _fmt_map(k_map, prec=3))
		if isinstance(R_map, Mapping):
			print("- R :", _fmt_map(R_map, prec=3))


def _example() -> None:
	"""Standalone run entry point."""

	use_model_encourage = False
	seed = 20260121
	n_rounds = 8

	contract = ContractConfig(eta=1.2, budget_B=5.0, floor_at_zero=True, scale_to_budget=True)
	rep_cfg = ReputationConfig(rho=0.3, mode="smooth")

	game = StackelbergGame.create_default_4players(
		leader_mode="reputation",
		contract=contract,
		rep_config=rep_cfg,
		seed=seed,
	)

	# Step-1: prepare the multi-round Si sequence.
	if use_model_encourage:
		try:
			from Model_Encourage import compute_incentive_Si_auto, IncentiveConfig
		except Exception:
			use_model_encourage = False
			print("[example] Model_Encourage was not found; automatically switching to internal Si generation")
		else:
			si_rounds: List[Mapping[ClientId, float]] = []
			for s in range(n_rounds):
				result = compute_incentive_Si_auto(
					config=IncentiveConfig(verbose=False, show_entropy_details=False, show_topsis_details=False),
					seed=seed + s,
				)
				si_rounds.append(result["Si"])  # type: ignore[arg-type]
	else:
		si_rounds = generate_sample_si_rounds_4players(n_rounds=n_rounds, seed=seed)

	print("\n[example] StackelbergGame standalone run")
	print("\nNote: this file can independently run the Stackelberg incentive-game module.")
	print("- The scoring module outputs Si (0~1) as an auditable contribution signal; this file does not reapply the gating penalty")
	print("- The leader publishes contract (a_i,k_i) and settles by R_i=a_i+k_i*Si-(eta/2)*Si^2")
	print("- effort_next is used as the next-round client training intensity")
	print("- x is the reputation state used for the next-round contract; in this example k_i/a_i increase with x")
	print("players(theta,c_cost,effort0):")
	for cid in game.client_ids:
		p = game.players[cid]
		print(f"  client={cid}: theta={p.theta:.3f}, c_cost={p.c_cost:.3f}, effort0={p.effort:.3f}")
	print(f"contract: eta={contract.eta}, B={contract.budget_B}, scale_to_budget={contract.scale_to_budget}")
	print(f"reputation: rho={rep_cfg.rho}, mode={rep_cfg.mode}")

	# Step-2: advance round by round.
	for si in si_rounds:
		out = game.step(si)
		_print_game_round(out)

	print("\n[example] done. final reputation x:", game.x)


if __name__ == "__main__":
	_example()

