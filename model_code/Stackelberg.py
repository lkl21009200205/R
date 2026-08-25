
r"""Stackelberg.py

把你在 Model_Encourage.py 得到的综合评分 S_i（已包含门控 g_i 与惩罚 P_i）
代入二次合约的 Stackelberg 激励博弈，实现单轮结算与多轮声誉扩展。

核心设定（与你的论文方案一致）：

1) 评分信号（上链/可审计）：S_i \in [0,1]
	- 由评分模块输出：S_i = g_i * P_i * \hat S_i
	- 合约层不再重复扣门控，直接按 S_i 结算

2) 领导者（二次合约）：
	R_i(S_i) = a_i + k_i S_i - (eta/2) S_i^2
	其中 a_i>=0, k_i>=0, eta>0

3) 追随者（闭式最优响应，用于确定下一轮训练强度）：
	E[S_i] \approx theta_i * e_i,	成本 c_i(e)= (c_i/2) e_i^2
	=> e_i^* = k_i*theta_i / (c_i + eta*theta_i^2)
	=> E[S_i] = k_i*theta_i^2 / (c_i + eta*theta_i^2)

4) 预算约束：sum_i R_i <= B
	- 选择 k_i/a_i 后，用真实 S_i 计算 R_i
	- 若超预算，按比例缩放（也可在上层做数值优化重算）

注意：本文件不依赖第三方库，便于你直接跑通与写论文。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import math
import random


ClientId = str


@dataclass(frozen=True)
class ContractConfig:
	"""二次合约配置。"""

	eta: float = 1.0
	budget_B: float = 10.0
	# 是否把负支付截断到0（合约实践里通常不允许扣款为负）
	floor_at_zero: bool = True
	# 若 sum(R_i) > B，是否按比例缩放到预算内
	scale_to_budget: bool = True


@dataclass(frozen=True)
class LeaderKKTConfig:
	"""领导者优化（KKT闭式）所需参数。

	你论文里是 max sum_i (v*E[S_i] - E[R_i]) s.t. sum_i E[R_i] <= B。
	这里给出一个可跑的闭式解版本（只优化 k_i；a_i 由规则给或置0）。
	"""

	v: float = 1.0
	# a_i 基线补贴：可置0，也可设为 a0*x_i
	a0: float = 0.0
	# 是否让 a_i、k_i 依赖声誉 x_i
	use_reputation: bool = True


@dataclass(frozen=True)
class ReputationConfig:
	"""多轮声誉更新配置。"""

	rho: float = 0.2
	mode: str = "smooth"  # "smooth" or "gating"


def follower_best_response_e(
	k_i: float,
	*,
	theta_i: float,
	c_i: float,
	eta: float,
) -> float:
	"""追随者最优努力 e_i^*（闭式）。"""
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
	"""根据合约强度计算期望得分 E[S_i]。"""
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
	"""二次合约支付 R_i(S_i)。

	R_i = a_i + k_i*S_i - eta/2 * S_i^2
	"""
	R = float(a_i + k_i * S_i - 0.5 * eta * (S_i ** 2))
	if floor_at_zero:
		return float(max(0.0, R))
	return R


def _safe_unit_interval(x: float) -> float:
	return float(max(0.0, min(1.0, x)))


def _normalize_keys(si_map: Mapping[ClientId, float]) -> Dict[ClientId, float]:
	# Model_Encourage 返回 key 为 str，这里仍做一次健壮化
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
	"""根据最优响应更新下一轮训练强度。

	e_{t+1} = (1-α) e_t + α e* + ξ_t
	其中 α=step_size，ξ~N(0, noise_std^2)
	"""
	alpha = float(max(0.0, min(1.0, step_size)))
	noise = rng.gauss(0.0, float(max(0.0, noise_std)))
	e_next = (1.0 - alpha) * float(e_prev) + alpha * float(e_star) + noise
	return float(max(0.0, min(float(max_effort), e_next)))


@dataclass
class PlayerParams:
	"""客户端参数。

	- theta：努力->得分的效率系数（越大表示同等努力更容易取得高分）
	- c_cost：努力成本系数（越大表示更“懒”/更贵）
	- effort：当前轮实际训练强度
	"""

	theta: float
	c_cost: float
	effort: float = 0.0
	# 努力强度更新步长与扰动项
	effort_step: float = 0.35
	effort_noise: float = 0.02


@dataclass
class LeaderParams:
	"""领导者（结算/评估节点）的策略参数。"""

	mode: str = "reputation"  # "reputation" or "kkt"
	# 规则版：a_i=a0*x_i, k_i=k0*x_i
	a0_rule: float = 0.2
	k0_rule: float = 2.0
	# KKT 版：v 与 a0 以及是否依赖声誉
	kkt: LeaderKKTConfig = LeaderKKTConfig(v=1.0, a0=0.0, use_reputation=True)


class StackelbergLeader:
	"""抽象领导者：给定声誉与玩家参数，产生合约 (a_i,k_i)。"""

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
			raise ValueError("leader_mode='kkt' 需要 players 参数（theta、c_cost）")
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
	"""把博弈论从主训练流程里“抽离出来”的模块。

	你在 main.py 每轮已经能拿到 result["Si"]。
		这里做：
		- leader 选合约 (a_i,k_i)
		- 用真实 Si 结算 R_i
		- 根据最优响应更新下一轮训练强度
		- 更新声誉 x_i(t)
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
		# 默认初始声誉：1.0（你也可以改成 0.5 更“保守”）
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
		"""按给定客户端集合初始化博弈状态。"""

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
		"""初始化 4 个参与方。"""

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
		"""推进一轮：输入评分 Si 和本轮实际训练强度，输出结算与下一轮训练强度。"""

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

		# 按当前声誉发布本轮结算合约
		a_map, k_map = self.leader.choose_contract(ids, x_rep=self.x, players=self.players)

		# 按本轮评分结算奖励
		settle = settle_payments_from_Si(si, a_map=a_map, k_map=k_map, contract=self.leader.contract)

		# 先更新声誉，再按新声誉确定下一轮训练强度
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
	"""把努力动态配置成“明显向最优 e* 收敛”的形态。

	- effort_step 越大：越快靠近 e*
	- effort_noise 越小：越不抖动（更容易看到收敛趋势）

	注意：如果你希望像现实一样有波动，把 effort_noise 调回 0.02~0.05。
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
	"""对齐 main.py 的调用口径：输入 compute_incentive_Si_auto 的 result，输出本轮博弈结算。

	用法（在 main 的 for 循环里）：
		stack_out, game = stackelberg_settle_from_model_encourage(result, game=game)
	
	关键点：
	- game 是跨轮状态，里面保存了每个参与方的 effort e(t) 和声誉 x(t)
	- 只要你每轮把上一轮返回的 game 传回去，就能看到 e 逐轮向 e* 靠近
	"""

	si = result.get("Si")
	if not isinstance(si, Mapping):
		raise ValueError("result 必须包含 result['Si'] 字典")
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
	"""更直接的接口：main.py 每轮已经拿到 Si 时可直接调用。

	你要的效果：随着 for 循环轮次推进，每个参与方 e(t) 都在向当轮最优 e*(k_i) 靠近。
	实现方式：
	- game 持久化保存 e(t)
	- 每轮 step 内部做 e_{t+1}=(1-α)e_t+α e*(t)+噪声
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
	"""用 KKT 闭式结构给出 (a_i, k_i)。

	推导要点（与你的“单轮静态”口径一致）：
	- d_i = theta_i^2 / (c_i + eta*theta_i^2)
	- E[S_i] = k_i * d_i
	- E[R_i] = a_i + k_i*E[S_i] - eta/2*E[S_i]^2
	       = a_i + k_i^2 * q_i
	  其中 q_i = d_i - (eta/2)*d_i^2 > 0
	- 领导者效用：U_i = v*E[S_i] - E[R_i] = v*d_i*k_i - a_i - q_i*k_i^2
	- 预算：sum_i E[R_i] <= B

	在 a_i 已给定（或置0）情况下，优化 k_i 是可分离的二次问题，
	给出闭式：k_i = v*d_i / (2*(1+lambda)*q_i)，lambda 由预算确定。
	"""

	ids = [str(i) for i in client_ids]
	eta = float(contract.eta)
	B = float(contract.budget_B)
	v = float(leader.v)
	if B <= 0:
		return ({cid: 0.0 for cid in ids}, {cid: 0.0 for cid in ids})

	# a_i：先按声誉规则给（也可全部为0）
	a_map: Dict[ClientId, float] = {}
	for cid in ids:
		x = 1.0
		if leader.use_reputation and x_rep is not None:
			x = float(x_rep.get(cid, 1.0))
		a_map[cid] = float(max(0.0, leader.a0 * x))

	# 留给 k 的预算（期望意义下）
	B_k = B - sum(a_map.values())
	if B_k <= 0:
		return (a_map, {cid: 0.0 for cid in ids})

	# 预计算 d_i 与 q_i
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

	# 若所有 q_i=0 或 d_i=0，则无法激励
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

	# 由 B_k = A/(1+lambda)^2 得：1+lambda = sqrt(A/B_k)，并且 lambda>=0
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
		# 可选：让 k 乘声誉（把“好人更值得奖励”写进策略）
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
	"""更贴近你论文“示例策略”的规则版：a_i=a0*x_i, k_i=k0*x_i。"""
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
	"""用真实 S_i 结算得到 R_i，并按预算 B 做可选缩放。"""

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
	"""更新声誉 x_i(t+1)。

	- mode="gating": x'=(1-rho)x + rho*1[g_i=1]
	- mode="smooth": x'=(1-rho)x + rho*S_i

	说明：如果你没单独传 g_i，可用 S_i>0 近似 1[g_i=1]。
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
	# 领导者选择合约的方式
	leader_mode: str = "kkt",  # "kkt" or "reputation"
		# KKT 所需参数
	theta: Optional[Mapping[ClientId, float]] = None,
	cost_c: Optional[Mapping[ClientId, float]] = None,
	# 声誉
	x_rep: Optional[Mapping[ClientId, float]] = None,
	# 合约
	contract: ContractConfig = ContractConfig(),
	leader_kkt: LeaderKKTConfig = LeaderKKTConfig(),
	# 规则版合约参数
	a0_rule: float = 0.0,
	k0_rule: float = 1.0,
) -> Dict[str, object]:
	"""单轮：输入 S_i -> 领导者选 (a_i,k_i) -> 结算支付 R_i。"""

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
			raise ValueError("leader_mode='kkt' 需要传入 theta 与 cost_c")
		a_map, k_map = leader_choose_contract_kkt(
			ids,
			theta=theta,
			cost_c=cost_c,
			x_rep=x_rep,
			contract=contract,
			leader=leader_kkt,
		)

	settle = settle_payments_from_Si(si, a_map=a_map, k_map=k_map, contract=contract)

	# 追随者理论响应（用于解释，不影响真实结算）
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
	# 规则版策略参数
	a0_rule: float = 0.0,
	k0_rule: float = 1.0,
	# KKT 所需
	theta: Optional[Mapping[ClientId, float]] = None,
	cost_c: Optional[Mapping[ClientId, float]] = None,
) -> Dict[str, object]:
	"""多轮运行：每轮输入评分 S_i(t)，更新声誉 x_i(t)，输出每轮结算。"""

	if len(si_rounds) == 0:
		return {"rounds": [], "x": {}}

	# 初始化声誉
	all_ids: List[ClientId] = []
	for si in si_rounds:
		for cid in si.keys():
			if str(cid) not in all_ids:
				all_ids.append(str(cid))
	# 默认初始声誉为 1（也可改成 0.5）
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
		# 门控效果：当 Si 低于阈值时直接置 0
		gating_threshold: float = 0.05,
) -> List[Dict[ClientId, float]]:
	"""生成样本 Si 序列（4 个参与方，多轮）。

	结合 main.py 时由评分模块提供 result['Si']。

	生成逻辑：
	- 每个参与方有一个“潜在贡献水平” q_i(t) 随轮次轻微漂移
	- 观测到的 S_i(t) = clamp(q_i(t) + 噪声)
	- 若 S_i < gating_threshold，则置 0
	"""

	rng = random.Random(seed)
	n_rounds = int(max(1, n_rounds))
	ids = ["1", "2", "3", "4"]

	# 初始潜在贡献（让 1/2 偏好，3 一般，4 偏“懒惰/恶意”）
	q = {
		"1": rng.uniform(0.60, 0.80),
		"2": rng.uniform(0.45, 0.70),
		"3": rng.uniform(0.20, 0.55),
		"4": rng.uniform(0.00, 0.20),
	}

	si_rounds: List[Dict[ClientId, float]] = []
	for t in range(n_rounds):
		# 随轮次推进，好的参与方略有上升，差的参与方缓慢上升或波动
		for cid in ids:
			if cid == "1":
				drift = rng.uniform(0.00, 0.03)
			elif cid == "2":
				drift = rng.uniform(-0.01, 0.03)
			elif cid == "3":
				drift = rng.uniform(-0.02, 0.02)
			else:
				# 4 号：偶尔“摆烂”导致本轮 Si≈0
				drift = rng.uniform(-0.03, 0.01)

			q[cid] = float(max(0.0, min(1.0, q[cid] + drift)))

		Si_t: Dict[ClientId, float] = {}
		for cid in ids:
			noise = rng.gauss(0.0, 0.04 if cid != "4" else 0.08)
			val = _safe_unit_interval(q[cid] + noise)
			if val < gating_threshold:
				val = 0.0
			# 保留 3 位小数，便于看输出
			Si_t[cid] = float(f"{val:.3f}")
		si_rounds.append(Si_t)

	return si_rounds


def _print_game_round(out: Mapping[str, object]) -> None:
	"""把单轮输出格式化打印。"""
	print_stackelberg_round(out)


def print_stackelberg_round(
	out: Mapping[str, object],
	*,
	round_title: Optional[str] = None,
	per_client: bool = True,
	show_maps: bool = False,
) -> None:
	"""打印每轮 Stackelberg 结算、声誉和下一轮训练强度。

	参数：
	- per_client: True 时按客户端逐个输出
	- show_maps: True 时额外打印整体 map（便于调试）
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
		round_title = f"第 {shown_round} 轮 (t={t})"
	print("\n" + "=" * 92)
	print(round_title)
	print("=" * 92)

	print("[结算汇总]")
	if scale < 0.999:
		print(f"- 预算约束触发：总支出按 scale={scale:.3f} 缩放到预算内")
	print(f"- 本轮总支出 total_R = {total_R:.6f}")
	print("- 合约形式：R_i = a_i + k_i*Si - (eta/2)*Si^2")

	prev = out.get("effort_prev") or {}
	star = out.get("effort_star") or {}
	next_ = out.get("effort_next") or {}

	if show_maps:
		print("\n[调试] 整体映射（map）")
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
		print("\n[逐客户端明细]")
		if not isinstance(si, Mapping):
			print("- Si 不是字典，无法逐客户端输出：", si)
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
			print(f"  评分信号 Si = {si_i:.6f}  （来自 Model_Encourage，已含门控/惩罚）")
			print(f"  合约参数: a_i={a_i:.6f}, k_i={k_i:.6f}  （k_i 越大，下一轮训练强度越高）")
			print(f"  支付结算: R_i={R_i:.6f}  （若触发预算缩放，已计入 scale={scale:.3f}）")
			print(f"  声誉更新: x_i(t+1)={x_i:.6f}  （用于下一轮合约策略）")
			if Se_i > 0:
				print(f"  期望评分: E[Si](下一轮合约)={Se_i:.6f}")

			if isinstance(prev, Mapping) and isinstance(star, Mapping) and isinstance(next_, Mapping):
				ep = float(prev.get(cid, 0.0))
				es = float(star.get(cid, 0.0))
				en = float(next_.get(cid, 0.0))
				gap_prev = abs(ep - es)
				gap_next = abs(en - es)
				trend = "收敛" if gap_next <= gap_prev + 1e-12 else "调整"
				print(f"  训练强度: e(t)={ep:.6f} -> e*(t+1)={es:.6f} -> e(t+1)={en:.6f}")
				print(f"           gap: |e-e*| {gap_prev:.6f} -> {gap_next:.6f}  ({trend})")
			else:
				print("  (未提供 effort_prev/effort_star/effort_next 字段)")
	else:
		# 兼容：不逐客户端时给一个简洁的总体输出
		print("\n[概览]")
		if isinstance(si, Mapping):
			print("- Si:", _fmt_map(si, prec=3))
		if isinstance(k_map, Mapping):
			print("- k :", _fmt_map(k_map, prec=3))
		if isinstance(R_map, Mapping):
			print("- R :", _fmt_map(R_map, prec=3))


def _example() -> None:
	"""独立运行入口。"""

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

	# Step-1: 准备多轮 Si 序列
	if use_model_encourage:
		try:
			from Model_Encourage import compute_incentive_Si_auto, IncentiveConfig
		except Exception:
			use_model_encourage = False
			print("[example] 未找到 Model_Encourage，自动切换为内部 Si 生成")
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
	print("\n说明：本文件可独立运行 Stackelberg 激励博弈模块。")
	print("- 评分模块输出 Si（0~1）作为可审计贡献信号；本文件不重复门控惩罚")
	print("- 领导者发布合约 (a_i,k_i)，按 R_i=a_i+k_i*Si-(eta/2)*Si^2 结算")
	print("- effort_next 作为下一轮客户端训练强度")
	print("- x 为声誉状态，用于下一轮合约（示例里让 k_i/a_i 随 x 增大）")
	print("players(theta,c_cost,effort0):")
	for cid in game.client_ids:
		p = game.players[cid]
		print(f"  client={cid}: theta={p.theta:.3f}, c_cost={p.c_cost:.3f}, effort0={p.effort:.3f}")
	print(f"contract: eta={contract.eta}, B={contract.budget_B}, scale_to_budget={contract.scale_to_budget}")
	print(f"reputation: rho={rep_cfg.rho}, mode={rep_cfg.mode}")

	# Step-2: 逐轮推进
	for si in si_rounds:
		out = game.step(si)
		_print_game_round(out)

	print("\n[example] done. final reputation x:", game.x)


if __name__ == "__main__":
	_example()

