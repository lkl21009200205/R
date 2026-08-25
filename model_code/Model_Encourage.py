"""Model_Encourage.py

多指标“懒惰客户端”激励评分：
- 指标：D-S证据Score(效益)、数据量(效益)、模型新颖性(效益)、本地训练延迟(成本)、端到端延迟(成本)
- 方法链：Min-Max归一化 -> 熵权法(客观权重) -> TOPSIS(综合评分) -> (可选)新颖性门控惩罚

说明：未传入真实指标时，本模块会按配置生成样本指标。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import math
import random


MetricType = Literal["benefit", "cost"]
NoveltyInputMode = Literal["novelty", "similarity"]


@dataclass(frozen=True)
class IncentiveConfig:
	"""激励评分的超参数配置。"""

	eps: float = 1e-12
	# 输出控制。
	verbose: bool = True
	show_normalized_matrix: bool = True
	show_entropy_details: bool = True
	show_topsis_details: bool = True
	# novelty_scores 输入语义：
	# - "novelty": 你传入的是“新颖性”（越大越好）
	# - "similarity": 你传入的是“相似度”（越大越差），内部会转换 novelty_eff = 1 - similarity
	novelty_input: NoveltyInputMode = "similarity"
	# 门控（懒惰惩罚）相关：只基于新颖性 nov（0~1），你后续可扩展到 PN 得分。
	enable_gating: bool = True
	tau_nov: float = 0.0  # 硬阈值：nov < tau_nov -> 直接置零
	lam: float = 2.0  # 指数惩罚强度：P=exp(-lam*(1-nov))
	# 未传入延迟指标时使用的生成范围（单位随数据定义，例如秒/毫秒）
	# 你当前指定：本地训练延迟 10~20；端到端延迟 0~5
	local_delay_range: Tuple[float, float] = (10.0, 20.0)
	e2e_delay_range: Tuple[float, float] = (0.0, 5.0)


def _banner(title: str) -> None:
	line = "=" * 88
	print(f"\n{line}")
	print(title)
	print(line)


def _sub_banner(title: str) -> None:
	line = "─" * 88
	print(f"\n{line}")
	print(title)
	print(line)


def _fmt(x: float, width: int = 10, prec: int = 6) -> str:
	"""统一的浮点格式化，便于表格对齐。"""
	try:
		return f"{float(x):{width}.{prec}f}"
	except Exception:
		return f"{x!s:>{width}}"


def _round_keep_2(x: float) -> float:
	"""除数据量外：生成指标保留两位小数。"""
	return float(f"{x:.2f}")


def _rand_uniform_2(rng: random.Random, lo: float, hi: float) -> float:
	"""U(lo, hi) 随机采样并保留两位小数。"""
	return _round_keep_2(rng.uniform(lo, hi))


def _min_max_normalize(
	values: Sequence[float],
	metric_type: MetricType,
	eps: float,
) -> List[float]:
	"""Min-Max 归一化到[0,1]。

	- benefit: 越大越好
	- cost: 越小越好
	"""

	if len(values) == 0:
		return []

	vmin = min(values)
	vmax = max(values)
	denom = (vmax - vmin) + eps

	if metric_type == "benefit":
		return [(v - vmin) / denom for v in values]
	if metric_type == "cost":
		return [(vmax - v) / denom for v in values]
	raise ValueError(f"Unknown metric_type: {metric_type}")


def _entropy_weights(z_matrix: List[List[float]], eps: float) -> List[float]:
	"""熵权法：输入归一化矩阵 z（N×M），输出权重 w（M）。"""

	n = len(z_matrix)
	if n == 0:
		return []
	m = len(z_matrix[0])
	if m == 0:
		return []

	# 列和
	col_sums = [0.0] * m
	for i in range(n):
		if len(z_matrix[i]) != m:
			raise ValueError("z_matrix must be rectangular")
		for j in range(m):
			col_sums[j] += z_matrix[i][j]

	# pij
	p = [[0.0] * m for _ in range(n)]
	for j in range(m):
		denom = col_sums[j] + eps
		for i in range(n):
			p[i][j] = z_matrix[i][j] / denom

	k = 1.0 / (math.log(n) + eps) if n > 1 else 0.0

	# ej
	e = [0.0] * m
	for j in range(m):
		s = 0.0
		for i in range(n):
			s += p[i][j] * math.log(p[i][j] + eps)
		e[j] = -k * s

	d = [1.0 - ej for ej in e]
	d_sum = sum(d) + eps
	return [dj / d_sum for dj in d]


def _entropy_details(z_matrix: List[List[float]], eps: float) -> Dict[str, object]:
	"""熵权法的详细中间量（用于 verbose 打印）。

	公式回顾：
	- p_ij = z_ij / sum_i z_ij
	- e_j = -k * sum_i p_ij ln(p_ij)
	- d_j = 1 - e_j
	- w_j = d_j / sum_j d_j
	其中 k = 1/ln(N)
	"""

	n = len(z_matrix)
	if n == 0:
		return {"p": [], "e": [], "d": [], "w": []}
	m = len(z_matrix[0]) if z_matrix[0] else 0
	if m == 0:
		return {"p": [], "e": [], "d": [], "w": []}

	col_sums = [0.0] * m
	for i in range(n):
		for j in range(m):
			col_sums[j] += z_matrix[i][j]

	p = [[0.0] * m for _ in range(n)]
	for j in range(m):
		denom = col_sums[j] + eps
		for i in range(n):
			p[i][j] = z_matrix[i][j] / denom

	k = 1.0 / (math.log(n) + eps) if n > 1 else 0.0
	e = [0.0] * m
	for j in range(m):
		s = 0.0
		for i in range(n):
			s += p[i][j] * math.log(p[i][j] + eps)
		e[j] = -k * s

	d = [1.0 - ej for ej in e]
	d_sum = sum(d) + eps
	w = [dj / d_sum for dj in d]
	return {"p": p, "e": e, "d": d, "w": w}


def _topsis_score(z_matrix: List[List[float]], w: Sequence[float], eps: float) -> List[float]:
	"""TOPSIS：输入归一化矩阵 z（N×M）与权重 w（M），输出贴近度 C（N）。"""

	n = len(z_matrix)
	if n == 0:
		return []
	m = len(z_matrix[0])
	if len(w) != m:
		raise ValueError("weights length must match z_matrix columns")

	# 加权矩阵 v
	v = [[z_matrix[i][j] * w[j] for j in range(m)] for i in range(n)]

	v_plus = [max(v[i][j] for i in range(n)) for j in range(m)]
	v_minus = [min(v[i][j] for i in range(n)) for j in range(m)]

	c = []
	for i in range(n):
		d_plus = 0.0
		d_minus = 0.0
		for j in range(m):
			d_plus += (v[i][j] - v_plus[j]) ** 2
			d_minus += (v[i][j] - v_minus[j]) ** 2
		d_plus = math.sqrt(d_plus)
		d_minus = math.sqrt(d_minus)
		c.append(d_minus / (d_plus + d_minus + eps))
	return c


def _topsis_details(z_matrix: List[List[float]], w: Sequence[float], eps: float) -> Dict[str, object]:
	"""TOPSIS 的详细中间量（用于 verbose 打印）。

	公式回顾：
	- v_ij = w_j * z_ij
	- v^+_j = max_i v_ij,  v^-_j = min_i v_ij
	- D^+_i = sqrt(sum_j (v_ij - v^+_j)^2)
	- D^-_i = sqrt(sum_j (v_ij - v^-_j)^2)
	- C_i = D^-_i / (D^+_i + D^-_i)
	"""

	n = len(z_matrix)
	if n == 0:
		return {"v": [], "v_plus": [], "v_minus": [], "d_plus": [], "d_minus": [], "c": []}
	m = len(z_matrix[0])
	if len(w) != m:
		raise ValueError("weights length must match z_matrix columns")

	v = [[z_matrix[i][j] * w[j] for j in range(m)] for i in range(n)]
	v_plus = [max(v[i][j] for i in range(n)) for j in range(m)]
	v_minus = [min(v[i][j] for i in range(n)) for j in range(m)]

	d_plus_list: List[float] = []
	d_minus_list: List[float] = []
	c_list: List[float] = []
	for i in range(n):
		d_plus = 0.0
		d_minus = 0.0
		for j in range(m):
			d_plus += (v[i][j] - v_plus[j]) ** 2
			d_minus += (v[i][j] - v_minus[j]) ** 2
		d_plus = math.sqrt(d_plus)
		d_minus = math.sqrt(d_minus)
		d_plus_list.append(d_plus)
		d_minus_list.append(d_minus)
		c_list.append(d_minus / (d_plus + d_minus + eps))

	return {
		"v": v,
		"v_plus": v_plus,
		"v_minus": v_minus,
		"d_plus": d_plus_list,
		"d_minus": d_minus_list,
		"c": c_list,
	}


def compute_incentive_Si(
	client_ids: Sequence[int] | Sequence[str],
	*,
	ds_scores: Optional[Sequence[float]] = None,
	data_sizes: Optional[Sequence[float]] = None,
	novelty_scores: Optional[Sequence[float]] = None,
	local_train_delays: Optional[Sequence[float]] = None,
	e2e_delays: Optional[Sequence[float]] = None,
	config: IncentiveConfig = IncentiveConfig(),
	seed: Optional[int] = 42,
) -> Dict[str, object]:
	"""计算最终激励评分 Si。

		默认指标生成规则：
	- D-S Score：默认全部 0.8
	- 数据量：默认全部 50000
	- 新颖性：默认全部 0.85
	- 本地训练延迟 / 端到端延迟：默认用随机数生成

	后续你把这些数组通过形参传进来即可，无需改算法主体。

	Returns:
		dict，包含：
		- "Si": {client_id: Si}
		- "Ci": {client_id: Ci}  # TOPSIS基础综合得分
		- "weights": {metric_name: wj}
		- "raw": {metric_name: {client_id: raw_value}}
	"""

	n_clients = len(client_ids)
	if n_clients == 0:
		return {"Si": {}, "Ci": {}, "weights": {}, "raw": {}}

	rng = random.Random(seed)

	def _ensure(values: Optional[Sequence[float]], default_factory) -> List[float]:
		"""把外部传入数组对齐到 N；若不传则用 default_factory() 生成样本值。"""
		if values is None:
			return [default_factory() for _ in range(n_clients)]
		if len(values) != n_clients:
			raise ValueError("All metric arrays must have same length as client_ids")
		return list(values)

	# =========================
		# 默认指标生成规则
	# - ds_score:     U(0.85, 0.95) 两位小数
	# - data_size:    randint(48000, 52000)（整数）
	# - novelty_scores:
	#     - 若 config.novelty_input == "similarity": U(0.10, 0.30) 两位小数（表示相似度，越大越差）
	#       实际参与计算的新颖性 novelty_eff = 1 - similarity
	#     - 若 config.novelty_input == "novelty":   U(0.10, 0.30) 两位小数（表示新颖性，越大越好）
	# - local_delay:  U(10, 20) 两位小数
	# - e2e_delay:    U(0, 5) 两位小数
	# =========================

	ds_scores_ = _ensure(ds_scores, lambda: _rand_uniform_2(rng, 0.85, 0.95))
	data_sizes_ = _ensure(data_sizes, lambda: float(rng.randint(48000, 52000)))
	# 这里先生成“raw novelty_scores”，它的语义由 config.novelty_input 决定
	novelty_scores_raw_ = _ensure(novelty_scores, lambda: _rand_uniform_2(rng, 0.10, 0.30))

	# 将 raw 转为“参与评价的新颖性” novelty_eff（越大越好，0~1）
	if config.novelty_input == "novelty":
		novelty_scores_eff_ = [float(max(0.0, min(1.0, x))) for x in novelty_scores_raw_]
	elif config.novelty_input == "similarity":
		novelty_scores_eff_ = [float(max(0.0, min(1.0, 1.0 - x))) for x in novelty_scores_raw_]
	else:
		raise ValueError(f"Unknown novelty_input: {config.novelty_input}")

	if local_train_delays is None:
		lo, hi = config.local_delay_range
		local_train_delays_ = [_rand_uniform_2(rng, lo, hi) for _ in range(n_clients)]
	else:
		if len(local_train_delays) != n_clients:
			raise ValueError("local_train_delays length mismatch")
		local_train_delays_ = list(local_train_delays)

	if e2e_delays is None:
		lo, hi = config.e2e_delay_range
		e2e_delays_ = [_rand_uniform_2(rng, lo, hi) for _ in range(n_clients)]
	else:
		if len(e2e_delays) != n_clients:
			raise ValueError("e2e_delays length mismatch")
		e2e_delays_ = list(e2e_delays)

	# 指标定义（你已要求去除所有稳定性指标）
	# 重要：下面的 metric_names/metric_types/raw_cols 的顺序，就是你后续要改/扩展指标时的“位置索引”。
	# 位置索引说明（M=5）：
	#   j=0 -> ds_score      (效益型)
	#   j=1 -> data_size     (效益型)
	#   j=2 -> novelty_eff   (效益型)  # 注意：这里用的是“新颖性”，若你传入相似度则会自动取 1-similarity
	#   j=3 -> local_delay   (成本型)
	#   j=4 -> e2e_delay     (成本型)
	metric_names = [
		"ds_score",  # benefit
		"data_size",  # benefit
		"novelty_eff",  # benefit
		"local_delay",  # cost
		"e2e_delay",  # cost
	]
	metric_types: List[MetricType] = [
		"benefit",
		"benefit",
		"benefit",
		"cost",
		"cost",
	]

	raw_cols: List[List[float]] = [
		ds_scores_,
		data_sizes_,
		novelty_scores_eff_,
		local_train_delays_,
		e2e_delays_,
	]

	# 逐列归一化
	z_cols: List[List[float]] = []
	for col, t in zip(raw_cols, metric_types):
		z_cols.append(_min_max_normalize(col, t, config.eps))

	# 转成 N×M
	z_matrix = [[z_cols[j][i] for j in range(len(metric_names))] for i in range(n_clients)]

	# 熵权法：根据“信息熵”自动分配客观权重。越能拉开差异的指标，权重通常越高。
	if config.show_entropy_details:
		entropy = _entropy_details(z_matrix, config.eps)
		weights = list(entropy["w"])  # type: ignore[assignment]
	else:
		weights = _entropy_weights(z_matrix, config.eps)

	# TOPSIS：理想解排序，得到基础综合评分 C_i ∈ [0,1]
	if config.show_topsis_details:
		topsis = _topsis_details(z_matrix, weights, config.eps)
		c_scores = list(topsis["c"])  # type: ignore[assignment]
	else:
		c_scores = _topsis_score(z_matrix, weights, config.eps)

	# 懒惰惩罚门控（当前仅用新颖性 novelty_eff）
	# 你现在把新颖性都默认 0.85，但后续接入真实 PN / PCA 检测后，nov 会出现差异，门控会更“有杀伤力”。
	si_scores: List[float] = []
	g_list: List[float] = []
	p_list: List[float] = []
	for i in range(n_clients):
		nov = novelty_scores_eff_[i]
		if not config.enable_gating:
			si_scores.append(c_scores[i])
			g_list.append(1.0)
			p_list.append(1.0)
			continue

		g = 1.0 if nov >= config.tau_nov else 0.0
		p = math.exp(-config.lam * (1.0 - nov))  # (0,1]
		g_list.append(g)
		p_list.append(p)
		si_scores.append(g * p * c_scores[i])

	# =========================
		# verbose 详细输出
	# =========================
	if config.verbose:
		_banner("多指标激励评分计算开始 (Entropy-Weight + TOPSIS + Lazy-Gating)")
		print(f"参与方数量 N = {n_clients}")
		print("指标集合: ds_score(效益), data_size(效益), novelty(效益), local_delay(成本), e2e_delay(成本)")
		print("指标位置: [0]=ds_score, [1]=data_size, [2]=novelty_eff, [3]=local_delay, [4]=e2e_delay")
		print("方法链: Min-Max归一化 -> 熵权法 -> TOPSIS -> (可选)新颖性门控惩罚")
		print(f"随机种子 seed = {seed}")
		print(f"门控 enable_gating = {config.enable_gating}, tau_nov = {config.tau_nov}, lam = {config.lam}")
		print(f"novelty_input = {config.novelty_input}  (内部用于评价/门控的是 novelty_eff)")

		_sub_banner("Step-1 原始指标 (Raw Metrics)")
		head = (
			f"{'client':>8}"
			f"{'ds_score':>12}"
			f"{'data_size':>12}"
			f"{'nov_raw':>12}"
			f"{'nov_eff':>12}"
			f"{'local_delay':>14}"
			f"{'e2e_delay':>14}"
		)
		print(head)
		print("-" * len(head))
		for i, cid in enumerate(client_ids):
			nov_raw = novelty_scores_raw_[i]
			nov_eff = novelty_scores_eff_[i]
			print(
				f"{str(cid):>8}"
				f"{_fmt(ds_scores_[i], 12, 4)}"
				f"{_fmt(data_sizes_[i], 12, 0)}"
				f"{_fmt(nov_raw, 12, 4)}"
				f"{_fmt(nov_eff, 12, 4)}"
				f"{_fmt(local_train_delays_[i], 14, 4)}"
				f"{_fmt(e2e_delays_[i], 14, 4)}"
			)

		_sub_banner("Step-2 Min-Max 归一化矩阵 Z (0~1)")
		if config.show_normalized_matrix:
			print(f"{'client':>8} {'z_ds':>10} {'z_n':>10} {'z_nov':>10} {'z_T':>10} {'z_D':>10}")
			print("-" * 62)
			for i, cid in enumerate(client_ids):
				row = z_matrix[i]
				print(
					f"{str(cid):>8}"
					f"{_fmt(row[0], 10, 6)}"
					f"{_fmt(row[1], 10, 6)}"
					f"{_fmt(row[2], 10, 6)}"
					f"{_fmt(row[3], 10, 6)}"
					f"{_fmt(row[4], 10, 6)}"
				)
		else:
			print("(已关闭 show_normalized_matrix)")

		_sub_banner("Step-3 熵权法 (Entropy Weight)")
		if config.show_entropy_details:
			# type: ignore[has-type]
			e = entropy["e"]  # type: ignore[assignment]
			d = entropy["d"]  # type: ignore[assignment]
			print(f"{'metric':<14} {'e_j':>12} {'d_j=1-e':>12} {'w_j':>12}")
			print("-" * 56)
			for name, ej, dj, wj in zip(metric_names, e, d, weights):
				print(f"{name:<14} {_fmt(float(ej),12,6)} {_fmt(float(dj),12,6)} {_fmt(float(wj),12,6)}")
		else:
			print("(使用简洁熵权计算，未输出中间量)")
			print(f"{'metric':<14} {'w_j':>12}")
			print("-" * 28)
			for name, wj in zip(metric_names, weights):
				print(f"{name:<14} {_fmt(float(wj),12,6)}")

		_sub_banner("Step-4 TOPSIS 理想解排序")
		if config.show_topsis_details:
			v_plus = topsis["v_plus"]  # type: ignore[assignment]
			v_minus = topsis["v_minus"]  # type: ignore[assignment]
			d_plus = topsis["d_plus"]  # type: ignore[assignment]
			d_minus = topsis["d_minus"]  # type: ignore[assignment]
			print("正理想解 v+ (逐指标最大):")
			print("  " + ", ".join(f"{metric_names[j]}={float(v_plus[j]):.6f}" for j in range(len(metric_names))))
			print("负理想解 v- (逐指标最小):")
			print("  " + ", ".join(f"{metric_names[j]}={float(v_minus[j]):.6f}" for j in range(len(metric_names))))
			print(f"\n{'client':>8} {'D+':>12} {'D-':>12} {'Ci':>12}")
			print("-" * 48)
			for i, cid in enumerate(client_ids):
				print(
					f"{str(cid):>8}"
					f"{_fmt(float(d_plus[i]), 12, 6)}"
					f"{_fmt(float(d_minus[i]), 12, 6)}"
					f"{_fmt(float(c_scores[i]), 12, 6)}"
				)
		else:
			print("(已关闭 show_topsis_details)")

		_sub_banner("Step-5 懒惰惩罚门控 (基于 novelty)")
		print("门控公式: S_i = g_i * P_i * C_i")
		print("  g_i = 1[novelty >= tau_nov]  (硬门控)")
		print("  P_i = exp(-lam * (1 - novelty)) (软惩罚, 越新颖惩罚越小)")
		print(f"\n{'client':>8} {'nov_eff':>12} {'g':>8} {'P':>12} {'Ci':>12} {'Si':>12}")
		print("-" * 72)
		for i, cid in enumerate(client_ids):
			print(
				f"{str(cid):>8}"
				f"{_fmt(novelty_scores_eff_[i], 12, 6)}"
				f"{_fmt(g_list[i], 8, 0)}"
				f"{_fmt(p_list[i], 12, 6)}"
				f"{_fmt(c_scores[i], 12, 6)}"
				f"{_fmt(si_scores[i], 12, 6)}"
			)

		_sub_banner("最终排名 (按 Si 降序)")
		ranked = sorted([(str(client_ids[i]), si_scores[i], c_scores[i]) for i in range(n_clients)], key=lambda x: x[1], reverse=True)
		for rank, (cid, si, ci) in enumerate(ranked, 1):
			star = "★" if rank == 1 else ""
			print(f"{rank:2d}. client={cid:>4} {star:2s}  Si={si:.6f}  Ci={ci:.6f}")

		print("\n" + "=" * 88)
		print("多指标激励评分计算完成")
		print("=" * 88)

	# 打包输出
	si_map = {str(cid): si_scores[i] for i, cid in enumerate(client_ids)}
	ci_map = {str(cid): c_scores[i] for i, cid in enumerate(client_ids)}
	w_map = {metric_names[j]: weights[j] for j in range(len(metric_names))}
	raw_map: Dict[str, Dict[str, float]] = {}
	for j, name in enumerate(metric_names):
		raw_map[name] = {str(cid): raw_cols[j][i] for i, cid in enumerate(client_ids)}
	# 额外补充：把 novelty 的 raw 和 eff 都返回，方便你后续接真实相似度/新颖性时对齐。
	raw_map["novelty_raw"] = {str(cid): novelty_scores_raw_[i] for i, cid in enumerate(client_ids)}
	raw_map["novelty_eff"] = {str(cid): novelty_scores_eff_[i] for i, cid in enumerate(client_ids)}

	return {"Si": si_map, "Ci": ci_map, "weights": w_map, "raw": raw_map}


def compute_incentive_Si_4clients(
	*,
	ds_scores: Optional[Sequence[float]] = None,
	data_sizes: Optional[Sequence[float]] = None,
	novelty_scores: Optional[Sequence[float]] = None,
	local_train_delays: Optional[Sequence[float]] = None,
	e2e_delays: Optional[Sequence[float]] = None,
	config: IncentiveConfig = IncentiveConfig(),
	seed: Optional[int] = 42,
) -> Dict[str, object]:
	"""便捷调用函数（默认生成长度为4的数组A，但客户端数量N由数组长度自动决定）。

	你提出的需求：
	- 先在函数内部生成一个长度为4的数组 A
	- 再读取 A（或你传入的数组）的长度，自动决定参与客户端数量 N
	
	因此：
	- 如果你后续把 ds_scores / data_sizes / novelty_scores 传成长度为 K 的数组，这里会自动按 K 个客户端算分
	- 如果你什么都不传，就用默认 A 的长度=4
	"""

	# A：参考长度（默认4个参与方）。注意：A 仅用于“确定默认N=4”，不用于指标取值。
	A = [0.0, 0.0, 0.0, 0.0]

	# 选择“决定N的参考数组”：优先用用户传入的数组，否则使用A
	ref = None
	for candidate in (ds_scores, data_sizes, novelty_scores, local_train_delays, e2e_delays):
		if candidate is not None:
			ref = candidate
			break
	if ref is None:
		ref = A

	n_clients = len(ref)
	client_ids = list(range(1, n_clients + 1))

	return compute_incentive_Si(
		client_ids,
		ds_scores=ds_scores,
		data_sizes=data_sizes,
		novelty_scores=novelty_scores,
		local_train_delays=local_train_delays,
		e2e_delays=e2e_delays,
		config=config,
		seed=seed,
	)


def compute_incentive_Si_auto(
	*,
	ds_scores: Optional[Sequence[float]] = None,
	data_sizes: Optional[Sequence[float]] = None,
	novelty_scores: Optional[Sequence[float]] = None,
	local_train_delays: Optional[Sequence[float]] = None,
	e2e_delays: Optional[Sequence[float]] = None,
	config: IncentiveConfig = IncentiveConfig(),
	seed: Optional[int] = 42,
) -> Dict[str, object]:
	"""更通用的便捷函数：不需要你传 client_ids，直接由数组长度推断 N。

	推断规则：
	- 依次检查 ds_scores/data_sizes/novelty_scores/local_train_delays/e2e_delays
	- 找到第一个非 None 的数组，其长度作为 N
	- 若全部为 None，则默认 N=4

	适用场景：你后面在别的函数里调用时，只要把某个指标数组传进来，N 就自动跟随。
	"""

	# 默认N=4的“参考长度数组”，仅用于确定N，不用于指标取值。
	A = [0.0, 0.0, 0.0, 0.0]
	ref = None
	for candidate in (ds_scores, data_sizes, novelty_scores, local_train_delays, e2e_delays):
		if candidate is not None:
			ref = candidate
			break
	if ref is None:
		ref = A

	n_clients = len(ref)
	client_ids = list(range(1, n_clients + 1))
	return compute_incentive_Si(
		client_ids,
		ds_scores=ds_scores,
		data_sizes=data_sizes,
		novelty_scores=novelty_scores,
		local_train_delays=local_train_delays,
		e2e_delays=e2e_delays,
		config=config,
		seed=seed,
	)



def example_run(n_clients: int = 4, seed: int = 42) -> None:
	"""使用默认生成指标运行一遍。

	
	- 默认打开 verbose，会打印完整的熵权法 + TOPSIS + 门控过程。
	"""

	if n_clients == 4:
		result = compute_incentive_Si_4clients(seed=seed)
		client_ids = ["1", "2", "3", "4"]
	else:
		client_ids_int = list(range(1, n_clients + 1))
		result = compute_incentive_Si(client_ids_int, seed=seed)
		client_ids = [str(x) for x in client_ids_int]

	# compute_incentive_Si 内部已输出详细日志，这里只做一个轻量的补充汇总。
	print("\n[Quick Summary]")
	print("weights:", result["weights"])
	si_items = sorted(result["Si"].items(), key=lambda x: x[1], reverse=True)
	for cid, si in si_items:
		print(f"  client={cid:>3}  Si={si:.6f}  Ci={result['Ci'][cid]:.6f}")


if __name__ == "__main__":
	example_run(n_clients=4, seed=123)

