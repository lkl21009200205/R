"""Model_Encourage.py

Multi-metric incentive score for lazy clients:
- Metrics: D-S evidence score (benefit), data size (benefit), model novelty (benefit), local training delay (cost), end-to-end delay (cost)
- Method chain: Min-Max normalization -> entropy weight method (objective weights) -> TOPSIS (composite score) -> optional novelty-gating penalty

Note: if real metrics are not provided, this module generates sample metrics according to the config.
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
	"""Hyperparameter configuration for incentive scoring."""

	eps: float = 1e-12
	# Output controls.
	verbose: bool = True
	show_normalized_matrix: bool = True
	show_entropy_details: bool = True
	show_topsis_details: bool = True
	# Input semantics for novelty_scores:
	# - "novelty": the input is novelty, where larger is better.
	# - "similarity": the input is similarity, where larger is worse; internally converted as novelty_eff = 1 - similarity.
	novelty_input: NoveltyInputMode = "similarity"
	# Gating (lazy-client penalty): currently based only on novelty nov (0~1), and can later be extended to PN scores.
	enable_gating: bool = True
	tau_nov: float = 0.0  # Hard threshold: nov < tau_nov -> set directly to zero.
	lam: float = 2.0  # Exponential penalty strength: P=exp(-lam*(1-nov)).
	# Sampling ranges used when delay metrics are not provided; units follow the data definition, such as seconds or milliseconds.
	# Current setting: local training delay 10~20; end-to-end delay 0~5.
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
	"""Format floating-point values consistently for table alignment."""
	try:
		return f"{float(x):{width}.{prec}f}"
	except Exception:
		return f"{x!s:>{width}}"


def _round_keep_2(x: float) -> float:
	"""Keep generated metrics to two decimal places, except for data size."""
	return float(f"{x:.2f}")


def _rand_uniform_2(rng: random.Random, lo: float, hi: float) -> float:
	"""Sample from U(lo, hi) and keep two decimal places."""
	return _round_keep_2(rng.uniform(lo, hi))


def _min_max_normalize(
	values: Sequence[float],
	metric_type: MetricType,
	eps: float,
) -> List[float]:
	"""Min-Max normalize values to [0,1].

	- benefit: larger is better.
	- cost: smaller is better.
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
	"""Entropy weight method: input normalized matrix z (N x M), output weights w (M)."""

	n = len(z_matrix)
	if n == 0:
		return []
	m = len(z_matrix[0])
	if m == 0:
		return []

	# Column sums.
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
	"""Detailed intermediate values for the entropy weight method, used for verbose printing.

	Formula recap:
	- p_ij = z_ij / sum_i z_ij
	- e_j = -k * sum_i p_ij ln(p_ij)
	- d_j = 1 - e_j
	- w_j = d_j / sum_j d_j
	where k = 1/ln(N)
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
	"""TOPSIS: input normalized matrix z (N x M) and weights w (M), output closeness C (N)."""

	n = len(z_matrix)
	if n == 0:
		return []
	m = len(z_matrix[0])
	if len(w) != m:
		raise ValueError("weights length must match z_matrix columns")

	# Weighted matrix v.
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
	"""Detailed intermediate values for TOPSIS, used for verbose printing.

	Formula recap:
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
	"""Compute the final incentive score Si.

	Default metric generation rules:
	- D-S Score: sampled by default unless provided.
	- data size: default is 50000.
	- novelty: default is 0.85.
	- local training delay / end-to-end delay: generated randomly by default.

	Later, pass these arrays as parameters without changing the main algorithm.

	Returns:
		dict containing:
		- "Si": {client_id: Si}
		- "Ci": {client_id: Ci}  # TOPSIS base composite score.
		- "weights": {metric_name: wj}
		- "raw": {metric_name: {client_id: raw_value}}
	"""

	n_clients = len(client_ids)
	if n_clients == 0:
		return {"Si": {}, "Ci": {}, "weights": {}, "raw": {}}

	rng = random.Random(seed)

	def _ensure(values: Optional[Sequence[float]], default_factory) -> List[float]:
		"""Align externally provided arrays to N; if omitted, generate sample values with default_factory()."""
		if values is None:
			return [default_factory() for _ in range(n_clients)]
		if len(values) != n_clients:
			raise ValueError("All metric arrays must have same length as client_ids")
		return list(values)

	# =========================
	# Default metric generation rules.
		# - ds_score:     U(0.85, 0.95), rounded to two decimal places.
		# - data_size:    randint(48000, 52000) as an integer.
		# - novelty_scores:
		#     - if config.novelty_input == "similarity": U(0.10, 0.30), rounded to two decimals; this is similarity, where larger is worse.
		#       The novelty actually used in scoring is novelty_eff = 1 - similarity.
		#     - if config.novelty_input == "novelty":   U(0.10, 0.30), rounded to two decimals; this is novelty, where larger is better.
		# - local_delay:  U(10, 20), rounded to two decimal places.
		# - e2e_delay:    U(0, 5), rounded to two decimal places.
	# =========================

	ds_scores_ = _ensure(ds_scores, lambda: _rand_uniform_2(rng, 0.85, 0.95))
	data_sizes_ = _ensure(data_sizes, lambda: float(rng.randint(48000, 52000)))
	# Generate raw novelty_scores first; their semantics are determined by config.novelty_input.
	novelty_scores_raw_ = _ensure(novelty_scores, lambda: _rand_uniform_2(rng, 0.10, 0.30))

	# Convert raw values into novelty_eff used for evaluation, where larger is better and values are in [0,1].
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

	# Metric definitions; all stability metrics have been removed as requested.
	# Important: the order of metric_names/metric_types/raw_cols below is the positional index for future metric changes or extensions.
	# Positional index notes (M=5):
	#   j=0 -> ds_score      (benefit)
	#   j=1 -> data_size     (benefit)
	#   j=2 -> novelty_eff   (benefit)  # This uses novelty; if similarity is provided, it is converted to 1-similarity.
	#   j=3 -> local_delay   (cost)
	#   j=4 -> e2e_delay     (cost)
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

	# Normalize column by column.
	z_cols: List[List[float]] = []
	for col, t in zip(raw_cols, metric_types):
		z_cols.append(_min_max_normalize(col, t, config.eps))

	# Convert to N x M.
	z_matrix = [[z_cols[j][i] for j in range(len(metric_names))] for i in range(n_clients)]

	# Entropy weight method: automatically assign objective weights by information entropy. Metrics that separate clients more usually receive higher weights.
	if config.show_entropy_details:
		entropy = _entropy_details(z_matrix, config.eps)
		weights = list(entropy["w"])  # type: ignore[assignment]
	else:
		weights = _entropy_weights(z_matrix, config.eps)

	# TOPSIS: rank against ideal solutions to obtain base composite score C_i in [0,1].
	if config.show_topsis_details:
		topsis = _topsis_details(z_matrix, weights, config.eps)
		c_scores = list(topsis["c"])  # type: ignore[assignment]
	else:
		c_scores = _topsis_score(z_matrix, weights, config.eps)

	# Lazy-client penalty gating, currently using only novelty_eff.
	# Novelty currently defaults to 0.85; after integrating real PN/PCA detection, nov will vary and the gate will become more discriminative.
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
	# Detailed verbose output.
	# =========================
	if config.verbose:
		_banner("Multi-Metric Incentive Scoring Start (Entropy Weight + TOPSIS + Lazy Gating)")
		print(f"Number of participants N = {n_clients}")
		print("Metric set: ds_score(benefit), data_size(benefit), novelty(benefit), local_delay(cost), e2e_delay(cost)")
		print("Metric positions: [0]=ds_score, [1]=data_size, [2]=novelty_eff, [3]=local_delay, [4]=e2e_delay")
		print("Method chain: Min-Max normalization -> entropy weight method -> TOPSIS -> optional novelty-gating penalty")
		print(f"Random seed = {seed}")
		print(f"Gating enable_gating = {config.enable_gating}, tau_nov = {config.tau_nov}, lam = {config.lam}")
		print(f"novelty_input = {config.novelty_input}  (novelty_eff is used internally for evaluation/gating)")

		_sub_banner("Step-1 Raw Metrics")
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

		_sub_banner("Step-2 Min-Max Normalized Matrix Z (0~1)")
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
			print("(show_normalized_matrix is disabled)")

		_sub_banner("Step-3 Entropy Weight Method")
		if config.show_entropy_details:
			# type: ignore[has-type]
			e = entropy["e"]  # type: ignore[assignment]
			d = entropy["d"]  # type: ignore[assignment]
			print(f"{'metric':<14} {'e_j':>12} {'d_j=1-e':>12} {'w_j':>12}")
			print("-" * 56)
			for name, ej, dj, wj in zip(metric_names, e, d, weights):
				print(f"{name:<14} {_fmt(float(ej),12,6)} {_fmt(float(dj),12,6)} {_fmt(float(wj),12,6)}")
		else:
			print("(Using compact entropy-weight calculation without intermediate output)")
			print(f"{'metric':<14} {'w_j':>12}")
			print("-" * 28)
			for name, wj in zip(metric_names, weights):
				print(f"{name:<14} {_fmt(float(wj),12,6)}")

		_sub_banner("Step-4 TOPSIS Ideal-Solution Ranking")
		if config.show_topsis_details:
			v_plus = topsis["v_plus"]  # type: ignore[assignment]
			v_minus = topsis["v_minus"]  # type: ignore[assignment]
			d_plus = topsis["d_plus"]  # type: ignore[assignment]
			d_minus = topsis["d_minus"]  # type: ignore[assignment]
			print("Positive ideal solution v+ (metric-wise maximum):")
			print("  " + ", ".join(f"{metric_names[j]}={float(v_plus[j]):.6f}" for j in range(len(metric_names))))
			print("Negative ideal solution v- (metric-wise minimum):")
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
			print("(show_topsis_details is disabled)")

		_sub_banner("Step-5 Lazy Penalty Gating (Based on novelty)")
		print("Gating formula: S_i = g_i * P_i * C_i")
		print("  g_i = 1[novelty >= tau_nov]  (hard gate)")
		print("  P_i = exp(-lam * (1 - novelty)) (soft penalty; higher novelty means lower penalty)")
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

		_sub_banner("Final Ranking (Descending by Si)")
		ranked = sorted([(str(client_ids[i]), si_scores[i], c_scores[i]) for i in range(n_clients)], key=lambda x: x[1], reverse=True)
		for rank, (cid, si, ci) in enumerate(ranked, 1):
			star = "*" if rank == 1 else ""
			print(f"{rank:2d}. client={cid:>4} {star:2s}  Si={si:.6f}  Ci={ci:.6f}")

		print("\n" + "=" * 88)
		print("Multi-metric incentive scoring complete")
		print("=" * 88)

	# Package outputs.
	si_map = {str(cid): si_scores[i] for i, cid in enumerate(client_ids)}
	ci_map = {str(cid): c_scores[i] for i, cid in enumerate(client_ids)}
	w_map = {metric_names[j]: weights[j] for j in range(len(metric_names))}
	raw_map: Dict[str, Dict[str, float]] = {}
	for j, name in enumerate(metric_names):
		raw_map[name] = {str(cid): raw_cols[j][i] for i, cid in enumerate(client_ids)}
	# Also return both raw and effective novelty values for future alignment with real similarity/novelty inputs.
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
	"""Convenience function that defaults to an array A of length 4 while inferring client count N from array length.

	Requested behavior:
	- First create an array A of length 4 inside the function.
	- Then read the length of A, or a provided array, to decide the number of participating clients N.
		
	Therefore:
	- If ds_scores / data_sizes / novelty_scores are later passed with length K, scoring automatically uses K clients.
	- If nothing is passed, the default A length of 4 is used.
	"""

	# A is the reference length, defaulting to 4 participants. It is only used to determine default N=4, not for metric values.
	A = [0.0, 0.0, 0.0, 0.0]

	# Choose the reference array that determines N: prefer the user-provided array; otherwise use A.
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
	"""More general convenience function: infer N directly from array length without requiring client_ids.

	Inference rules:
	- Check ds_scores/data_sizes/novelty_scores/local_train_delays/e2e_delays in order.
	- Use the length of the first non-None array as N.
	- If all are None, default to N=4.

	Use case: when calling this from other functions, passing any metric array automatically sets N.
	"""

	# Reference-length array for the default N=4 case; only used to determine N, not metric values.
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
	"""Run once with default generated metrics.

		
	- verbose is enabled by default and prints the full entropy-weight + TOPSIS + gating process.
	"""

	if n_clients == 4:
		result = compute_incentive_Si_4clients(seed=seed)
		client_ids = ["1", "2", "3", "4"]
	else:
		client_ids_int = list(range(1, n_clients + 1))
		result = compute_incentive_Si(client_ids_int, seed=seed)
		client_ids = [str(x) for x in client_ids_int]

	# compute_incentive_Si already prints detailed logs internally; only add a lightweight summary here.
	print("\n[Quick Summary]")
	print("weights:", result["weights"])
	si_items = sorted(result["Si"].items(), key=lambda x: x[1], reverse=True)
	for cid, si in si_items:
		print(f"  client={cid:>3}  Si={si:.6f}  Ci={result['Ci'][cid]:.6f}")


if __name__ == "__main__":
	example_run(n_clients=4, seed=123)

