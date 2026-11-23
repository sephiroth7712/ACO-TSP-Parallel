#!/usr/bin/env python3
"""Render the README animation: one Max-Min Ant System run on TSPLIB eil51.

This is a NumPy re-implementation of the update rules used by the OpenMP and MPI
builds (src/ACO_TSP_Threaded.cpp, src/ACO_TSP_MPI.cpp), not output of the C++
programs: every ant starts at city 0, the next city is drawn by roulette wheel with
weight tau^alpha * (1/d)^beta, only the iteration-best ant deposits Q/L on its
consecutive roads in both directions (never on the closing road back to city 0),
and every entry is clamped to [tau_min, tau_max]. Lengths are exact Euclidean.

The GIF (800 x 480 px, 131 frames, 14.87 s, looping) opens on the state after 2,000
iterations, replays the early pheromone web thinning into trails, plays one iteration
in slow motion (256 ants walk their tours, then evaporation, deposit and clamping each
get their own beat), fast-forwards to iteration 2,000 and traces the best tour back
into the opening frame. Apart from the slow-motion update steps, every pheromone state
on screen is a recorded state of the run, never an interpolation between iterations.

Before anything is rendered, the run is checked against every claim the captions
make (for example, that the final pheromone trail is exactly the best tour). If one
fails, the script exits 1 with "animate_aco.py: FAILED: <reason>". The GIF is written
to <out>.tmp, decoded and verified, and only then moved onto <out>, so a failed run
never replaces the committed file. Re-running reproduces the same bytes on the same
machine. The script prints the run's key numbers and the GIF's size and SHA-1.

    python3 scripts/animate_aco.py                 # writes output_images/mmas_eil51.gif
    python3 scripts/animate_aco.py --cache /tmp/run.npz --frames /tmp/frames
    python3 scripts/animate_aco.py --seed 0 --find-seed

Needs NumPy, matplotlib and Pillow only (no ffmpeg). Run it from any directory.
"""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TSP = REPO / "datasets" / "eil51.tsp"

# ---- run parameters (README example for eil51) ---------------------------------
# Constants, not flags: the checks and captions depend on them.
N_ANTS, N_ITERS = 256, 2000
ALPHA, BETA, RHO, TAU_MIN, TAU_MAX, Q = 1.0, 3.5, 0.1, 0.1, 2.0, 430.0
# The slow-motion iteration K is the first new overall best at iteration K_MIN or later
# whose tour, before the update, has >= K_MIN_CAP roads at tau_max and >= K_MIN_FLOOR at tau_min.
K_MIN, K_MIN_CAP, K_MIN_FLOOR = 30, 25, 1
EPS = 1e-6                                            # tolerance for "at tau_min" / "at tau_max"

# ---- canvas ----------------------------------------------------------------------
W, H, DPI = 800, 480, 100
PT = 72.0 / DPI                                       # px -> pt
MAP = (12, 12, 448, 456)                              # x, y, w, h in px
PX, PW = 476, 308                                     # panel x, width
C = dict(bg="#0d1117", border="#3d444d", fg="#e6edf3", muted="#9198a1", city="#c9d1d9",
         tour="#ffffff", ant="#4aa3ff", head="#cfe6ff", over="#c297ff", under="#33264a",
         zone="#1b1726", bar="#9198a1")
RAMP = [(0.00, "#1d232b"), (0.15, "#3b2433"), (0.40, "#8c3a2c"), (0.70, "#e2742a"), (1.00, "#ffc24f")]
W_MIN, W_SPAN = 0.5, 2.1                              # road width = 0.5 + 2.1 * s**1.5 px
OVER_SPAN = 2.2                                       # extra width at tau = tau_max + 0.8
TOUR_W, CLOSE_W, STREAK_W, HEAD_D, TRACE_HEAD_D = 1.2, 1.2, 1.1, 2.8, 7.0

# ---- frame schedule (delays D_* in centiseconds) -----------------------------------
WARM = [4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 24, 29, 35, 42, 50, 60]   # warm-up iterations shown
# Walk frames i = 0..WALK_STEPS share the arc length s_i = L_max * (WALK_A * u + (1 - WALK_A) * u**3),
# u = i / WALK_STEPS; streaks are clip(s_i - s_(i-1), STREAK_MIN, STREAK_MAX) px long.
WALK_STEPS, WALK_A, STREAK_MIN, STREAK_MAX = 56, 0.2, 6.0, 20.0
FF_N, TRACE_N = 22, 30                                # fast-forward snapshots, trace steps
D_POSTER, D_WARM0, D_WARM, D_WALK, D_WALK_END = 230, 50, 9, 5, 90
D_TWEEN, D_EVAP, D_DEP, D_CLAMP, D_FF, D_FF_END, D_TRACE = 6, 110, 140, 120, 8, 30, 4

# seed-9 regression values (warn only: another CPU or NumPy build may diverge)
SEED9 = dict(it1=565.17, before_k=462.14, k=61, hero=94, hero_len=455.73, q_over_l=0.9436,
             median=588.54, composition=(29, 3, 18), final=429.5303, found=802, tmin_after_29=1051,
             new_bests=[1, 4, 5, 9, 11, 14, 15, 25, 61, 158, 177, 186, 228, 600, 744, 802])


def fail(msg):
    """Exit with status 1, printing "animate_aco.py: FAILED: <msg>" to stderr."""
    sys.exit(f"animate_aco.py: FAILED: {msg}")


def load_tsp(path):
    """Read the city coordinates of a TSPLIB EUC_2D file as an (n, 2) float array."""
    xy, n, in_coords = [], None, False
    for line in open(path):
        tok = line.replace(":", " ").split()
        if not tok:
            continue
        if tok[0] == "EOF":
            break
        if in_coords:
            xy.append((float(tok[1]), float(tok[2])))
        elif tok[0] == "DIMENSION":
            n = int(tok[1])
        elif tok[0] == "NODE_COORD_SECTION":
            in_coords = True
    xy = np.array(xy, dtype=np.float64)
    if n is None or len(xy) != n:
        fail(f"{path}: expected {n} coordinates, read {len(xy)}")
    return xy


# ---- simulation --------------------------------------------------------------------
def simulate(xy, seed):
    """Run MMAS with the constants above and return the recorded state as a dict:

    taus  float32 (N_ITERS + 1, n(n-1)/2): pheromone after k iterations, upper triangle
          in np.triu_indices(n, 1) order (row 0 is the all-tau_max start)
    ibl   float64 (N_ITERS,): iteration-best length; 1-based iteration k is at index k - 1
    ibt   int16 (N_ITERS, n): iteration-best tours, each starting at city 0
    ants  {k: (tours, lens)}: all ants' tours and lengths for every 1-based iteration
          k >= K_MIN that sets a new overall best (the candidates for K)

    The RNG call order and the selection rule below fix the trajectory of each seed.
    """
    n, m = len(xy), N_ANTS
    rng = np.random.default_rng(seed)
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    if (d + np.eye(n) == 0).any():
        fail("duplicate city coordinates (the C++ isinf rule) are not supported")
    eta_b = np.where(d > 0, (1.0 / np.where(d > 0, d, 1.0)) ** BETA, 0.0)
    tau = np.full((n, n), TAU_MAX)
    iu = np.triu_indices(n, 1)
    ar = np.arange(m)
    taus = np.empty((N_ITERS + 1, len(iu[0])), np.float32)
    taus[0] = tau[iu]
    ibl = np.empty(N_ITERS)
    ibt = np.empty((N_ITERS, n), np.int16)
    ants = {}                                          # 1-based iteration -> (tours, lens)
    best = np.inf
    for it in range(N_ITERS):
        w_base = (tau ** ALPHA) * eta_b                # tau^alpha * (1/d)^beta for every road
        tours = np.zeros((m, n), int)
        visited = np.zeros((m, n), bool)
        visited[:, 0] = True                           # every ant starts at city 0
        cur = np.zeros(m, int)
        for step in range(1, n):
            # roulette wheel, all ants at once: the first city whose cumulative weight exceeds u
            w = np.where(visited, 0.0, w_base[cur])
            cs = np.cumsum(w, axis=1)
            u = rng.random(m) * cs[:, -1]
            nxt = np.minimum((cs <= u[:, None]).sum(1), n - 1)
            if visited[ar, nxt].any():
                fail(f"iteration {it + 1}: an ant picked a visited city")
            tours[:, step] = nxt
            visited[ar, nxt] = True
            cur = nxt
        p = xy[tours]
        lens = np.sqrt(((p - np.roll(p, -1, 1)) ** 2).sum(-1)).sum(1)   # exact, closing edge included
        k = int(np.argmin(lens))                       # the first argmin wins ties
        L = float(lens[k])
        bt = tours[k]
        ibl[it], ibt[it] = L, bt
        if it + 1 >= K_MIN and L < best:
            ants[it + 1] = (tours.copy(), lens.copy())
        best = min(best, L)
        # only the iteration-best ant deposits Q/L, in both directions as in the OpenMP/MPI builds
        delta = np.zeros((n, n))
        np.add.at(delta, (bt[:-1], bt[1:]), Q / L)     # closing edge bt[-1] -> bt[0] is not reinforced
        np.add.at(delta, (bt[1:], bt[:-1]), Q / L)
        tau = np.clip((1 - RHO) * tau + delta, TAU_MIN, TAU_MAX)
        if not np.array_equal(tau, tau.T):
            fail(f"iteration {it + 1}: pheromone matrix not symmetric")
        taus[it + 1] = tau[iu]
    return dict(taus=taus, ibl=ibl, ibt=ibt, ants=ants)


def full(v, n):
    """Expand an upper-triangle vector into a symmetric n x n matrix with a zero diagonal."""
    M = np.zeros((n, n))
    M[np.triu_indices(n, 1)] = v
    return M + M.T


def analyse(xy, run):
    """Pick the slow-motion iteration K and check every claim the captions make (exits on failure).

    Returns K, the hero ant (first argmin at K), the slow-motion matrices TK (before), Te (evaporated),
    Dm (deposit), Tr = Te + Dm and Tc (clamped), the best tour and the numbers that get printed.
    """
    n = len(xy)
    taus, ibl, ibt = run["taus"], run["ibl"], run["ibt"]
    gb = np.minimum.accumulate(ibl)                    # best length after k iterations is gb[k - 1]
    if np.any(np.diff(gb) > 0):
        fail("best-so-far increased")
    if taus.min() < TAU_MIN - EPS or taus.max() > TAU_MAX + EPS:
        fail("pheromone outside [tau_min, tau_max]")
    for t in ibt:
        if t[0] != 0 or not np.array_equal(np.sort(t), np.arange(n)):
            fail("an iteration-best tour is not a permutation starting at city 0")
    # K: the first candidate whose iteration-best tour, before the update, meets the K_MIN_CAP/K_MIN_FLOOR rule
    K = None
    for k in sorted(run["ants"]):
        tours, lens = run["ants"][k]
        bt = tours[int(np.argmin(lens))]
        v = full(taus[k - 1], n)[bt[:-1], bt[1:]]
        cap, flo = int((v >= TAU_MAX - EPS).sum()), int((v <= TAU_MIN + EPS).sum())
        if cap >= K_MIN_CAP and flo >= K_MIN_FLOOR:
            K = k
            break
    if K is None:
        fail("no slow-motion iteration satisfies the rule; try --find-seed")
    tours, lens = run["ants"][K]
    hero = int(np.argmin(lens))
    bt = tours[hero]
    Lh = float(lens[hero])
    # replay iteration K step by step; clamping must give back the recorded state after K
    TK = full(taus[K - 1], n)
    Te = (1 - RHO) * TK
    Dm = np.zeros((n, n))
    np.add.at(Dm, (bt[:-1], bt[1:]), Q / Lh)
    np.add.at(Dm, (bt[1:], bt[:-1]), Q / Lh)
    Tr = Te + Dm
    Tc = np.clip(Tr, TAU_MIN, TAU_MAX)
    if not np.allclose(Tc[np.triu_indices(n, 1)], taus[K], rtol=0, atol=EPS):
        fail("evaporate + deposit + clamp does not reproduce the recorded state")
    if not Lh < gb[K - 2]:
        fail(f"iteration {K} is not a new best overall")
    # "the trails converge on one tour": after the last iteration exactly 50 roads sit at tau_max,
    # all of them on the best tour (closing road included), and every other road at tau_min
    found = int(np.argmax(ibl == ibl.min())) + 1
    best = ibt[found - 1].astype(int)
    T = taus[N_ITERS]
    iu = np.triu_indices(n, 1)
    on = {frozenset(e) for e in zip(best, np.roll(best, -1))}
    capped = [frozenset((int(a), int(b))) for a, b, v in zip(iu[0], iu[1], T) if v >= TAU_MAX - EPS]
    between = int(((T > TAU_MIN + EPS) & (T < TAU_MAX - EPS)).sum())
    if len(capped) != n - 1 or not all(e in on for e in capped) or between:
        fail("the final trail is not exactly the best tour; try --find-seed")
    v = TK[bt[:-1], bt[1:]]                            # K's deposited roads: (at tau_max, at tau_min, between)
    comp = (int((v >= TAU_MAX - EPS).sum()), int((v <= TAU_MIN + EPS).sum()))
    comp = comp + (len(v) - comp[0] - comp[1],)
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    rounded = int(sum(int(d[a, b] + 0.5) for a, b in zip(best, np.roll(best, -1))))   # TSPLIB EUC_2D length
    return dict(gb=gb, K=K, hero=hero, Lh=Lh, ql=Q / Lh, tours=tours, lens=lens, TK=TK, Te=Te, Dm=Dm, Tr=Tr,
                Tc=Tc, found=found, best=best, final=float(gb[-1]), rounded=rounded, comp=comp,
                median=float(np.median(lens)), new_bests=[i + 1 for i in range(N_ITERS) if i == 0 or ibl[i] < gb[i - 1]],
                tmin_after_29=int((taus[29] <= TAU_MIN + EPS).sum()))


def regression_warnings(a, run, seed):
    """Print a warning (never fail) for each seed-9 value that differs from SEED9."""
    if seed != 9:
        return
    got = dict(it1=round(float(run["ibl"][0]), 2), before_k=round(float(a["gb"][a["K"] - 2]), 2), k=a["K"],
               hero=a["hero"], hero_len=round(a["Lh"], 2), q_over_l=round(a["ql"], 4), median=round(a["median"], 2),
               composition=a["comp"], final=round(a["final"], 4), found=a["found"],
               tmin_after_29=a["tmin_after_29"], new_bests=a["new_bests"])
    for key, want in SEED9.items():
        if got[key] != (tuple(want) if isinstance(want, tuple) else want):
            print(f"warning: seed 9 {key} = {got[key]}, expected {want} (trajectory differs on this platform)")


# ---- geometry helpers ----------------------------------------------------------------
def map_px(xy):
    """City coordinates -> px in the map box, same scale on both axes, y pointing down."""
    mx, my, mw, mh = MAP
    lo, hi = xy.min(0) - 2, xy.max(0) + 2
    s = min(mw / (hi[0] - lo[0]), mh / (hi[1] - lo[1]))
    c = (lo + hi) / 2
    return np.stack([mx + mw / 2 + (xy[:, 0] - c[0]) * s, my + mh / 2 - (xy[:, 1] - c[1]) * s], 1)


def closed(P, t):
    """Vertices of tour t closed back to its first city, and the arc length at each vertex."""
    pts = P[np.append(t, t[0])]
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return pts, cum


def point_at(pts, cum, s):
    """Point at arc length s along the polyline pts (cum from closed())."""
    i = min(max(int(np.searchsorted(cum, s, side="right")) - 1, 0), len(cum) - 2)
    return pts[i] + (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-12) * (pts[i + 1] - pts[i])


def sub_path(pts, cum, s0, s1):
    """Segments of the polyline from arc s0 to arc s1, through any vertices in between."""
    poly = [point_at(pts, cum, s0)] + [pts[j] for j in range(len(cum)) if s0 < cum[j] < s1] + [point_at(pts, cum, s1)]
    return [[poly[q], poly[q + 1]] for q in range(len(poly) - 1)]


# ---- frame list ------------------------------------------------------------------------
def build_frames(xy, run, a, seed):
    """Return the frames in display order (dicts of per-frame state) and the fast-forward iterations."""
    n = len(xy)
    iu = np.triu_indices(n, 1)
    P = map_px(xy)
    taus, gb, K = run["taus"], a["gb"], a["K"]
    F = []

    def add(**kw):
        F.append(kw)

    # poster: the state after the last iteration, with the whole best tour
    bpts, bcum = closed(P, a["best"])
    cap_end = ("Best tour after 2,000 iterations:", f"{a['final']:.2f}, first found at iteration {a['found']:,}")
    end = dict(tau=taus[N_ITERS], it=N_ITERS, best=a["final"], spark=N_ITERS, step="loop", cap=cap_end)
    add(seg="poster", tour=sub_path(bpts, bcum, 0, bcum[-1]), delay=D_POSTER, **end)
    # warm-up: recorded states after the iterations in WARM, no tour line
    for j, k in enumerate(WARM):
        add(seg="warm", tau=taus[k], it=k, best=float(gb[k - 1]), spark=k, step="loop", delay=D_WARM0 if j == 0 else D_WARM,
            cap=("Every road starts at τmax = 2. Each", "iteration, all roads lose 10% and the",
                 "iteration's shortest tour adds more"))
    # walk (iteration K): every ant is at the same arc length s, so the shortest tour gets home first
    paths = [closed(P, t) for t in a["tours"]]
    Lpx = np.array([c[-1] for _, c in paths])
    hpts, hcum = paths[a["hero"]]
    hero_tour = sub_path(hpts, hcum, 0, hcum[-1])
    s_of = lambda i: Lpx.max() * (WALK_A * (i / WALK_STEPS) + (1 - WALK_A) * (i / WALK_STEPS) ** 3)
    for i in range(WALK_STEPS + 1):
        s = s_of(i)
        ell = min(max(s - s_of(i - 1), STREAK_MIN), STREAK_MAX) if i else 0.0
        streaks, heads = [], []
        for (pts, cum), L in zip(paths, Lpx):
            if s >= L:
                continue                                       # this ant is home
            if i:
                streaks += sub_path(pts, cum, max(s - ell, 0.0), s)
            heads.append(point_at(pts, cum, s))
        home = s >= Lpx[a["hero"]]                             # from here on: hero tour, its length, new caption
        cap = ((f"Iteration {K}, slowed down: {N_ANTS} ants", "walk at equal speed; short roads with",
                "more pheromone are likelier picks") if not home else
               ("The first ant home has the shortest", f"tour of this iteration: {a['Lh']:.2f},", "also a new best overall"))
        add(seg="walk", tau=taus[K - 1], it=K, best=a["Lh"] if home else float(gb[K - 2]), spark=K if home else K - 1,
            step=1, streaks=streaks, heads=heads, tour=hero_tour if home else [], cap=cap,
            delay=D_WALK_END if i == WALK_STEPS else D_WALK)
    # update steps of iteration K: each f = 0.5 frame is a tween inside its labelled step, readouts frozen
    sl = dict(it=K, best=a["Lh"], spark=K)
    for f, dl in ((0.5, D_TWEEN), (1.0, D_EVAP)):
        add(seg="evap", tau=((1 - RHO * f) * a["TK"])[iu], step=2, tour=hero_tour, delay=dl,
            cap=("Every road keeps 90% of its pheromone", "(ρ = 0.1); the faintest roads drop", "below τmin"), **sl)
    for f, dl in ((0.5, D_TWEEN), (1.0, D_DEP)):
        add(seg="deposit", tau=(a["Te"] + f * a["Dm"])[iu], step=3, tour=hero_tour if f < 1 else [], tour_alpha=1 - f,
            closing=[[hpts[-2], hpts[-1]]] if f == 1 else [], delay=dl,
            cap=(f"That tour adds Q/L = {a['ql']:.2f} to each of", "its roads except the last one back",
                 "to start (dashed)"), **sl)
    for f, dl in ((0.5, D_TWEEN), (1.0, D_CLAMP)):
        add(seg="clamp", tau=(a["Tr"] + f * (a["Tc"] - a["Tr"]))[iu], step=4, delay=dl,
            cap=("Values above τmax are cut to 2 and", "values below τmin raised to 0.1"), **sl)
    # fast-forward: recorded states on a log schedule from K + 1 to N_ITERS
    ks = np.unique(np.round(np.geomspace(K + 1, N_ITERS, FF_N)).astype(int))
    for k in ks:
        add(seg="ff", tau=taus[k], it=int(k), best=float(gb[k - 1]), spark=int(k), step="loop",
            delay=D_FF_END if k == N_ITERS else D_FF,
            cap=(f"Repeat for {N_ITERS:,} iterations. In this run,", "the trails converge on one tour"))
    # trace: the best tour from city 0 up to arc j / TRACE_N; the poster (j = TRACE_N) completes the loop
    for j in range(1, TRACE_N):
        s = bcum[-1] * j / TRACE_N
        add(seg="trace", tour=sub_path(bpts, bcum, 0, s), heads=[point_at(bpts, bcum, s)], head_d=TRACE_HEAD_D,
            delay=D_TRACE, **end)
    return F, ks


# ---- renderer ----------------------------------------------------------------------------
def h2r(h):
    """'#rrggbb' -> RGB floats in [0, 1]."""
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float) / 255


def road_style(tau):
    """Fixed tau -> (RGB, width in px) for roads: with s = (tau - tau_min) / (tau_max - tau_min), colour
    RAMP at s**3 and width 0.5 + 2.1 * s**1.5. Values outside [tau_min, tau_max] (only in the slow-motion
    update steps) are purple: wide above tau_max, a dim tint below tau_min."""
    tau = np.asarray(tau, float)
    s = np.clip((tau - TAU_MIN) / (TAU_MAX - TAU_MIN), 0, 1)
    xs, cs = np.array([p for p, _ in RAMP]), np.array([h2r(c) for _, c in RAMP])
    rgb = np.stack([np.interp(s ** 3, xs, cs[:, i]) for i in range(3)], -1)
    w = W_MIN + W_SPAN * s ** 1.5
    over, under = tau > TAU_MAX + 1e-9, tau < TAU_MIN - 1e-9
    rgb[over] = h2r(C["over"])
    w[over] = W_MIN + W_SPAN + OVER_SPAN * np.clip((tau[over] - TAU_MAX) / 0.8, 0, 1)
    rgb[under] = h2r(C["under"])
    return rgb, w


KLO, KHI = -31, 5                                   # histogram bin k is centred on 2 * 0.9**(-k)


def hist_bins(tau):
    """Road counts per histogram bin KLO..KHI. One evaporation step moves a road one bin left."""
    k = np.round(np.log(np.asarray(tau, float) / TAU_MAX) / np.log(1 / (1 - RHO)))
    return np.bincount(np.clip(k, KLO, KHI).astype(int) - KLO, minlength=KHI - KLO + 1)


class Renderer:
    """One matplotlib figure, its artists created once and updated per frame by draw(). Units are px, y down."""

    def __init__(self, xy, gb, seed):
        n = len(xy)
        self.gb = gb
        plt.rcdefaults()                               # ignore any user matplotlibrc
        plt.rcParams["font.family"] = "DejaVu Sans"
        self.fig = fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=C["bg"])
        ax = self.ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_axis_off()
        P = self.P = map_px(xy)
        iu = np.triu_indices(n, 1)
        self.segs = np.stack([P[iu[0]], P[iu[1]]], 1)
        self.roads = ax.add_collection(LineCollection([], capstyle="round", zorder=1))
        self.tour = ax.add_collection(LineCollection([], colors=C["tour"], linewidths=TOUR_W * PT, capstyle="round",
                                                     joinstyle="round", zorder=3))
        self.closing = ax.add_collection(LineCollection([], colors=C["tour"], linewidths=CLOSE_W * PT,
                                                        linestyles=(0, (3, 3)), zorder=3))
        self.streaks = ax.add_collection(LineCollection([], colors=C["ant"], linewidths=STREAK_W * PT,
                                                        capstyle="round", joinstyle="round", zorder=4))
        self.heads = ax.scatter(np.zeros(0), np.zeros(0), c=C["head"], linewidths=0, zorder=5)
        ax.scatter(P[1:, 0], P[1:, 1], s=(4.6 * PT) ** 2, c=C["city"], edgecolors=C["bg"], linewidths=1.2 * PT, zorder=6)
        ax.scatter(P[:1, 0], P[:1, 1], s=(13 * PT) ** 2, facecolors="none", edgecolors=C["fg"], linewidths=2 * PT, zorder=7)
        ax.scatter(P[:1, 0], P[:1, 1], s=(5 * PT) ** 2, c=C["fg"], linewidths=0, zorder=7)
        T = self.text
        # left of the ring: seed 9's closing road 21 -> 0 leaves city 0 up and to the right
        T(P[0, 0] - 9, P[0, 1] - 7, "start", 13, C["fg"], weight="bold", ha="right")
        T(PX, 38, "Max-Min Ant System", 22, C["fg"], weight="bold")
        T(PX, 59, f"TSPLIB eil51 · {n} cities · {N_ANTS} ants · seed {seed}", 13, C["muted"])
        T(PX, 90, "ITERATION", 12, C["muted"], weight="bold")
        T(PX + 132, 90, "BEST TOUR SO FAR", 12, C["muted"], weight="bold")
        self.v_it = T(PX, 124, "", 32, C["fg"], weight="bold")
        self.v_best = T(PX + 132, 124, "", 32, C["fg"], weight="bold")
        T(PX, 150, "Exact tour length · iterations on a log scale", 12, C["muted"])
        self.sx0, self.sx1, self.sy0, self.sy1 = PX + 4, PX + PW - 8, 158, 192
        ax.plot([self.sx0, self.sx1], [195.5, 195.5], color=C["border"], lw=PT, zorder=9)
        for k, lab in ((1, "1"), (10, "10"), (100, "100"), (1000, "1,000")):
            x = self.kx(k)
            ax.plot([x, x], [195.5, 199.5], color=C["border"], lw=PT, zorder=9)
            T(x, 213, lab, 12, C["muted"], ha="left" if k == 1 else "center")
        self.ylo, self.yhi = gb[-1] - 6, gb[0] + 6
        self.spark = ax.plot([], [], color=C["fg"], lw=1.8 * PT, drawstyle="steps-post", zorder=10)[0]
        self.spark_dot = ax.scatter(np.zeros(0), np.zeros(0), s=(6 * PT) ** 2, c=C["fg"], linewidths=0, zorder=11)
        self.steps = []
        for i, lab in enumerate(("Ants build tours", "Pheromone evaporates", "Iteration-best tour deposits",
                                 "Clamp to [τmin, τmax]")):
            y = 238 + 21 * i
            self.steps.append((T(PX, y, str(i + 1), 15, C["fg"], weight="bold"), T(PX + 18, y, lab, 15, C["fg"])))
        self.cap = [T(PX, 330 + 18 * i, "", 14, C["fg"]) for i in range(3)]
        T(PX, 392, f"Pheromone on all {len(self.segs):,} roads · log scales", 12, C["muted"])
        hx0, hx1, hy0, hy1 = PX, PX + PW - 8, 400, 426
        bw = (hx1 - hx0) / (KHI - KLO + 1)
        wl, wh = hx0 + 3 * bw, hx0 + 32 * bw            # walls at k = -28.5 (tau_min) and k = +0.5 (tau_max)
        ax.add_patch(Rectangle((hx0, hy0), wl - hx0, hy1 - hy0, color=C["zone"], lw=0, zorder=8))
        ax.add_patch(Rectangle((wh, hy0), hx1 - wh, hy1 - hy0, color=C["zone"], lw=0, zorder=8))
        self.bars = [ax.add_patch(Rectangle((hx0 + (b + 0.1) * bw, hy1), 0.8 * bw, 0, lw=0, zorder=9,
                                            color=C["over"] if not (3 <= b <= 31) else C["bar"]))
                     for b in range(KHI - KLO + 1)]
        self.hy1, self.hh = hy1, hy1 - hy0
        # legend strip under the bars: the exact road colour of each pixel column's tau
        xs = np.arange(int(np.floor(wl)), int(np.ceil(wh)))
        kk = (xs + 0.5 - hx0) / bw - 0.5 + KLO
        rgb, _ = road_style(np.clip(TAU_MAX * (1 - RHO) ** (-kk), TAU_MIN, TAU_MAX))
        for x, c in zip(xs, rgb):
            ax.add_patch(Rectangle((x, hy1 + 2), 1.02, 5, color=c, lw=0, zorder=9))
        ax.add_patch(Rectangle((hx0, hy1 + 2), xs[0] - hx0, 5, color=C["under"], lw=0, zorder=9))
        ax.add_patch(Rectangle((xs[-1] + 1, hy1 + 2), hx1 - xs[-1] - 1, 5, color=C["over"], lw=0, zorder=9))
        for x in (wl, wh):
            ax.plot([x, x], [hy0, hy1 + 8], color=C["fg"], lw=1.5 * PT, zorder=10)
        T(wl, 446, "τmin 0.1", 12, C["fg"], ha="center")
        T(wh, 446, "τmax 2", 12, C["fg"], ha="center")
        T(PX, 469, "NumPy re-implementation of the OpenMP/MPI rules", 12, C["muted"])

    def text(self, x, y, s, px, color, weight="normal", ha="left"):
        return self.ax.text(x, y, s, fontsize=px * PT, color=color, fontweight=weight, ha=ha, va="baseline", zorder=10)

    def kx(self, k):
        return self.sx0 + (self.sx1 - self.sx0) * np.log10(k) / np.log10(N_ITERS)

    def ly(self, L):
        return self.sy1 - (self.sy1 - self.sy0) * (np.minimum(L, self.yhi) - self.ylo) / (self.yhi - self.ylo)

    def draw(self, f):
        """Render frame dict f as an (H, W, 3) uint8 array. Every per-frame property is set on every
        call, so the result never depends on the order in which frames are drawn."""
        tau = np.asarray(f["tau"], float)
        order = np.argsort(tau, kind="stable")             # strongest road drawn last, on top
        rgb, w = road_style(tau)
        self.roads.set_segments(self.segs[order])
        self.roads.set_color(rgb[order])
        self.roads.set_linewidth(w[order] * PT)
        self.tour.set_segments(f.get("tour", []))
        self.tour.set_alpha(f.get("tour_alpha", 1.0))
        self.closing.set_segments(f.get("closing", []))
        self.streaks.set_segments(f.get("streaks", []))
        heads = np.asarray(f.get("heads", []), float).reshape(-1, 2)
        self.heads.set_offsets(heads)
        self.heads.set_sizes([(f.get("head_d", HEAD_D) * PT) ** 2])
        self.v_it.set_text(f"{f['it']:,}")
        self.v_best.set_text(f"{f['best']:.2f}")
        k = f["spark"]
        ks = np.arange(1, k + 1)
        self.spark.set_data(self.kx(ks), self.ly(self.gb[:k]))
        self.spark_dot.set_offsets([[self.kx(k), self.ly(self.gb[k - 1])]])
        for i, (dg, lab) in enumerate(self.steps):
            active = f["step"] == i + 1
            col = C["fg"] if f["step"] == "loop" or active else C["muted"]
            dg.set_color(col)
            lab.set_color(col)
            lab.set_fontweight("bold" if active else "normal")
        for t, s in zip(self.cap, list(f["cap"]) + ["", "", ""]):
            t.set_text(s)
        for b, (r, c) in enumerate(zip(self.bars, hist_bins(tau))):
            h = self.hh * np.log1p(c) / np.log1p(len(tau))
            r.set_y(self.hy1 - h)
            r.set_height(h)
        self.fig.canvas.draw()
        img = np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()
        # the 1 px border is written into the pixels, not drawn, so all four edges are exact
        img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = np.round(h2r(C["border"]) * 255)
        return img


# ---- encode + verify -----------------------------------------------------------------------
def encode(imgs, delays, out):
    """Write the GIF with Pillow (delays in cs): one 128-colour palette, no dithering. optimize=True
    stores delta frames whose unchanged pixels use a spare transparent palette index, which is why
    the palette has 128 colours rather than 256."""
    stack = Image.fromarray(np.concatenate(imgs[::4], axis=0))     # every 4th frame, starting with the poster
    pal = stack.quantize(colors=128, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    q = [Image.fromarray(im).quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    out.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(out, format="GIF", save_all=True, append_images=q[1:], duration=[10 * d for d in delays], loop=0,
              disposal=1, optimize=True)
    return pal


def verify(out, imgs, delays, final):
    """Reopen the GIF at out (the .tmp file) and check its size, frame count, loop, total duration,
    per-frame quantization error against the rendered frames, and file size. The size warning names
    final, the path the GIF is then moved to. Returns the numbers."""
    g = Image.open(out)
    if g.size != (W, H) or g.n_frames != len(imgs) or g.info.get("loop") != 0:
        fail(f"{final} not written: size {g.size}, {g.n_frames} frames, loop {g.info.get('loop')}")
    total, worst, worst_share = 0, 0, 0.0
    for i in range(g.n_frames):
        g.seek(i)
        total += g.info["duration"]
        dec = np.asarray(g.convert("RGB")).astype(int)
        err = np.abs(dec - imgs[i].astype(int)).max(2)
        worst, worst_share = max(worst, int(err.max())), max(worst_share, float((err > 16).mean()))
    if total != 10 * sum(delays):
        fail(f"total duration {total} ms, expected {10 * sum(delays)} ms")
    if worst > 64 or worst_share > 0.015:
        fail(f"quantization error too large: max {worst}, {worst_share:.2%} of pixels off by more than 16")
    size = out.stat().st_size
    if size > 3_000_000:
        fail(f"{final} not written: the GIF is {size:,} bytes (cap 3,000,000)")
    if size > 2_600_000:
        print(f"warning: {final} is {size:,} bytes (target 2,600,000 or less)")
    return size, total, worst, worst_share


def main():
    """Simulate (or load the cache), check, render, encode, verify and report."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=9, metavar="INT", help="seed for np.random.default_rng (default: 9)")
    ap.add_argument("--out", type=Path, default=REPO / "output_images" / "mmas_eil51.gif", metavar="PATH",
                    help="GIF to write (default: output_images/mmas_eil51.gif in this repository)")
    ap.add_argument("--cache", type=Path, metavar="PATH.npz",
                    help="load the simulation from this .npz if it was saved with the same seed, otherwise "
                         "simulate and save it there (about 11 MB); the GIF is the same either way")
    ap.add_argument("--frames", type=Path, metavar="DIR",
                    help="also write every rendered frame as DIR/f000.png, f001.png, ... for inspection")
    ap.add_argument("--find-seed", action="store_true",
                    help="if the checks fail for --seed, print why and try the next seed (100 seeds at most), "
                         "then render the first seed that passes")
    args = ap.parse_args()
    t0 = time.time()
    xy = load_tsp(TSP)
    seed = args.seed
    while True:
        run = None
        if args.cache and args.cache.exists():                # reuse a saved run only if its seed matches
            z = np.load(args.cache, allow_pickle=False)
            if int(z["seed"]) == seed:
                ants = {int(k): (z["ant_tours"][i], z["ant_lens"][i]) for i, k in enumerate(z["ant_its"])}
                run = dict(taus=z["taus"], ibl=z["ibl"], ibt=z["ibt"], ants=ants)
        if run is None:
            run = simulate(xy, seed)
            if args.cache:                                     # a file handle keeps np.savez from appending .npz
                its = sorted(run["ants"])
                args.cache.parent.mkdir(parents=True, exist_ok=True)
                with open(args.cache, "wb") as fh:
                    np.savez(fh, seed=seed, taus=run["taus"], ibl=run["ibl"], ibt=run["ibt"], ant_its=np.array(its),
                             ant_tours=np.array([run["ants"][k][0] for k in its]),
                             ant_lens=np.array([run["ants"][k][1] for k in its]))
        if not args.find_seed:
            break
        try:                                                   # a failed check exits through fail()
            analyse(xy, run)
            break
        except SystemExit as e:                                # print the bare reason
            print(f"seed {seed}: {str(e).removeprefix('animate_aco.py: FAILED: ').removesuffix('; try --find-seed')}")
            seed += 1
            if seed >= args.seed + 100:                        # tried args.seed .. args.seed + 99
                fail("no seed passed within 100 tries")
    t_sim = time.time() - t0
    a = analyse(xy, run)
    regression_warnings(a, run, seed)
    F, ks = build_frames(xy, run, a, seed)
    R = Renderer(xy, a["gb"], seed)
    imgs = [R.draw(f) for f in F]
    t_render = time.time() - t0 - t_sim
    if args.frames:
        args.frames.mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(imgs):
            Image.fromarray(im).save(args.frames / f"f{i:03d}.png")
    delays = [f["delay"] for f in F]
    tmp = args.out.with_name(args.out.name + ".tmp")
    try:
        encode(imgs, delays, tmp)
        size, total, worst, share = verify(tmp, imgs, delays, args.out)
        os.replace(tmp, args.out)                              # a failed check never replaces the committed GIF
    except BaseException as e:                                 # fail() raises SystemExit; no .tmp is left behind
        tmp.unlink(missing_ok=True)
        if isinstance(e, Exception):
            fail(f"writing {args.out}: {type(e).__name__}: {e}")
        raise
    sha = hashlib.sha1(args.out.read_bytes()).hexdigest()
    if seed != 9:
        print(f"note: rendered with seed {seed}; update 'seed 9' in README.md")
    print(f"seed {seed}: iteration {a['K']} slowed down; best of its {N_ANTS} tours {a['Lh']:.2f} (ant {a['hero']}, "
          f"Q/L {a['ql']:.4f}); final best {a['final']:.2f} (TSPLIB-rounded {a['rounded']}) first found at iteration "
          f"{a['found']:,}; trail at {N_ITERS:,} = best tour")
    print(f"fast-forward iterations: {', '.join(map(str, ks))}")
    print(f"{args.out}: {size:,} bytes, {len(F)} frames, {total / 1000:.2f} s, loops forever, sha1 {sha}")
    print(f"quantization: max error {worst}, {share:.2%} of pixels off by more than 16 (worst frame)")
    print(f"time: simulation {t_sim:.1f} s, render {t_render:.1f} s, total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
