"""RECONSTRUCTION of brickwall/quantum_utils.py (the file was NOT supplied with the notebook).

Rebuilt from the paper (Scalable_Quantum_ML.pdf, Sec. II.A-C, III.D, V) and the notebook's call signatures.
Everything here is our reading of the paper, NOT Antonio's code; conventions are listed in BRICKWALL.md
("Questions for Antonio").

  particle_count(n, k)       k=None -> "scalable default": largest even k with 4k <= n... see below (GUESS; n=8 -> 2 as in the
                             notebook output "features=8, k=2").  Explicit k must be even with 2k <= n (k/2 magic blocks of 4 qubits
                             need 4*(k/2) = 2k <= n qubits).
  spreading_matrix(n)        n x n single-particle unitary W of the fixed Hadamard spreading U_had: Hadamard brick-wall of depth n-1,
                             chain-pair RBS angles pi/4, boundary pair (n-1,0) and all Rz = 0 (paper Sec. III.D).  For n a power of 2
                             this is the normalised Walsh-Hadamard transform (|W_ij|^2 = 1/n) up to signs/permutation; for other n
                             (e.g. 12) the brick-wall is still defined but may NOT have uniform moduli -> see uniformity_defect().
  UniformSpreading(W, wires) the many-body (Jordan-Wigner, passive fermionic-linear-optics) action of W on the Fock space; here
                             provided as a plain function many_body(W, state_dict) / phi0_sector().
  initial_weights(n, columns, rng)  -> (w, phases, extra): w ~ small normal, shape (columns, 3n/2); first n/2 entries per column are
                             RBS angles, last n are Rz angles (as indexed in make_circuit).  `phases` = zeros(n) data-phase offsets.
  shift_coefficients(k)      parallel parameter-shift rule coefficients (Sec. V): RBS in the k-particle sector has generator
                             eigenvalue differences {+-1/2 ... } -> frequencies 1..k/..; reconstructed as the equidistant-frequency
                             rule for frequencies 1..k (shifts x_m = (2m-1) pi / (4k)).  Only needed to retrain with PSR; NOT used
                             by our autograd training.
"""
from __future__ import annotations
import itertools
import numpy as np
from math import comb


def particle_count(n, k=None):
    if k is None:
        k = max(2, 2 * (n // 8))          # GUESS: n=8 -> 2 (matches notebook output), n=12 -> 2, n=16 -> 4
        k = min(k, 60)
    if k % 2 or k < 2 or 2 * k > n:
        raise ValueError(f"k={k} must be even, >=2, and 2k<=n={n}")
    return k


def pairs(n, col):
    out = [(j, j + 1) for j in range(col % 2, n - 1, 2)]
    return out + ([(n - 1, 0)] if col % 2 else [])


def givens_single_particle(n, a, b, phi):
    """Single-particle matrix of SingleExcitation(phi) on qubits (a,b): |01> -> cos(phi/2)|01> + sin(phi/2)|10>.
    Single-particle basis |e_j> = particle on qubit j.  'qubit a occupied' = e_a.  |10> = e_a, |01> = e_b."""
    c, s = np.cos(phi / 2), np.sin(phi / 2)
    G = np.eye(n)
    # e_b -> c e_b + s e_a ; e_a -> -s e_b + c e_a
    G[b, b], G[a, b], G[b, a], G[a, a] = c, s, -s, c
    return G


def hadamard_brickwall_matrix(n):
    """Single-particle matrix of the paper's 'Hadamard brick-wall' as literally described (n-1 columns of chain-pair RBS at pi/4).
    FINDING: this is orthogonal but NOT a uniform-modulus matrix (see uniformity_defect), so it does not satisfy Lemma 2."""
    W = np.eye(n)
    for col in range(n - 1):
        for (a, b) in [(j, j + 1) for j in range(col % 2, n - 1, 2)]:
            W = givens_single_particle(n, a, b, np.pi / 2) @ W
    return W


def _paley12():
    q = 11
    sq = {(i * i) % q for i in range(1, q)}
    chi = lambda a: 0 if a % q == 0 else (1 if a % q in sq else -1)
    Q = np.array([[chi(i - j) for j in range(q)] for i in range(q)], float)
    S = np.zeros((12, 12)); S[0, 1:] = 1; S[1:, 0] = -1; S[1:, 1:] = Q
    return np.eye(12) + S


def spreading_matrix(n):
    """Uniform-modulus single-particle unitary used for U_had: normalised Walsh-Hadamard for n=2^m (paper Eq. 15,
    H[i,j]=(-1)^<i,j>/sqrt n); normalised real Hadamard matrix of order 12 (Paley) for n=12; normalised DFT otherwise.
    Exactly |W_ij|^2 = 1/n, so Lemma 2 holds.  (Deviation: the paper realises WHT by a fixed-angle brick-wall; the literal
    chain-pair pi/4 brick-wall we tried is not uniform, see hadamard_brickwall_matrix.)"""
    if n & (n - 1) == 0:
        H = np.array([[(-1) ** bin(i & j).count("1") for j in range(n)] for i in range(n)], float)
        return H / np.sqrt(n)
    if n == 12:
        return _paley12() / np.sqrt(12)
    return np.fft.fft(np.eye(n)) / np.sqrt(n)


def uniformity_defect(W):
    n = len(W)
    return float(np.abs(np.abs(W) ** 2 - 1.0 / n).max())


def initial_weights(n, columns, rng):
    return rng.normal(0, 0.1, (columns, 3 * n // 2)), np.zeros(n), None


def shift_coefficients(k):
    m = np.arange(1, k + 1)
    shifts = (2 * m - 1) * np.pi / (4 * k)
    return shifts, np.ones(k) / k      # placeholder weights, see module docstring (not used for training)


# ---------------------------------------------------------------------------------------------------------------------
# particle-number sector helpers (all wires indexed 0..n-1; occupation matrix occ[D, n])
# ---------------------------------------------------------------------------------------------------------------------
def sector(n, k):
    occ = np.array([[1 if j in c else 0 for j in range(n)] for c in itertools.combinations(range(n), k)], dtype=np.int8)
    return occ


def sector_index(occ):
    return {tuple(r): i for i, r in enumerate(occ)}


def magic_input(n, k):
    """|Psi_in>: k/2 blocks on qubits (4g..4g+3), each (|1100> + |0011>)/sqrt2 (X a, X b, H c, CNOT c->d, c->a, c->b)."""
    comps = []
    for choice in itertools.product([0, 1], repeat=k // 2):
        occ = [0] * n
        for g, ch in enumerate(choice):
            for q in ((4 * g, 4 * g + 1) if ch == 0 else (4 * g + 2, 4 * g + 3)):
                occ[q] = 1
        comps.append(tuple(occ))
    return comps, 1 / np.sqrt(2 ** (k // 2))


def phi0_sector(n, k, W=None):
    """|phi0> = U_had |Psi_in> in the k-particle sector, amplitude of Fock state T = sum_S psi_in[S] det W[T,S]
    (Jordan-Wigner, ascending-order creation operators; checked against explicit gate circuits in p2_verify.py)."""
    if W is None:
        W = spreading_matrix(n)
    occ = sector(n, k)
    comps, amp = magic_input(n, k)
    cols = [np.flatnonzero(c) for c in comps]
    out = np.zeros(len(occ), dtype=complex)
    rows = [np.flatnonzero(r) for r in occ]
    for ti, T in enumerate(rows):
        for S in cols:
            out[ti] += amp * np.linalg.det(W[np.ix_(T, S)])
    return occ, out


class Sector:
    """numpy reference simulator in the k-particle sector (used for verification; training uses bw_torch)."""
    def __init__(self, n, columns, k=None, W=None):
        self.n, self.columns, self.k = n, columns, particle_count(n, k)
        self.occ, self.phi0 = phi0_sector(n, self.k, W)
        self.D = len(self.occ)
        idx = sector_index(self.occ)
        self.pair_idx = {}
        for col in range(columns):
            for (a, b) in pairs(n, col):
                if (a, b) in self.pair_idx:
                    continue
                i01, i10 = [], []
                for i, r in enumerate(self.occ):
                    if r[a] == 0 and r[b] == 1:
                        r2 = r.copy(); r2[a], r2[b] = 1, 0
                        i01.append(i); i10.append(idx[tuple(r2)])
                self.pair_idx[(a, b)] = (np.array(i01), np.array(i10))
        self.M = (self.occ * np.roll(self.occ, -1, axis=1)).astype(float)       # m_j = <n_j n_{j+1}>

    def state(self, x, w):
        """x: (n,), w: (columns, 3n/2) -> state in sector (complex), INCLUDING global phase (matches PennyLane exactly)."""
        n = self.n
        theta = -np.pi * x
        psi = self.phi0 * np.exp(-1j * theta.sum() / 2 + 1j * (self.occ @ theta))
        for col in range(self.columns):
            th = -w[col, n // 2:]
            psi = psi * np.exp(-1j * th.sum() / 2 + 1j * (self.occ @ th))
            for j, (a, b) in enumerate(pairs(n, col)):
                phi = -2 * w[col, j]
                c, s = np.cos(phi / 2), np.sin(phi / 2)
                i01, i10 = self.pair_idx[(a, b)]
                p01, p10 = psi[i01].copy(), psi[i10].copy()
                psi[i01] = c * p01 - s * p10
                psi[i10] = s * p01 + c * p10
        return psi

    def features(self, x, w):
        p = np.abs(self.state(x, w)) ** 2
        return 4 * (p @ self.M)
