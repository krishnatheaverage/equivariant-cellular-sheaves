"""
E5  Verify the H^1 / Hodge 1-Laplacian claims for Section 4.4.

Checks, by explicit construction of the coboundary maps delta^0, delta^1:
  (a) trivial scalar sheaf on a bare n-cycle:        dim H^1 = b_1 = 1
  (b) Z/2-frustrated (Mobius) scalar sheaf on cycle: holonomy = -1,
                                                     dim H^0 = dim H^1 = 0
  (c) filled hexagon (benzene complex), trivial:     dim H^1 = 0 (cycle bounded)
  (d) Hodge theorem cross-check: dim ker L_1 == dim H^1 in every case.
"""
import numpy as np

def nullity(M, tol=1e-9):
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s < tol)) + max(0, M.shape[1] - len(s))

def rank(M, tol=1e-9):
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol))

def cohomology(d0, d1, nV, nE, nF):
    """dims of H^0, H^1, H^2 from the cochain complex C^0 -d0-> C^1 -d1-> C^2."""
    # sanity: it must be a complex
    if d1.size and d0.size:
        assert np.allclose(d1 @ d0, 0, atol=1e-9), "delta^1 . delta^0 != 0"
    H0 = nullity(d0) if d0.size else nV
    rk0 = rank(d0) if d0.size else 0
    rk1 = rank(d1) if d1.size else 0
    H1 = (nE - rk1) - rk0          # dim ker d1 - dim im d0
    H2 = nF - rk1                  # coker d1
    # Hodge cross-check
    L1 = (d0 @ d0.T) + (d1.T @ d1)
    H1_hodge = nullity(L1)
    return H0, H1, H2, H1_hodge

def scalar_cycle(n, flip_edges=()):
    """delta^0 for a scalar sheaf on the n-cycle; flip_edges -> restriction -1."""
    edges = [(i, (i + 1) % n) for i in range(n)]
    d0 = np.zeros((n, n))                       # (nE x nV)
    transports = []
    for k, (i, j) in enumerate(edges):
        r_tail, r_head = 1.0, 1.0
        if k in flip_edges:
            r_head = -1.0                       # break the restriction map sign
        d0[k, i] += -1.0 * r_tail               # [tail:e] = -1
        d0[k, j] += +1.0 * r_head               # [head:e] = +1
        transports.append((r_head**-1) * r_tail)  # T_e : F(tail)->F(head)
    holonomy = float(np.prod(transports))
    return d0, edges, holonomy

print("=" * 64)
# (a) bare hexagon, trivial sheaf
d0, edges, hol = scalar_cycle(6)
nE = len(edges)
H0, H1, H2, H1h = cohomology(d0, np.zeros((0, nE)), 6, nE, 0)
print(f"(a) C6 trivial      holonomy={hol:+.0f}  H0={H0} H1={H1}  (kerL1={H1h})")
assert (H0, H1) == (1, 1) and H1h == 1

# (b) Mobius hexagon: one sign flip -> Z/2 holonomy
d0, edges, hol = scalar_cycle(6, flip_edges=(0,))
H0, H1, H2, H1h = cohomology(d0, np.zeros((0, nE)), 6, nE, 0)
print(f"(b) C6 Mobius       holonomy={hol:+.0f}  H0={H0} H1={H1}  (kerL1={H1h})")
assert (H0, H1) == (0, 0) and H1h == 0

# (c) filled hexagon (one 2-cell on all six edges), trivial sheaf
d0, edges, hol = scalar_cycle(6)
d1 = np.ones((1, 6))                            # boundary of the face = sum of edges
H0, H1, H2, H1h = cohomology(d0, d1, 6, 6, 1)
print(f"(c) C6 + 2-cell     holonomy={hol:+.0f}  H0={H0} H1={H1} H2={H2}  (kerL1={H1h})")
assert (H0, H1, H2) == (1, 0, 0) and H1h == 0

# (d) larger ring sweep: trivial bare cycle always gives b_1 = 1
for n in (4, 5, 8, 10):
    d0, edges, hol = scalar_cycle(n)
    H0, H1, _, H1h = cohomology(d0, np.zeros((0, n)), n, n, 0)
    assert (H0, H1, H1h) == (1, 1, 1)
print("(d) C_n trivial (n=4,5,8,10): dim H^1 = b_1 = 1 in all cases  [OK]")
print("=" * 64)
print("ALL E5 CHECKS PASSED")
