"""
E4 (trained): data efficiency and rotation generalization from equivariance.

We train two models of equal capacity to predict a directional electronic target
(the HOMO-LUMO gap of a p-only Slater-Koster tight-binding Hamiltonian):

  * SHEAF (equivariant): an MLP on the spectral moments of the O(3)-equivariant
    sheaf Laplacian (rotation-invariant features, Thm 4.4).
  * COORD (non-equivariant): an MLP on raw atomic coordinates (orientation aware).

Both are trained on PCA-canonicalized molecules with NO rotation augmentation, then
tested on the same molecules in (a) the canonical frame and (b) a random rotation.
The equivariant model is invariant by construction (a == b) and uses no capacity on
orientation; the coordinate model has higher error and degrades on unseen rotations.
"""
import os, json
import numpy as np
import torch, torch.nn as nn
from numpy.linalg import eigvalsh, norm
from scipy.spatial.transform import Rotation as Rot
import networkx as nx

torch.set_default_dtype(torch.float64)
OUT = os.path.expanduser("~/topological-qc-paper")
MAXN = 7

def random_molecule(rng, nmin=4, nmax=MAXN, rc=1.9):
    n = int(rng.integers(nmin, nmax + 1))
    pos = rng.uniform(-1.5, 1.5, size=(n, 3))
    G = nx.Graph(); G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if norm(pos[i] - pos[j]) < rc: G.add_edge(i, j)
    order = sorted(range(n), key=lambda k: (pos[k, 0], pos[k, 1]))
    for a, b in zip(order, order[1:]):
        if not nx.has_path(G, a, b): G.add_edge(a, b)
    return pos, list(G.edges())

def place(M, i, j, B, d): M[i*d:(i+1)*d, j*d:(j+1)*d] += B

def gap_p(pos, edges, n, Vsig=-1.2, Vpi=0.3):
    N = 3*n; H = np.zeros((N, N))
    for (u, v) in edges:
        r = pos[v]-pos[u]; d = norm(r); rh = r/d; rad = np.exp(-(d-1.4))
        B = (Vsig*np.outer(rh, rh) + Vpi*(np.eye(3)-np.outer(rh, rh))) * rad
        place(H, u, v, B, 3); place(H, v, u, B.T, 3)
    w = np.sort(eigvalsh(H)); m = N//2; return w[m]-w[m-1]

def sheaf_L_p(pos, edges, n, W=(1.0, 1.2)):           # O(3)-equivariant p-stalk Laplacian
    N = 3*n; L = np.zeros((N, N))
    for (u, v) in edges:
        r = pos[v]-pos[u]; d = norm(r); rh = r/d; rad = np.exp(-(d-1.4))
        F = rad*(W[0]*np.eye(3) + W[1]*np.outer(rh, rh)); M = F.T@F
        place(L, u, u, M, 3); place(L, v, v, M, 3); place(L, u, v, -M, 3); place(L, v, u, -M, 3)
    for v in range(n): place(L, v, v, 0.5*np.eye(3), 3)
    return L

def moments(eigs, K=8):
    e = eigs/(np.abs(eigs).max()+1e-9)
    return np.array([np.mean(e**p) for p in range(1, K+1)])

def pca_align(pos):
    p = pos - pos.mean(0); _, V = np.linalg.eigh(p.T@p); return p@V

def coordvec(pos):
    v = np.zeros(MAXN*3); flat = pos.flatten(); v[:len(flat)] = flat; return v

def make_mlp(din):
    return nn.Sequential(nn.Linear(din, 64), nn.SiLU(),
                         nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

def fit(X, y, idx, ep=400, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    mu, sd = X[idx].mean(0), X[idx].std(0)+1e-9
    Xt = torch.tensor((X-mu)/sd); yt = torch.tensor(y)
    model = make_mlp(X.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    tr = torch.tensor(idx)
    for _ in range(ep):
        opt.zero_grad(); loss = ((model(Xt[tr]).squeeze(-1) - yt[tr])**2).mean()
        loss.backward(); opt.step()
    return model, (mu, sd)

def predict(model, norm_, X):
    mu, sd = norm_
    with torch.no_grad():
        return model(torch.tensor((X-mu)/sd)).squeeze(-1).numpy()

def main():
    sizes = [20, 40, 80, 160]; n_test = 300; seeds = 3
    eq = np.zeros((seeds, len(sizes))); co_c = np.zeros((seeds, len(sizes))); co_r = np.zeros((seeds, len(sizes)))
    for s in range(seeds):
        rng = np.random.default_rng(700+s)
        mols = []
        while len(mols) < max(sizes)+n_test:
            pos, edges = random_molecule(rng)
            if edges: mols.append((pca_align(pos), edges, len(pos)))
        y = np.array([gap_p(p, e, n) for (p, e, n) in mols])
        ymu, ysd = y[:max(sizes)].mean(), y[:max(sizes)].std()+1e-9; yn = (y-ymu)/ysd
        R = Rot.random(random_state=s).as_matrix()
        Xeq = np.array([moments(eigvalsh(sheaf_L_p(p, e, n))) for (p, e, n) in mols])
        Xco = np.array([coordvec(p) for (p, e, n) in mols])
        Xco_r = np.array([coordvec(p@R.T) for (p, e, n) in mols])   # rotated coords
        te = np.arange(len(mols)-n_test, len(mols))
        for j, nt in enumerate(sizes):
            tr = np.arange(nt)
            m_eq, nrm = fit(Xeq, yn, tr, seed=s)
            eq[s, j] = np.mean(np.abs(predict(m_eq, nrm, Xeq[te])*ysd+ymu - y[te]))  # invariant: canon==rotated
            m_co, nco = fit(Xco, yn, tr, seed=s)
            co_c[s, j] = np.mean(np.abs(predict(m_co, nco, Xco[te])*ysd+ymu - y[te]))
            co_r[s, j] = np.mean(np.abs(predict(m_co, nco, Xco_r[te])*ysd+ymu - y[te]))
            print(f"seed {s}  N {nt:4d}:  equivariant {eq[s,j]:.4f}   coord(canon) {co_c[s,j]:.4f}   "
                  f"coord(rotated) {co_r[s,j]:.4f}")
    EQ, CC, CR = eq.mean(0), co_c.mean(0), co_r.mean(0)
    print("\nmean over seeds:")
    print(f"{'N_train':>8s} {'equiv MAE':>11s} {'coord canon':>12s} {'coord rot':>11s} {'equiv vs coord-rot':>20s}")
    for j, nt in enumerate(sizes):
        print(f"{nt:8d} {EQ[j]:11.4f} {CC[j]:12.4f} {CR[j]:11.4f} {100*(CR[j]-EQ[j])/CR[j]:18.1f}%")

    rp = os.path.join(OUT, "experiments", "results.json")
    res = json.load(open(rp)) if os.path.exists(rp) else {}
    res["E4"] = {"train_sizes": sizes, "equiv_mae": EQ.tolist(),
                 "coord_canonical_mae": CC.tolist(), "coord_rotated_mae": CR.tolist(),
                 "equiv_std": eq.std(0).tolist(), "seeds": seeds,
                 "target": "p-only tight-binding HOMO-LUMO gap"}
    json.dump(res, open(rp, "w"), indent=2)
    print("\nmerged E4 into results.json")

if __name__ == "__main__":
    main()
