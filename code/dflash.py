import time
from types import SimpleNamespace


# ==== DOMINO_TREE: static tree-48 verification (injected) ====
import os as _tree_os

def _tree_topology(device):
    import torch as _t
    _w = _tree_os.environ.get("DOMINO_TREE_WIDTHS", "")
    widths = [int(x) for x in _w.split(",")] if _w else [8, 6, 5, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]
    par, depth, rank = [-1], [0], [0]
    level_nodes = {0: [0]}
    nid = 1
    for d, w in enumerate(widths, start=1):
        prev = level_nodes[d - 1]
        cur = []
        for j in range(w):
            par.append(prev[j % len(prev)]); depth.append(d); rank.append(j)
            cur.append(nid); nid += 1
        level_nodes[d] = cur
    par_t = _t.tensor(par, device=device)
    depth_t = _t.tensor(depth, device=device)
    rank_t = _t.tensor(rank, device=device)
    n = len(par)
    anc = _t.zeros(n, n, dtype=_t.bool, device=device)
    for i in range(n):
        anc[i, i] = True
        p = par[i]
        while p >= 0:
            anc[i, p] = True
            p = par[p]
    md = max(depth)
    paths = _t.zeros(n, md + 1, dtype=_t.long, device=device)
    plen = _t.zeros(n, dtype=_t.long, device=device)
    for i in range(n):
        chain = [i]
        p = par[i]
        while p >= 0:
            chain.append(p)
            p = par[p]
        chain = chain[::-1]
        plen[i] = len(chain)
        for j, c in enumerate(chain):
            paths[i, j] = c
    return par_t, depth_t, rank_t, anc, md, paths, plen

_TREE_CACHE = {}

def _tree_get(device):
    key = str(device)
    if key not in _TREE_CACHE:
        _TREE_CACHE[key] = _tree_topology(device)
    return _TREE_CACHE[key]

def _tree_cache_gather(cache, base_len, path_idx):
    m = path_idx.shape[0]
    sel = base_len + path_idx
    if type(cache).__name__ == "StaticCache":
        for lyr in cache.layers:
            lyr.keys[:, :, base_len : base_len + m] = lyr.keys[:, :, sel]
            lyr.values[:, :, base_len : base_len + m] = lyr.values[:, :, sel]
        return
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for lyr in layers:
            lyr.keys[:, :, base_len : base_len + m] = lyr.keys[:, :, sel]
            lyr.values[:, :, base_len : base_len + m] = lyr.values[:, :, sel]
    else:
        for l in range(len(cache.key_cache)):
            cache.key_cache[l][:, :, base_len : base_len + m] = cache.key_cache[l][:, :, sel]
            cache.value_cache[l][:, :, base_len : base_len + m] = cache.value_cache[l][:, :, sel]
    cache.crop(base_len + m)
_TREE_SIDE_STREAM = None
_TREE_GATHER_EVT = None
_TREE_MASK_BUF = None
_TREE_ANC_BLOCK = None
_TREE_MASK_PREV = -1


def _true_tree_build(base_logits, hiddens, anchor_id, embed_fn, gru, proj, budget, topk, temperature=0.0):
    """best-first 真樹構建。
    base_logits: [L, V] (該 verify 塊的草稿 base), hiddens: [L, H], anchor_id: () long。
    返回 node_tokens[n], par[n], depth[n], anc[n,n](bool, 含自身), paths[n,maxd+1], plen[n]。
    node0 = anchor(depth0)。depth1 用 base_logits[0] 純 base(官方 prefix 語義),
    depth d>=2 用 base_logits[d-1] + proj([hiddens[d-1], h_parent])。"""
    import torch as _t
    device = base_logits.device
    L = base_logits.shape[0]
    max_depth = L  # 樹深至多 L(depth1..L 消耗 base_logits[0..L-1])

    anchor_emb = embed_fn(anchor_id.view(1, 1))            # [1,1,E]
    _, h_root = gru(anchor_emb)                            # [1,1,1024]
    h_root = h_root[:, 0, :]                               # [1,1024]

    toks = [int(anchor_id)]
    pars = [-1]
    deps = [0]
    scores = [0.0]
    h_list = [h_root[0]]                                   # 每節點 GRU 態 [1024]

    frontier = [0]
    # TREE_V2_FIXED: 無節點數早停(Codex#5); 層寬由 keep=budget 限制, 最終全局截斷
    for d in range(1, max_depth + 1):
        if not frontier:
            break
        F = len(frontier)
        if d == 1:
            dist = base_logits[0].unsqueeze(0).expand(F, -1)          # 純 base
        else:
            h_par = _t.stack([h_list[i] for i in frontier], dim=0)     # [F,1024]
            z = hiddens[d - 1].unsqueeze(0).expand(F, -1)              # [F,H]
            bias = proj(_t.cat([z, h_par], dim=-1))                    # [F,V]
            dist = base_logits[d - 1].unsqueeze(0) + bias
        _dist32 = dist.float()
        if temperature and temperature > 0:
            _dist32 = _dist32 / temperature
        lp = _t.log_softmax(_dist32, dim=-1)
        k = min(topk, lp.shape[-1])
        top_lp, top_id = lp.topk(k, dim=-1)                            # [F,k]
        par_score = _t.tensor([scores[i] for i in frontier], device=device)
        cand_score = par_score.unsqueeze(1) + top_lp                   # [F,k]
        flat = cand_score.view(-1)
        keep = min(budget, flat.shape[0])
        sel_score, sel_idx = flat.topk(keep)
        sel_f = (sel_idx // k).tolist()
        sel_tok = top_id.view(-1)[sel_idx].tolist()
        sel_sc = sel_score.tolist()
        # 該層節點的 GRU 態一次批量步進
        new_ids = []
        h_par_sel = _t.stack([h_list[frontier[f]] for f in sel_f], dim=0)  # [S,1024]
        if d < max_depth:
            emb_in = embed_fn(_t.tensor(sel_tok, device=device).view(-1, 1))   # [S,1,E]
            _, h_new = gru(emb_in, h_par_sel.unsqueeze(0))                     # [1,S,1024]
            h_new = h_new[0]
        else:
            h_new = h_par_sel  # Codex#4: 末層節點不再作父, 佔位保持索引對齊
        for j in range(len(sel_tok)):
            toks.append(sel_tok[j])
            pars.append(frontier[sel_f[j]])
            deps.append(d)
            scores.append(float(sel_sc[j]))
            h_list.append(h_new[j])
            new_ids.append(len(toks) - 1)
        frontier = new_ids

    # 全局 best-first: 非 root 節點按 score 取 top-budget, score 單調故父閉合; 並列邊界補閉合
    import math as _m
    assert all(_m.isfinite(s) for s in scores), "TREE_V2: non-finite score"  # Codex#10
    order = sorted(range(1, len(toks)), key=lambda i: -scores[i])
    chosen = set(order[:budget])
    changed = True
    while changed:
        changed = False
        for i in list(chosen):
            pp = pars[i]
            if pp > 0 and pp not in chosen:
                chosen.add(pp)
                changed = True
    while len(chosen) > budget:
        worst = min((i for i in chosen if not any(pars[j] == i for j in chosen)),
                    key=lambda i: scores[i])
        chosen.remove(worst)
    keep_ids = [0] + sorted(chosen, key=lambda i: (deps[i], -scores[i]))
    remap = {old: new for new, old in enumerate(keep_ids)}
    n = len(keep_ids)
    node_tokens = _t.tensor([toks[i] for i in keep_ids], dtype=_t.long, device=device)
    par = _t.tensor([remap[pars[i]] if pars[i] >= 0 else -1 for i in keep_ids],
                    dtype=_t.long, device=device)
    depth = _t.tensor([deps[i] for i in keep_ids], dtype=_t.long, device=device)
    md = int(depth.max())
    paths = _t.zeros(n, md + 1, dtype=_t.long, device=device)
    plen = _t.zeros(n, dtype=_t.long, device=device)
    anc = _t.zeros(n, n, dtype=_t.bool, device=device)
    par_l = par.tolist()
    for i in range(n):
        chain = [i]
        while par_l[chain[-1]] >= 0:
            chain.append(par_l[chain[-1]])
        chain.reverse()
        plen[i] = len(chain)
        paths[i, : len(chain)] = _t.tensor(chain, device=device)
        anc[i, chain] = True
    return node_tokens, par, depth, anc, paths, plen




def _true_tree_build_fast(base_logits, hiddens, anchor_id, embed_fn, gru, proj, budget, topk, temperature=0.0, lite_k0=0):
    """向量化構樹: 語義 == _true_tree_build(等價測試保證)。主循環零 host 同步。"""
    import torch as _t
    device = base_logits.device
    L = base_logits.shape[0]
    gh = gru.hidden_size
    cap = 1 + L * budget + 8
    tok = _t.zeros(cap, dtype=_t.long, device=device)
    par = _t.full((cap,), -1, dtype=_t.long, device=device)
    dep = _t.zeros(cap, dtype=_t.long, device=device)
    score = _t.zeros(cap, dtype=_t.float32, device=device)
    hbuf = _t.zeros(cap, gh, dtype=next(gru.parameters()).dtype, device=device)

    _lite = int(lite_k0) if lite_k0 else 0
    if _lite:
        _pv, _pi = base_logits.float().topk(_lite, dim=-1)      # [L,K0] 基座預截斷
        _w_in = proj[0]                                          # Linear(hidden+gh -> mid)
        _act = proj[1]
        _w_out = proj[2].weight                                  # [V, mid]
        _sub_w = _w_out[_pi.reshape(-1)].reshape(L, _lite, -1)   # [L,K0,mid]
    import os as _os_cell
    _cell = _os_cell.environ.get("TREE_V2_CELL", "1") == "1" and gru.num_layers == 1
    if _cell:
        _Wih = gru.weight_ih_l0    # [3H,E]
        _Whh = gru.weight_hh_l0    # [3H,H]
        _bih = getattr(gru, "bias_ih_l0", None)
        _bhh = getattr(gru, "bias_hh_l0", None)
        def _gru_step(x, h):
            xw = x @ _Wih.t()
            hw = h @ _Whh.t()
            if _bih is not None:
                xw = xw + _bih
            if _bhh is not None:
                hw = hw + _bhh
            xr, xz, xn = xw.chunk(3, dim=-1)
            hr, hz, hn = hw.chunk(3, dim=-1)
            r = _t.sigmoid(xr + hr)
            z = _t.sigmoid(xz + hz)
            n = _t.tanh(xn + r * hn)
            return (1 - z) * n + z * h
    tok[0] = anchor_id
    if _cell:
        hbuf[0] = _gru_step(embed_fn(anchor_id.view(1, -1)), hbuf[0].unsqueeze(0))[0]
    else:
        _, h0 = gru(embed_fn(anchor_id.view(1, 1)))
        hbuf[0] = h0[0, 0]
    cnt = 1
    fr = _t.zeros(1, dtype=_t.long, device=device)

    for d in range(1, L + 1):
        F = fr.shape[0]
        if F == 0 or cnt >= cap - budget:
            break
        if _lite:
            if d == 1:
                dist_s = _pv[0].unsqueeze(0).expand(F, -1)
            else:
                z = hiddens[d - 1].unsqueeze(0).expand(F, -1)
                mid = _act(_w_in(_t.cat([z, hbuf[fr]], dim=-1)))          # [F,mid]
                bias_s = _t.einsum("fm,km->fk", mid.float(), _sub_w[d - 1].float())
                dist_s = _pv[d - 1].unsqueeze(0) + bias_s
            _dist32 = dist_s
            if temperature and temperature > 0:
                _dist32 = _dist32 / temperature
            lp = _t.log_softmax(_dist32, dim=-1)
            k = min(topk, lp.shape[-1])
            top_lp, sub_idx = lp.topk(k, dim=-1)
            top_id = _pi[max(d - 1, 0)].unsqueeze(0).expand(F, -1).gather(1, sub_idx)
        else:
            if d == 1:
                dist = base_logits[0].unsqueeze(0).expand(F, -1)
            else:
                z = hiddens[d - 1].unsqueeze(0).expand(F, -1)
                bias = proj(_t.cat([z, hbuf[fr]], dim=-1))
                dist = base_logits[d - 1].unsqueeze(0) + bias
            _dist32 = dist.float()
            if temperature and temperature > 0:
                _dist32 = _dist32 / temperature
            lp = _t.log_softmax(_dist32, dim=-1)
            k = min(topk, lp.shape[-1])
            top_lp, top_id = lp.topk(k, dim=-1)
        cand = (score[fr].unsqueeze(1) + top_lp).view(-1)
        keep = min(budget, cand.shape[0])
        sel_score, sel_idx = cand.topk(keep)
        sel_par = fr[sel_idx // k]
        sel_tok = top_id.view(-1)[sel_idx]
        idx = _t.arange(cnt, cnt + keep, device=device)
        tok[idx] = sel_tok
        par[idx] = sel_par
        dep[idx] = d
        score[idx] = sel_score
        if d < L:
            if _cell:
                hbuf[idx] = _gru_step(embed_fn(sel_tok), hbuf[sel_par])
            else:
                emb_in = embed_fn(sel_tok.view(-1, 1))
                _, h_new = gru(emb_in, hbuf[sel_par].unsqueeze(0).contiguous())
                hbuf[idx] = h_new[0]
        cnt += keep
        fr = idx

    # 尾部拓撲: 單次 host 同步
    tok_l = tok[:cnt].tolist()
    par_l = par[:cnt].tolist()
    dep_l = dep[:cnt].tolist()
    sc_l = score[:cnt].tolist()
    import math as _m
    assert all(_m.isfinite(s) for s in sc_l), "TREE_V2_FAST: non-finite score"
    order = sorted(range(1, cnt), key=lambda i: -sc_l[i])
    chosen = set(order[:budget])
    changed = True
    while changed:
        changed = False
        for i in list(chosen):
            pp = par_l[i]
            if pp > 0 and pp not in chosen:
                chosen.add(pp)
                changed = True
    while len(chosen) > budget:
        worst = min((i for i in chosen if not any(par_l[j] == i for j in chosen)),
                    key=lambda i: sc_l[i])
        chosen.remove(worst)
    keep_ids = [0] + sorted(chosen, key=lambda i: (dep_l[i], -sc_l[i]))
    remap = {old: new for new, old in enumerate(keep_ids)}
    n = len(keep_ids)
    node_tokens = _t.tensor([tok_l[i] for i in keep_ids], dtype=_t.long, device=device)
    par_o = _t.tensor([remap[par_l[i]] if par_l[i] >= 0 else -1 for i in keep_ids],
                      dtype=_t.long, device=device)
    dep_o = _t.tensor([dep_l[i] for i in keep_ids], dtype=_t.long, device=device)
    md = int(dep_o.max())
    paths = _t.zeros(n, md + 1, dtype=_t.long, device=device)
    plen = _t.zeros(n, dtype=_t.long, device=device)
    anc = _t.zeros(n, n, dtype=_t.bool, device=device)
    po = par_o.tolist()
    for i in range(n):
        chain = [i]
        while po[chain[-1]] >= 0:
            chain.append(po[chain[-1]])
        chain.reverse()
        plen[i] = len(chain)
        paths[i, : len(chain)] = _t.tensor(chain, device=device)
        anc[i, chain] = True
    return node_tokens, par_o, dep_o, anc, paths, plen




_TREE_V2_TG = {}

class _TreeV2Graph:
    """構樹主循環的 CUDA Graph 封裝(單例 per 形狀鍵)。"""

    def __init__(self, L, V, H, budget, topk, lite, embed_fn, gru, proj, device):
        import torch as _t
        self.L, self.budget, self.topk, self.lite = L, budget, topk, lite
        gh = gru.hidden_size
        self.sb_base = _t.zeros(L, V, dtype=_t.float32, device=device)
        self.sb_hid = _t.zeros(L, H, dtype=next(proj.parameters()).dtype, device=device)
        self.sb_anchor = _t.zeros(1, dtype=_t.long, device=device)
        cap = 1 + L * budget + 8
        self.cap = cap
        self.tok = _t.zeros(cap, dtype=_t.long, device=device)
        self.par = _t.full((cap,), -1, dtype=_t.long, device=device)
        self.dep = _t.zeros(cap, dtype=_t.long, device=device)
        self.score = _t.zeros(cap, dtype=_t.float32, device=device)
        self.hbuf = _t.zeros(cap, gh, dtype=next(gru.parameters()).dtype, device=device)
        self._dconst = _t.arange(L + 2, dtype=_t.long, device=device)
        nB = budget + 1
        self.o_tok = _t.zeros(nB, dtype=_t.long, device=device)
        self.o_par = _t.zeros(nB, dtype=_t.long, device=device)
        self.o_dep = _t.zeros(nB, dtype=_t.long, device=device)
        self.o_anc = _t.zeros(nB, nB, dtype=_t.bool, device=device)
        self.o_paths = _t.zeros(nB, L + 1, dtype=_t.long, device=device)
        self.o_plen = _t.zeros(nB, dtype=_t.long, device=device)
        self.o_closed = _t.zeros(1, dtype=_t.bool, device=device)
        self._true1 = _t.ones(1, dtype=_t.bool, device=device)
        self._negone = _t.full((1,), -1, dtype=_t.long, device=device)
        Wih, Whh = gru.weight_ih_l0, gru.weight_hh_l0
        bih = getattr(gru, "bias_ih_l0", None)
        bhh = getattr(gru, "bias_hh_l0", None)

        def gru_step(x, h):
            xw = x @ Wih.t()
            hw = h @ Whh.t()
            if bih is not None:
                xw = xw + bih
            if bhh is not None:
                hw = hw + bhh
            xr, xz, xn = xw.chunk(3, dim=-1)
            hr, hz, hn = hw.chunk(3, dim=-1)
            r = _t.sigmoid(xr + hr)
            z = _t.sigmoid(xz + hz)
            n = _t.tanh(xn + r * hn)
            return (1 - z) * n + z * h

        w_in, act, w_out = proj[0], proj[1], proj[2].weight

        def body():
            base = self.sb_base
            if self.lite:
                pv, pi = base.topk(self.lite, dim=-1)
                sub_w = w_out[pi.reshape(-1)].reshape(L, self.lite, -1)
            self.tok.zero_(); self.par.fill_(-1); self.dep.zero_(); self.score.zero_()
            self.hbuf.zero_()
            self.tok[0] = self.sb_anchor[0]
            self.hbuf[0] = gru_step(embed_fn(self.sb_anchor.view(1, -1)).squeeze(1), self.hbuf[0].unsqueeze(0))[0]
            cnt = 1
            fr = _t.zeros(1, dtype=_t.long, device=self.sb_base.device)
            for d in range(1, L + 1):
                F = fr.shape[0]
                if self.lite:
                    if d == 1:
                        dist_s = pv[0].unsqueeze(0).expand(F, -1)
                    else:
                        z = self.sb_hid[d - 1].unsqueeze(0).expand(F, -1)
                        mid = act(w_in(_t.cat([z, self.hbuf[fr]], dim=-1)))
                        bias_s = _t.einsum("fm,km->fk", mid.float(), sub_w[d - 1].float())
                        dist_s = pv[d - 1].unsqueeze(0) + bias_s
                    lp = _t.log_softmax(dist_s, dim=-1)
                    k = min(self.topk, lp.shape[-1])
                    top_lp, sub_idx = lp.topk(k, dim=-1)
                    top_id = pi[max(d - 1, 0)].unsqueeze(0).expand(F, -1).gather(1, sub_idx)
                else:
                    if d == 1:
                        dist = base[0].unsqueeze(0).expand(F, -1)
                    else:
                        z = self.sb_hid[d - 1].unsqueeze(0).expand(F, -1)
                        bias = proj(_t.cat([z, self.hbuf[fr]], dim=-1))
                        dist = base[d - 1].unsqueeze(0) + bias
                    lp = _t.log_softmax(dist.float(), dim=-1)
                    k = min(self.topk, lp.shape[-1])
                    top_lp, top_id = lp.topk(k, dim=-1)
                cand = (self.score[fr].unsqueeze(1) + top_lp).view(-1)
                keep = min(self.budget, cand.shape[0])
                sel_score, sel_idx = cand.topk(keep)
                sel_par = fr[sel_idx // k]
                sel_tok = top_id.view(-1)[sel_idx]
                idx = _t.arange(cnt, cnt + keep, device=self.sb_base.device)
                self.tok[idx] = sel_tok
                self.par[idx] = sel_par
                self.dep[idx] = self._dconst[d]
                self.score[idx] = sel_score
                if d < L:
                    self.hbuf[idx] = gru_step(embed_fn(sel_tok.view(-1, 1)).squeeze(1), self.hbuf[sel_par])
                cnt += keep
                fr = idx
            self.cnt = cnt
            self._tail_static(cnt)

        # warmup ×2 然後捕獲
        for _ in range(2):
            body()
        _t.cuda.synchronize()
        self.graph = _t.cuda.CUDAGraph()
        with _t.cuda.graph(self.graph):
            body()
        _t.cuda.synchronize()

    def _tail_static(self, cnt):
        import torch as _t
        B, L = self.budget, self.L
        nB = B + 1
        dev = self.sb_base.device
        sc = self.score[1:cnt]
        _v, sel = sc.topk(B)
        sel = sel + 1                                     # 全局索引(去掉 root 偏移)
        mask = _t.zeros(cnt, dtype=_t.bool, device=dev)
        mask[0:1] = self._true1
        mask[sel] = self._true1.expand(sel.shape[0])
        # 閉合檢查(score 單調 ⇒ 理論恆真): 所有選中節點的父都被選中
        par_ok = mask[self.par[:cnt].clamp(min=0)] | ~mask
        self.o_closed[0] = par_ok.all()
        # 緊致重映射: keep 順序 = [0] + sel 按 score 降序(sel 已降序)
        keep = _t.cat([_t.zeros(1, dtype=_t.long, device=dev), sel])   # [nB]
        remap = _t.zeros(cnt, dtype=_t.long, device=dev)
        remap[keep] = _t.arange(nB, device=dev)
        self.o_tok.copy_(self.tok[keep])
        self.o_dep.copy_(self.dep[keep])
        par_g = self.par[keep].clamp(min=0)
        self.o_par.copy_(remap[par_g])
        self.o_par[0:1] = self._negone
        self.o_plen.copy_(self.o_dep + 1)
        # rev-paths: hop 鏈(新編號空間)
        rev = _t.zeros(nB, L + 1, dtype=_t.long, device=dev)
        cur = _t.arange(nB, device=dev)
        rev[:, 0] = cur
        pc = self.o_par.clamp(min=0)
        for t in range(1, L + 1):
            cur = pc[cur]
            rev[:, t] = cur
        # anc scatter(無效槽落到 0 = root, root 本就是全員祖先, 無害)
        self.o_anc.zero_()
        self.o_anc.scatter_(1, rev, self._true1.expand(rev.shape[0], rev.shape[1]))
        # paths[i, t] = rev[i, plen-1-t](t<plen), 其餘 0
        tt = _t.arange(L + 1, device=dev).unsqueeze(0)
        gidx = (self.o_plen.unsqueeze(1) - 1 - tt).clamp(min=0)
        self.o_paths.copy_(rev.gather(1, gidx) * (tt < self.o_plen.unsqueeze(1)).long())

    def run(self, base_logits, hiddens, anchor_id):
        self.sb_base.copy_(base_logits.float())
        self.sb_hid.copy_(hiddens)
        self.sb_anchor[0] = anchor_id
        self.graph.replay()
        return (self.o_tok, self.o_par, self.o_dep, self.o_anc, self.o_paths,
                self.o_plen, self.o_closed)


def _true_tree_build_graph(base_logits, hiddens, anchor_id, embed_fn, gru, proj, budget, topk, temperature=0.0, lite_k0=64):
    import torch as _t
    L, V = base_logits.shape
    H = hiddens.shape[-1]
    key = (L, V, H, budget, topk, lite_k0)
    tg = _TREE_V2_TG.get(key)
    if tg is None:
        tg = _TreeV2Graph(L, V, H, budget, topk, lite_k0, embed_fn, gru, proj, base_logits.device)
        _TREE_V2_TG[key] = tg
    o = tg.run(base_logits, hiddens, anchor_id)
    if bool(o[6][0]):
        return o[0], o[1], o[2], o[3], o[4], o[5]
    # 閉合違規(理論不可達): 回退 eager 慢版保正確
    return _true_tree_build_fast(base_logits, hiddens, anchor_id, embed_fn, gru, proj,
                                 budget, topk, temperature, lite_k0)
    tok_l = None  # unreachable




class _TreeV2GraphCTC(_TreeV2Graph):
    """CTC 頭真樹: 繼承 GRU 版的緩衝/尾部/捕獲框架, 重寫 body 的分佈計算。"""

    def __init__(self, L, V, H, budget, topk, lite, embed_fn, w, device):
        import torch as _t
        self.w = w
        self.dm = w["in_proj.weight"].shape[0]
        import os as _os
        self.nh = int(_os.environ.get("CTC_HEADS", "8"))
        self.n_layers = 1 + max(int(k.split(".")[1]) for k in w if k.startswith("layers."))
        # 祖先鏈各層 K/V 與殘差流緩存在 __init__ 分配(cap 個節點 × L+1 位)
        self._embed_fn = embed_fn
        super_init_ok = False
        # 手動執行父類 __init__ 的緩衝分配部分(不走 GRU 權重), 再捕獲
        self.L, self.budget, self.topk, self.lite = L, budget, topk, lite
        cap = 1 + L * budget + 8
        self.cap = cap
        self.sb_base = _t.zeros(L, V, dtype=_t.float32, device=device)
        self.sb_hid = _t.zeros(L, H, dtype=_t.float32, device=device)
        self.sb_anchor = _t.zeros(1, dtype=_t.long, device=device)
        self.tok = _t.zeros(cap, dtype=_t.long, device=device)
        self.par = _t.full((cap,), -1, dtype=_t.long, device=device)
        self.dep = _t.zeros(cap, dtype=_t.long, device=device)
        self.score = _t.zeros(cap, dtype=_t.float32, device=device)
        self._dconst = _t.arange(L + 2, dtype=_t.long, device=device)
        nB = budget + 1
        self.o_tok = _t.zeros(nB, dtype=_t.long, device=device)
        self.o_par = _t.zeros(nB, dtype=_t.long, device=device)
        self.o_dep = _t.zeros(nB, dtype=_t.long, device=device)
        self.o_anc = _t.zeros(nB, nB, dtype=_t.bool, device=device)
        self.o_paths = _t.zeros(nB, L + 1, dtype=_t.long, device=device)
        self.o_plen = _t.zeros(nB, dtype=_t.long, device=device)
        self.o_closed = _t.zeros(1, dtype=_t.bool, device=device)
        self._true1 = _t.ones(1, dtype=_t.bool, device=device)
        self._negone = _t.full((1,), -1, dtype=_t.long, device=device)
        # CTC 鏈緩存: 每層 K/V [cap, L+1, dm]
        self.kc = _t.zeros(self.n_layers, cap, L + 1, self.dm, dtype=_t.float32, device=device)
        self.vc = _t.zeros(self.n_layers, cap, L + 1, self.dm, dtype=_t.float32, device=device)
        if lite:
            self._w_sub = None  # lite 下 delta_out 子矩陣在 body 內取

        def body():
            _F = _t.nn.functional
            w = self.w
            base = self.sb_base
            dm, nh, hd = self.dm, self.nh, self.dm // self.nh
            if self.lite:
                pv, pi = base.topk(self.lite, dim=-1)
                sub_w = w["delta_out.weight"][pi.reshape(-1)].reshape(L, self.lite, -1)
            self.tok.zero_(); self.par.fill_(-1); self.dep.zero_(); self.score.zero_()
            self.kc.zero_(); self.vc.zero_()
            self.tok[0] = self.sb_anchor[0]

            def _pos_forward(node_rows, pos, h_pos, prev_tok):
                """為 node_rows(index 張量)在位置 pos 追加緩存並返回該位 delta。
                h_pos: [H]; prev_tok: [F] — 位置 pos 的 prev token(=鏈上第 pos 項)。"""
                F_ = node_rows.shape[0]
                x = _F.linear(
                    _t.cat([h_pos.unsqueeze(0).expand(F_, -1),
                            self._embed_fn(prev_tok).float()], dim=-1),
                    w["in_proj.weight"])                                   # [F,dm]
                for li in range(self.n_layers):
                    pre = f"layers.{li}."
                    h1 = _F.layer_norm(x, (dm,), w[pre + "ln1.weight"], w[pre + "ln1.bias"])
                    qkv = _F.linear(h1, w[pre + "qkv.weight"])
                    q, k, v = qkv.chunk(3, dim=-1)
                    self.kc[li, node_rows, pos] = k
                    self.vc[li, node_rows, pos] = v
                    K = self.kc[li, node_rows, : pos + 1]                  # [F,pos+1,dm]
                    Vv = self.vc[li, node_rows, : pos + 1]
                    qh = q.view(F_, nh, hd).unsqueeze(2)                   # [F,nh,1,hd]
                    kh = K.view(F_, pos + 1, nh, hd).transpose(1, 2)
                    vh = Vv.view(F_, pos + 1, nh, hd).transpose(1, 2)
                    a = _F.scaled_dot_product_attention(qh, kh, vh)
                    a = a.squeeze(2).reshape(F_, dm)
                    x = x + _F.linear(a, w[pre + "attn_out.weight"])
                    h2 = _F.layer_norm(x, (dm,), w[pre + "ln2.weight"], w[pre + "ln2.bias"])
                    x = x + _F.linear(_F.silu(_F.linear(h2, w[pre + "mlp_in.weight"])), w[pre + "mlp_out.weight"])
                x = _F.layer_norm(x, (dm,), w["out_ln.weight"], w["out_ln.bias"])
                return x                                                   # pre-delta 特徵 [F,dm]

            # root: 位置 0(prev=anchor), 建緩存
            r0 = _t.zeros(1, dtype=_t.long, device=self.sb_base.device)
            _pos_forward(r0, 0, self.sb_hid[0] * 0, self.sb_anchor)  # h 佔位不影響 K/V? 影響! 位置0輸入用 h[0]
            # 重做: 位置 0 的 h = parallel_hiddens[0]
            self.kc.zero_(); self.vc.zero_()
            _pos_forward(r0, 0, self.sb_hid[0], self.sb_anchor)
            cnt = 1
            fr = r0
            for d in range(1, L + 1):
                F_ = fr.shape[0]
                if d == 1:
                    dist_s = (pv[0].unsqueeze(0).expand(F_, -1) if self.lite
                              else base[0].unsqueeze(0).expand(F_, -1))
                    lp = _t.log_softmax(dist_s, dim=-1)
                    k_ = min(self.topk, lp.shape[-1])
                    top_lp, sub_idx = lp.topk(k_, dim=-1)
                    top_id = (pi[0].unsqueeze(0).expand(F_, -1).gather(1, sub_idx)
                              if self.lite else sub_idx)
                else:
                    feat = _pos_forward(fr, d - 1, self.sb_hid[d - 1], self.tok[fr])
                    mid = _F.silu(_F.linear(feat, w["delta_in.weight"]))
                    if self.lite:
                        bias_s = _t.einsum("fm,km->fk", mid, sub_w[d - 1])
                        dist_s = pv[d - 1].unsqueeze(0) + bias_s
                        lp = _t.log_softmax(dist_s, dim=-1)
                        k_ = min(self.topk, lp.shape[-1])
                        top_lp, sub_idx = lp.topk(k_, dim=-1)
                        top_id = pi[d - 1].unsqueeze(0).expand(F_, -1).gather(1, sub_idx)
                    else:
                        bias = _F.linear(mid, w["delta_out.weight"])
                        dist = base[d - 1].unsqueeze(0) + bias
                        lp = _t.log_softmax(dist.float(), dim=-1)
                        k_ = min(self.topk, lp.shape[-1])
                        top_lp, top_id = lp.topk(k_, dim=-1)
                cand = (self.score[fr].unsqueeze(1) + top_lp).view(-1)
                keep = min(self.budget, cand.shape[0])
                sel_score, sel_idx = cand.topk(keep)
                sel_par = fr[sel_idx // k_]
                sel_tok = top_id.view(-1)[sel_idx]
                idx = _t.arange(cnt, cnt + keep, device=self.sb_base.device)
                self.tok[idx] = sel_tok
                self.par[idx] = sel_par
                self.dep[idx] = self._dconst[d]
                self.score[idx] = sel_score
                # 鏈緩存繼承: 子行 = 父行(位置 0..d-1)
                self.kc[:, idx, :d] = self.kc[:, sel_par, :d]
                self.vc[:, idx, :d] = self.vc[:, sel_par, :d]
                cnt += keep
                fr = idx
            self.cnt = cnt
            self._tail_static(cnt)

        for _ in range(2):
            body()
        _t.cuda.synchronize()
        self.graph = _t.cuda.CUDAGraph()
        with _t.cuda.graph(self.graph):
            body()
        _t.cuda.synchronize()

    def run(self, base_logits, hiddens, anchor_id):
        self.sb_base.copy_(base_logits.float())
        self.sb_hid.copy_(hiddens.float())
        self.sb_anchor[0] = anchor_id
        self.graph.replay()
        return (self.o_tok, self.o_par, self.o_dep, self.o_anc, self.o_paths,
                self.o_plen, self.o_closed)


def _true_tree_build_ctc(base_logits, hiddens, anchor_id, embed_fn, w, budget, topk, lite_k0=64):
    import torch as _t
    L, V = base_logits.shape
    H = hiddens.shape[-1]
    key = ("ctc", L, V, H, budget, topk, lite_k0)
    tg = _TREE_V2_TG.get(key)
    if tg is None:
        tg = _TreeV2GraphCTC(L, V, H, budget, topk, lite_k0, embed_fn, w, base_logits.device)
        _TREE_V2_TG[key] = tg
    o = tg.run(base_logits, hiddens, anchor_id)
    assert bool(o[6][0]), "TREE_V2_CTC: closure violated"
    return o[0], o[1], o[2], o[3], o[4], o[5]




class _TreeV2GraphPAC(_TreeV2GraphCTC):
    """PAC 頭真樹(單層注意力, functional 鏡像 _pac_correct)。
    復用 CTC 版骨架, 重寫 _pos_forward 為 PAC 單層語義(無參 layer_norm)。"""

    def __init__(self, L, V, H, budget, topk, lite, embed_fn, w, device):
        import torch as _t
        self.w = w
        d3 = w["qkv_in.weight"].shape[0] // 3
        self.dm = d3
        self.nh = d3 // 64
        self.n_layers = 1
        self._embed_fn = embed_fn
        self.L, self.budget, self.topk, self.lite = L, budget, topk, lite
        cap = 1 + L * budget + 8
        self.cap = cap
        self.sb_base = _t.zeros(L, V, dtype=_t.float32, device=device)
        self.sb_hid = _t.zeros(L, H, dtype=_t.float32, device=device)
        self.sb_anchor = _t.zeros(1, dtype=_t.long, device=device)
        self.tok = _t.zeros(cap, dtype=_t.long, device=device)
        self.par = _t.full((cap,), -1, dtype=_t.long, device=device)
        self.dep = _t.zeros(cap, dtype=_t.long, device=device)
        self.score = _t.zeros(cap, dtype=_t.float32, device=device)
        self._dconst = _t.arange(L + 2, dtype=_t.long, device=device)
        nB = budget + 1
        self.o_tok = _t.zeros(nB, dtype=_t.long, device=device)
        self.o_par = _t.zeros(nB, dtype=_t.long, device=device)
        self.o_dep = _t.zeros(nB, dtype=_t.long, device=device)
        self.o_anc = _t.zeros(nB, nB, dtype=_t.bool, device=device)
        self.o_paths = _t.zeros(nB, L + 1, dtype=_t.long, device=device)
        self.o_plen = _t.zeros(nB, dtype=_t.long, device=device)
        self.o_closed = _t.zeros(1, dtype=_t.bool, device=device)
        self._true1 = _t.ones(1, dtype=_t.bool, device=device)
        self._negone = _t.full((1,), -1, dtype=_t.long, device=device)
        # 單層 K/V 鏈緩存 + x1 緩存(y 需要當前位 x1)
        self.kc = _t.zeros(1, cap, L + 1, self.dm, dtype=_t.float32, device=device)
        self.vc = _t.zeros(1, cap, L + 1, self.dm, dtype=_t.float32, device=device)

        def body():
            import torch.nn.functional as _F
            w = self.w
            base = self.sb_base
            dm, nh, hd = self.dm, self.nh, self.dm // self.nh
            if self.lite:
                pv, pi = base.topk(self.lite, dim=-1)
                sub_w = w["delta_out.weight"].float()[pi.reshape(-1)].reshape(L, self.lite, -1)
            self.tok.zero_(); self.par.fill_(-1); self.dep.zero_(); self.score.zero_()
            self.kc.zero_(); self.vc.zero_()
            self.tok[0] = self.sb_anchor[0]

            def _pos_forward(node_rows, pos, h_pos, prev_tok):
                F_ = node_rows.shape[0]
                x = _t.cat([h_pos.unsqueeze(0).expand(F_, -1),
                            self._embed_fn(prev_tok).float()], dim=-1)
                x1 = _F.layer_norm(x, (x.shape[-1],))
                qkv = _F.linear(x1, w["qkv_in.weight"].float())
                q, k, v = qkv.chunk(3, dim=-1)
                self.kc[0, node_rows, pos] = k
                self.vc[0, node_rows, pos] = v
                K = self.kc[0, node_rows, : pos + 1]
                Vv = self.vc[0, node_rows, : pos + 1]
                qh = q.view(F_, nh, hd).unsqueeze(2)
                kh = K.view(F_, pos + 1, nh, hd).transpose(1, 2)
                vh = Vv.view(F_, pos + 1, nh, hd).transpose(1, 2)
                a = _F.scaled_dot_product_attention(qh, kh, vh)
                a = a.squeeze(2).reshape(F_, dm)
                attn = _F.linear(a, w["attn_out.weight"].float())
                y = _t.cat([x1, attn], dim=-1)
                y = _F.layer_norm(y, (y.shape[-1],))
                return _F.silu(_F.linear(y, w["delta_in.weight"].float()))

            r0 = _t.zeros(1, dtype=_t.long, device=self.sb_base.device)
            _pos_forward(r0, 0, self.sb_hid[0], self.sb_anchor)
            cnt = 1
            fr = r0
            for d in range(1, L + 1):
                F_ = fr.shape[0]
                if d == 1:
                    dist_s = (pv[0].unsqueeze(0).expand(F_, -1) if self.lite
                              else base[0].unsqueeze(0).expand(F_, -1))
                    lp = _t.log_softmax(dist_s, dim=-1)
                    k_ = min(self.topk, lp.shape[-1])
                    top_lp, sub_idx = lp.topk(k_, dim=-1)
                    top_id = (pi[0].unsqueeze(0).expand(F_, -1).gather(1, sub_idx)
                              if self.lite else sub_idx)
                else:
                    mid = _pos_forward(fr, d - 1, self.sb_hid[d - 1], self.tok[fr])
                    if self.lite:
                        bias_s = _t.einsum("fm,km->fk", mid, sub_w[d - 1])
                        dist_s = pv[d - 1].unsqueeze(0) + bias_s
                        lp = _t.log_softmax(dist_s, dim=-1)
                        k_ = min(self.topk, lp.shape[-1])
                        top_lp, sub_idx = lp.topk(k_, dim=-1)
                        top_id = pi[d - 1].unsqueeze(0).expand(F_, -1).gather(1, sub_idx)
                    else:
                        bias = _F.linear(mid, w["delta_out.weight"].float())
                        dist = base[d - 1].unsqueeze(0) + bias
                        lp = _t.log_softmax(dist.float(), dim=-1)
                        k_ = min(self.topk, lp.shape[-1])
                        top_lp, top_id = lp.topk(k_, dim=-1)
                cand = (self.score[fr].unsqueeze(1) + top_lp).view(-1)
                keep = min(self.budget, cand.shape[0])
                sel_score, sel_idx = cand.topk(keep)
                sel_par = fr[sel_idx // k_]
                sel_tok = top_id.view(-1)[sel_idx]
                idx = _t.arange(cnt, cnt + keep, device=self.sb_base.device)
                self.tok[idx] = sel_tok
                self.par[idx] = sel_par
                self.dep[idx] = self._dconst[d]
                self.score[idx] = sel_score
                self.kc[:, idx, :d] = self.kc[:, sel_par, :d]
                self.vc[:, idx, :d] = self.vc[:, sel_par, :d]
                cnt += keep
                fr = idx
            self.cnt = cnt
            self._tail_static(cnt)

        for _ in range(2):
            body()
        _t.cuda.synchronize()
        self.graph = _t.cuda.CUDAGraph()
        with _t.cuda.graph(self.graph):
            body()
        _t.cuda.synchronize()


def _true_tree_build_pac(base_logits, hiddens, anchor_id, embed_fn, w, budget, topk, lite_k0=64):
    import torch as _t
    L, V = base_logits.shape
    H = hiddens.shape[-1]
    key = ("pac", L, V, H, budget, topk, lite_k0)
    tg = _TREE_V2_TG.get(key)
    if tg is None:
        tg = _TreeV2GraphPAC(L, V, H, budget, topk, lite_k0, embed_fn, w, base_logits.device)
        _TREE_V2_TG[key] = tg
    o = tg.run(base_logits, hiddens, anchor_id)
    assert bool(o[6][0]), "TREE_V2_PAC: closure violated"
    return o[0], o[1], o[2], o[3], o[4], o[5]


# ==== end DOMINO_TREE helpers ====

_PAC_HEAD = None

def _pac_load(device, dtype=None):
    global _PAC_HEAD
    if dtype is None:
        dtype = torch.float32
    if _PAC_HEAD is None:
        import os as _os
        _ckpt = _os.environ["PAC_HEAD_CKPT"]
        w = torch.load(_ckpt, map_location=device, weights_only=True)
        _PAC_HEAD = {k: v.to(device=device, dtype=dtype) for k, v in w.items()}
        _tot = sum(v.numel() for v in _PAC_HEAD.values())
        print(f"[pac] head LOADED ckpt={_ckpt} keys={len(_PAC_HEAD)} params={_tot/1e6:.1f}M", flush=True)
    return _PAC_HEAD


def _pac_correct(parallel_hiddens, base_logits, anchor_ids, target):
    """PAC one-shot correction, faithful to training contract.
    parallel_hiddens: [1, L, H]; base_logits: [1, L, V]; anchor_ids: [1, 1]."""
    import torch.nn.functional as _F
    w = _pac_load(parallel_hiddens.device)
    h = parallel_hiddens.float()
    L = h.shape[1]
    base = base_logits.float()
    def _round(chain_logits):
        prev_tok = chain_logits[:, :-1, :].argmax(dim=-1)
        prev_ids = torch.cat([anchor_ids, prev_tok], dim=1)
        prev_emb = target.model.embed_tokens(prev_ids).float()
        x = torch.cat([h, prev_emb], dim=-1)
        x1 = _F.layer_norm(x, (x.shape[-1],))
        qkv = _F.linear(x1, w["qkv_in.weight"])
        d3 = qkv.shape[-1] // 3
        q, k, v = qkv.split(d3, dim=-1)
        nh = d3 // 64
        def _hd(t):
            return t.view(1, L, nh, 64).transpose(1, 2)
        attn = _F.scaled_dot_product_attention(_hd(q), _hd(k), _hd(v), is_causal=True)
        attn = attn.transpose(1, 2).reshape(1, L, d3)
        attn = _F.linear(attn, w["attn_out.weight"])
        y = torch.cat([x1, attn], dim=-1)
        y = _F.layer_norm(y, (y.shape[-1],))
        return _F.linear(_F.silu(_F.linear(y, w["delta_in.weight"])), w["delta_out.weight"])

    def _round_from_ids(prev_ids):
        prev_emb = target.model.embed_tokens(prev_ids).float()
        x = torch.cat([h, prev_emb], dim=-1)
        x1 = _F.layer_norm(x, (x.shape[-1],))
        qkv = _F.linear(x1, w["qkv_in.weight"])
        d3 = qkv.shape[-1] // 3
        q, k, v = qkv.split(d3, dim=-1)
        nh = d3 // 64
        def _hd(t):
            return t.view(1, L, nh, 64).transpose(1, 2)
        attn = _F.scaled_dot_product_attention(_hd(q), _hd(k), _hd(v), is_causal=True)
        attn = attn.transpose(1, 2).reshape(1, L, d3)
        attn = _F.linear(attn, w["attn_out.weight"])
        y = torch.cat([x1, attn], dim=-1)
        y = _F.layer_norm(y, (y.shape[-1],))
        return _F.linear(_F.silu(_F.linear(y, w["delta_in.weight"])), w["delta_out.weight"])

    import os as _os2
    _sfx = int(_os2.environ.get("PAC_SUFFIX_START", "0"))
    if _os2.environ.get("PAC_ROLLOUT") == "1":
        prev_ids = torch.cat([anchor_ids, base.argmax(dim=-1)[:, :-1]], dim=1)
        final = base.clone()
        for _j in range(L):
            if _j >= _sfx:
                delta_j = _round_from_ids(prev_ids)
                final[:, _j] = base[:, _j] + delta_j[:, _j]
            if _j + 1 < L:
                prev_ids = prev_ids.clone()
                prev_ids[:, _j + 1] = final[:, _j].argmax(dim=-1)
        return final

    final = base + _round(base)
    for _it in range(int(_os2.environ.get("PAC_ITER_K", "1")) - 1):
        final = base + _round(final)
    return final


_CTC_HEAD = None

def _ctc_load(device, dtype=None):
    global _CTC_HEAD
    if dtype is None:
        dtype = torch.float32
    if _CTC_HEAD is None:
        import os as _os
        _ckpt = _os.environ["CTC_HEAD_CKPT"]
        w = torch.load(_ckpt, map_location=device, weights_only=True)
        _CTC_HEAD = {k: v.to(device=device, dtype=dtype) for k, v in w.items()}
        _tot = sum(v.numel() for v in _CTC_HEAD.values())
        print(f"[ctc] head LOADED ckpt={_ckpt} keys={len(_CTC_HEAD)} params={_tot/1e6:.1f}M", flush=True)
    return _CTC_HEAD


def _ctc_correct(parallel_hiddens, base_logits, anchor_ids, target):
    """CTC 塊內因果 mini-Transformer 修正(functional 鏡像 CausalTransformerCorrector)。
    parallel_hiddens: [1, L, H]; base_logits: [1, L, V]; anchor_ids: [1, 1]."""
    import os as _os
    import torch.nn.functional as _F
    w = _ctc_load(parallel_hiddens.device)
    h = parallel_hiddens.float()
    L = h.shape[1]
    base = base_logits.float()
    d_model = w["in_proj.weight"].shape[0]
    n_heads = int(_os.environ.get("CTC_HEADS", "8"))
    hd = d_model // n_heads
    n_layers = 1 + max(
        int(k.split(".")[1]) for k in w if k.startswith("layers.")
    )

    def _round(chain_logits):
        prev_tok = chain_logits[:, :-1, :].argmax(dim=-1)
        prev_ids = torch.cat([anchor_ids, prev_tok], dim=1)
        prev_emb = target.model.embed_tokens(prev_ids).float()
        x = _F.linear(torch.cat([h, prev_emb], dim=-1), w["in_proj.weight"])
        for li in range(n_layers):
            pre = f"layers.{li}."
            h1 = _F.layer_norm(x, (d_model,), w[pre + "ln1.weight"], w[pre + "ln1.bias"])
            qkv = _F.linear(h1, w[pre + "qkv.weight"])
            q, k, v = qkv.chunk(3, dim=-1)

            def _s(t):
                return t.view(1, L, n_heads, hd).transpose(1, 2)

            a = _F.scaled_dot_product_attention(_s(q), _s(k), _s(v), is_causal=True)
            a = a.transpose(1, 2).reshape(1, L, d_model)
            x = x + _F.linear(a, w[pre + "attn_out.weight"])
            h2 = _F.layer_norm(x, (d_model,), w[pre + "ln2.weight"], w[pre + "ln2.bias"])
            x = x + _F.linear(_F.silu(_F.linear(h2, w[pre + "mlp_in.weight"])), w[pre + "mlp_out.weight"])
        x = _F.layer_norm(x, (d_model,), w["out_ln.weight"], w["out_ln.bias"])
        return _F.linear(_F.silu(_F.linear(x, w["delta_in.weight"])), w["delta_out.weight"])

    def _round_from_ids(prev_ids):
        prev_emb = target.model.embed_tokens(prev_ids).float()
        x = _F.linear(torch.cat([h, prev_emb], dim=-1), w["in_proj.weight"])
        for li in range(n_layers):
            pre = f"layers.{li}."
            h1 = _F.layer_norm(x, (d_model,), w[pre + "ln1.weight"], w[pre + "ln1.bias"])
            qkv = _F.linear(h1, w[pre + "qkv.weight"])
            q, k, v = qkv.chunk(3, dim=-1)

            def _s(t):
                return t.view(1, L, n_heads, hd).transpose(1, 2)

            a = _F.scaled_dot_product_attention(_s(q), _s(k), _s(v), is_causal=True)
            a = a.transpose(1, 2).reshape(1, L, d_model)
            x = x + _F.linear(a, w[pre + "attn_out.weight"])
            h2 = _F.layer_norm(x, (d_model,), w[pre + "ln2.weight"], w[pre + "ln2.bias"])
            x = x + _F.linear(_F.silu(_F.linear(h2, w[pre + "mlp_in.weight"])), w[pre + "mlp_out.weight"])
        x = _F.layer_norm(x, (d_model,), w["out_ln.weight"], w["out_ln.bias"])
        return _F.linear(_F.silu(_F.linear(x, w["delta_in.weight"])), w["delta_out.weight"])

    _sfx = int(_os.environ.get("CTC_SUFFIX_START", "0"))
    if _os.environ.get("CTC_ROLLOUT") == "1":
        # 逐位 rollout(官方 GRU 部署同語義):修正即回填,已接受前綴=金鏈
        # CTC_SUFFIX_START=s: 前 s 位純 base 不修(官方 pure_draft_prefix_len 同款保護)
        prev_ids = torch.cat([anchor_ids, base.argmax(dim=-1)[:, :-1]], dim=1)
        final = base.clone()
        for _j in range(L):
            if _j < _sfx:
                pass
            else:
                delta_j = _round_from_ids(prev_ids)
                final[:, _j] = base[:, _j] + delta_j[:, _j]
            if _j + 1 < L:
                prev_ids = prev_ids.clone()
                prev_ids[:, _j + 1] = final[:, _j].argmax(dim=-1)
        return final

    final = base + _round(base)
    for _it in range(int(_os.environ.get("CTC_ITER_K", "1")) - 1):
        final = base + _round(final)
    return final



from typing import Callable, Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack


def is_domino_projector(projector_type):
    return projector_type == "domino"


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def cuda_time(device: torch.device | str | int | None = None) -> float:
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.synchronize()
        else:
            cuda_device = (
                torch.device(f"cuda:{device}")
                if isinstance(device, int)
                else torch.device(device)
            )
            if cuda_device.type == "cuda":
                torch.cuda.synchronize(cuda_device)
    return time.perf_counter()


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = self.config.dflash_config.get("target_layer_ids", build_target_layer_ids(config.num_target_layers, config.num_hidden_layers))
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = self.config.dflash_config.get("mask_token_id", None)
        self.projector_type = self.config.dflash_config.get("projector_type", None)
        self.pure_draft_prefix_len = self.config.dflash_config.get("pure_draft_prefix_len", 0)
        self.emb_dim = self.config.dflash_config["emb_dim"]
        self.gru_hidden_dim = self.config.dflash_config["gru_hidden_dim"]

        if not is_domino_projector(self.projector_type):
            raise ValueError(
                "This reviewer package only supports Domino checkpoints; "
                f"got projector_type={self.projector_type!r}."
            )

        self.prefix_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        )

        in_dim = config.hidden_size + self.gru_hidden_dim
        self.embed_proj = nn.Sequential(
            nn.Linear(in_dim, self.emb_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.emb_dim, config.vocab_size, bias=False),
        )

        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        input_ids: torch.Tensor,
        target: nn.Module,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        stop_token_ids: Optional[list[int] | int] = None,
        block_size: Optional[int] = None,
        graph_runner=None,
        use_bias: bool = True,
        return_dict: bool = False,
    ) -> torch.Tensor | SimpleNamespace:
        """Generate with Domino speculative decoding.

        This method currently supports a single sequence on one GPU, matching the
        draft checkpoints released with this repository.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "spec_generate currently supports input_ids with shape [1, seq_len]."
            )

        target_device = next(target.parameters()).device
        if target_device != self.device:
            raise ValueError(
                "The draft model and target model must be on the same device; "
                f"got draft={self.device}, target={target_device}."
            )

        input_ids = input_ids.to(self.device)
        block_size = int(block_size or self.block_size)
        mask_token_id = self.mask_token_id
        if mask_token_id is None:
            raise ValueError("The draft model config must define dflash_config.mask_token_id.")

        if isinstance(stop_token_ids, int):
            stop_token_ids = [stop_token_ids]
        elif stop_token_ids is not None:
            stop_token_ids = list(stop_token_ids)

        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + int(max_new_tokens)
        shift_label = bool(
            getattr(self.config, "dflash_config", {}).get("shift_label", False)
        )
        extra_buffer = block_size + 1 if shift_label else block_size

        output_ids = torch.full(
            (1, max_length + extra_buffer),
            mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=self.device).unsqueeze(0)
        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        prefill_start = cuda_time(self.device)
        _vg = None
        _vg_pastlen = num_input_tokens
        if (
            _tree_os.environ.get("DOMINO_VERIFY_GRAPH") == "1"
            and _tree_os.environ.get("DOMINO_TREE_GRU") == "1"
            and graph_runner is not None
            and block_size > 1
        ):
            from kernel.verify_graph import VerifyGraphRunner
            _tg5 = _tree_get(self.device)
            _n0 = _tg5[0].shape[0]
            _mdt0 = next(target.parameters()).dtype
            _vg = VerifyGraphRunner(
                target, _n0, max_length + _n0 + 8, self.device, _mdt0,
            ).capture()
            _vg.cache.reset()
            _vg_block = torch.where(
                _tg5[3],
                torch.zeros((), dtype=_mdt0, device=self.device),
                torch.full((), torch.finfo(_mdt0).min, dtype=_mdt0, device=self.device),
            ).view(1, 1, _n0, _n0)
            past_key_values_target = _vg.cache
        _lvg = None
        if (
            _tree_os.environ.get("DOMINO_LINEAR_VG") == "1"
            and graph_runner is not None
            and block_size > 1
        ):
            assert _vg is None, "tree-vg and linear-vg are mutually exclusive"
            from kernel.verify_graph import VerifyGraphRunner
            _nl = block_size + 1
            _mdt0 = next(target.parameters()).dtype
            _lvg = VerifyGraphRunner(
                target, _nl, max_length + _nl + 8, self.device, _mdt0,
            ).capture()
            _lvg.cache.reset()
            _lvg_block = torch.where(
                torch.tril(torch.ones(_nl, _nl, dtype=torch.bool, device=self.device)),
                torch.zeros((), dtype=_mdt0, device=self.device),
                torch.full((), torch.finfo(_mdt0).min, dtype=_mdt0, device=self.device),
            ).view(1, 1, _nl, _nl)
            past_key_values_target = _lvg.cache
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            cache_position=(torch.arange(num_input_tokens, device=self.device) if _vg is not None else None),
            logits_to_keep=1,
            output_hidden_states=block_size > 1,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        if block_size > 1:
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )
        time_to_first_token = cuda_time(self.device) - prefill_start

        decode_start = cuda_time(self.device)
        global _TREE_SIDE_STREAM, _TREE_GATHER_EVT
        _tree_mask_buf = None
        _tree_prev_win = None
        start = num_input_tokens
        acceptance_lengths: list[int] = []
        draft_prefill = True
        prefix_len = int(self.pure_draft_prefix_len)

        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            k_draft = block_size if shift_label else block_size - 1
            verify_ids = torch.full(
                (1, k_draft + 1),
                mask_token_id,
                dtype=torch.long,
                device=self.device,
            )
            verify_ids[:, 0] = output_ids[:, start]
            verify_position_ids = position_ids[:, start : start + k_draft + 1]

            if block_size > 1:
                if not is_domino_projector(self.projector_type):
                    raise ValueError(
                        "This package only supports Domino checkpoints; "
                        f"got projector_type={self.projector_type!r}."
                    )
                if not use_bias:
                    raise ValueError("Domino generation requires use_bias=True.")

                noise_embedding = target.model.embed_tokens(block_output_ids)
                parallel_hiddens = self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[
                        :, past_key_values_draft.get_seq_length() : start + block_size
                    ],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )
                if not shift_label:
                    parallel_hiddens = parallel_hiddens[:, -block_size + 1 :, :]
                past_key_values_draft.crop(start)

                base_logits = target.lm_head(parallel_hiddens)
                if _tree_os.environ.get("DOMINO_TREE_V2") == "1":
                    if draft_prefill:
                        draft_prefill = False
                        decode_start = cuda_time(self.device)
                    _b2 = int(_tree_os.environ.get("TREE_V2_BUDGET", "31"))
                    _k2 = int(_tree_os.environ.get("TREE_V2_TOPK", "8"))
                    assert int(self.pure_draft_prefix_len) == 1, "TREE_V2 assumes prefix_len==1"  # Codex#2
                    assert past_key_values_target.get_seq_length() == start, "TREE_V2: past/start desync"  # Codex#11
                    _prof = _tree_os.environ.get("TREE_V2_PROF") == "1"
                    if _prof:
                        import time as _tm
                        torch.cuda.synchronize(); _p0 = _tm.perf_counter()
                    _use_fast = _tree_os.environ.get("TREE_V2_FAST", "1") == "1"
                    _use_graph = _use_fast and _tree_os.environ.get("TREE_V2_GRAPH", "1") == "1" and temperature == 0
                    _lite_env = int(_tree_os.environ.get("TREE_V2_LITE", "64")) if _use_fast else 0
                    if _tree_os.environ.get("TREE_V2_HEAD", "gru") == "pac":
                        node_tokens, par_t, depth_t, anc, paths_t, plen_t = _true_tree_build_pac(
                            base_logits[0, :k_draft, :],
                            parallel_hiddens[0, :k_draft, :],
                            output_ids[0, start],
                            target.model.embed_tokens,
                            _pac_load(self.device),
                            _b2,
                            _k2,
                            _lite_env,
                        )
                    elif _tree_os.environ.get("TREE_V2_HEAD", "gru") == "ctc":
                        node_tokens, par_t, depth_t, anc, paths_t, plen_t = _true_tree_build_ctc(
                            base_logits[0, :k_draft, :],
                            parallel_hiddens[0, :k_draft, :],
                            output_ids[0, start],
                            target.model.embed_tokens,
                            _ctc_load(self.device),
                            _b2,
                            _k2,
                            _lite_env,
                        )
                    elif _use_graph:
                        _builder = _true_tree_build_graph
                        _bkw = {"lite_k0": _lite_env}
                        node_tokens, par_t, depth_t, anc, paths_t, plen_t = _builder(
                        base_logits[0, :k_draft, :],
                        parallel_hiddens[0, :k_draft, :],
                        output_ids[0, start],
                        target.model.embed_tokens,
                        self.prefix_gru,
                        self.embed_proj,
                        _b2,
                        _k2,
                        temperature,
                        **_bkw,
                    )
                    if _prof:
                        torch.cuda.synchronize(); _p1 = _tm.perf_counter()
                    n_nodes = node_tokens.shape[0]
                    tree_ids = node_tokens.unsqueeze(0)
                    tree_pos = (start + depth_t).unsqueeze(0)
                    past_len = past_key_values_target.get_seq_length()
                    mdt = parallel_hiddens.dtype
                    neg = torch.finfo(mdt).min
                    mask4d = torch.zeros(
                        1, 1, n_nodes, past_len + n_nodes, dtype=mdt, device=self.device
                    )
                    mask4d[:, :, :, past_len:] = torch.where(
                        anc,
                        torch.zeros((), dtype=mdt, device=self.device),
                        torch.full((), neg, dtype=mdt, device=self.device),
                    )
                    output = target(
                        tree_ids,
                        position_ids=tree_pos,
                        attention_mask=mask4d,
                        past_key_values=past_key_values_target,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    if _prof:
                        torch.cuda.synchronize(); _p2 = _tm.perf_counter()
                    posterior_t = sample(output.logits, temperature)[0]
                    ok = node_tokens == posterior_t[par_t.clamp(min=0)]
                    ok[0] = True
                    acc = ~((anc & ~ok.unsqueeze(0)).any(-1))
                    score = torch.where(acc, depth_t, torch.full_like(depth_t, -1))
                    best = score.argmax()
                    acceptance_length = int(plen_t[best].item()) - 1
                    path_idx = paths_t[best, : acceptance_length + 1]
                    output_ids[:, start : start + acceptance_length + 1] = node_tokens[path_idx].unsqueeze(0)
                    output_ids[:, start + acceptance_length + 1] = posterior_t[best]
                    acceptance_lengths.append(acceptance_length + 1)
                    _tree_cache_gather(past_key_values_target, past_len, path_idx)
                    start += acceptance_length + 1
                    target_hidden = extract_context_feature(
                        output.hidden_states, self.target_layer_ids
                    )[:, path_idx, :]
                    if _prof:
                        torch.cuda.synchronize(); _p3 = _tm.perf_counter()
                        global _PROF_ACC, _PROF_N
                        try:
                            _PROF_ACC
                        except NameError:
                            _PROF_ACC = [0.0, 0.0, 0.0]; _PROF_N = 0
                        _PROF_ACC[0] += _p1 - _p0; _PROF_ACC[1] += _p2 - _p1; _PROF_ACC[2] += _p3 - _p2
                        _PROF_N += 1
                        if _PROF_N % 60 == 0:
                            print("[prof] n=%d build=%.1fms fwd+mask=%.1fms post=%.1fms" % (
                                _PROF_N, 1e3*_PROF_ACC[0]/_PROF_N, 1e3*_PROF_ACC[1]/_PROF_N, 1e3*_PROF_ACC[2]/_PROF_N), flush=True)
                    if stop_token_ids is not None:
                        stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
                        if torch.isin(output_ids[:, num_input_tokens:start], stop_tensor).any():
                            break
                    continue
                if _tree_os.environ.get("DOMINO_TREE") == "1":
                    if draft_prefill:
                        draft_prefill = False
                        decode_start = cuda_time(self.device)
                    par_t, depth_t, rank_t, anc, max_d, paths_t, plen_t = _tree_get(self.device)
                    n_nodes = par_t.shape[0]
                    tree_logits = base_logits[:, :k_draft, :]
                    if _tree_os.environ.get("DOMINO_CTC_HEAD") == "1":
                        tree_logits = _ctc_correct(
                            parallel_hiddens[:, :k_draft, :],
                            tree_logits,
                            output_ids[:, start : start + 1],
                            target,
                        )
                    elif _tree_os.environ.get("DOMINO_PAC_HEAD") == "1":
                        tree_logits = _pac_correct(
                            parallel_hiddens[:, :k_draft, :],
                            tree_logits,
                            output_ids[:, start : start + 1],
                            target,
                        )
                    topk_ids = tree_logits.topk(8, dim=-1).indices[0]
                    node_tokens = torch.empty(n_nodes, dtype=torch.long, device=self.device)
                    node_tokens[0] = output_ids[0, start]
                    node_tokens[1:] = topk_ids[depth_t[1:] - 1, rank_t[1:]]
                    tree_ids = node_tokens.unsqueeze(0)
                    tree_pos = (start + depth_t).unsqueeze(0)
                    past_len = past_key_values_target.get_seq_length()
                    mdt = parallel_hiddens.dtype
                    neg = torch.finfo(mdt).min
                    global _TREE_MASK_BUF, _TREE_ANC_BLOCK, _TREE_MASK_PREV
                    if (
                        _TREE_MASK_BUF is None
                        or _TREE_MASK_BUF.dtype != mdt
                        or _TREE_MASK_BUF.shape[-1] < max_length + n_nodes + 8
                    ):
                        _TREE_MASK_BUF = torch.zeros(
                            1, 1, n_nodes, max_length + n_nodes + 8, dtype=mdt, device=self.device
                        )
                        _TREE_ANC_BLOCK = torch.where(
                            anc,
                            torch.zeros((), dtype=mdt, device=self.device),
                            torch.full((), neg, dtype=mdt, device=self.device),
                        ).view(1, 1, n_nodes, n_nodes)
                        _TREE_MASK_PREV = -1
                    if _TREE_MASK_PREV >= 0:
                        _TREE_MASK_BUF[:, :, :, _TREE_MASK_PREV:past_len] = 0
                    _TREE_MASK_PREV = past_len
                    _TREE_MASK_BUF[:, :, :, past_len : past_len + n_nodes] = _TREE_ANC_BLOCK
                    mask4d = _TREE_MASK_BUF[:, :, :, : past_len + n_nodes]
                    output = target(
                        tree_ids,
                        position_ids=tree_pos,
                        attention_mask=mask4d,
                        past_key_values=past_key_values_target,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    posterior_t = sample(output.logits, temperature)[0]
                    ok = node_tokens == posterior_t[par_t.clamp(min=0)]
                    ok[0] = True
                    acc = ~((anc & ~ok.unsqueeze(0)).any(-1))
                    score = torch.where(acc, depth_t, torch.full_like(depth_t, -1))
                    best = score.argmax()
                    acceptance_length = int(plen_t[best].item()) - 1
                    path_idx = paths_t[best, : acceptance_length + 1]
                    output_ids[:, start : start + acceptance_length + 1] = node_tokens[path_idx].unsqueeze(0)
                    output_ids[:, start + acceptance_length + 1] = posterior_t[best]
                    acceptance_lengths.append(acceptance_length + 1)
                    _tree_cache_gather(past_key_values_target, past_len, path_idx)
                    start += acceptance_length + 1
                    target_hidden = extract_context_feature(
                        output.hidden_states, self.target_layer_ids
                    )[:, path_idx, :]
                    if stop_token_ids is not None:
                        stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
                        if torch.isin(output_ids[:, num_input_tokens:start], stop_tensor).any():
                            break
                    continue
                if prefix_len > 0:
                    prefix_token_ids = sample(base_logits[:, :prefix_len], temperature)
                    verify_ids[:, 1 : 1 + prefix_len] = prefix_token_ids

                if _tree_os.environ.get("DOMINO_CTC_HEAD") == "1":
                    _final = _ctc_correct(
                        parallel_hiddens[:, :k_draft, :],
                        base_logits[:, :k_draft, :],
                        verify_ids[:, :1],
                        target,
                    )
                    verify_ids[:, 1:] = sample(_final, temperature)
                elif _tree_os.environ.get("DOMINO_PAC_HEAD") == "1":
                    _final = _pac_correct(
                        parallel_hiddens[:, :k_draft, :],
                        base_logits[:, :k_draft, :],
                        verify_ids[:, :1],
                        target,
                    )
                    verify_ids[:, 1:] = sample(_final, temperature)
                elif _tree_os.environ.get("DOMINO_DISABLE_HEAD") == "1":
                    verify_ids[:, 1 + prefix_len :] = sample(
                        base_logits[:, prefix_len:k_draft, :], temperature
                    )
                elif graph_runner is not None:
                    graph_prefix_ids = verify_ids[:, : 1 + prefix_len].contiguous()
                    graph_parallel_hiddens = parallel_hiddens[
                        :, prefix_len:k_draft, :
                    ].contiguous()
                    graph_base_logits = base_logits[:, prefix_len:k_draft, :].contiguous()
                    verify_ids[:, 1 + prefix_len :] = graph_runner(
                        graph_prefix_ids,
                        graph_parallel_hiddens,
                        graph_base_logits,
                    )
                    if _tree_os.environ.get("DOMINO_TREE_GRU") == "1":  # _TREE_GRU_BRANCH
                        if draft_prefill:
                            draft_prefill = False
                            decode_start = cuda_time(self.device)
                        par_t, depth_t, rank_t, anc, max_d, paths_t, plen_t = _tree_get(self.device)
                        n_nodes = par_t.shape[0]
                        gtopk = graph_runner.static_topk[0]  # [steps, 8]
                        if prefix_len > 0:
                            pref_topk = base_logits[0, :prefix_len, :].topk(16, dim=-1).indices
                            topk_ids = torch.cat([pref_topk, gtopk], dim=0)[:k_draft]
                        else:
                            topk_ids = gtopk[:k_draft]
                        node_tokens = torch.empty(n_nodes, dtype=torch.long, device=self.device)
                        node_tokens[0] = output_ids[0, start]
                        chain_ids = verify_ids[0, 1:]
                        _ch_at = torch.cat([chain_ids.new_zeros(1), chain_ids])[
                            torch.clamp(depth_t, max=chain_ids.shape[0])
                        ]
                        _eq = topk_ids == torch.cat(
                            [chain_ids, chain_ids.new_zeros(1)]
                        )[: topk_ids.shape[0]].unsqueeze(1)
                        _cum = torch.cumsum(_eq.long(), dim=1)
                        _r0 = (rank_t[1:] - 1).clamp(min=0)
                        _d0 = depth_t[1:] - 1
                        _adj = _r0 + _cum[_d0, _r0]
                        _adj = _adj.clamp(max=topk_ids.shape[1] - 1)
                        node_tokens[1:] = topk_ids[_d0, _adj]
                        lvl_first = torch.zeros(n_nodes, dtype=torch.bool, device=self.device)
                        lvl_first[1:] = rank_t[1:] == 0
                        node_tokens = torch.where(
                            lvl_first,
                            torch.cat([chain_ids.new_zeros(1), chain_ids])[
                                torch.clamp(depth_t, max=chain_ids.shape[0])
                            ],
                            node_tokens,
                        )
                        node_tokens[0] = output_ids[0, start]
                        tree_ids = node_tokens.unsqueeze(0)
                        tree_pos = (start + depth_t).unsqueeze(0)
                        past_len = (
                            _vg_pastlen if _vg is not None
                            else past_key_values_target.get_seq_length()
                        )
                        mdt = parallel_hiddens.dtype
                        neg = torch.finfo(mdt).min
                        if _tree_mask_buf is None:
                            _tree_mask_buf = torch.zeros(
                                1, 1, n_nodes, max_length + n_nodes + 8,
                                dtype=mdt, device=self.device,
                            )
                            _tree_anc_neg = torch.where(
                                anc,
                                torch.zeros((), dtype=mdt, device=self.device),
                                torch.full((), neg, dtype=mdt, device=self.device),
                            )
                        if _tree_prev_win is not None:
                            _tree_mask_buf[:, :, :, _tree_prev_win : _tree_prev_win + n_nodes] = 0
                        _tree_mask_buf[:, :, :, past_len : past_len + n_nodes] = _tree_anc_neg
                        _tree_prev_win = past_len
                        mask4d = _tree_mask_buf[:, :, :, : past_len + n_nodes]
                        if _TREE_GATHER_EVT is not None:
                            torch.cuda.current_stream().wait_event(_TREE_GATHER_EVT)
                        if _vg is not None:
                            _lg, _hs = _vg(tree_ids, tree_pos, _vg_block, past_len)
                            posterior_t = sample(_lg, temperature)[0]
                        else:
                            output = target(
                                tree_ids,
                                position_ids=tree_pos,
                                attention_mask=mask4d,
                                past_key_values=past_key_values_target,
                                use_cache=True,
                                output_hidden_states=True,
                            )
                            posterior_t = sample(output.logits, temperature)[0]
                        ok = node_tokens == posterior_t[par_t.clamp(min=0)]
                        ok[0] = True
                        acc = ~((anc & ~ok.unsqueeze(0)).any(-1))
                        score = torch.where(acc, depth_t, torch.full_like(depth_t, -1))
                        best = score.argmax()
                        acceptance_length = int(plen_t[best].item()) - 1
                        path_idx = paths_t[best, : acceptance_length + 1]
                        output_ids[:, start : start + acceptance_length + 1] = node_tokens[path_idx].unsqueeze(0)
                        output_ids[:, start + acceptance_length + 1] = posterior_t[best]
                        acceptance_lengths.append(acceptance_length + 1)
                        if _TREE_SIDE_STREAM is None:
                            _TREE_SIDE_STREAM = torch.cuda.Stream()
                            _TREE_GATHER_EVT = torch.cuda.Event()
                        _TREE_SIDE_STREAM.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(_TREE_SIDE_STREAM):
                            _tree_cache_gather(past_key_values_target, past_len, path_idx)
                        _TREE_GATHER_EVT.record(_TREE_SIDE_STREAM)
                        _vg_pastlen = past_len + acceptance_length + 1
                        start += acceptance_length + 1
                        target_hidden = extract_context_feature(
                            _hs if _vg is not None else output.hidden_states,
                            self.target_layer_ids,
                        )[:, path_idx, :]
                        if stop_token_ids is not None:
                            stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
                            if torch.isin(output_ids[:, num_input_tokens:start], stop_tensor).any():
                                break
                        continue
                else:
                    realized_prefix_ids = verify_ids[:, : 1 + prefix_len]
                    realized_prefix_embeds = target.model.embed_tokens(realized_prefix_ids)
                    _, gru_hidden = self.prefix_gru(realized_prefix_embeds)

                    for i in range(prefix_len, k_draft):
                        z_i = parallel_hiddens[:, i : i + 1, :]
                        s_i = gru_hidden.transpose(0, 1)

                        bias = self.embed_proj(torch.cat([z_i, s_i], dim=-1))
                        current_token_id = sample(
                            base_logits[:, i : i + 1, :] + bias,
                            temperature,
                        )
                        verify_ids[:, i + 1 : i + 2] = current_token_id

                        if i + 1 < k_draft:
                            new_embed = target.model.embed_tokens(current_token_id)
                            _, gru_hidden = self.prefix_gru(new_embed, gru_hidden)

                if draft_prefill:
                    draft_prefill = False
                    decode_start = cuda_time(self.device)

            if _lvg is not None:
                _llg, _lhs = _lvg(
                    verify_ids,
                    verify_position_ids,
                    _lvg_block[:, :, : verify_ids.shape[1], : verify_ids.shape[1]],
                    start,
                )
                _lout_logits = _llg[:, : verify_ids.shape[1], :]
                posterior = sample(_lout_logits, temperature)
            else:
                output = target(
                    verify_ids,
                    position_ids=verify_position_ids,
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    output_hidden_states=block_size > 1,
                )

                posterior = sample(output.logits, temperature)
            acceptance_length = (
                (verify_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            output_ids[:, start : start + acceptance_length + 1] = verify_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]

            acceptance_lengths.append(int(acceptance_length) + 1)
            start += int(acceptance_length) + 1
            if _lvg is None:
                past_key_values_target.crop(start)
            if block_size > 1:
                target_hidden = extract_context_feature(
                    _lhs if _lvg is not None else output.hidden_states,
                    self.target_layer_ids,
                )[:, : acceptance_length + 1, :]

            if stop_token_ids is not None:
                stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
                if torch.isin(output_ids[:, num_input_tokens:start], stop_tensor).any():
                    break

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != mask_token_id]
        if stop_token_ids is not None:
            stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_tensor
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0].item() + 1
                ]

        if not return_dict:
            return output_ids

        num_output_tokens = output_ids.shape[1] - num_input_tokens
        total_decode_time = cuda_time(self.device) - decode_start
        time_per_output_token = (
            total_decode_time / num_output_tokens if num_output_tokens > 0 else 0.0
        )
        return SimpleNamespace(
            output_ids=output_ids,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            time_to_first_token=time_to_first_token,
            time_per_output_token=time_per_output_token,
            acceptance_lengths=acceptance_lengths,
        )
