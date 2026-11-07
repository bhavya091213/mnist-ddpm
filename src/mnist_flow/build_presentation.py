"""Build the single-file HTML presentation.

python -m mnist_flow.build_presentation --results results --runs runs \
    --runs-viz runs_viz --out presentation/index.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger("build_presentation")

GITHUB_URL = "https://github.com/bhavya091213/mnist-ddpm"
TIE_REL = 0.10          # DDIM vs RF counts as a tie inside 10% relative FID
FLOOR_MARGIN = 3.0      # "within X of the floor" in FID points
_EPOCH_RE = re.compile(r"^(?P<model>[a-z]+)_epoch(?P<epoch>\d+)\.png$")


# --------------------------------------------------------------------------- io
def _uri(path: Path) -> str:
    mime = "image/gif" if path.suffix.lower() == ".gif" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _opt_uri(path: Path) -> str | None:
    if path.is_file():
        return _uri(path)
    log.warning("missing optional image: %s", path)
    return None


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def _epoch_uris(png_dir: Path, model: str) -> list[str]:
    found: dict[int, Path] = {}
    for p in png_dir.glob(f"{model}_epoch*.png"):
        m = _EPOCH_RE.match(p.name)
        if m and m["model"] == model:
            found[int(m["epoch"])] = p
    return [_uri(p) for _, p in sorted(found.items())]


def _f1(x: float) -> str:
    return f"{x:.1f}"


# ------------------------------------------------------------------ data helpers
def _index_rows(rows: list[dict]) -> dict[tuple[str, str, int], dict]:
    return {(r["model"], r["sampler"], int(r["nfe"])): r for r in rows}


def _nfe_grid(metrics: dict) -> list[int]:
    grid = metrics.get("meta", {}).get("nfe_grid")
    if grid:
        return [int(n) for n in grid]
    return sorted({int(r["nfe"]) for r in metrics["rows"]
                   if r["sampler"] in ("ddim", "euler", "ancestral_strided")})


def _classify(ddim: float, rf: float) -> str:
    if abs(ddim - rf) <= TIE_REL * max(ddim, rf):
        return "tie"
    return "rf" if rf < ddim else "ddim"


def _fmt_list(ns: list[int]) -> str:
    return ", ".join(str(n) for n in ns)


def crossover(idx: dict, grid: list[int]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {"rf": [], "ddim": [], "tie": []}
    for n in grid:
        a, b = idx.get(("ddpm", "ddim", n)), idx.get(("rf", "euler", n))
        if a and b:
            out[_classify(a["fid"], b["fid"])].append(n)
    return out


def first_within_floor(idx: dict, model: str, sampler: str, grid: list[int],
                       floor: float) -> int | None:
    for n in grid:
        r = idx.get((model, sampler, n))
        if r and r["fid"] <= floor + FLOOR_MARGIN:
            return n
    return None


# ------------------------------------------------------------------------ blurbs
def steps_blurb(idx: dict, grid: list[int]) -> str:
    c = crossover(idx, grid)
    parts = []
    if c["rf"]:
        parts.append(f"Rectified flow wins at {_fmt_list(c['rf'])} steps.")
    if c["ddim"]:
        parts.append(f"DDPM with DDIM wins at {_fmt_list(c['ddim'])} step"
                     f"{'s' if len(c['ddim']) > 1 else ''}.")
    if c["tie"]:
        parts.append(f"The two are within {int(TIE_REL * 100)}% of each other at "
                     f"{_fmt_list(c['tie'])} steps.")
    one_d, one_r = idx.get(("ddpm", "ddim", 1)), idx.get(("rf", "euler", 1))
    if one_d and one_r:
        parts.append(
            f"At one step DDIM scores {_f1(one_d['fid'])} and rectified flow "
            f"{_f1(one_r['fid'])}: its single Euler step lands on the average image, "
            f"a blur that covers few classes (coverage entropy "
            f"{one_r['coverage_entropy']:.2f} against {one_d['coverage_entropy']:.2f}).")
    strided = [n for n in grid if ("ddpm", "ancestral_strided", n) in idx
               and ("ddpm", "ddim", n) in idx
               and idx[("ddpm", "ancestral_strided", n)]["fid"] > idx[("ddpm", "ddim", n)]["fid"]]
    if strided:
        parts.append(f"Strided ancestral sampling is worse than DDIM at "
                     f"{_fmt_list(strided)} steps.")
    return " ".join(parts)


def takeaways(metrics: dict, idx: dict, grid: list[int]) -> list[str]:
    floor = metrics.get("floor", {}).get("fid")
    out: list[str] = []
    if floor is not None:
        out.append(f"The floor is FID {floor:.2f}: real training images scored against "
                   f"real test images. No generator can be expected to beat it.")
    c = crossover(idx, grid)
    low = [n for n in grid if n <= 5]
    if low:
        w = {n: ("rf" if n in c["rf"] else "ddim" if n in c["ddim"] else "tie") for n in low}
        bits = []
        for n in low:
            a, b = idx.get(("ddpm", "ddim", n)), idx.get(("rf", "euler", n))
            if a and b:
                who = {"rf": "rectified flow", "ddim": "DDPM (DDIM)", "tie": "a tie"}[w[n]]
                bits.append(f"{n} step{'s' if n > 1 else ''}: {who} "
                            f"({_f1(a['fid'])} DDIM vs {_f1(b['fid'])} RF)")
        if bits:
            out.append("With five steps or fewer: " + "; ".join(bits) + ".")
    high = [n for n in grid if n >= 50]
    hb = []
    for n in high:
        a, b = idx.get(("ddpm", "ddim", n)), idx.get(("rf", "euler", n))
        if a and b:
            hb.append(f"{n} steps {_f1(a['fid'])} vs {_f1(b['fid'])}")
    if hb:
        out.append("With 50 steps or more the models are close (DDIM vs RF): "
                   + "; ".join(hb) + ".")
    if floor is not None:
        cost = []
        for label, model, sampler in (("DDPM DDIM", "ddpm", "ddim"), ("RF Euler", "rf", "euler")):
            n = first_within_floor(idx, model, sampler, grid, floor)
            cost.append(f"{label} needs {n} steps" if n else f"{label} does not get there")
        ref = idx.get(("ddpm", "ancestral", 1000))
        tail = (f" Full 1000-step ancestral DDPM reaches {ref['fid']:.2f}." if ref else "")
        out.append(f"Reaching within {FLOOR_MARGIN:.0f} FID points of the floor: "
                   + "; ".join(cost) + "." + tail)
    out.append("Caveat: this is rectified flow without reflow (no 2-RF). Its learned "
               "paths are straight only on average, which is why one step collapses "
               "to a blur, and results come from one seed and one training run per model.")
    return out


# --------------------------------------------------------------------- loss chart
def _loss_svg(log_rows: list[dict], label: str, cls: str) -> str:
    w, h, pad = 420, 220, 34
    ys = [r["loss"] for r in log_rows]
    xs = [r["epoch"] for r in log_rows]
    lo, hi = min(ys), max(ys)
    span = (hi - lo) or 1.0
    x0, x1 = min(xs), max(xs)
    xspan = (x1 - x0) or 1
    pts = " ".join(
        f"{pad + (x - x0) / xspan * (w - 2 * pad):.1f},"
        f"{h - pad - (y - lo) / span * (h - 2 * pad):.1f}" for x, y in zip(xs, ys))
    return (
        f'<figure class="loss"><svg viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="{html.escape(label)} training loss per epoch">'
        f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" class="ax"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" class="ax"/>'
        f'<polyline points="{pts}" class="ln {cls}"/>'
        f'<text x="{pad}" y="{pad - 10}" class="tk">{hi:.3f}</text>'
        f'<text x="{pad}" y="{h - pad + 20}" class="tk">epoch {x0}</text>'
        f'<text x="{w - pad}" y="{h - pad + 20}" class="tk" text-anchor="end">epoch {x1}</text>'
        f'<text x="{w - pad}" y="{pad + 4}" class="tk" text-anchor="end">'
        f'final {ys[-1]:.3f}</text></svg>'
        f'<figcaption>{html.escape(label)}</figcaption></figure>')


# ----------------------------------------------------------------------- params
def _param_count(cfg: dict) -> int | None:
    try:  # torch is only needed for this badge
        from mnist_flow.config import Config
        from mnist_flow.unet import UNet
        from mnist_flow.utils import count_params
        return count_params(UNet(Config.from_dict(cfg)))
    except Exception as exc:  # noqa: BLE001 - badge is optional
        log.warning("could not count params: %s", exc)
        return None


# ------------------------------------------------------------------------- html
CSS = """
:root{--paper:#eef1f5;--panel:#fff;--ink:#17212b;--mute:#566474;--rule:#cdd5de;--acc:#1d3fd8;--ddpm:#33404d}
*{box-sizing:border-box}
html{font-size:18px;scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font-family:Palatino,"Palatino Linotype","Book Antiqua",Georgia,serif;line-height:1.6}
main,header,footer{max-width:1180px;margin:0 auto;padding:0 28px}
header{padding-top:72px;padding-bottom:36px}
h1{font-size:3.4rem;line-height:1.05;margin:0 0 .5em;letter-spacing:-.01em}
h1 .vs{color:var(--acc);font-style:italic}
h2{font-size:2rem;margin:0 0 .4em;line-height:1.15}
h3{font-size:1.15rem;margin:0 0 .3em}
section{padding:56px 0 20px;border-top:2px solid var(--ink)}
.thesis{font-size:1.35rem;max-width:44ch;margin:0 0 1.2em}
.blurb{max-width:64ch;margin:0 0 1.4em}
.badges{display:flex;flex-wrap:wrap;gap:8px;padding:0;margin:0;list-style:none;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:.8rem}
.badges li{border:1.5px solid var(--ink);padding:3px 11px;border-radius:99px;background:var(--panel)}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:24px}
.card{background:var(--panel);border-left:6px solid var(--ddpm);padding:20px 24px}
.card.rf{border-left-color:var(--acc)}
.eq{font-family:"SF Mono",Menlo,Consolas,monospace;font-size:.86rem;background:var(--paper);padding:10px 12px;margin:.6em 0;white-space:pre-wrap}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}
.three{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}
.panel{background:var(--panel);padding:16px 18px}
.panel h3{display:flex;align-items:center;gap:8px}
.sw{width:12px;height:12px;display:inline-block;background:var(--ddpm)}
.sw.rf{background:var(--acc)}
img.px{width:100%;max-width:484px;height:auto;image-rendering:pixelated;image-rendering:crisp-edges;display:block;margin:0 auto}
img.fig{width:100%;height:auto;display:block;background:#fff}
.ctl{display:flex;flex-wrap:wrap;align-items:center;gap:14px;margin:14px 0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:.9rem}
.ctl input[type=range]{flex:1 1 220px;accent-color:var(--acc);height:28px}
.ctl output{font-weight:700;min-width:7ch}
button{font:inherit;font-family:system-ui,sans-serif;font-size:.9rem;padding:7px 18px;border:2px solid var(--ink);background:var(--ink);color:#fff;cursor:pointer;border-radius:0}
button:hover{background:var(--acc);border-color:var(--acc)}
:focus-visible{outline:3px solid var(--acc);outline-offset:2px}
.stats{font-family:system-ui,-apple-system,sans-serif;font-size:.88rem;margin-top:10px;color:var(--mute)}
.stats b{color:var(--ink);font-size:1.3rem;display:block}
.loss svg{width:100%;height:auto;background:var(--panel)}
.loss .ax{stroke:var(--rule);stroke-width:1.5}
.loss .ln{fill:none;stroke-width:3}
.loss .ln.d{stroke:var(--ddpm)}.loss .ln.r{stroke:var(--acc)}
.loss .tk{font:11px system-ui,sans-serif;fill:var(--mute)}
.loss figcaption,.note{font-family:system-ui,sans-serif;font-size:.8rem;color:var(--mute);margin-top:4px}
figure{margin:0}
table{border-collapse:collapse;width:100%;font-family:system-ui,sans-serif;font-size:.8rem;background:var(--panel)}
th,td{padding:5px 10px;text-align:right;border-bottom:1px solid var(--rule)}
th:first-child,td:first-child{text-align:left}
details{margin:18px 0}summary{cursor:pointer;font-family:system-ui,sans-serif;font-size:.9rem}
ol.take{padding-left:1.3em;max-width:70ch}ol.take li{margin-bottom:.8em}
footer{padding:44px 28px 80px;font-family:system-ui,sans-serif;font-size:.85rem;color:var(--mute);border-top:2px solid var(--ink);margin-top:50px}
a{color:var(--acc)}
@media (max-width:900px){h1{font-size:2.4rem}.cards,.pair,.three{grid-template-columns:1fr}main,header,footer{padding-left:16px;padding-right:16px}}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
"""

JS = """
(function(){
var D=JSON.parse(document.getElementById('data').textContent);
function $(i){return document.getElementById(i)}
var tr=D.epochs;
if(tr&&$('s-ddpm')){
 var sd=$('s-ddpm'),sr=$('s-rf'),lock=$('lock'),play=$('play'),timer=null;
 function show(m,v){$('img-'+m).src=tr[m][v-1]||'';$('o-'+m).textContent='epoch '+v}
 function upd(m){show(m,+(m==='ddpm'?sd:sr).value)}
 function onIn(m){return function(){if(lock.checked){var o=m==='ddpm'?sr:sd,me=m==='ddpm'?sd:sr;o.value=Math.min(me.value,o.max);upd(m==='ddpm'?'rf':'ddpm')}upd(m)}}
 sd.addEventListener('input',onIn('ddpm'));sr.addEventListener('input',onIn('rf'));
 function stop(){clearInterval(timer);timer=null;play.textContent='Play'}
 play.addEventListener('click',function(){
  if(timer){stop();return}
  play.textContent='Pause';
  timer=setInterval(function(){
   var mx=Math.max(+sd.max,+sr.max),n=(+sd.value%mx)+1;
   sd.value=Math.min(n,sd.max);sr.value=Math.min(n,sr.max);upd('ddpm');upd('rf');
  },500)});
 upd('ddpm');upd('rf');
}
var ns=D.nfe,sn=$('s-nfe');
function cell(id,r){
 if(!r){$('img-'+id).removeAttribute('src');$('st-'+id).textContent='not run';return}
 $('img-'+id).src=r.uri;
 $('st-'+id).innerHTML='<b>FID '+r.fid.toFixed(1)+'</b>coverage entropy '+r.cov.toFixed(2)+' / confidence '+r.conf.toFixed(2);
}
function upN(){var n=ns[+sn.value];$('o-nfe').textContent=n+(n===1?' step':' steps');
 cell('ddim',D.rows['ddpm|ddim|'+n]);cell('strided',D.rows['ddpm|ancestral_strided|'+n]);cell('euler',D.rows['rf|euler|'+n]);}
if(sn){sn.max=ns.length-1;sn.addEventListener('input',upN);upN();}
})();
"""


def _epoch_section(d_n: int, r_n: int, loss_html: str) -> str:
    def panel(m: str, title: str, cls: str, n: int) -> str:
        return (f'<div class="panel"><h3><span class="sw {cls}"></span>{title}</h3>'
                f'<img class="px" id="img-{m}" alt="{title} samples, 8 by 8 grid" width="242" height="242">'
                f'<div class="ctl"><label for="s-{m}">Epoch</label>'
                f'<input type="range" id="s-{m}" min="1" max="{n}" value="1">'
                f'<output id="o-{m}" for="s-{m}">epoch 1</output></div></div>')
    return f"""
<section id="training"><h2>Training progress</h2>
<p class="blurb">Both models sample the same 64 starting noises after every epoch, using the EMA weights and 50 steps. Drag the sliders to watch digits emerge. Rectified flow tends to produce legible shapes earlier because its regression target is a plain velocity.</p>
<div class="pair">{panel('ddpm', 'DDPM', 'd', d_n)}{panel('rf', 'Rectified flow', 'rf', r_n)}</div>
<div class="ctl"><label><input type="checkbox" id="lock"> Lock sliders together</label><button id="play" type="button">Play</button></div>
<div class="pair">{loss_html}</div>
<p class="note">Loss values are not comparable across models: DDPM regresses unit-variance noise, rectified flow regresses a velocity field. Compare the shapes of the curves, not their heights.</p>
</section>"""


def _steps_section(blurb: str, has_ref: bool, has_chart: bool, table: str) -> str:
    def panel(k: str, title: str, cls: str) -> str:
        return (f'<div class="panel"><h3><span class="sw {cls}"></span>{title}</h3>'
                f'<img class="px" id="img-{k}" alt="{title} samples" width="242" height="242">'
                f'<div class="stats" id="st-{k}"></div></div>')
    ref = ('<div class="panel"><h3>DDPM ancestral, 1000 steps (reference)</h3>'
           '<img class="px" id="img-ref" alt="DDPM ancestral 1000 step samples" width="242" height="242">'
           '<div class="stats" id="st-ref"></div></div>') if has_ref else ""
    chart = ('<div class="panel"><h3>FID against steps</h3>'
             '<img class="fig" id="img-chart" alt="FID versus number of function evaluations"></div>'
             ) if has_chart else ""
    extra = f'<div class="pair" style="margin-top:20px">{ref}{chart}</div>' if (ref or chart) else ""
    return f"""
<section id="steps"><h2>Steps against quality</h2>
<p class="blurb">{html.escape(blurb)}</p>
<div class="ctl"><label for="s-nfe">Sampling steps</label><input type="range" id="s-nfe" min="0" max="0" value="0"><output id="o-nfe" for="s-nfe"></output></div>
<div class="three">{panel('ddim', 'DDPM, DDIM', 'd')}{panel('strided', 'DDPM, strided ancestral', 'd')}{panel('euler', 'Rectified flow, Euler', 'rf')}</div>
{extra}
<details><summary>All numbers</summary>{table}</details>
</section>"""


def _table(metrics: dict) -> str:
    head = "<tr><th>run</th><th>steps</th><th>FID</th><th>coverage entropy</th><th>confidence</th></tr>"
    body = []
    rows = [metrics["floor"]] if metrics.get("floor") else []
    rows += metrics["rows"]
    for r in rows:
        body.append(f"<tr><td>{html.escape(str(r.get('name', '')))}</td><td>{r.get('nfe', '')}</td>"
                    f"<td>{_f1(r['fid'])}</td><td>{r['coverage_entropy']:.2f}</td>"
                    f"<td>{r['mean_confidence']:.2f}</td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


def _badges(meta: dict, params: int | None) -> str:
    cfg = (meta.get("configs") or {}).get("ddpm") or {}
    items = []
    if params:
        items.append(f"{params / 1e6:.1f}M-parameter shared UNet")
    if cfg:
        items += [f"{cfg['epochs']} epochs", f"batch {cfg['batch_size']}",
                  f"Adam, lr {cfg['lr']:g}", f"EMA {cfg['ema_decay']:g}"]
    if meta.get("nfe_grid"):
        items.append(f"{len(meta['nfe_grid'])} step budgets, {meta['nfe_grid'][0]} to {meta['nfe_grid'][-1]}")
    if meta.get("n"):
        items.append(f"same noise bank, {meta['n']:,} samples each")
    if meta.get("classifier_test_accuracy"):
        items.append(f"same classifier ({meta['classifier_test_accuracy'] * 100:.1f}% test accuracy)")
    return '<ul class="badges">' + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>"


def _methods_section() -> str:
    return """
<section id="methods"><h2>How each model learns</h2>
<div class="cards">
<div class="card"><h3>DDPM</h3>
<p>Corrupt a clean image with a fixed amount of Gaussian noise, then train the network to name the noise.</p>
<div class="eq">xₜ = √āₜ x₀ + √(1 − āₜ) ε,  ε ~ N(0, I)
loss = E ‖ ε − εθ(xₜ, t) ‖²</div>
<p>Sampling walks 1000 noise levels back to an image, or skips levels with DDIM.</p></div>
<div class="card rf"><h3>Rectified flow</h3>
<p>Draw a straight line between noise and a clean image, then train the network to predict the direction of travel.</p>
<div class="eq">xₜ = (1 − t) x₀ + t ε
loss = E ‖ (ε − x₀) − vθ(xₜ, t) ‖²</div>
<p>Sampling integrates the learned velocity from noise to image with Euler steps.</p></div>
</div></section>"""


def build(results: Path, runs: Path, runs_viz: Path, out: Path,
          docs_out: Path | None = None) -> Path:
    """Build the presentation and return the output path."""
    results, runs, runs_viz, out = Path(results), Path(runs), Path(runs_viz), Path(out)
    metrics = _load_json(results / "metrics.json")
    if metrics is None:
        raise FileNotFoundError(f"metrics.json not found in {results}")
    meta, rows = metrics.get("meta", {}), metrics["rows"]
    idx, grid = _index_rows(rows), _nfe_grid(metrics)
    fig = results / "figures"

    data_rows: dict[str, dict] = {}
    for r in rows:
        p = fig / f"grid_{r['name']}.png"
        uri = _opt_uri(p)
        if uri:
            data_rows[f"{r['model']}|{r['sampler']}|{int(r['nfe'])}"] = {
                "uri": uri, "fid": r["fid"], "cov": r["coverage_entropy"],
                "conf": r["mean_confidence"]}

    data: dict[str, Any] = {"nfe": grid, "rows": data_rows, "epochs": None}
    sections = [_methods_section()]

    # training progress
    epochs_dir = runs_viz / "epoch_samples"
    d_eps = _epoch_uris(epochs_dir, "ddpm") if epochs_dir.is_dir() else []
    r_eps = _epoch_uris(epochs_dir, "rf") if epochs_dir.is_dir() else []
    logs = {m: (_load_json(runs / f"train_{m}.json") or _load_json(runs_viz / f"train_{m}.json"))
            for m in ("ddpm", "rf")}
    if d_eps and r_eps:
        data["epochs"] = {"ddpm": d_eps, "rf": r_eps}
        loss_html = "".join(_loss_svg(logs[m], lbl, c) for m, lbl, c in
                            (("ddpm", "DDPM loss", "d"), ("rf", "Rectified flow loss", "r"))
                            if logs[m])
        sections.append(_epoch_section(len(d_eps), len(r_eps), loss_html))
    else:
        log.warning("no per-epoch grids under %s; skipping training section", epochs_dir)

    # steps vs quality
    ref_row = idx.get(("ddpm", "ancestral", 1000))
    has_ref = bool(ref_row and f"ddpm|ancestral|1000" in data_rows)
    chart_uri = _opt_uri(fig / "fid_vs_nfe.png")
    data["chart"] = chart_uri
    sections.append(_steps_section(steps_blurb(idx, grid), has_ref, bool(chart_uri), _table(metrics)))

    # trajectories
    strip, gif = _opt_uri(fig / "trajectory_strip.png"), _opt_uri(fig / "trajectory.gif")
    if strip or gif:
        s = ('<section id="paths"><h2>Same noise, two paths</h2>'
             '<p class="blurb">Each model starts from the same noise and ends at an image. '
             'Rectified flow is trained to travel in straight lines, so its intermediate '
             'frames fade smoothly toward the digit. The diffusion path bends: most of the '
             'structure appears late, once the noise level is low.</p>')
        if strip:
            s += f'<img class="fig" alt="Trajectory strip for both models" src="{strip}">'
        if gif:
            s += (f'<img class="fig" style="max-width:640px;margin-top:20px" '
                  f'alt="Animated trajectories" src="{gif}">')
        sections.append(s + "</section>")

    items = "".join(f"<li>{html.escape(t)}</li>" for t in takeaways(metrics, idx, grid))
    sections.append(f'<section id="takeaways"><h2>Takeaways</h2><ol class="take">{items}</ol></section>')

    secs = [r["seconds"] for m in logs.values() if m for r in m]
    per_epoch = f"about {sum(secs) / len(secs):.0f} s per epoch" if secs else "per-epoch timing not available"
    cfg = (meta.get("configs") or {}).get("ddpm") or {}
    params = _param_count(cfg) if cfg else None
    footer = (f"<footer>Trained on {html.escape(str(meta.get('device', 'a GPU')))} with torch "
              f"{html.escape(str(meta.get('torch_version', '?')))}, {per_epoch}. Evaluation uses "
              f"EMA weights and a noise bank seeded {meta.get('noise_seed', '?')}. Code and "
              f'results: <a href="{GITHUB_URL}">{GITHUB_URL}</a></footer>')
    if has_ref:
        data["ref"] = data_rows["ddpm|ancestral|1000"]

    header = (f'<header><h1>DDPM <span class="vs">vs</span> Rectified Flow on MNIST</h1>'
              f'<p class="thesis">{html.escape(_thesis(idx, grid))}</p>{_badges(meta, params)}</header>')
    js_data = json.dumps(data).replace("</", "<\\/")
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>DDPM vs Rectified Flow on MNIST</title><style>{CSS}</style></head><body>'
            f'{header}<main>{"".join(sections)}</main>{footer}'
            f'<script id="data" type="application/json">{js_data}</script>'
            f'<script>{JS}{_REF_JS if has_ref or chart_uri else ""}</script></body></html>')
    for target in (out, docs_out):
        if target:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            Path(target).write_text(page, encoding="utf-8")
    return out


_REF_JS = """
(function(){var D=JSON.parse(document.getElementById('data').textContent);
var r=document.getElementById('img-ref');
if(r&&D.ref){r.src=D.ref.uri;document.getElementById('st-ref').innerHTML='<b>FID '+D.ref.fid.toFixed(1)+'</b>coverage entropy '+D.ref.cov.toFixed(2)+' / confidence '+D.ref.conf.toFixed(2)}
var c=document.getElementById('img-chart');if(c&&D.chart)c.src=D.chart})();
"""


def _thesis(idx: dict, grid: list[int]) -> str:
    c = crossover(idx, grid)
    if c["rf"] and c["tie"]:
        return ("Straight-line training buys rectified flow quality at very few steps, "
                "and the advantage disappears once the budget grows.")
    return "Same network, same data, same noise: how do the two samplers compare as the step budget changes?"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--runs-viz", default="runs_viz")
    ap.add_argument("--out", default="presentation/index.html")
    ap.add_argument("--docs-out", default="docs/index.html", help="empty string disables")
    a = ap.parse_args()
    out = build(Path(a.results), Path(a.runs), Path(a.runs_viz), Path(a.out),
                Path(a.docs_out) if a.docs_out else None)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
