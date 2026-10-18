#!/usr/bin/env python3
# Generate animated GIFs for dsv4_offload docs, styled per diagram-design skill.
# Toolchain: cairosvg (SVG -> PNG) + Pillow (frames -> GIF). No ffmpeg on this box.
import os
import cairosvg
from PIL import Image

OUT = "/home/userlist/ww/dsv4_offload/docs/diagrams"
TMP = "/tmp/diagrams/frames"
os.makedirs(TMP, exist_ok=True)

W, H = 960, 660
PAPER = "#f5f5f5"
INK = "#2d3142"
MUTED = "#4f5d75"
ACCENT = "#bf4520"
LABEL = "#5a6580"
MONO = "'Geist Mono', ui-monospace, monospace"
SANS = "'Geist', system-ui, sans-serif"
SERIF = "'Instrument Serif', serif"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, size=9, fill=LABEL, font=MONO, anchor="middle",
        weight=None, ls=None, opacity=None):
    p = [f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-family="{font}"']
    if anchor:
        p.append(f' text-anchor="{anchor}"')
    if weight:
        p.append(f' font-weight="{weight}"')
    if ls is not None:
        p.append(f' letter-spacing="{ls}"')
    if opacity is not None:
        p.append(f' opacity="{opacity:.3f}"')
    p.append(f'>{esc(s)}</text>')
    return "".join(p)


DEFS = (
    f'<marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
    f'<polygon points="0 0, 8 3, 0 6" fill="{MUTED}"/></marker>'
    f'<marker id="arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
    f'<polygon points="0 0, 8 3, 0 6" fill="{ACCENT}"/></marker>'
)


def svg(body):
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">'
            f'<defs>{DEFS}</defs>'
            f'<rect width="{W}" height="{H}" fill="{PAPER}"/>{body}</svg>')


def render(body, path):
    cairosvg.svg2png(bytestring=svg(body).encode("utf-8"), write_to=path,
                     output_width=W, output_height=H)


def make_gif(paths, out, duration=90):
    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=128)
              for p in paths]
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True, disposal=2)


def emit(name, bodies, duration, poster_index):
    paths = []
    for i, b in enumerate(bodies):
        p = os.path.join(TMP, f"{name}_{i:03d}.png")
        render(b, p)
        paths.append(p)
    make_gif(paths, os.path.join(OUT, f"{name}.gif"), duration)
    Image.open(paths[poster_index]).convert("RGB").save(
        os.path.join(OUT, f"{name}.png"))
    for p in paths:
        os.remove(p)
    print("wrote", name, len(bodies), "frames")


# =====================================================================
# shared sequence-diagram geometry (from docs/diagrams/sequence.svg)
# =====================================================================
SEQ = [
    dict(n=1,  x1=124, y1=140, x2=296, y2=140, dash=False, accent=False,
         label="forward(hidden_states)", lx=146, ly=120, lw=128, tx=210, ty=131),
    dict(n=2,  x1=304, y1=172, x2=476, y2=172, dash=False, accent=False,
         label="dispatch(indices)", lx=338, ly=152, lw=104, tx=390, ty=163),
    dict(n=3,  x1=484, y1=204, x2=836, y2=204, dash=False, accent=False,
         label="lookup(slot)", lx=562, ly=184, lw=76, tx=600, ty=195),
    dict(n=4,  x1=836, y1=236, x2=484, y2=236, dash=True, accent=False,
         label="hit / miss", lx=736, ly=216, lw=68, tx=770, ty=227),
    dict(n=5,  x1=484, y1=316, x2=656, y2=316, dash=False, accent=False,
         label="fetch(expert ids)", lx=518, ly=296, lw=104, tx=570, ty=307),
    dict(n=6,  x1=656, y1=348, x2=484, y2=348, dash=True, accent=False,
         label="MXFP4 w + scales", lx=522, ly=328, lw=96, tx=570, ty=339),
    dict(n=7,  x1=484, y1=380, x2=836, y2=380, dash=False, accent=False,
         label="H2D 20-51 GB/s", lx=554, ly=360, lw=92, tx=600, ty=371),
    dict(n=8,  x1=484, y1=424, x2=836, y2=424, dash=False, accent=True,
         label="npu_grouped_matmul", lx=542, ly=404, lw=108, tx=596, ty=415),
    dict(n=9,  x1=836, y1=456, x2=484, y2=456, dash=True, accent=False,
         label="per-expert outputs", lx=706, ly=436, lw=108, tx=760, ty=447),
    dict(n=10, x1=476, y1=488, x2=304, y2=488, dash=True, accent=False,
         label="routed outputs", lx=346, ly=468, lw=88, tx=390, ty=479),
    dict(n=11, x1=296, y1=520, x2=124, y2=520, dash=True, accent=False,
         label="hidden_states", lx=168, ly=500, lw=84, tx=210, ty=511),
]
LIFELINES = [120, 300, 480, 660, 840]
BARS = [(116, 136, 392), (296, 136, 392), (476, 168, 332), (656, 312, 44), (836, 200, 300)]
BAR_START = [1, 1, 2, 5, 3]
ACTORS = [
    (44, "RUNNER", 40, "vLLM Runner", "scheduler", "external"),
    (224, "MOE", 28, "Offload MoE", "gate / router", "backend"),
    (404, "ENGINE", 40, "Offload Engine", "ExpertOffloadEngine", "focal"),
    (584, "HOST", 32, "Host Bank", "packed MXFP4", "store"),
    (764, "CACHE", 36, "Slot Cache", "NPU bf16 LRU", "backend"),
]


def seq_base(op=1.0):
    s = (f'<rect x="460" y="256" width="400" height="136" rx="4" '
         f'fill="rgba(45,49,66,0.02)" stroke="rgba(45,49,66,0.22)" stroke-width="1" '
         f'opacity="{op}"/>')
    for x in LIFELINES:
        s += (f'<line x1="{x}" y1="104" x2="{x}" y2="552" stroke="rgba(45,49,66,0.20)" '
              f'stroke-width="1" stroke-dasharray="3,3" opacity="{op}"/>')
    return s


def seq_bars(op=1.0, upto=11):
    s = ""
    for (bx, by, bh), st in zip(BARS, BAR_START):
        if st <= upto:
            s += (f'<rect x="{bx}" y="{by}" width="8" height="{bh}" '
                  f'fill="rgba(45,49,66,0.06)" stroke="{MUTED}" stroke-width="0.8" '
                  f'opacity="{op}"/>')
    return s


def msg_line(m, frac=1.0, op=1.0):
    xe = m["x1"] + (m["x2"] - m["x1"]) * frac
    ye = m["y1"] + (m["y2"] - m["y1"]) * frac
    stroke = ACCENT if m["accent"] else MUTED
    sw = 1.2 if m["accent"] else 1.0
    dash = ' stroke-dasharray="5,4"' if m["dash"] else ""
    mk = ""
    if frac >= 0.999:
        mk = (' marker-end="url(#arrow-accent)"' if m["accent"]
              else ' marker-end="url(#arrow)"')
    return (f'<line x1="{m["x1"]}" y1="{m["y1"]}" x2="{xe:.1f}" y2="{ye:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{dash}{mk} opacity="{op:.3f}"/>')


def msg_label(m, op=1.0):
    fc = ACCENT if m["accent"] else LABEL
    return (f'<g opacity="{op:.3f}">'
            f'<rect x="{m["lx"]}" y="{m["ly"]}" width="{m["lw"]}" height="12" rx="2" '
            f'fill="{PAPER}"/>'
            f'{txt(m["tx"], m["ty"], m["label"], size=8, fill=fc, ls="0.06em")}</g>')


def seq_fragtab():
    return (f'<rect x="460" y="256" width="44" height="16" rx="2" fill="{PAPER}" '
            f'stroke="rgba(45,49,66,0.22)" stroke-width="1"/>'
            + txt(482, 268, "OPT", size=8, fill=MUTED, ls="0.12em")
            + txt(472, 286, "[cache miss]", size=8, fill=MUTED, anchor="start",
                  ls="0.04em"))


def seq_actors():
    s = ""
    for x, tag, tagw, title, sub, kind in ACTORS:
        if kind == "focal":
            fill, stroke = "rgba(191,69,32,0.08)", ACCENT
            tagsc, tagfc = "rgba(191,69,32,0.40)", "rgba(191,69,32,0.80)"
        elif kind == "store":
            fill, stroke = "rgba(45,49,66,0.05)", MUTED
            tagsc, tagfc = "rgba(45,49,66,0.40)", "rgba(45,49,66,0.80)"
        elif kind == "external":
            fill, stroke = "rgba(45,49,66,0.03)", "rgba(45,49,66,0.30)"
            tagsc, tagfc = "rgba(45,49,66,0.40)", "rgba(45,49,66,0.80)"
        else:
            fill, stroke = "#ffffff", INK
            tagsc, tagfc = "rgba(45,49,66,0.40)", "rgba(45,49,66,0.80)"
        s += (f'<rect x="{x}" y="48" width="152" height="56" rx="6" fill="{fill}" '
              f'stroke="{stroke}" stroke-width="1"/>')
        s += (f'<rect x="{x+8}" y="54" width="{tagw}" height="12" rx="2" '
              f'fill="transparent" stroke="{tagsc}" stroke-width="0.8"/>')
        s += txt(x + 8 + tagw / 2, 63, tag, size=7, fill=tagfc, ls="0.08em")
        s += txt(x + 76, 77, title, size=12, fill=INK, font=SANS, weight=600)
        s += txt(x + 76, 93, sub, size=9, fill=MUTED)
    return s


def seq_legend():
    s = ('<line x1="40" y1="584" x2="920" y2="584" stroke="rgba(45,49,66,0.10)" '
         'stroke-width="0.8"/>')
    s += txt(40, 604, "LEGEND", size=8, fill=MUTED, anchor="start", ls="0.14em")
    s += (f'<line x1="120" y1="600" x2="154" y2="600" stroke="{MUTED}" stroke-width="1" '
          f'marker-end="url(#arrow)"/>')
    s += txt(164, 604, "CALL", size=8, fill=MUTED, anchor="start", ls="0.14em")
    s += (f'<line x1="250" y1="600" x2="284" y2="600" stroke="{MUTED}" stroke-width="1" '
          f'stroke-dasharray="5,4" marker-end="url(#arrow)"/>')
    s += txt(294, 604, "RETURN", size=8, fill=MUTED, anchor="start", ls="0.14em")
    s += (f'<line x1="400" y1="600" x2="434" y2="600" stroke="{ACCENT}" stroke-width="1.2" '
          f'marker-end="url(#arrow-accent)"/>')
    s += txt(444, 604, "HEADLINE", size=8, fill=MUTED, anchor="start", ls="0.14em")
    s += ('<rect x="560" y="594" width="12" height="12" rx="2" '
          'fill="rgba(45,49,66,0.02)" stroke="rgba(45,49,66,0.22)" stroke-width="1"/>')
    s += txt(580, 604, "OPT FRAGMENT", size=8, fill=MUTED, anchor="start", ls="0.14em")
    s += ('<line x1="720" y1="600" x2="752" y2="600" stroke="rgba(45,49,66,0.20)" '
          'stroke-width="1" stroke-dasharray="3,3"/>')
    s += txt(762, 604, "LIFELINE", size=8, fill=MUTED, anchor="start", ls="0.14em")
    return s


# =====================================================================
# (4) sequence reveal: one message at a time
# =====================================================================
def reveal_body(shown, grow, frac):
    s = seq_base() + seq_bars(upto=(shown if grow is None else max(shown, grow)))
    for m in SEQ:
        if m["n"] <= shown:
            s += msg_line(m, 1.0, 1.0)
        elif m["n"] == grow:
            s += msg_line(m, frac, 1.0)
    for m in SEQ:
        if m["n"] <= shown:
            s += msg_label(m, 1.0)
        elif m["n"] == grow:
            s += msg_label(m, min(1.0, frac * 1.4))
    return s + seq_fragtab() + seq_actors() + seq_legend()


def build_reveal():
    bodies = [reveal_body(0, None, 0)]
    for i in range(1, 12):
        bodies.append(reveal_body(i - 1, i, 0.55))
        bodies.append(reveal_body(i - 1, i, 1.00))
        bodies.append(reveal_body(i, None, 0))   # settled
        bodies.append(reveal_body(i, None, 0))   # hold so it can be read
    bodies += [reveal_body(11, None, 0)] * 3     # longer final hold
    emit("sequence-reveal", bodies, 150, len(bodies) - 1)


# =====================================================================
# (1) decode step: guided tour, one phase highlighted at a time
# =====================================================================
PHASES = [
    ("GATE & ROUTE", [1, 2]),
    ("SLOT LOOKUP", [3, 4]),
    ("H2D FETCH", [5, 6, 7]),
    ("GROUPED GEMM", [8]),
    ("WRITE-BACK", [9, 10, 11]),
]
PHASE_FRAMES = 9


def decode_body(pi, f):
    active = set(PHASES[pi][1])
    s = txt(44, 20, f"DECODE STEP {pi+1} / 5", size=8, fill=MUTED,
            anchor="start", ls="0.16em")
    s += txt(44, 40, PHASES[pi][0], size=17, fill=INK, font=SERIF, anchor="start")
    segx = 916 - (5 * 30 + 4 * 6)
    for k in range(5):
        on = k == pi
        s += (f'<rect x="{segx + k*36}" y="22" width="30" height="5" rx="2.5" '
              f'fill="{ACCENT if on else "rgba(45,49,66,0.15)"}"/>')
    s += txt(916, 40, "offload MoE forward", size=8, fill=MUTED,
             anchor="end", ls="0.10em")
    s += seq_base()
    for (bx, by, bh) in BARS:
        s += (f'<rect x="{bx}" y="{by}" width="8" height="{bh}" '
              f'fill="rgba(45,49,66,0.06)" stroke="{MUTED}" stroke-width="0.8"/>')
    for m in SEQ:
        s += msg_line(m, 1.0, 1.0 if m["n"] in active else 0.13)
    for m in SEQ:
        s += msg_label(m, 1.0 if m["n"] in active else 0.13)
    for j, n in enumerate(PHASES[pi][1]):
        m = SEQ[n - 1]
        local = (f - 0.10 * j) / 0.62
        if 0.0 < local < 1.0:
            px = m["x1"] + (m["x2"] - m["x1"]) * local
            py = m["y1"] + (m["y2"] - m["y1"]) * local
            s += (f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6.5" fill="{ACCENT}" '
                  f'opacity="0.15"/>')
            s += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="{ACCENT}"/>'
    return s + seq_fragtab() + seq_actors() + seq_legend()


def build_decode():
    bodies = []
    for pi in range(len(PHASES)):
        for k in range(PHASE_FRAMES):
            bodies.append(decode_body(pi, k / (PHASE_FRAMES - 1)))
        bodies.append(decode_body(pi, 1.0))   # hold at phase end
        bodies.append(decode_body(pi, 1.0))
    emit("decode-step", bodies, 150, len(bodies) - 1)


# =====================================================================
# (3) storage flow: particles along H2D / evict / refill arrows
# =====================================================================
ZONES = [(40, 64, 880, 116), (40, 240, 880, 116), (40, 416, 880, 116)]
# arrow: x1,y1,x2,y2, color, dashed, offset, target-zone
FLOWS = [
    (660, 240, 660, 180, ACCENT, False, 0.00, 0),
    (780, 180, 780, 240, MUTED,  True,  0.33, 1),
    (660, 416, 660, 356, MUTED,  False, 0.66, 1),
]


def storage_body(p):
    flashes = [0.0, 0.0, 0.0]
    for (x1, y1, x2, y2, col, dash, off, tgt) in FLOWS:
        f = (p + off) % 1.0
        if f > 0.85:
            flashes[tgt] = max(flashes[tgt], (f - 0.85) / 0.15)
    s = txt(44, 22, "STORAGE · DSV4-PLUGIN", size=8, fill=MUTED,
            anchor="start", ls="0.18em")
    s += txt(44, 44, "Expert weights flow across three tiers", size=16,
             fill=INK, font=SERIF, anchor="start")
    s += txt(916, 22, "H2D \u00b7 EVICT \u00b7 REFILL", size=8, fill=MUTED,
             anchor="end", ls="0.14em")
    # zones
    for i, (zx, zy, zw, zh) in enumerate(ZONES):
        s += (f'<rect x="{zx}" y="{zy}" width="{zw}" height="{zh}" rx="6" '
              f'fill="rgba(45,49,66,0.02)" stroke="rgba(45,49,66,0.10)" stroke-width="1"/>')
        if flashes[i] > 0.01:
            s += (f'<rect x="{zx}" y="{zy}" width="{zw}" height="{zh}" rx="6" '
                  f'fill="rgba(191,69,32,{0.05*flashes[i]:.3f})" stroke="{ACCENT}" '
                  f'stroke-width="1.2" opacity="{0.85*flashes[i]:.3f}"/>')
    # arrows
    for (x1, y1, x2, y2, col, dash, off, tgt) in FLOWS:
        d = ' stroke-dasharray="4,3"' if dash else ""
        mk = 'url(#arrow-accent)' if col == ACCENT else 'url(#arrow)'
        sw = 1.2 if col == ACCENT else 1.0
        s += (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
              f'stroke-width="{sw}"{d} marker-end="{mk}"/>')
    # arrow labels (mask + text)
    s += (f'<rect x="600" y="204" width="120" height="12" rx="2" fill="{PAPER}"/>'
          + txt(660, 213, "H2D 19.8-51.3 GB/s", size=8, ls="0.06em"))
    s += (f'<rect x="742" y="204" width="76" height="12" rx="2" fill="{PAPER}"/>'
          + txt(780, 213, "EVICT (LRU)", size=8, ls="0.06em"))
    s += (f'<rect x="618" y="380" width="84" height="12" rx="2" fill="{PAPER}"/>'
          + txt(660, 389, "REFILL", size=8, ls="0.06em"))
    # zone labels + notes
    s += txt(64, 82, "HBM \u00b7 ACCELERATOR \u00b7 128 GB", size=8, fill=MUTED,
             anchor="start", ls="0.14em")
    s += txt(64, 258, "PINNED HOST RAM \u00b7 628 GB", size=8, fill=MUTED,
             anchor="start", ls="0.14em")
    s += txt(64, 434, "L3 NVMe \u00b7 COLD CAPACITY", size=8, fill=MUTED,
             anchor="start", ls="0.14em")
    s += txt(64, 170, "dense + KV + graph \u00b7 ~19-29 GB resident", size=9,
             fill=LABEL, anchor="start")
    s += txt(64, 346, "94% of all weights \u00b7 eviction target", size=9,
             fill=LABEL, anchor="start")
    s += txt(64, 522, "NVMe-backed \u00b7 prewarmed before use", size=9,
             fill=LABEL, anchor="start")
    # nodes
    s += _node(64, 92, 248, "HOT", 28, "Hot Expert Slots", "bf16 \u00b7 32-80 GB", True)
    s += _node(64, 268, 248, "POOL", 36, "Packed Expert Pool", "MXFP4 \u00b7 ~150 GB", False)
    s += _node(64, 444, 248, "COLD", 36, "Cold Expert Tier", "on-demand refill", False)
    # particle trails
    for (x1, y1, x2, y2, col, dash, off, tgt) in FLOWS:
        f = (p + off) % 1.0
        for k, d in enumerate((0.0, 0.05, 0.10)):
            ff = f - d
            if 0.0 <= ff <= 1.0:
                px = x1 + (x2 - x1) * ff
                py = y1 + (y2 - y1) * ff
                r = 3.6 - k * 0.9
                op = 1.0 - k * 0.38
                s += (f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r:.1f}" '
                      f'fill="{col}" opacity="{op:.2f}"/>')
    # legend
    s += ('<line x1="40" y1="580" x2="920" y2="580" stroke="rgba(45,49,66,0.10)" '
          'stroke-width="0.8"/>')
    s += txt(40, 600, "LEGEND", size=8, fill=MUTED, anchor="start", ls="0.14em")
    s += ('<rect x="120" y="588" width="12" height="12" rx="2" '
          'fill="rgba(191,69,32,0.08)" stroke="#bf4520" stroke-width="1"/>')
    s += txt(140, 600, "FOCAL", size=8, fill=MUTED, anchor="start", ls="0.14em")
    s += ('<rect x="290" y="588" width="12" height="12" rx="2" '
          'fill="rgba(45,49,66,0.05)" stroke="#4f5d75" stroke-width="1"/>')
    s += txt(310, 600, "STORE", size=8, fill=MUTED, anchor="start", ls="0.14em")
    s += ('<line x1="630" y1="594" x2="652" y2="594" stroke="#bf4520" stroke-width="1.2"/>')
    s += txt(662, 600, "TRANSFER IN", size=8, fill=MUTED, anchor="start", ls="0.14em")
    s += ('<line x1="800" y1="594" x2="822" y2="594" stroke="#4f5d75" stroke-width="1" '
          'stroke-dasharray="4,3"/>')
    s += txt(832, 600, "EVICT", size=8, fill=MUTED, anchor="start", ls="0.14em")
    return s


def _node(x, y, w, tag, tagw, title, sub, focal):
    if focal:
        fill, stroke = "rgba(191,69,32,0.08)", ACCENT
        tagsc, tagfc = "rgba(191,69,32,0.40)", "rgba(191,69,32,0.80)"
    else:
        fill, stroke = "rgba(45,49,66,0.05)", MUTED
        tagsc, tagfc = "rgba(45,49,66,0.40)", "rgba(45,49,66,0.80)"
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="56" rx="6" fill="{fill}" '
         f'stroke="{stroke}" stroke-width="1"/>')
    s += (f'<rect x="{x+8}" y="{y+6}" width="{tagw}" height="12" rx="2" '
          f'fill="transparent" stroke="{tagsc}" stroke-width="0.8"/>')
    s += txt(x + 8 + tagw / 2, y + 15, tag, size=7, fill=tagfc, ls="0.08em")
    s += txt(x + w / 2, y + 29, title, size=12, fill=INK, font=SANS, weight=600)
    s += txt(x + w / 2, y + 45, sub, size=9, fill=MUTED)
    return s


def build_storage():
    n = 30
    bodies = [storage_body(i / n) for i in range(n)]
    emit("storage-flow", bodies, 160, 0)


# =====================================================================
# (2) LRU expert-slot evolution (new diagram)
# =====================================================================
LRU_STEPS = [7, 3, 7, 11, 3, 7, 15, 11, 2, 7, 15, 9, 3, 2, 13, 7, 9, 15]
NSLOT = 5
SLOT_W, SLOT_H, SLOT_GAP, SLOT_X0, SLOT_Y = 160, 88, 18, 44, 100
SLOT_STEP = SLOT_W + SLOT_GAP


def simulate():
    slots = [None] * NSLOT
    rec = []
    hits = misses = 0
    hist = []
    for s, e in enumerate(LRU_STEPS):
        pre_slots, pre_rec = list(slots), list(rec)
        if e in slots:
            hit, victim = True, None
            idx = slots.index(e)
            rec.remove(e)
            rec.append(e)
            hits += 1
        else:
            hit = False
            misses += 1
            if None in slots:
                idx = slots.index(None)
                victim = None
            else:
                victim = rec.pop(0)
                idx = slots.index(victim)
            slots[idx] = e
            rec.append(e)
        hist.append(dict(step=s, e=e, hit=hit, victim=victim, idx=idx,
                         pre=list(pre_slots), post=list(slots),
                         rec_pre=pre_rec, rec_post=list(rec),
                         hits=hits, misses=misses))
    return hist


def slot_card(i, e, state, note, op=1.0):
    x = SLOT_X0 + i * SLOT_STEP
    y = SLOT_Y
    if state == "empty":
        fill, stroke, sw, dash = "rgba(45,49,66,0.02)", "rgba(45,49,66,0.14)", 1.0, ' stroke-dasharray="4,3"'
    elif state == "victim":
        fill, stroke, sw, dash = "rgba(191,69,32,0.10)", ACCENT, 1.3, ' stroke-dasharray="4,3"'
    elif state == "hit":
        fill, stroke, sw, dash = "rgba(45,49,66,0.08)", ACCENT, 1.4, ""
    elif state == "new":
        fill, stroke, sw, dash = "rgba(191,69,32,0.08)", ACCENT, 1.4, ""
    else:
        fill, stroke, sw, dash = "rgba(45,49,66,0.05)", MUTED, 1.0, ""
    s = f'<g opacity="{op:.3f}">'
    s += (f'<rect x="{x}" y="{y}" width="{SLOT_W}" height="{SLOT_H}" rx="6" fill="{fill}" '
          f'stroke="{stroke}" stroke-width="{sw}"{dash}/>')
    s += (f'<rect x="{x+10}" y="{y+10}" width="34" height="14" rx="2" fill="transparent" '
          f'stroke="rgba(45,49,66,0.30)" stroke-width="0.8"/>')
    s += txt(x + 27, y + 20, f"SLOT {i}", size=7, fill="rgba(45,49,66,0.70)", ls="0.06em")
    if e is None:
        s += txt(x + SLOT_W / 2, y + 54, "\u2014", size=20, fill="rgba(45,49,66,0.25)",
                 font=SANS)
        s += txt(x + SLOT_W / 2, y + 72, "empty", size=8, fill="rgba(45,49,66,0.35)", ls="0.10em")
    else:
        s += txt(x + SLOT_W / 2, y + 56, f"E{e}", size=22, fill=INK, font=SANS, weight=600)
        s += txt(x + SLOT_W / 2, y + 74, note, size=8,
                 fill=ACCENT if state in ("victim", "new", "hit") else LABEL, ls="0.06em")
    s += "</g>"
    return s


HIST = simulate()
ORD_Y = 224
ORD_W, ORD_GAP = 56, 8


def lru_body(hi, phase, f):
    h = HIST[hi]
    post = phase == 2
    slots = h["post"] if post else h["pre"]
    rec = h["rec_post"] if post else h["rec_pre"]
    pulse = 0.0
    if phase == 0:
        pulse = 0.35 + 0.25 * f
    elif phase == 1:
        pulse = 1.0
    # ---- header
    s = txt(44, 22, "CACHE \u00b7 DSV4-PLUGIN", size=8, fill=MUTED,
            anchor="start", ls="0.18em")
    s += txt(44, 44, "LRU expert-slot evolution", size=16, fill=INK,
             font=SERIF, anchor="start")
    total = hi + 1
    rate = 100.0 * h["hits"] / total
    s += txt(916, 22, f"STEP {hi+1:02d} / {len(LRU_STEPS)}", size=8, fill=MUTED,
             anchor="end", ls="0.14em")
    s += txt(916, 40, f"HIT {rate:.0f}%", size=11, fill=ACCENT, font=MONO,
             anchor="end", ls="0.06em")
    s += txt(916, 54, f"{h['hits']} hit \u00b7 {h['misses']} miss", size=8,
             fill=MUTED, anchor="end", ls="0.08em")
    # ---- slot row
    s += txt(44, 86, "HBM \u00b7 HOT EXPERT SLOTS \u00b7 bf16 LRU", size=8,
             fill=MUTED, anchor="start", ls="0.14em")
    for i, v in enumerate(slots):
        state, note = "filled", ""
        if v is None:
            state = "empty"
        else:
            if not h["hit"] and not post and v == h["victim"]:
                state, note = "victim", "LRU victim"
            elif i == h["idx"] and post:
                state = "hit" if h["hit"] else "new"
                note = "hit \u00b7 MRU" if h["hit"] else "just loaded"
            elif i == h["idx"] and h["hit"]:
                state, note = "hit", "hit \u00b7 MRU"
            elif i == h["idx"] and not h["hit"] and h["victim"] is None:
                state, note = "new", "loading"
            else:
                note = f"recency {rec.index(v)+1}"
        s += slot_card(i, v, state, note)
    # pulse ring on the affected slot
    if pulse > 0.01 and not (phase == 0 and h["idx"] is None):
        cx = SLOT_X0 + h["idx"] * SLOT_STEP + SLOT_W / 2
        cy = SLOT_Y + SLOT_H / 2
        col = ACCENT
        s += (f'<rect x="{SLOT_X0 + h["idx"]*SLOT_STEP - 5}" y="{SLOT_Y - 5}" '
              f'width="{SLOT_W + 10}" height="{SLOT_H + 10}" rx="9" fill="none" '
              f'stroke="{col}" stroke-width="2" opacity="{0.55*pulse:.3f}"/>')
    # ---- LRU order strip
    s += txt(44, 214, "LRU ORDER \u00b7 OLDEST \u2192 NEWEST", size=8, fill=MUTED,
             anchor="start", ls="0.14em")
    for k, v in enumerate(rec):
        x = 44 + k * (ORD_W + ORD_GAP)
        newest = k == len(rec) - 1
        s += (f'<rect x="{x}" y="{ORD_Y}" width="{ORD_W}" height="24" rx="4" '
              f'fill="{"rgba(191,69,32,0.08)" if newest else "rgba(45,49,66,0.05)"}" '
              f'stroke="{ACCENT if newest else MUTED}" stroke-width="1"/>')
        s += txt(x + ORD_W / 2, ORD_Y + 16, f"E{v}", size=10, fill=INK, font=MONO, weight=600)
    if rec:
        s += txt(44 + len(rec) * (ORD_W + ORD_GAP), ORD_Y + 16, "MRU", size=7,
                 fill=ACCENT, anchor="start", ls="0.10em")
    empty = NSLOT - len(rec)
    if empty > 0:
        s += txt(44 + NSLOT * (ORD_W + ORD_GAP) + 24, ORD_Y + 16,
                 f"{empty} slot{'s' if empty>1 else ''} empty", size=8, fill=MUTED,
                 anchor="start", ls="0.06em")
    s += ('<line x1="44" y1="272" x2="916" y2="272" stroke="rgba(45,49,66,0.10)" '
          'stroke-width="0.8"/>')
    return s


def build_lru():
    bodies = []
    for hi in range(len(HIST)):
        bodies.append(lru_body(hi, 0, 0.0))   # pre : pulse grows (request arrives)
        bodies.append(lru_body(hi, 1, 0.0))   # pre : solid (resolving)
        bodies.append(lru_body(hi, 2, 0.0))   # post: resolved state
        bodies.append(lru_body(hi, 2, 0.0))   # hold resolved state
    emit("lru-slots", bodies, 180, len(bodies) - 1)


if __name__ == "__main__":
    build_reveal()
    build_decode()
    build_storage()
    build_lru()
    print("done")
