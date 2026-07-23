"""Строит results/gates/grimmsnarl_hybrid_vs_v064_comparison.md из двух JSON:
  results/gates/v064_core_gate_vs_grimmsnarl_hybrid_N1000.json
    (candidate=v064, полный CORE_PAIRS пул, включая h2h vs ext_prvsiyan_grimmsnarl_hybrid)
  results/gates/ext_prvsiyan_grimmsnarl_hybrid_vs_pool.json
    (candidate=ext_prvsiyan_grimmsnarl_hybrid, тот же пул + отдельный h2h vs v064_starmie_scaling)

Разведочный документ, не участвует в решении о сабмите.
"""
from __future__ import annotations
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATES = os.path.join(ROOT, "results", "gates")

v064_path = os.path.join(GATES, "v064_core_gate_vs_grimmsnarl_hybrid_N1000.json")
hyb_path = os.path.join(GATES, "ext_prvsiyan_grimmsnarl_hybrid_vs_pool.json")

v064_data = json.load(open(v064_path))
hyb_data = json.load(open(hyb_path))

v064_opp = v064_data["opponents"]
hyb_opp = hyb_data["opponents"]

# label, opponent key in v064 json, opponent key in hyb json
ROWS = [
    ("v017_alakazam_heur",        "v017", "v017"),
    ("v016_alakazam",             "v016", "v016"),
    ("v009a_crustle",             "v009a", "v009a"),
    ("v044_archaludon_abomasnow", "v044", "v044"),
    ("v049_bench_sticky_guard",   "v049", "v049"),
    ("ext_prvsiyan_alak_v8",      "ext_prvsiyan_alak_v8", "ext_prvsiyan_alak_v8"),
    ("ext_prvsiyan_alak_v9",      "ext_prvsiyan_alak_v9", "ext_prvsiyan_alak_v9"),
    ("ext_maulikgajera_lucario",  "ext_maulikgajera_lucario", "ext_maulikgajera_lucario"),
    ("ext_nursrijan_lucario",     "ext_nursrijan_lucario", "ext_nursrijan_lucario"),
    ("v063_forensic86 (h2h)",     "v063_h2h", "v063_h2h"),
]


def fmt(v, k):
    o = v.get(k)
    if not o or o.get("n_games", 0) == 0:
        return "n/a", "n/a", "n/a"
    wr = o["win_rate"]
    lo = o["wilson95_lo"]
    hi = o["wilson95_hi"]
    n = o["n_games"]
    crashes = o.get("crashes", 0)
    exc = o.get("exc_agent", 0)
    illegal = o.get("illegal_agent", 0)
    flag = ""
    if crashes or exc or illegal:
        flag = f" [!] crashes={crashes} exc={exc} illegal={illegal}"
    return f"{wr:.3f}", f"[{lo:.3f}, {hi:.3f}]", f"n={n}{flag}"


lines = []
lines.append("# v064 vs ext_prvsiyan_grimmsnarl_hybrid — сравнительный гейт (2026-07-23)\n")
lines.append(
    "Разведочный прогон (\"ради прикола\"): наш v064_starmie_scaling против "
    "внешнего публичного агента prvsiyan/ptcg-rmy-grimmsnarl-replay-hybrid-v1 "
    "(attention-BC на 125 публичных реплеях Rmy Grimmsnarl/Froslass) "
    "по одному и тому же CORE_PAIRS пулу оппонентов, N=1000 на пару, cg_native движок.\n"
)
lines.append(
    "Оба кандидата используют **одну и ту же деку Grimmsnarl** (naunju-стиль), "
    "поэтому сравнение по общим оппонентам — это прямое сравнение силы пилота, "
    "а не архетипа.\n"
)

lines.append("## Прямой head-to-head (v064 vs ext_prvsiyan_grimmsnarl_hybrid)\n")
lines.append("| Направление | WR | Wilson95 CI | N |")
lines.append("|---|---|---|---|")
wr, ci, n = fmt(v064_opp, "ext_prvsiyan_grimmsnarl_hybrid")
lines.append(f"| v064 (candidate) vs hybrid, seed=v064-run | {wr} | {ci} | {n} |")
wr2, ci2, n2 = fmt(hyb_opp, "v064_h2h")
lines.append(f"| hybrid (candidate) vs v064, seed=hybrid-run | {wr2} | {ci2} | {n2} |")
lines.append("")
lines.append(
    "Обе строки измеряют один и тот же матчап с независимыми сидами/раскладкой "
    "первого хода (seat-balanced внутри каждого прогона). Согласованность между "
    "строками — sanity-check методологии.\n"
)

lines.append("## Сравнительная таблица по общим оппонентам пула\n")
lines.append("| Оппонент | v064 WR [CI] | hybrid WR [CI] | Кто сильнее |")
lines.append("|---|---|---|---|")
for label, k_v064, k_hyb in ROWS:
    if k_v064 in ("v063_h2h",):
        continue  # особый случай — v063, не общий third-party оппонент, добавим отдельно если нужно
    wr_v, ci_v, n_v = fmt(v064_opp, k_v064)
    wr_h, ci_h, n_h = fmt(hyb_opp, k_hyb)
    verdict = "?"
    try:
        fv, fh = float(wr_v), float(wr_h)
        if fv > fh + 0.02:
            verdict = "v064"
        elif fh > fv + 0.02:
            verdict = "hybrid"
        else:
            verdict = "≈ ничья"
    except ValueError:
        pass
    lines.append(f"| {label} | {wr_v} {ci_v} | {wr_h} {ci_h} | {verdict} |")

lines.append("")
lines.append("*(v064 WR — доля побед v064 против общего оппонента; hybrid WR — доля побед "
              "hybrid против того же общего оппонента; оба прогона N=1000, независимые. "
              "\"Кто сильнее\" — эвристическое сравнение WR против общего оппонента, "
              "не прямой h2h — см. раздел выше для прямого h2h.)*\n")

lines.append("## Диагностика 0/4 (первый смок-тест)\n")
lines.append(
    "Причина: **не баг порта**, а гонка по времени подготовки файлов. `model.pt` "
    "(4 МБ) был дописан в `submissions/external_prvsiyan_grimmsnarl_hybrid/` в "
    "10:51, а самый первый смок-запуск гейта (n=4) стартовал до этого — "
    "`_checkpoint_path()` в `main.py` кидал `FileNotFoundError` при импорте модуля "
    "(до входа в защищённый `try/except` вокруг `agent_fn()` в "
    "`gate_v064_regression.py::_play_one_game`), гейт классифицировал это как "
    "`worker_crash` в каждой из 4 игр → `wins=0, n=4 → WR=0.000`.\n"
)
lines.append(
    "Подтверждение фикса (не требовался — файл появился сам, но проверено явно):\n"
    "- Ручная трассировка (`scripts/diag_grimmsnarl_hybrid_trace.py`, 8 игр напрямую, "
    "single-process, монки-патч `bc_agent.predict_action`): **8/8 побед**, "
    "**527/527 решений через модельный путь, 0 через `_fallback_action`, 0 исключений**.\n"
    "- Полный прогон N=1000 (`ext_prvsiyan_grimmsnarl_hybrid_vs_pool.json`, ключ `v017`): "
    f"WR={hyb_opp.get('v017',{}).get('win_rate')}, crashes=0, exc_agent=0, illegal_agent=0 — "
    "согласуется с обеими независимыми проверками.\n"
)

out_path = os.path.join(GATES, "grimmsnarl_hybrid_vs_v064_comparison.md")
with open(out_path, "w") as f:
    f.write("\n".join(lines))
print(f"wrote {out_path}")
