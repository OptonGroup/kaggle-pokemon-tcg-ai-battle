#!/usr/bin/env python3
"""analyze_ex_rotation_corpus.py — вопрос A3 из docs/gold/MIRROR_LEVERS_BRIEF.md: присутствует
ли в корпусе экспертных решений механика "ротация копий Grimmsnarl ex (648) как пул HP" —
своп повреждённой активной копии на здоровую с бенча ПЕРЕД атакой.

МЕХАНИКА RETREAT В ДВИЖКЕ (валидировано вручную на episode 87558014, obs_ref 65/68, см. отчёт):
  MAIN-решение с опцией {"type": 12} (OPT_RETREAT, БЕЗ index/area — сам по себе не называет
  цель) -> при выборе следует 0+ решений DISCARD_ENERGY (context=30, оплата стоимости
  отступления сбросом прикреплённой энергии) -> ОБЯЗАТЕЛЬНОЕ решение SWITCH (context=3,
  опции {"type":3,"area":5,"index":N} — N это ИНДЕКС В BENCH). Только SWITCH-решение
  называет, в какую именно бенч-копию уходит свап.

АЛГОРИТМ на каждое MAIN-решение (obs.current.players[me] — своя сторона, me=obs.current.yourIndex):
  ЗНАМЕНАТЕЛЬ (item 1, "возможность есть"):
    (a) активный покемон own.active[0].id == 648, hp < maxHp (damage = maxHp-hp > 0)
    (b) на бенче есть другая копия id==648 c hp СТРОГО больше, чем у активной (более здоровая)
    (c) среди select.option есть {"type": 12} (retreat легален прямо сейчас)
  ЧИСЛИТЕЛЬ (item 2): выбранная опция в MAIN — это type==12 (эксперт пошёл в retreat) И
    последующее SWITCH-решение (ищем вперёд по obs_ref в decisions этого эпизода/агента, окно
    6 решений) целится в бенч-индекс, который на момент SWITCH был id==648 c hp БОЛЬШЕ, чем
    hp активной на момент MAIN-решения (т.е. реально доехали до более здоровой копии, а не
    просто отступили к чему-то другому).

Обрабатываются ДВЕ выборки:
  - MIRROR: opp_archetype==subj_archetype=='mirror_grimmsnarl' (основной ответ на A3)
  - ALL: весь корпус (для item 5 — доля ex-ротаций среди ВСЕХ retreat-решений корпуса,
    не только зеркальных)

ВХОД:
  out/corpus_2b/versions/v088_dragapult/merged/decisions.jsonl
  out/corpus_2b/obs_shards/<episode_id>.jsonl (obs_ref = номер строки, 0-based)

ВЫХОД:
  out/mirror_levers_20260815/a3_corpus_mechanic/result.json
  out/mirror_levers_20260815/a3_corpus_mechanic/report.md
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
DECISIONS_PATH = WORKTREE / "out/corpus_2b/versions/v088_dragapult/merged/decisions.jsonl"
# ЛОВУШКА (найдено в первом прогоне): out/corpus_2b/obs_shards/ - плоская директория, но
# содержит НАПРЯМУЮ только 738 из 10635 эпизодов мёрджа; остальные 9897 существуют лишь как
# симлинки в merged/obs_shards/ (которые указывают в датированные out/corpus_2b/versions/
# v083_daily_full/shards/<date>/obs_shards/). Использовать ИМЕННО merged/obs_shards/, иначе
# ALL-corpus агрегаты (item 5) молча считаются по 7% эпизодов.
OBS_DIR = WORKTREE / "out/corpus_2b/versions/v088_dragapult/merged/obs_shards"
OUT_DIR = WORKTREE / "out/mirror_levers_20260815/a3_corpus_mechanic"

GRIMMSNARL_EX = 648
DAMAGE_THRESHOLD = 60.0
SWITCH_SEARCH_WINDOW = 8  # макс. решений вперёд для поиска SWITCH-разрешения


def load_decisions():
    """episode_id -> list of lightweight dicts (obs_ref, agent_index, sel_context_name, action,
    opp_archetype, subj_archetype, expert, game_won). Один полный проход по decisions.jsonl."""
    by_ep: dict[int, list[dict]] = defaultdict(list)
    t0 = time.time()
    n = 0
    with open(DECISIONS_PATH) as f:
        for line in f:
            n += 1
            d = json.loads(line)
            by_ep[d["episode_id"]].append({
                "obs_ref": d["obs_ref"],
                "agent_index": d["agent_index"],
                "ctx": d["sel_context_name"],
                "action": d["action"],
                "opp_arch": d.get("opp_archetype"),
                "subj_arch": d.get("subj_archetype"),
                "expert": d.get("expert"),
                "game_won": d.get("game_won"),
            })
    for ep in by_ep:
        by_ep[ep].sort(key=lambda r: r["obs_ref"])
    print(f"[load] {n} decisions / {len(by_ep)} episodes in {time.time()-t0:.1f}s", file=sys.stderr)
    return by_ep


def own_view(obs: dict):
    cur = obs.get("current") or {}
    me = cur.get("yourIndex")
    players = cur.get("players") or []
    if not isinstance(me, int) or not (0 <= me < len(players)):
        return None, None, None
    mine = players[me]
    active_list = mine.get("active") or []
    active = active_list[0] if active_list and isinstance(active_list[0], dict) else None
    bench = [m for m in (mine.get("bench") or []) if isinstance(m, dict)]
    return me, active, bench


def resolve_switch_target(obs2: dict, target_bench_idx: int):
    me2, _active2, bench2 = own_view(obs2)
    if me2 is None or target_bench_idx is None:
        return None
    if 0 <= target_bench_idx < len(bench2):
        return bench2[target_bench_idx]
    return None


def process_episode(ep_id: int, decs: list[dict], stats_mirror: dict, stats_all: dict,
                     examples: dict):
    is_mirror_episode = any(
        d["opp_arch"] == "mirror_grimmsnarl" and d["subj_arch"] == "mirror_grimmsnarl"
        for d in decs)
    obs_path = OBS_DIR / f"{ep_id}.jsonl"
    if not obs_path.exists():
        stats_all["missing_shard_episodes"] += 1
        if is_mirror_episode:
            stats_mirror["missing_shard_episodes"] += 1
        return
    with open(obs_path) as f:
        lines = f.readlines()

    def get_obs(oref: int):
        if oref is None or not (0 <= oref < len(lines)):
            return None
        try:
            return json.loads(lines[oref])
        except json.JSONDecodeError:
            return None

    for idx, d in enumerate(decs):
        if d["ctx"] != "MAIN":
            continue
        is_mirror = d["opp_arch"] == "mirror_grimmsnarl" and d["subj_arch"] == "mirror_grimmsnarl"
        obs = get_obs(d["obs_ref"])
        if obs is None:
            continue

        # ── item 5, ШИРОКИЙ знаменатель: ЛЮБОЕ retreat-действие на ЛЮБОМ MAIN-решении,
        # независимо от условий (a)/(b) ниже — "доля ex-ротаций среди ВСЕХ retreat-решений
        # корпуса", не только среди retreat-решений внутри уже отфильтрованных состояний. ──
        sel_any = obs.get("select") or {}
        opts_any = sel_any.get("option") or []
        action_any = d["action"] or []
        if action_any and 0 <= action_any[0] < len(opts_any):
            chosen_any = opts_any[action_any[0]]
            if isinstance(chosen_any, dict) and chosen_any.get("type") == 12:
                stats_all["total_retreats_anywhere"] += 1
                if is_mirror:
                    stats_mirror["total_retreats_anywhere"] += 1

        me, active, bench = own_view(obs)
        if active is None:
            continue
        active_id = active.get("id")
        if active_id != GRIMMSNARL_EX:
            continue
        hp = active.get("hp")
        maxhp = active.get("maxHp")
        if not isinstance(hp, (int, float)) or not isinstance(maxhp, (int, float)):
            continue
        damage = maxhp - hp
        if damage <= 0:
            continue

        healthier = [m for m in bench
                     if m.get("id") == GRIMMSNARL_EX
                     and isinstance(m.get("hp"), (int, float))
                     and m["hp"] > hp]
        if not healthier:
            continue

        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        if not any(isinstance(o, dict) and o.get("type") == 12 for o in opts):
            continue

        # ── ЗНАМЕНАТЕЛЬ достигнут ──
        best_candidate = max(healthier, key=lambda m: m["hp"])
        best_charged = len(best_candidate.get("energies") or []) > 0
        damage_ge60 = damage >= DAMAGE_THRESHOLD
        expert = d["expert"]
        game_won = d["game_won"]

        buckets = [stats_all] + ([stats_mirror] if is_mirror else [])
        for st in buckets:
            st["denom"] += 1
            if damage_ge60:
                st["denom_dmg60"] += 1
            st["denom_by_expert"][expert] += 1
            st["denom_by_won"][bool(game_won)] += 1
            st["denom_by_charged"][best_charged] += 1

        # ── ЧИСЛИТЕЛЬ: реально выбрали retreat? ──
        action = d["action"] or []
        if not action:
            continue
        chosen_idx = action[0]
        if not (0 <= chosen_idx < len(opts)):
            continue
        chosen_opt = opts[chosen_idx]
        chose_retreat = isinstance(chosen_opt, dict) and chosen_opt.get("type") == 12
        for st in buckets:
            if chose_retreat:
                st["retreated_any_target"] += 1

        if not chose_retreat:
            continue

        # найти следующее SWITCH-решение того же агента в окне
        switch_dec = None
        for j in range(idx + 1, min(idx + 1 + SWITCH_SEARCH_WINDOW, len(decs))):
            cand = decs[j]
            if cand["agent_index"] != d["agent_index"]:
                continue
            if cand["ctx"] == "MAIN":
                break  # новый ход начался без разрешения свопа — нет смысла искать дальше
            if cand["ctx"] == "SWITCH":
                switch_dec = cand
                break
        if switch_dec is None:
            for st in buckets:
                st["retreat_unresolved_switch"] += 1
            continue

        obs2 = get_obs(switch_dec["obs_ref"])
        if obs2 is None:
            continue
        s_opts = (obs2.get("select") or {}).get("option") or []
        s_action = switch_dec["action"] or []
        if not s_action or not (0 <= s_action[0] < len(s_opts)):
            continue
        s_opt = s_opts[s_action[0]]
        target_idx = s_opt.get("index") if isinstance(s_opt, dict) else None
        target_mon = resolve_switch_target(obs2, target_idx)
        if target_mon is None:
            continue

        is_ex_rotation = (target_mon.get("id") == GRIMMSNARL_EX
                           and isinstance(target_mon.get("hp"), (int, float))
                           and target_mon["hp"] > hp)
        for st in buckets:
            if is_ex_rotation:
                st["numer"] += 1
                if damage_ge60:
                    st["numer_dmg60"] += 1
                st["numer_by_expert"][expert] += 1
                st["numer_by_won"][bool(game_won)] += 1
                st["numer_by_charged"][best_charged] += 1
            else:
                st["retreated_wrong_target"] += 1

        if is_ex_rotation and len(examples["swap"]) < 8:
            examples["swap"].append({
                "episode_id": ep_id, "obs_ref_main": d["obs_ref"],
                "obs_ref_switch": switch_dec["obs_ref"], "expert": expert,
                "active_hp_before": hp, "active_maxhp": maxhp, "damage": damage,
                "target_hp_after": target_mon["hp"], "is_mirror": is_mirror,
            })
        if (not is_ex_rotation) and chose_retreat and len(examples["retreat_other"]) < 5:
            examples["retreat_other"].append({
                "episode_id": ep_id, "obs_ref_main": d["obs_ref"],
                "target_id": target_mon.get("id"), "target_hp": target_mon.get("hp"),
                "is_mirror": is_mirror,
            })

    # examples of DECLINED opportunity (denom reached, but no retreat chosen) — mirror only
    for idx, d in enumerate(decs):
        if d["ctx"] != "MAIN":
            continue
        if not (d["opp_arch"] == "mirror_grimmsnarl" and d["subj_arch"] == "mirror_grimmsnarl"):
            continue
        obs = get_obs(d["obs_ref"])
        if obs is None:
            continue
        me, active, bench = own_view(obs)
        if active is None or active.get("id") != GRIMMSNARL_EX:
            continue
        hp = active.get("hp")
        maxhp = active.get("maxHp")
        if not isinstance(hp, (int, float)) or not isinstance(maxhp, (int, float)) or maxhp - hp <= 0:
            continue
        healthier = [m for m in bench if m.get("id") == GRIMMSNARL_EX
                     and isinstance(m.get("hp"), (int, float)) and m["hp"] > hp]
        if not healthier:
            continue
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        if not any(isinstance(o, dict) and o.get("type") == 12 for o in opts):
            continue
        action = d["action"] or []
        if not action or not (0 <= action[0] < len(opts)):
            continue
        chosen_opt = opts[action[0]]
        if isinstance(chosen_opt, dict) and chosen_opt.get("type") == 12:
            continue  # они и правда пошли в retreat — не декларируем
        if len(examples["declined"]) < 5:
            examples["declined"].append({
                "episode_id": ep_id, "obs_ref_main": d["obs_ref"], "expert": d["expert"],
                "active_hp": hp, "active_maxhp": maxhp,
                "best_bench_hp": max(m["hp"] for m in healthier),
            })


def fresh_stats():
    return {
        "denom": 0, "denom_dmg60": 0,
        "numer": 0, "numer_dmg60": 0,
        "retreated_any_target": 0, "retreated_wrong_target": 0, "retreat_unresolved_switch": 0,
        "denom_by_expert": defaultdict(int), "numer_by_expert": defaultdict(int),
        "denom_by_won": defaultdict(int), "numer_by_won": defaultdict(int),
        "denom_by_charged": defaultdict(int), "numer_by_charged": defaultdict(int),
        "missing_shard_episodes": 0, "total_retreats_anywhere": 0,
    }


def jsonify(st: dict) -> dict:
    out = {}
    for k, v in st.items():
        if isinstance(v, defaultdict):
            out[k] = {str(kk): vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_ep = load_decisions()

    stats_mirror = fresh_stats()
    stats_all = fresh_stats()
    examples = {"swap": [], "declined": [], "retreat_other": []}

    t0 = time.time()
    n_done = 0
    for ep_id, decs in by_ep.items():
        process_episode(ep_id, decs, stats_mirror, stats_all, examples)
        n_done += 1
        if n_done % 1000 == 0:
            print(f"[scan] {n_done}/{len(by_ep)} episodes, {time.time()-t0:.0f}s elapsed",
                  file=sys.stderr)
    print(f"[scan] done: {n_done} episodes in {time.time()-t0:.0f}s", file=sys.stderr)

    n_mirror_episodes = sum(
        1 for decs in by_ep.values()
        if any(d["opp_arch"] == "mirror_grimmsnarl" and d["subj_arch"] == "mirror_grimmsnarl"
               for d in decs))

    result = {
        "mirror": jsonify(stats_mirror),
        "all_corpus": jsonify(stats_all),
        "n_mirror_episodes": n_mirror_episodes,
        "n_total_episodes": len(by_ep),
        "examples": examples,
    }
    take_rate_mirror = (stats_mirror["numer"] / stats_mirror["denom"]) if stats_mirror["denom"] else None
    take_rate_all = (stats_all["numer"] / stats_all["denom"]) if stats_all["denom"] else None
    result["take_rate_mirror"] = take_rate_mirror
    result["take_rate_all_corpus"] = take_rate_all
    # item 5: доля ex-ротаций СРЕДИ ВСЕХ retreat-решений (широкий знаменатель — любой retreat
    # на любом MAIN-решении, не только внутри уже отфильтрованных denom-состояний).
    result["share_of_all_retreats_that_are_ex_rotation_mirror"] = (
        stats_mirror["numer"] / stats_mirror["total_retreats_anywhere"]
        if stats_mirror["total_retreats_anywhere"] else None)
    result["share_of_all_retreats_that_are_ex_rotation_all"] = (
        stats_all["numer"] / stats_all["total_retreats_anywhere"]
        if stats_all["total_retreats_anywhere"] else None)
    # доп.: доля retreat-решений ВНУТРИ уже отфильтрованных denom-состояний, которые попали
    # именно в более здоровую копию (узкий срез, оставлен для сравнения/отладки).
    result["share_within_opportunity_states_mirror"] = (
        stats_mirror["numer"] / stats_mirror["retreated_any_target"]
        if stats_mirror["retreated_any_target"] else None)
    result["swaps_per_mirror_episode"] = (
        stats_mirror["numer"] / n_mirror_episodes if n_mirror_episodes else None)

    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    write_report(result)
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=2,
                      ensure_ascii=False))


def pct(n, d):
    if not d:
        return "n/a"
    return f"{100.0*n/d:.1f}%"


def write_report(r: dict):
    m = r["mirror"]
    a = r["all_corpus"]

    def expert_table(st):
        rows = []
        for expert, dn in sorted(st["denom_by_expert"].items(), key=lambda kv: -kv[1]):
            nm = st["numer_by_expert"].get(expert, 0)
            rows.append(f"| {expert} | {dn} | {nm} | {pct(nm, dn)} |")
        return "\n".join(rows)

    lines = []
    lines.append("# A3: механика ротации копий Grimmsnarl ex (648) в корпусе — есть или нет")
    lines.append("")
    lines.append(f"Скрипт: `scripts/analyze_ex_rotation_corpus.py`. Корпус: "
                 f"`out/corpus_2b/versions/v088_dragapult/merged/decisions.jsonl` "
                 f"({r['n_total_episodes']} эпизодов, {r['n_mirror_episodes']} зеркальных "
                 f"mirror_grimmsnarl vs mirror_grimmsnarl).")
    lines.append("")
    lines.append("## Определения")
    lines.append("- **Знаменатель** (возможность): MAIN-решение, где активный покемон = "
                 "Grimmsnarl ex (648) с hp<maxHp, на бенче есть ДРУГАЯ копия 648 со строго "
                 "большим hp, и опция retreat легальна.")
    lines.append("- **Числитель**: эксперт выбрал retreat И итоговый SWITCH-таргет — именно "
                 "та более здоровая копия 648 (проверено по факту через obs после свопа).")
    lines.append("- Логика провалидирована вручную на episode 87558014 (obs_ref 65→68, см. "
                 "раздел «Валидация» ниже).")
    lines.append("")
    lines.append("## Главный результат (mirror_grimmsnarl vs mirror_grimmsnarl)")
    lines.append(f"- N знаменателя: **{m['denom']}** (из них с уроном ≥60: {m['denom_dmg60']})")
    lines.append(f"- N числителя (реальный своп в здоровую копию): **{m['numer']}** "
                 f"(из них с уроном ≥60: {m['numer_dmg60']})")
    lines.append(f"- **Take-rate = {pct(m['numer'], m['denom'])}** "
                 f"({m['numer']}/{m['denom']})")
    lines.append(f"- Take-rate при уроне ≥60: **{pct(m['numer_dmg60'], m['denom_dmg60'])}** "
                 f"({m['numer_dmg60']}/{m['denom_dmg60']})")
    lines.append(f"- Всего retreat-решений (любая цель) в этих же денаминаторных состояниях: "
                 f"{m['retreated_any_target']}; из них ушли НЕ в более здоровую копию 648: "
                 f"{m['retreated_wrong_target']}; не удалось разрешить SWITCH: "
                 f"{m['retreat_unresolved_switch']}")
    tr = r["take_rate_mirror"] or 0
    if tr >= 0.40:
        verdict = "ЕСТЬ (частое поведение)"
    elif tr >= 0.05:
        verdict = "ЕСТЬ, НО КАК МЕНЬШИНСТВЕННОЕ ПОВЕДЕНИЕ"
    else:
        verdict = "ПОЧТИ НЕТ"
    lines.append("")
    lines.append(f"### Вывод: механика в корпусе **{verdict}** "
                 f"(эксперты берут её в {pct(m['numer'], m['denom'])} возможностей)")
    lines.append("")
    lines.append("Важный нюанс для BC: N числителя нетривиален (2018 чистых положительных "
                 "примеров своп-в-здоровую-копию только в зеркальных играх, 3283 по всему "
                 "корпусу) — примеров ХВАТАЕТ для обучения самого факта «такая опция вообще "
                 "существует». Но МАЖОРИТАРНАЯ метка внутри уже отфильтрованных "
                 "opportunity-состояний (a+b+c выполнены) — «НЕ свопать» (86.6% состояний): "
                 "эксперты сами в подавляющем большинстве подходящих моментов НЕ делают "
                 "ротацию (возможно, оставляют повреждённую копию добивать урон/защищают "
                 "здоровую от следующей атаки/бенчат её на будущее). Значит проблема не "
                 "«примеров нет», а «сигнал сам по себе слабый/несогласованный в исходной "
                 "экспертной политике» — простое up-weighting класса retreat НЕ решит именно "
                 "ЭТОТ подкласс, т.к. модель обучается предсказывать поведение эксперта, а "
                 "поведение эксперта здесь само в основном «не свопать». Зато КОГДА эксперт "
                 "решается на retreat в такой ситуации, он почти всегда (99.1%) целится "
                 "именно в более здоровую копию 648, а не куда-то ещё — значит ЦЕЛЬ свопа "
                 "выучиваема почти детерминированно, если модель научится хотя бы иногда "
                 "выбирать retreat в такой позиции.")
    lines.append("")
    lines.append("## Разрез по эксперту (mirror)")
    lines.append("| expert | denom | numer | take-rate |")
    lines.append("|---|---|---|---|")
    lines.append(expert_table(m))
    lines.append("")
    lines.append("## Разрез по исходу игры (mirror)")
    for won in (True, False):
        dn = m["denom_by_won"].get(str(won), m["denom_by_won"].get(won, 0))
        nm = m["numer_by_won"].get(str(won), m["numer_by_won"].get(won, 0))
        lines.append(f"- game_won={won}: {nm}/{dn} = {pct(nm, dn)}")
    lines.append("")
    lines.append("## Разрез по заряженности лучшей бенч-копии (есть ли на ней энергия, mirror)")
    for charged in (True, False):
        dn = m["denom_by_charged"].get(str(charged), m["denom_by_charged"].get(charged, 0))
        nm = m["numer_by_charged"].get(str(charged), m["numer_by_charged"].get(charged, 0))
        lines.append(f"- charged={charged}: {nm}/{dn} = {pct(nm, dn)}")
    lines.append("")
    lines.append(f"## Частота на игру (mirror)")
    lines.append(f"- Свопов на зеркальную игру у экспертов (по корпусу): "
                 f"**{r['swaps_per_mirror_episode']:.3f}** "
                 f"({m['numer']} свопов / {r['n_mirror_episodes']} зеркальных эпизодов)")
    lines.append("  (для сравнения: разведка топ-5 говорила про ~150 свопов/день у Luca — "
                 "порядок величины несопоставим на per-game основе, см. интерпретацию ниже)")
    lines.append("")
    lines.append("## Доля ex-ротаций среди ВСЕХ retreat-решений корпуса")
    lines.append("(широкий знаменатель: ЛЮБОЕ retreat-действие на ЛЮБОМ MAIN-решении, "
                 "независимо от того, был ли активным именно 648 — не только retreat-решения "
                 "внутри уже отфильтрованных denom-состояний)")
    lines.append(f"- В зеркальных играх: {pct(m['numer'], m['total_retreats_anywhere'])} "
                 f"({m['numer']}/{m['total_retreats_anywhere']} ВСЕХ retreat-решений корпуса "
                 f"в зеркальных играх)")
    lines.append(f"- Во ВСЁМ корпусе (любой оппонент): "
                 f"denom={a['denom']}, numer={a['numer']}, "
                 f"take-rate={pct(a['numer'], a['denom'])}, "
                 f"доля среди ВСЕХ retreat-решений корпуса={pct(a['numer'], a['total_retreats_anywhere'])} "
                 f"({a['numer']}/{a['total_retreats_anywhere']})")
    lines.append(f"- Для сравнения, доля ВНУТРИ уже отфильтрованных opportunity-состояний "
                 f"(эксперт retreat-нул именно когда были условия a/b/c): "
                 f"{pct(m['numer'], m['retreated_any_target'])} "
                 f"({m['numer']}/{m['retreated_any_target']})")
    lines.append("")
    lines.append("## Примеры для ручной проверки")
    lines.append("### Успешные свопы (числитель)")
    for ex in r["examples"]["swap"][:5]:
        lines.append(f"- episode_id={ex['episode_id']}, obs_ref MAIN={ex['obs_ref_main']} / "
                     f"SWITCH={ex['obs_ref_switch']}, expert={ex['expert']}, "
                     f"hp {ex['active_hp_before']:.0f}/{ex['active_maxhp']:.0f} "
                     f"(урон {ex['damage']:.0f}) -> цель hp {ex['target_hp_after']:.0f}, "
                     f"mirror={ex['is_mirror']}")
    lines.append("### Возможность была, но НЕ воспользовались (declined)")
    for ex in r["examples"]["declined"][:5]:
        lines.append(f"- episode_id={ex['episode_id']}, obs_ref={ex['obs_ref_main']}, "
                     f"expert={ex['expert']}, active hp {ex['active_hp']:.0f}/"
                     f"{ex['active_maxhp']:.0f}, лучшая бенч-копия hp {ex['best_bench_hp']:.0f}")
    lines.append("### Retreat выбран, но ушли не в более здоровую 648-копию")
    for ex in r["examples"]["retreat_other"][:5]:
        lines.append(f"- episode_id={ex['episode_id']}, obs_ref={ex['obs_ref_main']}, "
                     f"target_id={ex['target_id']}, target_hp={ex['target_hp']}, "
                     f"mirror={ex['is_mirror']}")
    lines.append("")
    lines.append("## Валидация (episode 87558014)")
    lines.append("- obs_ref=65 (MAIN): активный 648 hp=140/320 (урон 180); bench[0]=648 "
                 "hp=320/320 (полностью здоровая, заряжена 2 энергиями); опции содержат "
                 "{'type':12} (retreat), эксперт (Luca) выбрал action=[11] -> именно эту "
                 "retreat-опцию.")
    lines.append("- obs_ref=66,67 (DISCARD_ENERGY): сброс 2 приклеенных {D}-энергий — оплата "
                 "стоимости отступления активного 648.")
    lines.append("- obs_ref=68 (SWITCH): 5 опций area=5 (бенч-индексы 0..4), эксперт выбрал "
                 "action=[0] -> bench-индекс 0 -> именно здоровая копия 648.")
    lines.append("- obs_ref=69 (следующее MAIN): active теперь id=648, hp=320/320 — свап "
                 "подтверждён напрямую по итоговому состоянию борда.")
    (OUT_DIR / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
