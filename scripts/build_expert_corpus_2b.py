#!/usr/bin/env python3
"""
build_expert_corpus_2b.py — собрать decision-level (obs, action) корпус решений
эталонов-игроков (Luca / szlachetny snieg / Dominic Peel, топ-5 разведки 07-23)
для behavioral cloning (план 2B, docs/gold/PLAN_WR60_20260809.md, T1).

Источник: out/top5_recon/2026-07-23/episodes/*.json — 1176 полных реплеев
(формат kaggle-environments), собранных ранее разведкой топ-5 игроков
(см. out/top5_recon/2026-07-23/index.csv, out/top_players_ledger.csv —
1273 строки = 1176 уникальных episode, часть встречается дважды: по разу на
каждого топ-игрока в партии).

Пары (obs, action) строятся ТОЛЬКО через src/replays/parse.py::parse_replay —
единственную канонизированную реализацию off-by-one склейки
(obs = steps[i][player].observation, chosen = steps[i+1][player].action),
см. в памяти project_offbyone_forensic_bug. Этот скрипт НЕ переизобретает
склейку obs/action — только (а) отбирает решения нужной стороны и
(б) восстанавливает для них служебные ссылки на исходный реплей.

Отбор стороны эталона: pair['player_name'] (parse_replay уже вычисляет его из
info.Agents[player_index]['Name']) сверяется с колонкой team в
out/top_players_ledger.csv для того же episode_id — оттуда же берём
opp_arche/win, не дублируя классификацию архетипов.

★ ЛОВУШКА #1 (обнаружена и обойдена эмпирически, см. also тело файла ниже):
parse_replay() возвращает pair['episodeId'] = data['id'] — это UUID
(например 'c6c9bb52-8629-11f1-80eb-0242ac130203'), А НЕ числовой episodeId!
Причина: `episode_id = data.get("id") or data.get("info", {}).get("EpisodeId", 0)`
внутри parse_replay — data['id'] всегда truthy (непустая строка), поэтому
правая часть `or` никогда не срабатывает. Числовой episode_id, которым
проиндексирован top_players_ledger.csv и названы файлы реплеев — это ТОЛЬКО
info.EpisodeId / имя файла. Мы берём его из имени файла (авторитетно) и
НИКОГДА не доверяем pair['episodeId'].

★ ЛОВУШКА #2: у parse_replay() нет поля "какой это шаг steps[i]" в
возвращаемой паре — только сам obs_dict. Индекс шага в исходном steps[]
восстанавливаем БЕЗ переизобретения склейки: двухуказательным сканом
raw_steps[i][agent_index]['observation'] == obs_dict (структурное сравнение
словарей — просто поиск, а не логика выбора решения) по возрастанию i, в том
же порядке, в котором parse_replay() отдаёт пары для фиксированного
agent_index. Поле obs['step'] (внутри самого observation) для этой цели НЕ
используется — эмпирически бывает None на части решений (например
DRAW_COUNT), то есть ненадёжно как есть. Если совпадение не нашлось (не
ожидается, но не гарантировано), step_idx=None — доля таких случаев видна в
stats.json → step_idx_match_rate.

★ ЛОВУШКА #3: 5 из 1176 эпизодов — игры ДВУХ эталонов друг против друга
(например Luca vs Dominic Peel); в top_players_ledger.csv на такой эпизод —
две строки (по одной на каждую сторону) с РАЗНЫМ opp_arche/win. Обе стороны
такого эпизода обрабатываются независимо и обе попадают в корпус.

Выход:
  out/corpus_2b/decisions.jsonl
      По строке на решение (лёгкий, БЕЗ obs). Схема одной строки:
      episode_id, expert, agent_index, turn_idx, step_idx, sel_context,
      sel_context_name, n_options, action, min_count, max_count,
      opp_archetype, game_won, obs_ref.
  out/corpus_2b/obs_shards/<episode_id>.jsonl
      Полный obs_dict, по строке на решение, ТОЛЬКО для решений эталонов
      этого эпизода (может быть до 2 сторон на файл — ловушка #3). obs_ref в
      decisions.jsonl — это 0-based номер строки в соответствующем файле
      данного episode_id.
  out/corpus_2b/stats.json
      Игры/стороны/решения на эксперта, распределение sel_context, доля
      побед (по ledger), распределение матчапов (opp_arche, по ledger).

Использование:
  .venv/bin/python scripts/build_expert_corpus_2b.py [--limit N] [--workers 2]
      [--experts "Luca" "szlachetny snieg" "Dominic Peel"]
      [--episodes-dir out/top5_recon/2026-07-23/episodes]
      [--ledger out/top_players_ledger.csv]
      [--out-dir out/corpus_2b]

Требует .venv/bin/python (>=3.10 — parse.py использует PEP604 `str | os.PathLike`,
падает TypeError на системном /usr/bin/python3 3.9.6).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from replays.parse import _context_name, parse_replay  # noqa: E402 (needs sys.path insert first)

DEFAULT_EXPERTS = ["Luca", "szlachetny snieg", "Dominic Peel", "__Taichicchi__"]
# Корпус 2B — ТОЛЬКО решения на grimmsnarl-toolbox деке: __Taichicchi__ 27 из своих
# 200 сторон отыграл на tr_mewtwo (совсем другая дека), такие решения в BC-корпус
# попадать не должны. Фильтр по колонке subj_arche леджера, снимается --archetype "".
DEFAULT_ARCHETYPE = "mirror_grimmsnarl"
DEFAULT_EPISODES_DIR = REPO_ROOT / "out/top5_recon/2026-07-23/episodes"
DEFAULT_LEDGER = REPO_ROOT / "out/top_players_ledger.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "out/corpus_2b"

# Populated once per worker process via ProcessPoolExecutor(initializer=...) (or
# directly in the parent process when --workers<=1) so we don't re-pickle the
# ledger dict on every single task.
_LEDGER: dict[tuple[int, str], dict] = {}
_EXPERTS: set[str] = set()
_OBS_SHARD_DIR: Path | None = None
_ARCHETYPE: str = ""

# Известные листы эталонов относительно submissions/v076_final/deck.csv (проверено
# 08-09 по колонке subj_deck леджера): ВСЕ эталоны выкинули Xerosic's Machinations
# (1197). luca_toolscrapper = наш лист −1197 +1137 (Tool Scrapper), 495 сторон;
# peel_handfan = −1197 −1122(Pokégear) +2×1161 (Handheld Fan), 202 стороны.
_DECK_KEYS = {
    "7 7 7 7 7 7 7 7 7 7 104 104 112 112 112 112 646 646 646 646 647 647 647 648 648 648 "
    "860 860 1079 1079 1079 1080 1086 1086 1086 1086 1097 1097 1097 1122 1137 1152 1152 1152 "
    "1152 1182 1182 1219 1219 1219 1219 1227 1227 1227 1227 1231 1259 1259 1259 1259": "luca_toolscrapper",
    "7 7 7 7 7 7 7 7 7 7 104 104 112 112 112 112 646 646 646 646 647 647 647 648 648 648 "
    "860 860 1079 1079 1079 1080 1086 1086 1086 1086 1097 1097 1097 1152 1152 1152 1152 1161 "
    "1161 1182 1182 1219 1219 1219 1219 1227 1227 1227 1227 1231 1259 1259 1259 1259": "peel_handfan",
}


def _deck_key(subj_deck: str) -> str:
    """Метка варианта листа эксперта. Неизвестные листы получают 'other:<hash>' —
    молча склеивать разные деки в один корпус нельзя (см. project_deck_policy_entanglement)."""
    s = " ".join(subj_deck.split())
    if s in _DECK_KEYS:
        return _DECK_KEYS[s]
    if not s:
        return "unknown"
    import hashlib
    return "other:" + hashlib.md5(s.encode()).hexdigest()[:8]


def _init_worker(ledger: dict[tuple[int, str], dict], experts: set[str], obs_shard_dir: Path,
                 archetype: str) -> None:
    global _LEDGER, _EXPERTS, _OBS_SHARD_DIR, _ARCHETYPE
    _LEDGER = ledger
    _EXPERTS = experts
    _OBS_SHARD_DIR = obs_shard_dir
    _ARCHETYPE = archetype


def load_ledger(path: Path) -> dict[tuple[int, str], dict]:
    """out/top_players_ledger.csv -> {(episode_id, team): row_dict}. Keys are unique
    (verified: max 1 row per (episode, team) pair over all 1273 rows)."""
    out: dict[tuple[int, str], dict] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[(int(row["episode"]), row["team"])] = row
    return out


def discover_episode_files(episodes_dir: Path) -> dict[int, Path]:
    """episode_id (from filename stem, authoritative — see ЛОВУШКА #1) -> path."""
    out: dict[int, Path] = {}
    for p in sorted(episodes_dir.glob("*.json")):
        try:
            eid = int(p.stem)
        except ValueError:
            continue
        out[eid] = p
    return out


def _match_step_indices(steps: list, agent_pairs: list[dict], agent_index: int) -> list[int | None]:
    """Recover, for each pair in `agent_pairs` (already in ascending-i order, as
    parse_replay emits them per fixed agent_index), the raw `steps` array index i such
    that steps[i][agent_index]['observation'] is structurally equal to that pair's
    obs_dict. Pure lookup by content equality — does NOT redecide which steps are
    decisions (that stays parse_replay's job); see ЛОВУШКА #2 in the module docstring.
    """
    n_pairs = len(agent_pairs)
    result: list[int | None] = [None] * n_pairs
    ptr = 0
    n_steps = len(steps)
    for i in range(n_steps - 1):  # same bound parse_replay itself uses
        if ptr >= n_pairs:
            break
        frame = steps[i]
        if agent_index >= len(frame):
            continue
        obs_i = frame[agent_index].get("observation")
        if obs_i == agent_pairs[ptr]["obs_dict"]:
            result[ptr] = i
            ptr += 1
    return result


def _process_episode(task: tuple[int, Path]) -> dict:
    """Runs in a worker process (or inline if --workers<=1). Returns a dict with the
    decision records + obs-shard lines for this one episode (writes the shard file
    itself — safe in parallel, one file per episode_id, no cross-worker overlap)."""
    episode_id, path = task
    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception as e:  # pragma: no cover (defensive — corrupt file)
        return {"episode_id": episode_id, "error": f"failed to load JSON: {e}", "decisions": [], "warnings": []}

    steps = raw.get("steps", [])

    try:
        pairs = parse_replay(path)
    except Exception as e:  # pragma: no cover (defensive)
        return {"episode_id": episode_id, "error": f"parse_replay failed: {e}", "decisions": [], "warnings": []}

    by_agent: dict[int, list[dict]] = defaultdict(list)
    for p in pairs:
        by_agent[p["player_index"]].append(p)

    decisions: list[dict] = []
    shard_lines: list[dict] = []
    warnings: list[str] = []

    for agent_index, agent_pairs in sorted(by_agent.items()):
        if not agent_pairs:
            continue
        player_name = agent_pairs[0]["player_name"]
        if player_name not in _EXPERTS:
            continue

        ledger_row = _LEDGER.get((episode_id, player_name))
        if ledger_row is None:
            warnings.append(f"episode={episode_id}: no ledger row for team={player_name!r} (agent_index={agent_index})")
            continue

        subj_arche = ledger_row.get("subj_arche") or None
        if _ARCHETYPE and subj_arche != _ARCHETYPE:
            continue  # чужая дека эксперта (напр. __Taichicchi__ на tr_mewtwo) — не наш корпус

        deck_key = _deck_key(ledger_row.get("subj_deck") or "")
        opp_has_crustle = (ledger_row.get("opp_has_crustle") or "0") not in ("0", "", None)
        opp_archetype = ledger_row.get("opp_arche") or None
        ledger_win_raw = ledger_row.get("win")
        ledger_won = bool(int(ledger_win_raw)) if ledger_win_raw not in (None, "") else None

        step_indices = _match_step_indices(steps, agent_pairs, agent_index)
        n_unmatched = sum(1 for s in step_indices if s is None)
        if n_unmatched:
            warnings.append(
                f"episode={episode_id} team={player_name!r} agent_index={agent_index}: "
                f"{n_unmatched}/{len(agent_pairs)} decisions got no step_idx match"
            )

        for pair, step_idx in zip(agent_pairs, step_indices):
            obs = pair["obs_dict"]
            sel = obs.get("select") or {}
            cur = obs.get("current")
            turn_idx = cur.get("turn") if isinstance(cur, dict) else None

            won = pair["won"]
            if ledger_won is not None and won != ledger_won:
                warnings.append(
                    f"episode={episode_id} team={player_name!r}: parse_replay won={won} "
                    f"!= ledger win={ledger_won} (mismatch — using parse_replay value)"
                )

            obs_ref = len(shard_lines)
            shard_lines.append(obs)

            decisions.append({
                "episode_id": episode_id,
                "expert": player_name,
                "agent_index": agent_index,
                "turn_idx": turn_idx,
                "step_idx": step_idx,
                "sel_context": pair["select_context"],
                "sel_context_name": _context_name(pair["select_context"]),
                "n_options": pair["n_options"],
                "action": pair["chosen_option_indices"],
                "min_count": sel.get("minCount"),
                "max_count": sel.get("maxCount"),
                "opp_archetype": opp_archetype,
                "opp_has_crustle": opp_has_crustle,
                "subj_archetype": subj_arche,
                "deck_key": deck_key,
                "game_won": won,
                "obs_ref": obs_ref,
            })

    if shard_lines:
        shard_path = _OBS_SHARD_DIR / f"{episode_id}.jsonl"
        with open(shard_path, "w") as f:
            for obs in shard_lines:
                f.write(json.dumps(obs) + "\n")

    return {"episode_id": episode_id, "decisions": decisions, "warnings": warnings}


def build(args: argparse.Namespace) -> dict:
    ledger = load_ledger(args.ledger)
    experts = set(args.experts)
    episode_files = discover_episode_files(args.episodes_dir)

    per_expert_ledger_ids: dict[str, set[int]] = defaultdict(set)
    for (eid, team), row in ledger.items():
        if team not in experts:
            continue
        if args.archetype and (row.get("subj_arche") or None) != args.archetype:
            continue
        per_expert_ledger_ids[team].add(eid)

    all_target_ids = sorted(set().union(*per_expert_ledger_ids.values())) if per_expert_ledger_ids else []
    missing_files = [eid for eid in all_target_ids if eid not in episode_files]
    target_ids = [eid for eid in all_target_ids if eid in episode_files]
    if args.limit and args.limit > 0:
        target_ids = target_ids[: args.limit]
    target_id_set = set(target_ids)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    obs_shard_dir = args.out_dir / "obs_shards"
    obs_shard_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(eid, episode_files[eid]) for eid in target_ids]

    print(
        f"[build_expert_corpus_2b] experts={sorted(experts)} episodes_dir={args.episodes_dir} "
        f"target_episodes={len(tasks)} (of {len(all_target_ids)} in ledger for these experts, "
        f"{len(missing_files)} missing files) workers={args.workers}",
        file=sys.stderr,
    )

    t0 = time.time()
    results: list[dict] = []
    progress_every = max(1, len(tasks) // 10) if tasks else 1

    if args.workers <= 1:
        _init_worker(ledger, experts, obs_shard_dir, args.archetype)
        for i, task in enumerate(tasks, 1):
            results.append(_process_episode(task))
            if i % progress_every == 0 or i == len(tasks):
                print(f"[build_expert_corpus_2b] ...{i}/{len(tasks)}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(ledger, experts, obs_shard_dir, args.archetype),
        ) as ex:
            for i, res in enumerate(ex.map(_process_episode, tasks), 1):
                results.append(res)
                if i % progress_every == 0 or i == len(tasks):
                    print(f"[build_expert_corpus_2b] ...{i}/{len(tasks)}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"[build_expert_corpus_2b] parsed {len(results)} episodes in {elapsed:.1f}s", file=sys.stderr)

    decisions_path = args.out_dir / "decisions.jsonl"
    n_decisions = 0
    n_step_matched = 0
    n_warnings = 0
    n_errors = 0
    context_totals: Counter = Counter()
    decisions_per_expert: Counter = Counter()
    sides_with_decisions: dict[str, set[int]] = defaultdict(set)
    context_per_expert: dict[str, Counter] = defaultdict(Counter)
    deck_key_totals: Counter = Counter()
    deck_key_per_expert: dict[str, Counter] = defaultdict(Counter)
    opp_arche_decisions: Counter = Counter()
    n_decisions_won_games = 0

    with open(decisions_path, "w") as out_f:
        for res in results:
            if res.get("error"):
                n_errors += 1
                print(f"[build_expert_corpus_2b] ERROR episode={res['episode_id']}: {res['error']}", file=sys.stderr)
                continue
            for w in res.get("warnings", []):
                n_warnings += 1
                print(f"[build_expert_corpus_2b] WARN {w}", file=sys.stderr)
            for d in res["decisions"]:
                out_f.write(json.dumps(d) + "\n")
                n_decisions += 1
                if d["step_idx"] is not None:
                    n_step_matched += 1
                context_totals[d["sel_context_name"]] += 1
                decisions_per_expert[d["expert"]] += 1
                context_per_expert[d["expert"]][d["sel_context_name"]] += 1
                sides_with_decisions[d["expert"]].add(d["episode_id"])
                deck_key_totals[d["deck_key"]] += 1
                deck_key_per_expert[d["expert"]][d["deck_key"]] += 1
                opp_arche_decisions[d["opp_archetype"] or "UNKNOWN"] += 1
                if d["game_won"]:
                    n_decisions_won_games += 1

    per_expert_stats = {}
    for expert in sorted(experts):
        rows = [(eid, row) for (eid, team), row in ledger.items()
                if team == expert and eid in target_id_set
                and (not args.archetype or (row.get("subj_arche") or None) == args.archetype)]
        n_games = len(rows)
        n_wins = sum(1 for _, row in rows if (row.get("win") or "0") not in ("0", "", None))
        matchup_counts = Counter((row.get("opp_arche") or "UNKNOWN") for _, row in rows)
        n_dec = decisions_per_expert.get(expert, 0)
        n_sides = len(sides_with_decisions.get(expert, set()))
        per_expert_stats[expert] = {
            "n_games_in_ledger_total": len(per_expert_ledger_ids.get(expert, set())),
            "n_games_processed": n_games,
            "n_wins": n_wins,
            "win_rate": (n_wins / n_games) if n_games else None,
            "n_sides_with_decisions": n_sides,
            "n_decisions": n_dec,
            "decisions_per_side_mean": (n_dec / n_sides) if n_sides else None,
            "sel_context_distribution": dict(context_per_expert.get(expert, Counter()).most_common()),
            "matchup_distribution": dict(matchup_counts.most_common()),
            "deck_key_distribution": dict(deck_key_per_expert.get(expert, Counter()).most_common()),
        }

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "episodes_dir": str(args.episodes_dir),
            "ledger": str(args.ledger),
            "out_dir": str(args.out_dir),
            "experts": sorted(experts),
            "limit": args.limit,
            "workers": args.workers,
        },
        "n_episode_files_found_total": len(episode_files),
        "n_ledger_rows_total": len(ledger),
        "n_ledger_rows_for_experts_total": len(all_target_ids),
        "n_ledger_rows_missing_file": len(missing_files),
        "n_episodes_processed": len(results),
        "n_episodes_with_parse_error": n_errors,
        "n_decisions_total": n_decisions,
        "step_idx_match_rate": (n_step_matched / n_decisions) if n_decisions else None,
        "n_warnings": n_warnings,
        "elapsed_sec": elapsed,
        "sel_context_distribution_total": dict(context_totals.most_common()),
        "deck_key_distribution_total": dict(deck_key_totals.most_common()),
        "opp_archetype_decisions_total": dict(opp_arche_decisions.most_common()),
        "decisions_in_won_games_share": (n_decisions_won_games / n_decisions) if n_decisions else None,
        "per_expert": per_expert_stats,
    }

    stats_path = args.out_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[build_expert_corpus_2b] wrote {n_decisions} decisions -> {decisions_path}", file=sys.stderr)
    print(f"[build_expert_corpus_2b] wrote stats -> {stats_path}", file=sys.stderr)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Собрать decision-level (obs, action) корпус решений эталонов "
                     "(Luca / szlachetny snieg / Dominic Peel) из out/top5_recon для BC (план 2B)."
    )
    parser.add_argument("--episodes-dir", type=Path, default=DEFAULT_EPISODES_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--experts", nargs="+", default=DEFAULT_EXPERTS)
    parser.add_argument("--archetype", default=DEFAULT_ARCHETYPE,
                        help="брать только стороны с этим subj_arche (пусто = все деки эксперта)")
    parser.add_argument("--limit", type=int, default=0, help="ограничить числом эпизодов (0 = все; smoke: --limit 20)")
    parser.add_argument("--workers", type=int, default=2, help="параллельных процессов (машина занята — по умолчанию 2)")
    args = parser.parse_args()

    if not args.episodes_dir.is_dir():
        parser.error(f"episodes-dir not found: {args.episodes_dir}")
    if not args.ledger.is_file():
        parser.error(f"ledger not found: {args.ledger}")

    build(args)


if __name__ == "__main__":
    main()
