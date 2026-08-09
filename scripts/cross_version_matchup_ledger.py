#!/usr/bin/env python3
"""Кросс-сабмитный матчап-леджер: пулит live-игры всех версий одной родословной пилота.

Мотивация (2026-07-25): одиночный сабмит даёт 80-110 игр => 5-15 игр на архетип,
SE 12-20пп. Матчап-вопросы ("lucario — реальная дыра или шум?") на таком N
неразрешимы. Но правки v062->v075 демонстрируемо НЕ двигают общий WR (все версии
в пределах шума друг от друга) — это и есть основание считать их одним оценщиком
и пулить. Скрипт сначала ПРОВЕРЯЕТ это допущение (тест гетерогенности), и только
потом печатает пулированную матчап-статистику.

Контроль силы оппонента: сырой WR по архетипу смешан с рейтингом тех, кто на нём
играет. Стратификация по 100-очковым бендам + Mantel-Haenszel (см. память
project_gate_stratified_measurement — пулинг сырых счётчиков = парадокс Симпсона).

Usage:
    .venv/bin/python scripts/cross_version_matchup_ledger.py [--min-n 15] [--csv out.csv]
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent

# submission_id -> версия. Порядок = хронология родословной пилота.
SUBMISSION_VERSION = {
    "54887093": "v062", "54887100": "v062",
    "54896368": "v063", "54896381": "v063",
    "54910947": "v064", "54911093": "v064",
    "54924307": "v070", "54924311": "v070",
    "54931792": "v070+P0", "54938788": "v070+P0",
    "54950230": "v072", "54950236": "v072",
    "54959231": "v073", "54959237": "v073",
    "54969284": "v074", "54969288": "v074",
    "54974320": "v075", "54974326": "v075",
}
VERSION_ORDER = ["v062", "v063", "v064", "v070", "v070+P0", "v072",
                 "v073", "v074", "v075"]

# Директории с реплеями. Дубли эпизодов между директориями схлопываются по episode_id.
# ВНИМАНИЕ: out/s54957625_wr_check НЕ включена — это разведка ЧУЖОГО сабмита
# (субъект играет Crustle-деку, в списке наших сабмитов id отсутствует). Её our_reward
# = награда чужого агента; включение контаминирует леджер.
# out/v062_full_download включена, но её replays/ вычищены => все строки NO_REPLAY.
SOURCE_DIRS = [
    "v062_full_download", "v063_full_download", "v064_full_download",
    "v070_wr_check", "v070p0_wr_check", "v072_wr_check",
    "v073_full_download", "v074_wr_check", "v075_wr_check",
]

BAD_ARCHE = {"NO_REPLAY", "NO_DECK_DATA", ""}


def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - half) / d, (c + half) / d)


def chi2_indep(table):
    """table: list of [w, l]. Возвращает (chi2, dof, p). Строки с n=0 выкидываются."""
    rows = [r for r in table if sum(r) > 0]
    if len(rows) < 2:
        return (0.0, 0, 1.0)
    tot_w = sum(r[0] for r in rows)
    tot_l = sum(r[1] for r in rows)
    n = tot_w + tot_l
    if tot_w == 0 or tot_l == 0:
        return (0.0, len(rows) - 1, 1.0)
    stat = 0.0
    for w, l in rows:
        rn = w + l
        for obs, col_tot in ((w, tot_w), (l, tot_l)):
            exp = rn * col_tot / n
            stat += (obs - exp) ** 2 / exp
    dof = len(rows) - 1
    return (stat, dof, float(chi2_dist.sf(stat, dof)))


def load_leaderboard():
    """Свежайший снапшот паблик-лидерборда среди всех out/*/ -> {team_name: score}."""
    snaps = sorted(ROOT.glob("out/*/pokemon-tcg-ai-battle-publicleaderboard-*.csv"))
    if not snaps:
        return {}, None
    latest = max(snaps, key=lambda p: p.name.split("publicleaderboard-")[-1])
    lb = {}
    with open(latest, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            lb[row["TeamName"].strip()] = float(row["Score"])
    return lb, latest.name


def load_games():
    lb, lb_name = load_leaderboard()
    games = {}   # episode_id -> dict
    missing_lb = 0
    for d in SOURCE_DIRS:
        path = ROOT / "out" / d / "archetype_index.csv"
        if not path.exists():
            print(f"[warn] нет {path} — пропущена", file=sys.stderr)
            continue
        for row in csv.DictReader(open(path)):
            opp = (row.get("opp_team_name") or "").strip()
            if not opp:          # self-validation
                continue
            arche = row.get("archetype", "")
            if arche in BAD_ARCHE:
                continue
            ver = SUBMISSION_VERSION.get(row["submission_id"])
            if ver is None:
                continue
            ep = row["episode_id"]
            score = lb.get(opp)
            if score is None:
                missing_lb += 1
            games[ep] = {
                "version": ver,
                "sub": row["submission_id"],
                "arche": arche,
                "cluster": row.get("cluster", ""),
                "win": 1 if float(row["our_reward"]) > 0 else 0,
                "opp": opp,
                "opp_score": score,
                "band": None if score is None else min(int(score // 100) * 100, 1200),
            }
    return list(games.values()), lb_name, missing_lb


def wl(rows):
    w = sum(g["win"] for g in rows)
    return w, len(rows) - w


def fmt_ci(w, n):
    lo, hi = wilson(w, n)
    return f"[{lo*100:4.1f},{hi*100:5.1f}]"


def mantel_haenszel(rows, target):
    """Архетип `target` против остального поля, стратифицированно по бендам оппонента.

    Возвращает (OR_MH, p, n_target_used, n_other_used, strata_used).
    Игры без рейтинга оппонента (не в текущем паблик-лидерборде) исключаются.
    """
    by_band = defaultdict(lambda: {"t": [0, 0], "o": [0, 0]})   # [win, loss]
    for g in rows:
        if g["band"] is None:
            continue
        cell = by_band[g["band"]]["t" if g["arche"] == target else "o"]
        cell[0 if g["win"] else 1] += 1
    num = den = 0.0
    nt = no = 0
    strata = 0
    # тест Кохрана-Мантеля-Хензеля
    cmh_num = cmh_var = 0.0
    for band, c in sorted(by_band.items()):
        a, b = c["t"]        # target: win, loss
        cc, d = c["o"]       # rest:   win, loss
        n = a + b + cc + d
        if a + b == 0 or cc + d == 0 or n == 0:
            continue
        strata += 1
        nt += a + b
        no += cc + d
        num += a * d / n
        den += b * cc / n
        r1, r2 = a + b, cc + d
        c1, c2 = a + cc, b + d
        cmh_num += a - r1 * c1 / n
        if n > 1:
            cmh_var += r1 * r2 * c1 * c2 / (n * n * (n - 1))
    if den == 0 or cmh_var == 0:
        return (float("nan"), 1.0, nt, no, strata)
    orr = num / den
    z = cmh_num / (cmh_var ** 0.5)
    p = 2 * norm.sf(abs(z))
    return (orr, p, nt, no, strata)


def main():
    argv = sys.argv[1:]
    min_n = 15
    if "--min-n" in argv:
        min_n = int(argv[argv.index("--min-n") + 1])
    rows, lb_name, missing_lb = load_games()
    print(f"Игр в леджере: {len(rows)} (self-val исключены). "
          f"Лидерборд-снапшот: {lb_name}; без рейтинга оппонента: {missing_lb}")

    # ---------- A. WR по версиям + тест гетерогенности ----------
    print("\n=== A. WR по версиям (проверка допущения о пулинге) ===")
    print(f"{'версия':<12}{'W-L':>9}{'N':>6}{'WR':>8}   95% Wilson")
    by_ver = defaultdict(list)
    for g in rows:
        by_ver[g["version"]].append(g)
    table = []
    for v in VERSION_ORDER:
        if v not in by_ver:
            continue
        w, l = wl(by_ver[v])
        n = w + l
        table.append([w, l])
        print(f"{v:<12}{f'{w}-{l}':>9}{n:>6}{w/n*100:7.1f}%   {fmt_ci(w, n)}")
    stat, dof, p = chi2_indep(table)
    print(f"\nГетерогенность между версиями: chi2={stat:.2f}, dof={dof}, p={p:.3f}")
    print("  p>0.05 => версии неразличимы по WR => пулинг оправдан "
          "(правки ниже шумового пола, родословная = один оценщик)")

    # ---------- B. Пулированный матчап-леджер ----------
    print("\n=== B. Пулированный WR по архетипам (все версии вместе) ===")
    by_arche = defaultdict(list)
    for g in rows:
        by_arche[g["arche"]].append(g)
    overall_w, overall_l = wl(rows)
    overall = overall_w / (overall_w + overall_l)
    print(f"{'архетип':<20}{'W-L':>9}{'N':>5}{'WR':>8}  {'95% Wilson':<16}"
          f"{'ср.рейт.опп':>12}{'z vs поле':>11}")
    ranked = sorted(by_arche.items(), key=lambda kv: -len(kv[1]))
    for arche, gs in ranked:
        w, l = wl(gs)
        n = w + l
        scored = [g["opp_score"] for g in gs if g["opp_score"] is not None]
        mean_s = sum(scored) / len(scored) if scored else float("nan")
        # z архетипа против остального поля (сырой, без стратификации)
        rest = [g for g in rows if g["arche"] != arche]
        rw, rl = wl(rest)
        p1, p2 = w / n, rw / (rw + rl)
        pp = (w + rw) / (n + rw + rl)
        se = (pp * (1 - pp) * (1 / n + 1 / (rw + rl))) ** 0.5
        z = (p1 - p2) / se if se > 0 else 0.0
        mark = "" if n >= min_n else "  (шум, N мал)"
        print(f"{arche:<20}{f'{w}-{l}':>9}{n:>5}{w/n*100:7.1f}%  {fmt_ci(w, n):<16}"
              f"{mean_s:12.1f}{z:11.2f}{mark}")
    print(f"{'ИТОГО':<20}{f'{overall_w}-{overall_l}':>9}"
          f"{overall_w+overall_l:>5}{overall*100:7.1f}%  "
          f"{fmt_ci(overall_w, overall_w+overall_l)}")

    # ---------- C. Архетип x версия (двигала ли хоть одна правка матчап?) ----------
    print("\n=== C. Архетип x версия — двигала ли хоть одна правка конкретный матчап? ===")
    vers = [v for v in VERSION_ORDER if v in by_ver]
    hdr = "".join(f"{v:>11}" for v in vers)
    print(f"{'архетип':<20}{hdr}{'chi2':>8}{'p':>8}")
    for arche, gs in ranked:
        if len(gs) < min_n:
            continue
        cells, tab = [], []
        for v in vers:
            sub = [g for g in gs if g["version"] == v]
            w, l = wl(sub)
            tab.append([w, l])
            cells.append(f"{w}-{l}" if w + l else "·")
        stat, dof, p = chi2_indep(tab)
        line = "".join(f"{c:>11}" for c in cells)
        flag = "  <-- значимо" if p < 0.01 else ""
        print(f"{arche:<20}{line}{stat:8.1f}{p:8.3f}{flag}")

    # ---------- D. Стратифицированный тест (контроль силы оппонента) ----------
    print("\n=== D. Mantel-Haenszel: архетип vs остальное поле, "
          "стратификация по 100-очковым бендам рейтинга оппонента ===")
    print(f"{'архетип':<20}{'OR_MH':>8}{'p':>9}{'N_arche':>9}{'N_rest':>8}{'страт':>7}")
    for arche, gs in ranked:
        if len(gs) < min_n:
            continue
        orr, p, nt, no, strata = mantel_haenszel(rows, arche)
        flag = "  <-- реальная дыра" if (p < 0.01 and orr < 1) else (
               "  <-- реальный плюс" if (p < 0.01 and orr > 1) else "")
        print(f"{arche:<20}{orr:8.2f}{p:9.4f}{nt:9d}{no:8d}{strata:7d}{flag}")
    print("OR<1 => матчап хуже поля при равном рейтинге оппонента; OR>1 => лучше.")

    # ---------- E. WR по бендам (пулировано) ----------
    print("\n=== E. Пулированный WR по бендам рейтинга оппонента ===")
    by_band = defaultdict(list)
    for g in rows:
        if g["band"] is not None:
            by_band[g["band"]].append(g)
    print(f"{'бенд':<12}{'W-L':>9}{'N':>5}{'WR':>8}  95% Wilson")
    for band in sorted(by_band):
        w, l = wl(by_band[band])
        n = w + l
        print(f"{f'{band}-{band+100}':<12}{f'{w}-{l}':>9}{n:>5}"
              f"{w/n*100:7.1f}%  {fmt_ci(w, n)}")

    # ---------- F. Прямая стандартизация + цена матчапа в общем WR ----------
    print("\n=== F. Прямая стандартизация по бендам + цена матчапа для общего WR ===")
    band_rows = defaultdict(list)
    for g in rows:
        if g["band"] is not None:
            band_rows[g["band"]].append(g)

    def standardize(sel):
        gs = [g for g in rows if sel(g) and g["band"] is not None]
        exp = obs = 0.0
        n = 0
        for g in gs:
            peers = [x for x in band_rows[g["band"]] if not sel(x)]
            if not peers:
                continue
            exp += sum(x["win"] for x in peers) / len(peers)
            obs += g["win"]
            n += 1
        if n == 0:
            return None
        delta = (obs - exp) / n * 100
        se = (0.25 / n) ** 0.5 * 100
        return (n, obs / n * 100, exp / n * 100, delta, delta / se, n / len(rows))

    # архетипы + отдельной строкой настоящая Crustle-стена (метка crustle склеивает деки)
    selectors = [(a, (lambda a: (lambda g: g["arche"] == a))(a))
                 for a, gs in ranked if len(gs) >= min_n]
    selectors.append(("[кластер] Crustle-стена",
                      lambda g: "Crustle" in (g["cluster"] or "")))
    print(f"{'матчап':<28}{'N':>5}{'факт':>8}{'ожид':>8}{'дельта':>10}{'z':>7}"
          f"{'доля':>7}{'цена в общем WR':>17}")
    scored = []
    for name, sel in selectors:
        r = standardize(sel)
        if r is None:
            continue
        scored.append((r[3], name, r))
    drag = 0.0
    for delta, name, (n, obs, exp, d, z, share) in sorted(scored):
        cost = d * share
        if name.startswith("[кластер]"):
            pass       # не суммируем: пересекается с меткой crustle
        elif d < 0:
            drag += cost
        print(f"{name:<28}{n:>5}{obs:7.1f}%{exp:7.1f}%{d:+9.1f}пп{z:+7.1f}"
              f"{share*100:6.1f}%{cost:+16.2f}пп")
    print(f"\nСуммарная просадка по минусовым матчапам: {drag:+.2f}пп общего WR")
    print(f"=> подтяжка их до среднего по полю при равном рейтинге ≈ {-drag:+.1f}пп WR")

    if "--csv" in argv:
        out = argv[argv.index("--csv") + 1]
        with open(out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=[
                "version", "sub", "arche", "cluster", "win", "opp", "opp_score", "band"])
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nСырой леджер записан: {out} ({len(rows)} строк)")


if __name__ == "__main__":
    main()
