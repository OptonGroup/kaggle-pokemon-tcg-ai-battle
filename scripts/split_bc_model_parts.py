#!/usr/bin/env python3
"""split_bc_model_parts.py — разрез сгенерированного bc_model_2b.py на part-модули (v087).

Зачем: пик памяти КОМПИЛЯЦИИ CPython у файла-литерала ~155MB на 1MB исходника и платится
за модуль целиком. Замер в Docker (2 vCPU, образ pokemon-tcg-sim-torch): монолит v086
4.9MB -> пик 714MB (проверенная Kaggle-зона: v083/v085/v086 прошли live-валидацию);
монолит v087 10.7MB -> пик 1.66GB — за пределами доказанной зоны, риск OOM. Разрез на
части <= --max-part-bytes уменьшает пик до max(пик части) + удержанные структуры.

Механика: вход — сгенерированный export_bc_2b.py файл вида
    <docstring/шапка>
    MODELS = {
      "main": (
        ((...),(...)),   # одна строка = одно дерево
        ...
      ),
      ...
    }
    <хвост: predict_group и пр.>
Части пишутся как bc_model_p<N>.py с `TREES_<group>_<i> = (...)` (строки-литералы деревьев
копируются БАЙТ-В-БАЙТ), головной файл собирает MODELS конкатенацией кортежей и несёт
исходный хвост без изменений. Эквивалентность проверяется отдельным скриптом
(scripts/verify_bc_model_split.py) глубоким сравнением MODELS монолита и сборки.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="монолитный bc_model_2b.py от экспортёра")
    ap.add_argument("--out-dir", required=True, help="каталог сабмита (part-файлы лягут рядом)")
    ap.add_argument("--max-part-bytes", type=int, default=3_000_000)
    args = ap.parse_args()

    src = Path(args.src).read_text()
    out_dir = Path(args.out_dir)

    m = re.search(r"^MODELS = \{$", src, re.M)
    if not m:
        sys.exit("FATAL: не найдено 'MODELS = {' на отдельной строке")
    head = src[:m.start()]
    rest = src[m.end() + 1:]
    close = re.search(r"^\}$", rest, re.M)
    if not close:
        sys.exit("FATAL: не найдена закрывающая '}' словаря MODELS")
    body = rest[:close.start()]
    tail = rest[close.end() + 1:]

    # разбор тела: '  "group": (' ... строки деревьев ... '  ),'
    groups: list[tuple[str, list[str]]] = []
    cur: str | None = None
    cur_lines: list[str] = []
    for line in body.splitlines():
        gm = re.match(r'^  "(\w+)": \($', line)
        if gm:
            cur = gm.group(1)
            cur_lines = []
            continue
        if line == "  ),":
            if cur is None:
                sys.exit("FATAL: '),' вне группы")
            groups.append((cur, cur_lines))
            cur = None
            continue
        if cur is not None:
            if not line.strip():
                continue
            cur_lines.append(line)
    if cur is not None:
        sys.exit(f"FATAL: группа {cur} не закрыта")

    # нарезка каждой группы на чанки <= max-part-bytes, упаковка чанков в part-файлы
    chunks: list[tuple[str, int, list[str]]] = []  # (group, chunk_idx, lines)
    for g, lines in groups:
        acc: list[str] = []
        size = 0
        idx = 0
        for ln in lines:
            if acc and size + len(ln) > args.max_part_bytes:
                chunks.append((g, idx, acc))
                idx += 1
                acc, size = [], 0
            acc.append(ln)
            size += len(ln) + 1
        if acc:
            chunks.append((g, idx, acc))

    # part-файлы: жадно пакуем чанки, не превышая max-part-bytes на файл
    parts: list[list[tuple[str, int, list[str]]]] = []
    cur_part: list[tuple[str, int, list[str]]] = []
    cur_size = 0
    for ch in chunks:
        ch_size = sum(len(l) + 1 for l in ch[2])
        if cur_part and cur_size + ch_size > args.max_part_bytes:
            parts.append(cur_part)
            cur_part, cur_size = [], 0
        cur_part.append(ch)
        cur_size += ch_size
    if cur_part:
        parts.append(cur_part)

    part_names = []
    for i, part in enumerate(parts, 1):
        name = f"bc_model_p{i}"
        part_names.append(name)
        with open(out_dir / f"{name}.py", "w") as f:
            f.write(f'"""Часть {i}/{len(parts)} данных BC-модели — сгенерировано '
                    f'scripts/split_bc_model_parts.py, НЕ РЕДАКТИРОВАТЬ."""\n')
            for g, idx, lines in part:
                f.write(f"TREES_{g}_{idx} = (\n")
                f.write("\n".join(lines) + "\n")
                f.write(")\n")

    # головной файл: шапка + сборка MODELS + исходный хвост
    per_group: dict[str, list[str]] = {}
    for i, part in enumerate(parts, 1):
        for g, idx, _ in part:
            per_group.setdefault(g, []).append(f"bc_model_p{i}.TREES_{g}_{idx}")
    order = [g for g, _ in groups]
    with open(out_dir / "bc_model_2b.py", "w") as f:
        f.write(head)
        f.write("# ── v087: данные деревьев вынесены в part-модули (пик компиляции CPython\n")
        f.write("# ~155MB/MB исходника платится ЗА МОДУЛЬ; монолит 10.7MB давал пик 1.66GB,\n")
        f.write("# сборка из частей возвращает пик в проверенную Kaggle-зону <=714MB) ──\n")
        f.write("".join(f"import {n}\n" for n in part_names))
        f.write("\nMODELS = {\n")
        for g in order:
            f.write(f'  "{g}": ' + " + ".join(per_group[g]) + ",\n")
        f.write("}\n")
        f.write(tail)

    total = sum(len(l) + 1 for _, _, ls in chunks for l in ls)
    print(f"частей: {len(parts)}; групп-чанков: {len(chunks)}; байт данных: {total}")
    for i, part in enumerate(parts, 1):
        sz = sum(len(l) + 1 for _, _, ls in part for l in ls)
        print(f"  bc_model_p{i}.py: {sz/1e6:.2f}MB  [{', '.join(f'{g}#{x}' for g, x, _ in part)}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
