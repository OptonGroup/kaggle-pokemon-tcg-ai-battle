"""v074_strat_meta.py — стратифицированное сравнение двух арм гейта (мета-анализ разности долей).

Зачем отдельный скрипт: пары гейта — это СТРАТЫ с очень разными WR (0.98 у v017 против 0.26
у lucario). Если в армах разное число прогонов на пару, пулинг сырых счётчиков по всем парам
даёт парадокс Симпсона (на прогоне v074 он выдал фиктивные −1.14 пп с p=0.0005 там, где код
заведомо идентичен). Правильный способ — дельта по каждой паре + объединение с
обратно-дисперсионным весом; здесь именно он.

Читает out/v074_gate/aggregate.json и aggregate_p1.json (их пишет scripts/v074_gate_aggregate.py).
"""
import json, math, sys

def strat(pairs, keys, label):
    """Мета-анализ разности долей с фиксированным эффектом (обратно-дисперсионные веса).
    Пары — независимые страты; пулинг сырых счётчиков по стратам НЕВАЛИДЕН, если в армах
    разное число прогонов на пару (парадокс Симпсона)."""
    num = den = 0.0
    rows = []
    for p in pairs:
        if p['key'] not in keys:
            continue
        p1, n1 = p['cand_pooled'], p['cand_n']
        p2, n2 = p['base_pooled'], p['base_n']
        var = p1*(1-p1)/n1 + p2*(1-p2)/n2
        if var <= 0:
            continue
        w = 1.0/var
        num += w*(p1-p2); den += w
        rows.append((p['key'], 100*(p1-p2), n1, n2))
    if den == 0:
        print(f'{label}: нет данных'); return
    D = num/den; se = math.sqrt(1/den); z = D/se
    pv = math.erfc(abs(z)/math.sqrt(2))
    print(f'{label}')
    for k, d, n1, n2 in rows:
        print(f'    {k:32s} Δ={d:+5.2f}пп  (n={n1:.0f} vs {n2:.0f})')
    print(f'    ИТОГО стратифицированно: Δ={100*D:+.2f}пп  95%CI [{100*(D-1.96*se):+.2f}, {100*(D+1.96*se):+.2f}]  p={pv:.4f}\n')

A = json.load(open('out/v074_gate/aggregate.json'))['pairs']
A = [p for p in A if p['key'] != 'v070_h2h']
arch = {'v044', 'v036_h2h', 'ext_masamikobayashi_archaludon'}
allk = {p['key'] for p in A}
print('===== P0: v074 (Full Metal Lab) против v073 =====')
strat(A, arch, 'archaludon (деки с картой 1244 — единственное место, где код может сработать)')
strat(A, arch | {'v049'}, 'archaludon + v049 (3 копии 1244)')
strat(A, allk - arch - {'v049'}, 'КОНТРОЛЬ: пары без карты 1244 (поведение кода идентично v073)')

B = json.load(open('out/v074_gate/aggregate_p1.json'))['pairs']
luc = {'ext_maulikgajera_lucario', 'ext_nursrijan_lucario'}
allb = {p['key'] for p in B}
print('===== P1: v074b (Premium Power Pro) против v074 =====')
strat(B, luc, 'lucario (единственное место, где код может сработать)')
strat(B, allb - luc, 'КОНТРОЛЬ: пары без линии Lucario (поведение кода идентично v074)')
