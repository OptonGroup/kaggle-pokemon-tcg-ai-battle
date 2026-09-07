"""turn_order.py — deck-agnostic first/second-move choice (форум 723591: первый ход
+10пп WR, 55.2% vs 44.8%, право выбора первого хода реализуется в 91.5% случаев).

Контекст: `submissions/v076_final/scoring_grimmsnarl.py:855-856` (и тот же паттерн во
всём шлейфе v036/v046/v048/v050 — жёстко проверено) захардкожен на "choose second":

    if ctx == SelectContext.IS_FIRST:
        return (10000, "choose second") if opt.type == OptionType.NO else (0, "go first")

Это никогда не было обосновано в коде (нет комментария/тикета) и противоречит форумной
находке. Отдельно от submissions/v076_final (который эта функция НЕ трогает и НЕ
заменяет — v076_final остаётся как есть), эта функция даёт deck-agnostic реализацию
выбора очерёдности хода для новых/будущих агентов. Дефолт здесь — "first" (меняет
дефолт с текущего "second").

Раньше та же гипотеза уже была закодирована в `src/exit/pimc_agent.py` (H3-3,
`PIMCAgentV2._choose_first_move_if_applicable`, тоже "всегда first"), но: (a) это
другой, экспериментальный PIMC-агент, не deck-agnostic scoring-путь и не сабмит;
(b) PLAN.md отмечает H3-3 как "закодирован, но НЕ валидирован" (smoke-формат
battle_start не экспонирует deck-selection). Эффект первого хода деко-специфичен
(из брифа: Alakazam +8.9пп, Dragapult +1.5пп) — "всегда first" это pooled-эвристика,
не per-deck решение; per-deck калибровка, если понадобится, — отдельная задача поверх
этой функции.

Механика выбора в движке (cg_native / cg, подтверждено v076_final и pimc_agent.py):
SelectContext.IS_FIRST (=41) — это ОТДЕЛЫЙ SelectType.YES_NO (=9) select в начале
эпизода, предъявляется ТОЛЬКО игроку, выигравшему монетку (движок сам разыгрывает
подбрасывание, RNG не управляется извне). Опции — YES_NO пара: OptionType.YES (=1) =
"да, хочу ходить первым", OptionType.NO (=2) = "нет" (то есть уступить первый ход
сопернику). Это НЕ часть обычного MAIN-выбора хода — отдельный контекст до
SETUP_ACTIVE_POKEMON.

Модуль намеренно НЕ импортирует cg_native/cg (см. аналогичный комментарий в
pimc_agent.py) — дублирует нужные int-константы, чтобы оставаться дешёвым для
импорта и не тянуть платформенно-специфичный нативный движок в модули, которым он
не нужен.
"""
from __future__ import annotations

# SelectContext.IS_FIRST / SelectType.YES_NO / OptionType.{YES,NO} — см. src/cg_native/api.py.
_SELECT_CONTEXT_IS_FIRST: int = 41
_SELECT_TYPE_YES_NO: int = 9
_OPTION_TYPE_YES: int = 1
_OPTION_TYPE_NO: int = 2

DEFAULT_TURN_ORDER: str = "first"  # было де-факто "second" во всём шлейфе v036...v076_final


def _ctx_of(obs) -> int | None:
    """obs может быть либо cg(_native).api-style объектом (obs.select.context), либо
    сырым obs_dict (obs["select"]["context"]) — гейты/воркеры гоняют игры через
    сырые dict'ы (см. scripts/gate_scout_guard.py:_play_one_game), а score_* функции
    сабмитов — через обёрнутые объекты. Поддерживаем оба, без импорта cg_native."""
    sel = obs.get("select") if isinstance(obs, dict) else getattr(obs, "select", None)
    if sel is None:
        return None
    ctx = sel.get("context") if isinstance(sel, dict) else getattr(sel, "context", None)
    return int(ctx) if ctx is not None else None


def is_turn_order_select(obs) -> bool:
    """True, если текущий select — это выбор "ходить первым?" (SelectContext.IS_FIRST)."""
    return _ctx_of(obs) == _SELECT_CONTEXT_IS_FIRST


def choose_turn_order(obs) -> str:
    """Deck-agnostic выбор очерёдности хода. Возвращает "first" или "second".

    Игнорирует содержимое obs (deck-agnostic по контракту задачи) — чистая функция
    от факта "это IS_FIRST-select", не от карт/архетипа. Вызывающая сторона обязана
    вызывать её только когда `is_turn_order_select(obs)` истинно; здесь дефолт
    возвращается безусловно для простоты интеграции.
    """
    return DEFAULT_TURN_ORDER


def turn_order_option_index(obs, choice: str | None = None) -> int | None:
    """Находит индекс опции в obs.select.option, соответствующий choice ("first"/
    "second"; по умолчанию — choose_turn_order(obs)). Работает и с сырыми
    list[dict]-опциями, и с объектами Option (opt.type). Возвращает None, если
    подходящей опции нет (например obs — не IS_FIRST select) или список опций пуст.
    """
    if choice is None:
        choice = choose_turn_order(obs)
    want_type = _OPTION_TYPE_YES if choice == "first" else _OPTION_TYPE_NO

    sel = obs.get("select") if isinstance(obs, dict) else getattr(obs, "select", None)
    if sel is None:
        return None
    opts = sel.get("option") if isinstance(sel, dict) else getattr(sel, "option", None)
    if not opts:
        return None

    for i, o in enumerate(opts):
        t = o.get("type") if isinstance(o, dict) else getattr(o, "type", None)
        if t is not None and int(t) == want_type:
            return i
    return None
