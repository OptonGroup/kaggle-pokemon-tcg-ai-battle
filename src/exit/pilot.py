"""WP4 — BC-prior / policy-prior (обучение + инференс).

Обучает policy-prior на (obs → action) парах, стартуя с ЛУЧШИХ игр
(выигранные PIMC-партии / топ-реплеи). Инференс даёт ранжирование опций для
использования: (а) как rollout policy внутри PIMC (WP3), (б) как standalone-агент.

УРОК ПРОЕКТА (критично): "quality > volume". Сабмит v008 (×3 данных) РЕГРЕССировал
vs v007. Для BC берём ЛУЧШИЕ игры (выигранные, топ-игроки), а не все подряд.

НА ЧЁМ УЖЕ ПОСТРОЕНО (переиспользовать, не дублировать):
  - src/il/features_iono.py  — 66-мерные фичи per-option для Iono (из raw obs_dict, без libcg)
  - src/il/features_v011.py  — фичи для Crustle/Typhlosion
  - src/il/features.py       — 49-мерные общие фичи (v004)
  - src/il/dataset.py        — построение датасета из реплеев (obs, action) + skill-weighting
  - src/il/train.py          — LightGBM lambdarank обучение (ranker)
  - scripts/train_iono_il.py — пайплайн обучения Iono IL модели
  - src/replays/parse.py     — парсинг реплеев (stdlib-only, работает на macOS без libcg)

Backbone: LightGBM lambdarank — уже доказан в v007/v009a, уже есть в Docker-образе.
torch НЕ нужен (нет в Docker). Если в будущем понадобится MLP — добавить torch в
docker/Dockerfile и переключить model_type="mlp".

Формат модели: LightGBM ranker (.txt). Инференс укладывается в Q-DEPS и Q-TIME.

Пример использования:
  from src.exit.pilot import Pilot
  from src.il.features_iono import build_feature_matrix  # переиспользуем

  pilot = Pilot(feature_fn=build_feature_matrix, model_type="lightgbm")
  pilot.train(trajectories, min_reward=1.0)   # только выигранные
  pilot.save("models/pilot_iter1.txt")
  action = pilot.agent(obs_dict)              # инференс

  # Или: обучить напрямую из реплеев (seed-данные)
  pilot = Pilot.train_from_replays(
      replay_dirs=["data/episodes/top_players"],
      player_filter={"onechan1"},             # quality: только лучший игрок
  )
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .harness import Trajectory

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Дефолтный дек (Iono Electric Swarm) — для standalone-режима
IONO_DECK: list[int] = [
    265, 265, 265, 268, 268, 268, 269, 269, 269, 270, 270, 270, 271, 271, 271,
    1233, 1233, 1233, 1233, 1227, 1227, 1227, 1227, 1086, 1086, 1086,
    1121, 1121, 1121, 1254, 1254, 1254, 1097, 1097, 1152, 1152, 1110, 1118,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
]


# ---------------------------------------------------------------------------
# Вспомогательные метрики (без lightgbm, только numpy)
# ---------------------------------------------------------------------------

def _top1_accuracy(
    scores: "object",  # np.ndarray
    y: "object",       # np.ndarray
    groups: "object",  # np.ndarray
    ctx: "object",     # np.ndarray
    main_ctx: int = 0,
) -> dict[str, float]:
    """Top-1 accuracy overall и для MAIN-контекста (ctx=0)."""
    import numpy as np
    scores = np.asarray(scores)
    y = np.asarray(y)
    groups = np.asarray(groups)
    ctx = np.asarray(ctx)

    correct_overall = correct_main = n_total = n_main = 0
    pos = 0
    for gsz, gctx in zip(groups, ctx):
        sg = scores[pos: pos + gsz]
        yg = y[pos: pos + gsz]
        pos += gsz
        if len(sg) == 0:
            continue
        top_idx = int(np.argmax(sg))
        hit = bool(yg[top_idx] == 1)
        n_total += 1
        correct_overall += int(hit)
        if int(gctx) == main_ctx:
            n_main += 1
            correct_main += int(hit)

    return {
        "overall": correct_overall / n_total if n_total else 0.0,
        "main": correct_main / n_main if n_main else 0.0,
        "n_total": n_total,
        "n_main": n_main,
    }


def _option0_accuracy(y: "object", groups: "object", ctx: "object", main_ctx: int = 0) -> dict:
    """Baseline: всегда выбираем option 0."""
    import numpy as np
    y = np.asarray(y)
    groups = np.asarray(groups)
    ctx = np.asarray(ctx)

    correct_overall = correct_main = n_total = n_main = 0
    pos = 0
    for gsz, gctx in zip(groups, ctx):
        yg = y[pos: pos + gsz]
        pos += gsz
        if len(yg) == 0:
            continue
        hit = bool(yg[0] == 1)
        n_total += 1
        correct_overall += int(hit)
        if int(gctx) == main_ctx:
            n_main += 1
            correct_main += int(hit)

    return {
        "overall": correct_overall / n_total if n_total else 0.0,
        "main": correct_main / n_main if n_main else 0.0,
    }


# ---------------------------------------------------------------------------
# Главный класс
# ---------------------------------------------------------------------------

class Pilot:
    """BC-prior policy: обучение на (obs→action) + инференс ранжирования опций.

    Args:
        feature_fn:  Функция извлечения фич per-option: (obs_dict) -> np.ndarray
                     формы [n_options, FEATURE_DIM]. По умолчанию —
                     build_feature_matrix из src/il/features_iono.py (66d, без libcg).
        model_type:  "lightgbm" (LightGBM lambdarank). MLP не реализован (torch нет
                     в Docker-образе); если нужен — добавить torch в docker/Dockerfile.
        deck:        Дек для standalone-режима (60 card IDs).

    Атрибуты:
        model:  Обученная LightGBM Booster или None до train/load.
    """

    def __init__(
        self,
        feature_fn: Callable[[dict], object] | None = None,
        model_type: str = "lightgbm",
        deck: list[int] | None = None,
    ):
        self.feature_fn = feature_fn
        self.model_type = model_type
        self.deck = deck or list(IONO_DECK)
        self.model = None  # LightGBM Booster или None

    # -----------------------------------------------------------------------
    # Вспомогательные методы
    # -----------------------------------------------------------------------

    def _get_feature_fn(self) -> Callable[[dict], "object"]:
        """Возвращает функцию извлечения фич (ленивый импорт)."""
        if self.feature_fn is not None:
            return self.feature_fn
        # Пробуем src.il (local dev, repo root в sys.path) → il (Docker PYTHONPATH=src)
        try:
            from src.il.features_iono import build_feature_matrix
            return build_feature_matrix
        except ImportError:
            pass
        try:
            from il.features_iono import build_feature_matrix  # type: ignore[import]
            return build_feature_matrix
        except ImportError as e:
            raise ImportError(
                "Не удалось импортировать build_feature_matrix из features_iono. "
                "Укажите feature_fn= явно или проверьте PYTHONPATH."
            ) from e

    @staticmethod
    def _episode_to_val(game_id: int, val_frac: float = 0.15) -> bool:
        """Детерминированный сплит train/val по game_id (по хешу)."""
        h = int(hashlib.md5(str(game_id).encode()).hexdigest(), 16)
        return (h % 100) < int(val_frac * 100)

    # -----------------------------------------------------------------------
    # Обучение
    # -----------------------------------------------------------------------

    def train(
        self,
        trajectories: "list[Trajectory]",
        min_reward: float = 1.0,
        skill_weight: bool = True,
        num_trees: int = 300,
        val_frac: float = 0.15,
    ) -> dict:
        """Обучает LightGBM lambdarank на траекториях.

        КРИТИЧНО (quality>volume): берём только партии с final_reward >= min_reward.
        Сплит train/val — детерминировано по game_id (не по строкам — нет data leak).

        Args:
            trajectories: Список объектов с полями: obs_dict, chosen, n_options,
                          final_reward, game_id. Совместимо с harness.Trajectory и
                          с конвертированными парами из parse_replay() (см. train_from_replays).
            min_reward:   Минимальный final_reward для включения (1.0 = только победы).
            skill_weight: Взвешивать примеры ×2 для победителей (skill-aware).
            num_trees:    Число деревьев LightGBM (lambdarank, без early stopping).
            val_frac:     Доля для валидации (сплит по game_id).

        Returns:
            dict: {n_train_groups, n_val_groups, n_train_rows, skip_errors, num_trees,
                   top1_val_overall, top1_val_main, top1_baseline_overall, top1_baseline_main}.
        """
        import numpy as np
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm нужен для Pilot.train(). "
                "Запускайте в Docker (lightgbm уже установлен там)."
            ) from exc

        feature_fn = self._get_feature_fn()

        # Фильтр: только выигранные (CRITICAL: quality > volume)
        winning = [
            t for t in trajectories
            if getattr(t, "final_reward", 0.0) >= min_reward
        ]
        if not winning:
            raise ValueError(
                f"Нет траекторий с final_reward >= {min_reward}. "
                f"Всего: {len(trajectories)}. "
                "Только выигранные партии идут в BC (quality > volume — урок v008)."
            )

        # Проверяем разнообразие game_id для сплита
        game_ids = [getattr(t, "game_id", 0) for t in winning]
        unique_game_ids = set(game_ids)
        all_same = len(unique_game_ids) == 1

        train_X, train_y, train_g, train_w, train_ctx = [], [], [], [], []
        val_X,   val_y,   val_g,   val_w,   val_ctx   = [], [], [], [], []

        skip_errors = 0
        total = 0

        for idx, traj in enumerate(winning):
            obs      = traj.obs_dict
            chosen   = traj.chosen
            n_opts   = traj.n_options
            game_id  = getattr(traj, "game_id", 0)

            if n_opts == 0 or not chosen:
                skip_errors += 1
                continue

            try:
                X_group = feature_fn(obs)
            except Exception:
                skip_errors += 1
                continue

            if X_group.shape[0] != n_opts:
                skip_errors += 1
                continue

            y_group = np.zeros(n_opts, dtype=np.int8)
            for ci in chosen:
                if 0 <= ci < n_opts:
                    y_group[ci] = 1

            w = 2.0 if skill_weight else 1.0
            ctx = int((obs.get("select") or {}).get("context") or 0)

            # Сплит по game_id (hash-детерминированный); fallback — по индексу
            if all_same:
                is_val = (idx % 100) < int(val_frac * 100)
            else:
                is_val = self._episode_to_val(game_id, val_frac)

            target_X   = val_X   if is_val else train_X
            target_y   = val_y   if is_val else train_y
            target_g   = val_g   if is_val else train_g
            target_w   = val_w   if is_val else train_w
            target_ctx = val_ctx if is_val else train_ctx

            target_X.append(X_group)
            target_y.append(y_group)
            target_g.append(n_opts)
            target_w.append(np.full(n_opts, w, dtype=np.float32))
            target_ctx.append(ctx)
            total += 1

        if total < 10:
            raise ValueError(
                f"Слишком мало валидных траекторий: {total} (skip_errors={skip_errors}). "
                "Проверьте данные."
            )
        if not train_X:
            raise ValueError(
                "Все траектории попали в val. Увеличьте данные или уменьшите val_frac."
            )

        # Сборка массивов
        X_tr = np.vstack(train_X).astype(np.float32)
        y_tr = np.concatenate(train_y).astype(np.int32)
        g_tr = np.array(train_g, dtype=np.int32)
        w_tr = np.concatenate(train_w).astype(np.float32)

        has_val = bool(val_X)
        if has_val:
            X_va = np.vstack(val_X).astype(np.float32)
            y_va = np.concatenate(val_y).astype(np.int32)
            g_va = np.array(val_g, dtype=np.int32)
            c_va = np.array(val_ctx, dtype=np.int32)

        # Feature names (для LightGBM interpretability)
        feat_names = None
        try:
            from src.il.features_iono import FEATURE_NAMES
            feat_names = FEATURE_NAMES
        except ImportError:
            try:
                from il.features_iono import FEATURE_NAMES  # type: ignore[import]
                feat_names = FEATURE_NAMES
            except ImportError:
                pass

        # LightGBM lambdarank — те же гиперпараметры что в train_iono_il.py (C6 config)
        dtrain = lgb.Dataset(
            X_tr, label=y_tr, group=g_tr, weight=w_tr,
            feature_name=feat_names, free_raw_data=False,
        )

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3],
            "num_leaves": 15,
            "learning_rate": 0.005,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
            "n_jobs": -1,
        }

        callbacks = [lgb.log_evaluation(period=50)]

        if has_val:
            dval = lgb.Dataset(
                X_va, label=y_va, group=g_va,
                feature_name=feat_names, reference=dtrain, free_raw_data=False,
            )
            booster = lgb.train(
                params, dtrain, num_boost_round=num_trees,
                valid_sets=[dval], valid_names=["val"],
                callbacks=callbacks,
            )
        else:
            booster = lgb.train(
                params, dtrain, num_boost_round=num_trees,
                callbacks=callbacks,
            )

        self.model = booster

        # Метрики
        metrics: dict = {
            "n_train_groups": len(g_tr),
            "n_val_groups":   len(g_va) if has_val else 0,
            "n_train_rows":   len(X_tr),
            "skip_errors":    skip_errors,
            "num_trees":      booster.num_trees(),
        }

        if has_val:
            scores_va = booster.predict(X_va)
            acc = _top1_accuracy(scores_va, y_va, g_va, c_va)
            bas = _option0_accuracy(y_va, g_va, c_va)
            metrics.update({
                "top1_val_overall":       acc["overall"],
                "top1_val_main":          acc["main"],
                "top1_baseline_overall":  bas["overall"],
                "top1_baseline_main":     bas["main"],
                "n_val_main_groups":      acc["n_main"],
            })

        return metrics

    @classmethod
    def train_from_replays(
        cls,
        replay_dirs: list[str],
        feature_fn: Callable | None = None,
        deck: list[int] | None = None,
        player_filter: set[str] | None = None,
        min_reward: float = 1.0,
        num_trees: int = 300,
        val_frac: float = 0.15,
        skip_contexts: set[int] | None = None,
    ) -> "Pilot":
        """Удобный вход: реплеи → Trajectory-like объекты → train().

        Применяет фильтр "только выигранные + только указанные игроки" (quality > volume).

        Args:
            replay_dirs:    Директории с *-replay.json.
            player_filter:  Имена игроков для включения; None = все.
                            Пример: {"onechan1"} — только лучший Iono-игрок.
            min_reward:     Порог победы (1.0 = только победы).
            skip_contexts:  Пропускать контексты (по умолчанию {41, 42}).

        Returns:
            Обученный Pilot.
        """
        from src.replays.parse import parse_replay, walk_replays

        if skip_contexts is None:
            skip_contexts = {41, 42}  # IS_FIRST, MULLIGAN

        dirs = [Path(d) for d in replay_dirs]
        paths = walk_replays(dirs)

        # Дедупликация по имени файла
        seen: set[str] = set()
        unique: list[Path] = []
        for p in paths:
            if p.name not in seen:
                seen.add(p.name)
                unique.append(p)

        # Лёгкий Trajectory-like без импорта harness (избегаем circular import)
        class _T:
            __slots__ = ("obs_dict", "chosen", "n_options", "final_reward", "game_id")

            def __init__(self, obs_dict, chosen, n_options, final_reward, game_id):
                self.obs_dict    = obs_dict
                self.chosen      = chosen
                self.n_options   = n_options
                self.final_reward = final_reward
                self.game_id     = game_id

        trajs: list[_T] = []
        parse_errors = skipped_scope = 0

        for path in unique:
            try:
                pairs = parse_replay(path)
            except Exception:
                parse_errors += 1
                continue

            for pair in pairs:
                if pair["select_context"] in skip_contexts:
                    continue
                if player_filter is not None and pair["player_name"] not in player_filter:
                    skipped_scope += 1
                    continue

                final_reward = 1.0 if pair["won"] else -1.0
                trajs.append(_T(
                    obs_dict      = pair["obs_dict"],
                    chosen        = pair["chosen_option_indices"],
                    n_options     = pair["n_options"],
                    final_reward  = final_reward,
                    game_id       = pair["episodeId"],
                ))

        print(
            f"[Pilot.train_from_replays] replays={len(unique)} "
            f"parse_errors={parse_errors} skipped_scope={skipped_scope} "
            f"total_pairs={len(trajs)}"
        )

        pilot = cls(feature_fn=feature_fn, deck=deck)
        metrics = pilot.train(trajs, min_reward=min_reward, num_trees=num_trees, val_frac=val_frac)
        print(f"[Pilot.train_from_replays] train metrics: {metrics}")
        return pilot

    # -----------------------------------------------------------------------
    # Инференс
    # -----------------------------------------------------------------------

    def score(self, obs_dict: dict) -> list[float]:
        """Ранжирующие скоры на каждую опцию. Выше = предпочтительнее.

        Returns:
            list[float] длины n_options.

        Raises:
            RuntimeError: если модель не обучена.
        """
        if self.model is None:
            raise RuntimeError("Pilot.score(): модель не загружена. Вызовите train() или load().")

        import numpy as np
        feature_fn = self._get_feature_fn()
        X = feature_fn(obs_dict)
        if X.shape[0] == 0:
            return []
        scores = self.model.predict(np.asarray(X, dtype=np.float32))
        return scores.tolist()

    def agent(self, obs_dict: dict) -> list[int]:
        """Never-throw entry point — совместим с Agent.policy контрактом.

        Первый вызов (select is None) → self.deck (60 card IDs).
        Иначе → argmax по score() (топ maxCount опций).
        Fallback на первые k индексов если модель недоступна или упала.
        """
        sel = obs_dict.get("select") if obs_dict else None
        if sel is None:
            return list(self.deck)

        n_options = len(sel.get("option") or [])
        min_count = int(sel.get("minCount") or 1)
        max_count = int(sel.get("maxCount") or 1)

        if n_options == 0:
            return []

        # Пробуем модель
        if self.model is not None:
            try:
                scores = self.score(obs_dict)
                if len(scores) == n_options:
                    import numpy as np
                    order = [int(i) for i in np.argsort(scores)[::-1]]
                    order = [i for i in order if 0 <= i < n_options]
                    k = max(1, min_count)
                    k = min(k, max_count, n_options)
                    result = order[:k]
                    # Валидация: нужный размер + без дублей
                    if len(result) >= min_count and len(set(result)) == len(result):
                        return result
            except Exception:
                pass

        # Fallback — тривиальный (never-throw)
        k = max(1, min_count)
        k = min(k, max_count, n_options)
        return list(range(k))

    # -----------------------------------------------------------------------
    # Сериализация
    # -----------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Сохраняет LightGBM-модель в .txt файл."""
        if self.model is None:
            raise RuntimeError("Pilot.save(): нет модели. Сначала вызовите train().")
        if self.model_type != "lightgbm":
            raise NotImplementedError(
                f"save() реализован только для lightgbm, не для {self.model_type!r}."
            )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: str) -> "Pilot":
        """Загружает LightGBM-модель из .txt файла. Ленивый import. Возвращает self."""
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("lightgbm нужен для Pilot.load().") from exc
        self.model = lgb.Booster(model_file=str(path))
        self.model_type = "lightgbm"
        return self


# ---------------------------------------------------------------------------
# Упаковка в Kaggle-сабмит
# ---------------------------------------------------------------------------

def export_to_submission(
    pilot: Pilot,
    deck: list[int],
    out_dir: str,
    feature_module: str = "features_iono.py",
) -> str:
    """Упаковывает обученный Pilot в директорию сабмита.

    Создаёт out_dir/{main.py, <feature_module>, model.txt, deck.csv, EN_Card_Data.csv, cg/}.
    main.py — self-contained: never-throw agent + LightGBM scorer + __file__ guard.
    (Kaggle грузит main.py через exec() БЕЗ __file__ — guard обязателен, см. MEMORY.)

    feature_module — файл фич (из src/il/), например "features_iono.py".
    В нём патчится _DATA_PATH для корректной работы в сабмите (multi-path resolution,
    как в features_v009a.py). EN_Card_Data.csv копируется co-located в out_dir/.

    Args:
        pilot:          Обученный Pilot (model != None).
        deck:           60 card IDs.
        out_dir:        submissions/vNNN_pilot/.
        feature_module: Файл фич для копирования из src/il/.

    Returns:
        Абсолютный путь к созданному out_dir.

    Raises:
        RuntimeError: если pilot.model не обучен.
        FileNotFoundError: если feature_module или src/cg/ не найдены.
    """
    if pilot.model is None:
        raise RuntimeError(
            "export_to_submission: pilot.model не обучен. Вызовите pilot.train() сначала."
        )
    if len(deck) != 60:
        raise ValueError(f"deck должен содержать 60 карт, получено {len(deck)}.")

    repo = Path(_REPO)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1. Копируем cg/ (game engine) — источник: src/cg/
    cg_src = repo / "src" / "cg"
    if not cg_src.is_dir():
        raise FileNotFoundError(f"src/cg/ не найдена: {cg_src}")
    cg_dst = out / "cg"
    if cg_dst.exists():
        shutil.rmtree(str(cg_dst))
    shutil.copytree(str(cg_src), str(cg_dst))

    # 2. deck.csv (один ID на строку)
    with open(out / "deck.csv", "w") as f:
        for cid in deck:
            f.write(f"{cid}\n")

    # 3. Сохраняем модель → model.txt
    pilot.save(str(out / "model.txt"))

    # 4. Копируем и патчим feature_module
    #    Патч: заменяем хрупкий _DATA_PATH на v009a-style multi-path resolution.
    feat_src = repo / "src" / "il" / feature_module
    if not feat_src.exists():
        raise FileNotFoundError(
            f"Feature module не найден: {feat_src}. "
            f"Укажите правильный feature_module= (напр. 'features_iono.py')."
        )
    feat_text = feat_src.read_text(encoding="utf-8")
    feat_text = _patch_features_data_path(feat_text)
    (out / feature_module).write_text(feat_text, encoding="utf-8")

    # 5. Копируем EN_Card_Data.csv co-located (для Kaggle runtime)
    csv_src = repo / "data" / "EN_Card_Data.csv"
    if csv_src.exists():
        shutil.copy(str(csv_src), str(out / "EN_Card_Data.csv"))
    else:
        print(
            f"[export_to_submission] WARN: {csv_src} не найдена — "
            "сабмит будет искать CSV по fallback-путям."
        )

    # 6. Генерируем main.py
    feat_stem = feature_module.removesuffix(".py")
    main_text = _generate_main_py(feat_stem=feat_stem, feature_module=feature_module)
    (out / "main.py").write_text(main_text, encoding="utf-8")

    # 7. Smoke-тест: exec() main.py без __file__ (эмуляция Kaggle)
    _smoke_test_main(out / "main.py", deck)

    print(f"[export_to_submission] Готово: {out}")
    return str(out)


def _patch_features_data_path(src_text: str) -> str:
    """Заменяет хрупкий _DATA_PATH в features_*.py на v009a-style multi-path resolution.

    Исходная строка (из features_iono.py):
        _DATA_PATH = Path(__file__).parent.parent.parent / "data" / "EN_Card_Data.csv"

    Заменяем на функцию с fallback-кандидатами (как в features_v009a.py), которая
    ищет EN_Card_Data.csv: сначала в той же директории (submission / Kaggle agent),
    затем по dev-пути через __file__, затем в /kaggle_simulations/agent, затем в CWD.
    """
    OLD = '_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "EN_Card_Data.csv"'
    NEW = '''def _find_card_data_csv() -> Path:
    """Find EN_Card_Data.csv: submission dir first, then repo dev path, then Kaggle agent."""
    try:
        _base = Path(__file__).parent
    except NameError:
        _base = Path("/kaggle_simulations/agent")
    candidates = [
        _base / "EN_Card_Data.csv",                              # submission / Kaggle agent dir
        _base.parent.parent.parent / "data" / "EN_Card_Data.csv",  # repo dev path
        Path("/kaggle_simulations/agent/EN_Card_Data.csv"),      # Kaggle agent dir fallback
        Path("EN_Card_Data.csv"),                                 # CWD fallback
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # если ничего нет — вернём первый (упадёт при первом чтении)

_DATA_PATH = _find_card_data_csv()'''

    if OLD in src_text:
        return src_text.replace(OLD, NEW, 1)

    # Fallback: если точная строка не найдена, пробуем патч по PREFIX
    # (на случай форматирования или изменения отступов)
    import re
    pat = re.compile(
        r'^_DATA_PATH\s*=\s*Path\(__file__\).*EN_Card_Data\.csv.*$',
        re.MULTILINE,
    )
    if pat.search(src_text):
        return pat.sub(NEW, src_text, count=1)

    # Вставляем функцию в конец, а OLD добавляем как строку-комментарий
    print(
        "[export_to_submission] WARN: не удалось запатчить _DATA_PATH в features файле — "
        "убедитесь, что EN_Card_Data.csv co-located в директории сабмита."
    )
    return src_text


def _generate_main_py(feat_stem: str, feature_module: str) -> str:
    """Генерирует self-contained main.py для сабмита.

    Содержит: __file__ guard, load deck, load LightGBM model, import features, agent().
    Следует структуре submissions/v009a_crustle/main.py.
    """
    return f'''\
"""
Pilot IL agent — сгенерировано src/exit/pilot.py:export_to_submission().
LightGBM lambdarank policy prior, обученный на выигранных партиях (quality > volume).
"""
from __future__ import annotations

import os
import sys

# ── Path bootstrap — __file__ guard для Kaggle exec() ────────────────────────
# Kaggle загружает main.py через exec() без __file__. Порядок резолюции:
#   1. /kaggle_simulations/agent  (Kaggle runtime — всегда верно там)
#   2. __file__-based dir          (local / importlib путь)
#   3. os.getcwd()                 (последний резерв)
_KAGGLE_AGENT_DIR = "/kaggle_simulations/agent"
if os.path.isdir(_KAGGLE_AGENT_DIR):
    _DIR = _KAGGLE_AGENT_DIR
else:
    try:
        _DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _DIR = os.getcwd()
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import numpy as np

# ── Счётчики для диагностики ─────────────────────────────────────────────────
_n_model_decisions: int = 0
_n_trivial_fallback: int = 0
_n_heuristic_fallback: int = 0  # совместимость с gate_runner
_exception_count: int = 0


def _find_file(name: str) -> str:
    """Ищет файл в директории сабмита, CWD и в /kaggle_simulations/agent."""
    candidates = [
        os.path.join(_DIR, name),
        name,
        os.path.join("/kaggle_simulations/agent", name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(_DIR, name)


# ── Дек ──────────────────────────────────────────────────────────────────────
_deck_path = _find_file("deck.csv")
with open(_deck_path, "r") as _f:
    _csv_lines = [ln.strip() for ln in _f if ln.strip()]
my_deck: list[int] = [int(x) for x in _csv_lines[:60]]
assert len(my_deck) == 60, f"deck.csv: {{len(my_deck)}} карт (ожидалось 60)"

# ── LightGBM модель ───────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    _model_path = _find_file("model.txt")
    _ranker: lgb.Booster | None = lgb.Booster(model_file=_model_path)
except Exception as _load_err:
    _ranker = None
    print(f"[pilot] WARNING: не удалось загрузить модель: {{_load_err}}", file=sys.stderr)

# ── Фичи ─────────────────────────────────────────────────────────────────────
# {feature_module} лежит co-located; sys.path уже содержит _DIR.
try:
    from {feat_stem} import build_feature_matrix
    _features_ok = True
except Exception as _feat_err:
    _features_ok = False
    print(f"[pilot] WARNING: импорт {feature_module} не удался: {{_feat_err}}", file=sys.stderr)


# ── Core scorer ──────────────────────────────────────────────────────────────

def _il_decide(obs_dict: dict) -> list[int]:
    """Ранжирует опции LightGBM и возвращает лучший выбор."""
    if _ranker is None or not _features_ok:
        raise RuntimeError("модель или фичи недоступны")

    sel = obs_dict.get("select") or {{}}
    opts = sel.get("option") or []
    n_opts = len(opts)
    min_count = int(sel.get("minCount") or 1)
    max_count = int(sel.get("maxCount") or 1)

    if n_opts == 0:
        return []

    X = build_feature_matrix(obs_dict)
    if X.shape[0] != n_opts:
        raise ValueError(f"feature matrix shape mismatch: {{X.shape[0]}} != {{n_opts}}")

    scores = _ranker.predict(X.astype(np.float32))
    order = [int(i) for i in np.argsort(scores)[::-1]]
    order = [i for i in order if 0 <= i < n_opts]

    k = max(min_count, 1)
    k = min(k, max_count, n_opts)
    return order[:k]


# ── Публичная точка входа ─────────────────────────────────────────────────────

def agent(obs_dict: dict) -> list[int]:
    """Never-throw entry point.

    1. Deck-selection (select is None) → my_deck.
    2. IL model → best-ranked option(s).
    3. Trivial fallback (первые k индексов) — если модель упала.
    """
    global _n_model_decisions, _n_trivial_fallback, _n_heuristic_fallback, _exception_count

    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return my_deck

    n_options = len((sel.get("option") or []))
    min_count = int(sel.get("minCount") or 1)
    max_count = int(sel.get("maxCount") or 1)

    if n_options == 0:
        _n_trivial_fallback += 1
        return []

    # 1. IL model
    try:
        result = _il_decide(obs_dict)
        if result and all(0 <= i < n_options for i in result) and len(result) >= min_count:
            _n_model_decisions += 1
            return result
    except Exception as exc:
        _exception_count += 1
        if _exception_count <= 3:
            print(
                f"[pilot] exception #{{_exception_count}}: {{exc}}  "
                f"(model={{_n_model_decisions}} fallback={{_n_trivial_fallback}})",
                file=sys.stderr,
            )

    # 2. Trivial fallback — always legal
    _n_trivial_fallback += 1
    k = max(1, min_count)
    k = min(k, max_count, n_options)
    return list(range(k))
'''


def _smoke_test_main(main_path: Path, deck: list[int]) -> None:
    """Эмулирует Kaggle exec() без __file__ — проверяет что NameError не возникает.

    Выполняется в изолированном namespace (без __file__, как на Kaggle).
    Импорт lightgbm может упасть если модель требует Linux — это OK (пропускаем).
    """
    src = main_path.read_text(encoding="utf-8")
    ns: dict = {}
    # Не устанавливаем __file__ — эмуляция Kaggle exec()
    try:
        exec(compile(src, str(main_path), "exec"), ns)  # noqa: S102
    except ImportError:
        # lightgbm/libcg.so может не быть на macOS — это ожидаемо
        pass
    except NameError as exc:
        raise RuntimeError(
            f"export_to_submission: main.py упал с NameError (как на Kaggle): {exc}\n"
            "Проверьте __file__ guard в _generate_main_py()."
        ) from exc
    except Exception:
        # Другие ошибки (напр. FileNotFoundError для model.txt) допустимы при smoke-тесте
        pass
