# Pokemon TCG AI Battle

Kaggle simulation competition — build an AI agent that plays Pokemon Trading Card Game.

> **Execution planner: see [PLAN.md](PLAN.md)** — the single source of truth for what to do next
> (phases, acceptance criteria, submission gate, guardrails, status log). Read it before starting work.

## ПРАВИЛА ПРОЕКТА (постоянные, NON-NEGOTIABLE)

1. **Общение — на русском.** Всегда отвечать пользователю по-русски в этом проекте.
2. **Загрузка на Kaggle — только с явного одобрения человека.** Никогда не запускать
   `scripts/submit.sh` / `kaggle competitions submit` без прямого подтверждения пользователя
   именно на сабмит ("залить"/"submit"/"да, грузи"). Слова вроде "continue"/"продолжай" — НЕ
   являются одобрением на загрузку. Локальный гейт можно гонять свободно; сам аплоад необратим,
   жжёт дневной лимит (5/сутки) и влияет на правило последних-2 — поэтому всегда ждать "ок".

- Competition: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
- Organizer: The Pokémon Company
- Deadline: **August 16, 2026** (then ~2 weeks of additional games, final leaderboard ~Sep 1)
- Prize: Knowledge tier (the $240K `pokemon-tcg-ai-battle-challenge-strategy` is a separate hackathon)
- API docs: https://matsuoinstitute.github.io/cabt/
- Rule differences from official TCG: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/708586

## Task

Build an agent that beats opponents in Pokemon TCG matches via the **cabt Engine** (C library wrapped in Python).

```python
def agent(obs_dict: dict) -> list[int]:
    # First call: obs.select is None → return list of 60 card IDs (the deck)
    # All other calls: return list of chosen option indices
    # Each index must be in range [0, len(obs.select.option))
    # List length must be in [obs.select.minCount, obs.select.maxCount], no duplicates
```

## Submission Rules

- **Format**: `.tar.gz` created with `tar -czvf submission.tar.gz *` from inside the submission dir
- `main.py` must be at the **top level** (not nested)
- Must include `deck.csv` (60 card IDs, one per line)
- Must include the `cg/` directory (game engine)
- **5 submissions per day** max; only the **latest 2** are tracked for final evaluation
- Each submission plays self-validation first, then joins matchmaking pool

## Ranking System

- Skill rating: Gaussian N(μ, σ²), initialized at μ₀=600
- Win → μ increases, Loss → μ decreases, Draw → both μ move toward their mean
- Only best-scoring submission shown on public leaderboard
- Episode replays downloadable from Submissions tab

## Project Structure

```
data/                       # Competition data (large PDFs gitignored)
  EN_Card_Data.csv          # English card database (Card ID, Name, HP, Type, moves, etc.)
  JP_Card_Data.csv          # Japanese card database
  sample_submission/        # Organizer's sample code + game engine (reference only)
  episodes/                 # Episode replay JSONs (gitignored, download from Submissions tab)
                            # Format: {episodeId}.json (full replay), {episodeId}-0/1.json (agent logs)
                            # rewards: [-1,1] means player0 lost, player1 won

src/
  cg/                       # Game engine Python wrappers
  cards.py                  # Card data loading utilities
  simulate.py               # Self-play simulation harness (Linux only)
  agents/
    random_agent.py         # Baseline random agent

submissions/                # Versioned submission directories (what gets submitted)
  v001_random/              # v001: baseline random agent
    main.py
    deck.csv
    cg/

scripts/
  submit.sh                 # ./scripts/submit.sh <version> [message]

results/
  submissions.csv           # Track submission history + scores
  *.tar.gz                  # Submission archives (gitignored)
```

## Key API

```python
# Game engine (Linux x86_64 only — use Kaggle notebook or Docker for testing)
from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, all_card_data, all_attack
from cg.api import search_begin, search_step, search_end  # for MCTS

# Card data
cards = all_card_data()   # -> list[CardData]  (id, name, hp, type, attacks, etc.)
attacks = all_attack()    # -> list[Attack]

# Self-play (Linux only)
obs_dict, start = battle_start(deck0, deck1)  # deck = list[60 card IDs]
obs_dict = battle_select([option_index])
battle_finish()

# Tree search — built-in MCTS support via game engine
state = search_begin(obs, your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active)
state = search_step(state.searchId, [option_index])
search_end()
```

## Platform Notes

- `libcg.so` is **Linux x86_64 only** — won't load on macOS ARM
- For local simulation: use Kaggle notebook, or Docker `--platform linux/amd64`
- We use Python 3.11 locally (via uv). Kaggle runs 3.10+.

## Workflow

```bash
# Setup
uv sync

# Submit a version (creates .tar.gz and uploads)
./scripts/submit.sh v002_heuristic "heuristic agent"

# Check submissions
.venv/bin/kaggle competitions submissions -c pokemon-tcg-ai-battle
```

## Versioning Convention

`v{NNN}_{description}` — e.g. `v001_random`, `v002_heuristic`, `v003_mcts`

Record each submission in `results/submissions.csv`.

## Strategy Roadmap

1. **v001** ✅ Random baseline — score **600.0** (2026-06-16, starting μ)
2. **v002** — Heuristic: always attack if possible, pick highest damage, retreat wisely
3. **v003** — MCTS using `search_begin`/`search_step` engine API
4. **v004** — MCTS + opponent hand/deck modeling
5. **v005+** — RL / self-play training
6. **v012** Iono heuristic — local gate 0.70 vs v009a but **LIVE 499.2**: local gates vs own agents don't predict ladder
7. **v007/v009a** (Crustle IL) = proven champions (**930.7/905.9** live), re-submitted 2026-07-03
8. **2026-07-03**: plan v2 (PLAN.md F0-F5) — meta rotated (Grimmsnarl toolbox top), deck decision by fresh data, then IL pilot v2 → PIMC → ExIt
9. **2026-07-05**: plan v3 (PLAN.md §S, PIMC v1) — ЗАКРЫТ в тот же день: S4a mirror ≈0.50, S4b field −0.40/−0.15 (rollout обеих сторон = Crustle-IL)
10. **2026-07-11**: **plan v4 CURRENT (PLAN.md §H, HYBRID PIMC)** — аудит off-by-one реплеев (форум 717141) → retrain prior → анкерный пул predicted-μ → PIMC v2 с раздельными rollout-политиками → live-probe → опц. ExIt-lite/deck-swap
