# Pokemon TCG AI Battle

Kaggle simulation competition: build an AI agent that plays Pokemon Trading Card Game.
Competition: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle

## Task

Build an agent that beats opponents in Pokemon TCG matches. The game engine is a compiled C library (`libcg.so` / `cg.dll`) provided by the organizers.

**Agent interface** (`submissions/<version>/main.py`):
```python
def agent(obs_dict: dict) -> list[int]:
    # returns list of option indices (selections)
    # on first call (obs.select is None) must return list of 60 card IDs (the deck)
```

**Deck rules**: exactly 60 cards, standard Pokemon TCG deck-building rules.

## Project Structure

```
data/                       # Competition data files (large PDFs gitignored)
  EN_Card_Data.csv          # English card database
  JP_Card_Data.csv          # Japanese card database
  sample_submission/        # Organizer's sample code + game engine

src/
  cg/                       # Game engine Python wrappers (copied from sample_submission)
  cards.py                  # Card data loading utilities
  simulate.py               # Self-play simulation harness
  agents/
    random_agent.py         # Baseline

submissions/                # Versioned submission directories
  v001_random/              # v001: baseline random agent
    main.py
    deck.csv
    cg/

scripts/
  submit.sh                 # ./scripts/submit.sh <version> [message]

results/
  submissions.csv           # Track submission history + scores
  *.zip                     # Submission zips (gitignored)
```

## Key API

```python
# Game engine (Linux only — use Kaggle or Docker for testing)
from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, all_card_data, all_attack, search_begin, search_step, search_end

# Card data
all_card_data()   # -> list[CardData]
all_attack()      # -> list[Attack]

# Self-play
obs_dict, start = battle_start(deck0, deck1)  # deck = list[60 card IDs]
obs_dict = battle_select([option_index])
battle_finish()

# Tree search (MCTS support)
state = search_begin(obs, your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active)
state = search_step(state.searchId, [option_index])
search_end()
```

## Platform Notes

- **libcg.so is Linux x86_64 only** — won't load on macOS ARM. Test on Kaggle notebooks or in Docker (`--platform linux/amd64`).
- Kaggle runs Python 3.10+. We use 3.11 locally via uv.
- Submission = zip of `main.py` + `deck.csv` + `cg/` directory.

## Workflow

```bash
# Setup
uv sync

# Submit a version
./scripts/submit.sh v002_heuristic "heuristic agent"

# Check leaderboard
kaggle competitions submissions -c pokemon-tcg-ai-battle
```

## Versioning Convention

`v{NNN}_{description}` — e.g. `v001_random`, `v002_heuristic`, `v003_mcts`

Track scores in `results/submissions.csv`.

## Strategy Ideas

1. **v001** (done): Random baseline
2. **v002**: Rule-based heuristic (always attack if possible, prefer high damage)
3. **v003**: MCTS using `search_begin/search_step` API
4. **v004**: MCTS + opponent modeling
