"""[v084] Bundled archetype deck-ID templates for search_gs.py's hidden-info
determinization (PLAN_v084_SEARCH.md §2.1 step 3).

Data only -- plain lists of 60 int card IDs, no logic, stdlib-only (nothing
to import at all). Snapshotted 2026-08-12 from real 60-card deck.csv files
already present in this repo's tree (our own agents' decks + publicly-visible
opponent decklists already used as gate-pool fixtures elsewhere in this repo,
e.g. scripts/scout_gate_pool2.py's hardcoded pool). This is card-ID data, not
executable code or model weights -- distinct from the "no competitor code/
weights" rule (PLAN_v084_SEARCH.md task item, §6.3), and explicitly the data
source the plan names for this step ("реальные листы из
submissions/external_*/deck.csv и data/deck_scan*/").

Provenance (source file -> archetype, matches _detect_matchup_raw's labels in
main.py):
  CRUSTLE    <- submissions/v009a_crustle/deck.csv           (our own IL agent's list)
  ARCHALUDON <- submissions/v044_archaludon_abomasnow/deck.csv (our own agent's list)
  LUCARIO    <- decks/meta/MegaLucario_live.csv               (live-verified recon)
  TR_MEWTWO  <- decks/meta/TRMewtwo_live.csv                  (live-verified recon)
  ABOMASNOW  <- decks/meta/Abomasnow_live.csv                 (live-verified recon)
  STARMIE    <- decks/meta/Starmie_live.csv                   (live-verified recon)
  ALAKAZAM   <- submissions/external_jazivxt_codex/deck.csv   (Magist, live 994.2)

"mirror" has no entry here -- search_gs.py uses this submission's own
deck.csv (my_deck, passed in by main.py) as the opponent template for that
matchup instead, since we know it exactly.

Not used at runtime for anything except as one candidate hypothesis in a
Monte-Carlo determinization of hidden opponent cards -- never revealed to the
opponent, never influences their play, purely an internal belief used to run
our own search rollouts against a plausible deck shape.
"""

CRUSTLE = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 11, 11, 11, 11, 14, 14, 14, 14, 18, 18, 18, 18, 344, 344, 344, 344, 345, 345, 345, 345, 1086, 1086, 1086, 1086, 1147, 1147, 1147, 1147, 1159, 1212, 1212, 1212, 1212, 1227, 1227, 1227, 1227, 1235, 1235, 1235, 1235]

ARCHALUDON = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 57, 169, 169, 169, 169, 190, 190, 190, 190, 666, 666, 666, 666, 1097, 1097, 1097, 1121, 1121, 1121, 1121, 1122, 1122, 1122, 1122, 1147, 1147, 1147, 1147, 1152, 1152, 1152, 1152, 1159, 1182, 1182, 1182, 1185, 1185, 1185, 1185, 1213, 1227, 1227, 1227, 1227, 1244, 1244, 1244, 1244]

LUCARIO = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1182, 1182, 1182, 1192, 1192, 1192, 1192, 1102, 1102, 1102, 1102, 1142, 1142, 1142, 1142, 1252, 674, 674, 1159, 1227, 1227, 1227, 1227, 675, 675, 673, 673, 678, 678, 678, 678, 1152, 1152, 1141, 1141, 1141, 1141, 677, 677, 677, 677, 676, 676, 676, 1123, 1123]

TR_MEWTWO = [1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 1175, 1094, 1094, 1094, 1119, 1119, 1159, 1227, 1227, 1097, 1152, 1152, 1152, 1152, 1217, 1216, 1216, 1216, 1216, 414, 414, 15, 15, 15, 15, 1257, 1257, 1218, 1218, 1218, 431, 431, 463, 463, 1219, 1220, 1220, 401, 401, 401, 401, 400, 400, 400, 400, 1134, 1134, 1134, 1134, 432]

ABOMASNOW = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 721, 721, 1227, 1227, 1227, 1227, 723, 723, 723, 723, 1145, 1145, 1163, 1163, 1092, 722, 722, 722, 722, 1262, 1262, 1219, 1219, 1219, 1219, 1121, 1121]

STARMIE = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 666, 666, 666, 666, 1086, 1086, 1086, 1086, 1122, 1122, 1122, 1122, 1145, 1145, 1145, 1145, 1229, 1229, 1229, 1229, 1189, 1189, 1189, 1189, 17, 17, 17, 17, 1030, 1030, 1030, 1031, 1031, 1031, 1256, 1256, 1256, 1101, 1101, 1101, 1227, 1227, 1227, 1225, 1225, 1225, 1097, 1159]

ALAKAZAM = [5, 5, 19, 19, 19, 19, 65, 65, 66, 66, 66, 66, 305, 741, 741, 741, 741, 742, 742, 742, 742, 743, 743, 743, 743, 1079, 1079, 1079, 1081, 1081, 1081, 1086, 1086, 1086, 1086, 1097, 1097, 1097, 1129, 1152, 1152, 1152, 1152, 1182, 1182, 1182, 1184, 1197, 1197, 1197, 1225, 1225, 1225, 1225, 1227, 1231, 1231, 1231, 1231, 1247]

TEMPLATES = {
    "crustle": CRUSTLE,
    "archaludon": ARCHALUDON,
    "lucario": LUCARIO,
    "tr_mewtwo": TR_MEWTWO,
    "abomasnow": ABOMASNOW,
    "starmie": STARMIE,
    "alakazam": ALAKAZAM,
}

for _name, _ids in TEMPLATES.items():
    assert len(_ids) == 60, f"{_name}: {len(_ids)} card IDs (expected 60)"
del _name, _ids
