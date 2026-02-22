# Codenames — Two Players vs AI Spymasters

Two human players share a terminal, taking turns as Red and Blue operatives.
Two separate AI Spymasters (one per team) generate clues independently using
the full reasoning pipeline: GloVe vectors, ConceptNet, and FOPL logic.

## Folder Structure

```
game play/
├── codenames_game.py          ← Main game file (run this)
├── README.md
└── logs/
    ├── game_TIMESTAMP.json    ← Individual log per game
    └── all_games_history.jsonl← Cumulative log across all sessions
```

## How to Run

From the project root:

```bash
python "game play/codenames_game.py"
```

Startup takes 30–60 seconds while both AI Spymasters load GloVe and ConceptNet.

## How a Game Works

1. A random 25-word board is generated (9 Red, 8 Blue, 7 Neutral, 1 Assassin)
2. Red Spymaster (AI) thinks and announces a clue + count
3. Red operative (human) guesses words on the board
4. Blue Spymaster (AI) thinks and announces a clue + count
5. Blue operative (human) guesses
6. Teams alternate until one team finds all their words, or someone hits the assassin

The board is **not** revealed with team colours at the start — operatives play blind,
just like real Codenames. Revealed words appear struck-through on the board as the
game progresses.

## AI Spymaster Design

Each team has its own independent Spymaster agent with separate state — they cannot
see each other's targets or reasoning. Each agent uses a three-stage pipeline:

| Stage | Method | What it does |
|---|---|---|
| 1 | GloVe vectors | Finds words mathematically close to all team words |
| 2 | ConceptNet graph | Finds words that share structured relationships with multiple targets |
| 3 | FOPL reasoning | Verifies the clue is safe (not linked to assassin or opponent words) |

The best candidate across both stages is chosen and announced.

## Guess Rules

| Guess Result     | What Happens                                      |
|-----------------|---------------------------------------------------|
| Your team's word | ✓ Correct — you may keep guessing                |
| Opponent's word  | ✗ Turn ends immediately                          |
| Neutral word     | ~ Turn ends immediately                          |
| Assassin word    | ☠ Instant loss for your team                     |

**Guess limits:**
- **Turn 1** for each team: exactly `count` guesses (no bonus)
- **Turn 2+** for each team: `count + 1` guesses (standard Codenames bonus guess)

Type `PASS` at any time to end your turn early.

## Log Files

Every game is automatically saved to `logs/`. Logs are never overwritten — they
accumulate across all sessions.

- **`game_TIMESTAMP.json`** — Full log of one game: board layout, every spymaster
  clue with its reasoning metadata (source, targets, score), every player guess
  and outcome, and the final result.

- **`all_games_history.jsonl`** — One JSON record per line, one per game. Useful
  for analysing spymaster performance across many games.

### Example log entries

```json
{ "event": "SPYMASTER_CLUE", "data": {
    "team": "RED", "clue": "SATELLITES", "count": 3,
    "targets": ["MISSILE", "SATURN", "SPY"],
    "source": "vector", "score": 0.57 }}

{ "event": "PLAYER_GUESS", "data": {
    "team": "RED", "word": "MISSILE", "result": "correct" }}

{ "event": "GAME_OVER", "data": {
    "winner": "BLUE", "reason": "RED hit the assassin" }}
```
