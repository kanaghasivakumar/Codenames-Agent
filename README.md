# Codenames AI Spymaster

A knowledge-graph-powered Codenames AI built for the KRR-L (Knowledge Representation, Reasoning, and Learning) course. Two human players compete against AI Spymasters that reason over a structured ConceptNet knowledge graph to generate clues — no large language models, no neural embeddings, just pure symbolic reasoning over explicit world knowledge.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Setup & Installation](#setup--installation)
4. [Building the Knowledge Graph](#building-the-knowledge-graph)
5. [Running the Game](#running-the-game)
6. [How It Works](#how-it-works)
7. [Knowledge Representations Used](#knowledge-representations-used)
8. [Reasoning Methods](#reasoning-methods)
9. [Why This Approach vs an LLM](#why-this-approach-vs-an-llm)

---

## Project Overview

Codenames is a word-association game where a Spymaster must give a single-word clue that links multiple words on a 5x5 board to their team, while avoiding the opponent's words and a deadly assassin word. This makes it a rich testbed for knowledge representation and reasoning — the agent must understand semantic relationships between words, reason about safety constraints, and select the most informative clue.

This project builds that agent from the ground up using ConceptNet, a large open-source commonsense knowledge graph, processed into a targeted JSON graph and queried with a custom reasoning engine. The Spymaster also maintains a per-player profile, learning over time which relation types each player responds well to and adapting its clue strategy accordingly.

---

## Folder Structure

```
KRR-L Project V2/
|
+-- data/
|   +-- codenames_words.txt         <- Word pool for generating boards (~400 Codenames words)
|   +-- common_words.txt            <- Allowed clue vocabulary (~50,000 common English words)
|   +-- conceptnet_graph.json       <- Built knowledge graph (generated, not in repo)
|   +-- conceptnet-assertions-5.7.0.csv.gz  <- Raw ConceptNet download (not in repo, see below)
|
+-- src/
|   +-- build_knowledge_graph.py    <- One-time script: builds conceptnet_graph.json
|   +-- reasoning_engine.py         <- Core AI: graph traversal, clue scoring, penalty model
|   +-- user_profile.py             <- Per-player weight profiles: decay and reward on each turn
|
+-- utils/
|   +-- constants.py                <- Relation weights and penalty constants (single source of truth)
|
+-- profiles/
|   +-- <name>.json                 <- Per-player profile files, auto-created on first game
|
+-- game play/
|   +-- codenames_game.py           <- Interactive two-player terminal game
|   +-- logs/
|       +-- game_TIMESTAMP.json     <- Per-game structured log
|       +-- all_games_history.jsonl <- Cumulative log across all sessions
|
+-- main.py                         <- Standalone AI demo (no human input, Red=AI, Blue=simulated)
```

---

## Setup & Installation

### 1. Create and activate a virtual environment

```bash
python -m venv codenames_env
# Windows:
codenames_env\Scripts\activate
# Mac/Linux:
source codenames_env/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The project has minimal dependencies — the reasoning engine uses only Python's standard library. `requirements.txt` covers anything needed for the game interface.

---

## Building the Knowledge Graph

This is a **one-time setup step**. The graph is not committed to the repo because the source file is ~1.5GB.

### Step 1 — Download ConceptNet

Go to: **https://github.com/commonsense/conceptnet5/wiki/Downloads**

Under the section **"Assertions"**, find the entry:

> *gzipped, tab-separated text file*

Click that hyperlink to download **`conceptnet-assertions-5.7.0.csv.gz`**.

Place it in the `data/` folder:

```
data/conceptnet-assertions-5.7.0.csv.gz
```

### Step 2 — Run the build script

From the **project root**:

```bash
python src/build_knowledge_graph.py
```

This will take **2-5 minutes**. It streams through the full ConceptNet file and extracts only the edges relevant to Codenames words. When it finishes you will see:

```
Parsing Complete. Saving N edges to data/conceptnet_graph.json...
Success! Knowledge Graph is ready.
```

The output `data/conceptnet_graph.json` is what the game loads at runtime. It is much smaller than the raw file and loads in seconds.

### What the build script does

The raw ConceptNet file contains ~35 million assertions across all languages and concepts. The build script applies three filters to make it useful for Codenames:

1. **English only** — keeps only edges where both endpoints are English concepts (`/c/en/`)
2. **Relevant relations only** — keeps only the 17 relation types that capture meaningful semantic links: IsA, HasA, UsedFor, AtLocation, HasProperty, PartOf, Causes, CapableOf, Antonym, DistinctFrom, SimilarTo, MadeOf, ReceivesAction, HasPrerequisite, HasSubevent, CreatedBy, LocatedNear
3. **Game relevance** — keeps only edges where at least one endpoint is a Codenames board word, and the other endpoint is either another board word or a word in the common vocabulary

Each edge also stores a **normalized weight** computed per relation type using a qualitatively-motivated alpha smoothing factor: `normalized_weight = weight + alpha * (1 - weight)`. Relations like IsA and AtLocation use alpha=0 (no smoothing — their raw ConceptNet weights are already reliable). Compositional relations like UsedFor and MadeOf use alpha=0.8. This encodes the insight that lower-confidence edges in certain relation types are more recoverable than in others.

The result is a compact, game-focused knowledge graph stored as a JSON dictionary: `{ "word": [ {start, relation, end, weight, normalized_weight}, ... ] }`.

---

## Running the Game

### Interactive two-player game (recommended)

From the project root:

```bash
python "game play/codenames_game.py"
```

Two humans share the terminal, taking turns as Red and Blue operatives. Both Spymasters are AI. The board assignments are hidden — play fully blind just like real Codenames. At the end of each game, each named player receives a full decision tree showing every clue given, which relations connected it to the targets, and how weights shifted after each turn. A per-player relation profile is also printed summarising which relation types that player responds well to versus struggles with.

### AI demo mode (no human input)

```bash
python main.py
```

Runs a fully automated game: Red team is AI-driven, Blue team randomly reveals one of its words each turn. Useful for testing the reasoning engine without playing.

---

## How It Works

### Game flow

1. A random 25-word board is generated: 9 Red words, 8 Blue words, 7 Neutral words, 1 Assassin
2. Red AI Spymaster analyses the graph and announces a clue + count
3. Red human operative guesses words on the board
4. Blue AI Spymaster does the same for Blue's words
5. Blue human operative guesses
6. Teams alternate until one team finds all their words, or someone hits the Assassin

### How the AI Spymaster finds a clue

The Spymaster searches the knowledge graph for candidate clue words that connect to as many of the team's target words as possible, then scores each candidate using a penalty-bonus model:

1. For each target word, traverse all outgoing edges in the graph and accumulate a positive score for each candidate clue: `score += normalized_weight * relation_weight`
2. Apply **soft penalties** for bad word proximity — rather than hard-rejecting any clue that touches a bad word, each connection to an opponent word, neutral word, or the assassin subtracts from the candidate's score proportionally to connection strength. Only assassin connections above a strength threshold of 0.7 are a hard ban.
3. Apply a **coverage bonus**: a clue covering N target words is multiplied by `2^(N-1)`, so a clue covering 3 words scores 4x higher than the same clue covering 1 word. This strongly incentivises multi-word clues and prevents games from devolving into one-word-per-turn play.
4. If strict safety produces no candidates, the engine retries with neutral word penalties relaxed, ensuring a clue is always returned when the graph has any coverage at all.

### Adaptive learning

After each operative turn, the Spymaster updates the player's relation weight profile stored in `profiles/<name>.json`:

- If a word was guessed correctly, the relation that connected the clue to that word is **rewarded**: `weight = min(default_cap, weight * 1.05)`
- If a word was missed, the relation **decays**: `weight = max(0.05, weight * 0.95)`

A floor of 0.05 prevents any relation from becoming permanently unusable. A cap at the default weight prevents rewards from inflating weights above their intended design values. Future clues use these updated weights, so the Spymaster genuinely adapts to each player's style over multiple games.

### End-of-game analytics

At the end of each game, two per-player summaries are printed:

**Decision tree** — a full turn-by-turn trace showing every clue given, which relation types were used and how many times, each target word with hit/miss outcome, and the relation weight before and after the update. A pivot marker is shown when the Spymaster abandoned a relation type it had used the previous turn — but only when that previous turn included at least one miss, distinguishing forced adaptation from natural progression.

**Relation profile** — hit rate per relation type accumulated across all turns of the game, bucketed into strong (>=60%), moderate (40-60%), and weak (<40%) categories, with a concrete example clue for each.

---

## Knowledge Representations Used

### 1. Semantic Network (ConceptNet Graph)

The core representation is a **directed weighted semantic network** — a graph where nodes are concepts (words/phrases) and edges are typed semantic relations with numeric weights. This is a classical KR formalism that makes implicit world knowledge explicit and machine-queryable.

Example edges in the graph:
```
SHARK  --[IsA]-->        fish         (weight: 4.0, normalized_weight: 4.0)
SHARK  --[AtLocation]--> ocean        (weight: 2.3, normalized_weight: 2.3)
SHARK  --[CapableOf]-->  bite         (weight: 1.8, normalized_weight: 2.44)
OCEAN  --[IsA]-->        body of water
```

This lets the engine ask: *"what concept connects SHARK and WHALE and WAVE?"* by finding shared neighbors across all three.

### 2. Weighted Relation Schema

Not all relationships are equally useful for Codenames. The engine applies a **relation weighting schema** defined in `utils/constants.py` that encodes domain knowledge about what makes a good clue:

| Relation        | Weight | Rationale |
|---|---|---|
| IsA             | 0.5    | Definitional — reliable but often broad |
| AtLocation      | 0.5    | Physical context — reliable and guessable |
| PartOf          | 1.0    | Compositional — unambiguous |
| UsedFor         | 1.0    | Functional purpose — clear and specific |
| HasA            | 1.0    | Ownership/composition — concrete and guessable |
| SimilarTo       | 1.0    | Similarity — moderately reliable |
| CapableOf       | 1.0    | Ability — moderately guessable |
| Causes          | 1.0    | Causal — often intuitive |
| MadeOf          | 1.0    | Material — concrete and guessable |
| HasSubevent     | 1.0    | Event structure — moderately guessable |
| CreatedBy       | 1.0    | Origin — often well-known |
| LocatedNear     | 1.0    | Proximity — moderately guessable |
| HasProperty     | 0.75   | Descriptive — moderately guessable |
| Antonym         | 0.75   | Opposite — can be useful in context |
| ReceivesAction  | 0.75   | Passive role — moderately reliable |
| HasPrerequisite | 0.75   | Dependency — moderately guessable |
| DistinctFrom    | 1.0    | Distinction — contextually useful |

These weights are the starting point. Per-player profiles modify them over time through the reward and decay mechanism.

### 3. Per-Player Weight Profiles

Each named player's current relation weights are persisted as a JSON file in `profiles/`. This is a **learned representation** of an individual's cognitive style — which semantic relation types they process reliably when guessing Codenames clues. The profile is loaded at the start of each game and saved after each turn.

### 4. Clue Validity Constraints

The engine encodes a set of **hard constraints** for clue validity as a rule system:

- A clue must be longer than 2 characters
- A clue must not be a substring of any target word (and vice versa) — prevents identity clues like giving "CAT" for "CATFISH"
- A clue must not be a stop word (a, the, of, etc.)
- A clue must not be too similar to a previously used clue in the same game — prevents repetition

These are logical constraints applied as filters before any scoring takes place.

---

## Reasoning Methods

### Graph Traversal and Coverage Reasoning

The primary reasoning method is **graph traversal with coverage scoring**: for each candidate clue word, the engine computes how many of the team's target words it connects to under the allowed relations, and accumulates a positive score from those connections. This is a form of abductive reasoning — finding the best explanation (clue) that accounts for multiple observations (target words).

Formally, for a candidate clue C and targets T1...Tn:
```
coverage(C) = { Ti | edge(Ti, r, C) exists for some relation r }
pos_score(C) = sum of (normalized_weight * relation_weight) for each covered target
```

### Safety Reasoning (Soft Penalty Model)

Rather than hard-rejecting any clue that touches a bad word, the engine applies **graded penalties** based on category and connection strength:

```
penalty(C) = sum over assassin neighbors:  connection_strength * 10.0
           + sum over opponent neighbors:  connection_strength * 4.0
           + sum over neutral neighbors:   connection_strength * 0.5  (only if strength > 0.6)

net_score(C) = pos_score(C) - penalty(C)
```

Only assassin connections with normalized_weight above 0.7 trigger a hard ban. Everything else is a cost the engine weighs against the benefit, allowing clues that weakly touch neutral words when the positive signal is strong enough.

### Coverage Bonus (Utility-Based Decision Making)

Candidate clues are ranked after applying a coverage multiplier:

```
final_score(C) = net_score(C) * (2 ^ (|coverage(C)| - 1))
```

This exponential bonus means a clue covering 3 words will almost always outrank a clue covering 1 word, even if the per-word scores are lower. Clues are first sorted by coverage count, then by final score within the same count.

### Adaptive Inductive Reasoning

The weight update system is a form of **inductive reasoning**: from specific observations (this player guessed SHARK correctly via IsA, missed ICE via MadeOf) the system generalises a rule (this player is stronger on definitional relations than material ones) and updates its representation accordingly. This learned generalisation then governs future clue selection — closing the loop between representation, reasoning, and learning.

### Pivot Detection (Meta-Reasoning)

The end-of-game decision tree includes a **pivot detector**: a lightweight form of meta-reasoning where the system reflects on its own strategy history. A pivot is flagged when the Spymaster switches dominant relation type between turns, but only when the previous turn included at least one miss, distinguishing a genuine strategic response to failure from coincidental variation.

---

## Why This Approach vs an LLM

A large language model like GPT-4 could also play Codenames — so why build a symbolic system instead?

### Transparency and Explainability

Every clue this system produces has a traceable reasoning chain: *"OCEAN connects SHARK, WHALE, and WAVE via AtLocation and IsA relationships with weights 2.3, 4.0, and 1.5."* An LLM produces a clue with no accessible explanation of why — it is a black box. For a course project demonstrating knowledge representation and reasoning, explainability is the whole point.

### Controllable Safety

The safety constraint — never give a clue that strongly connects to the assassin — is enforced by an explicit symbolic penalty model with a hard threshold. An LLM cannot make this guarantee; it may occasionally produce unsafe associations that it cannot detect.

### Structured World Knowledge

ConceptNet encodes structured commonsense knowledge that LLMs only have implicitly in their weights. The relation types (IsA, UsedFor, AtLocation) give the system a vocabulary for *kinds* of connections, enabling the relation weighting scheme and per-player profiling. An LLM has no mechanism to say "I prefer definitional connections over vague ones" — or to learn that a specific player does.

### No Hallucination

The system only produces clues that exist in the graph with documented connections. It cannot invent a relationship that does not exist in ConceptNet. LLMs hallucinate — they may confidently produce a clue based on a spurious or false association.

### Limitations

To be fair, this approach also has weaknesses compared to an LLM:

- **Coverage**: ConceptNet does not contain every word or relationship. An LLM has seen vastly more text and may find connections this system misses.
- **Cultural and contextual knowledge**: Puns, pop culture references, and context-dependent meanings are poorly captured in a formal graph.
- **Clue intuitiveness**: The system may generate clues targeting words the human operative does not find intuitive, even if the graph connection is technically correct.

The ideal system would combine both: use symbolic reasoning for safety guarantees and explainability, and use neural embeddings or an LLM as a fallback for coverage.

---

## Logs

Every game played through `game play/codenames_game.py` is automatically saved to `game play/logs/`. Logs are structured JSON and never overwritten.

- **`game_TIMESTAMP.json`** — full record of one game: board layout, every spymaster clue with score, targets, and relations used, every player guess and outcome, winner
- **`all_games_history.jsonl`** — one JSON object per line, one per game, accumulates across all sessions