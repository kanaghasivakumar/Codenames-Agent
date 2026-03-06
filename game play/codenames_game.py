"""
Codenames V2 — Two Players vs AI Spymasters
============================================
Uses the V2 ConceptNet graph reasoning engine (no GloVe vectors).
Both spymasters are AI. Two humans alternate as operatives.
Board assignments are hidden — play fully blind.

Run from the project root:
    python "game play/codenames_game.py"
"""

import os
import sys
import random
import itertools
import datetime
import json
from pathlib import Path


# ── Path setup ────────────────────────────────────────────────────────────────
GAMEPLAY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(GAMEPLAY_DIR)

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))

from src.reasoning_engine import ReasoningEngine
from src.user_profile import UserProfile

# ── Config ────────────────────────────────────────────────────────────────────
WORD_LIST_PATH = os.path.join(PROJECT_DIR, 'data', 'codenames_words.txt')
LOG_DIR        = os.path.join(GAMEPLAY_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL COLOURS
# ══════════════════════════════════════════════════════════════════════════════

def _c(code, t): return f'\033[{code}m{t}\033[0m'
def RED(t):      return _c('91;1', t)
def BLUE(t):     return _c('94;1', t)
def GRAY(t):     return _c('90',   t)
def YELLOW(t):   return _c('93',   t)
def GREEN(t):    return _c('92;1', t)
def BOLD(t):     return _c('1',    t)
def STRIKE(t):   return _c('9',    t)
def DIM(t):      return _c('2',    t)

def team_color(team, text):
    return RED(text) if team == 'RED' else BLUE(text)

def divider(char='─', w=72):
    print(DIM(char * w))


# ══════════════════════════════════════════════════════════════════════════════
# BOARD DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def print_board(board, red_words, blue_words, assassin, revealed, reveal_all=False):
    """5x5 board. Unrevealed words are plain. Revealed are struck-through + coloured.
    reveal_all=True shows full team assignments for every word (end-of-game reveal)."""
    W = 11  # fits longest Codenames words; border chars kept plain to avoid ANSI width bugs

    def BLACK(t): return f'\033[30;1m{t}\033[0m'

    def cell(word):
        w = word.upper()
        is_assassin = (w == assassin.upper())
        is_red      = (w in red_words)
        is_blue     = (w in blue_words)
        is_revealed = (w in revealed)

        if reveal_all:
            if is_assassin:
                base = STRIKE(f'{w:^{W}}') if is_revealed else f'{w:^{W}}'
                return BLACK(base)
            if is_red:
                base = STRIKE(f'{w:^{W}}') if is_revealed else BOLD(f'{w:^{W}}')
                return RED(base)
            if is_blue:
                base = STRIKE(f'{w:^{W}}') if is_revealed else BOLD(f'{w:^{W}}')
                return BLUE(base)
            base = STRIKE(f'{w:^{W}}') if is_revealed else f'{w:^{W}}'
            return GRAY(base)

        # Normal gameplay view
        if w not in revealed:
            return BOLD(f'{w:^{W}}')
        if is_assassin:
            return RED(STRIKE(f'{w:^{W}}'))
        if is_red:
            return RED(STRIKE(f'{w:^{W}}'))
        if is_blue:
            return BLUE(STRIKE(f'{w:^{W}}'))
        return GRAY(STRIKE(f'{w:^{W}}'))

    # Border chars are NOT wrapped in BOLD — ANSI codes around box-drawing
    # chars cause Windows Terminal to miscalculate cursor position, breaking
    # the right-side border. Plain chars render correctly on all terminals.
    border = '─' * (W * 5 + 4)
    print()
    print('┌' + border + '┐')
    for row in range(5):
        cells = [cell(board[row * 5 + col]) for col in range(5)]
        print('│' + ' '.join(cells) + '│')
    print('└' + border + '┘')
    print()


def print_scores(red_rem, blue_rem):
    bar_r = RED ('█' * red_rem  + '░' * (9 - red_rem))
    bar_b = BLUE('█' * blue_rem + '░' * (8 - blue_rem))
    print(f'  {RED("RED")}  {bar_r} {BOLD(str(red_rem))} left   '
          f'{BLUE("BLUE")} {bar_b} {BOLD(str(blue_rem))} left')
    print()


# ══════════════════════════════════════════════════════════════════════════════
# JSON ENCODER — handles any non-serialisable numerics
# ══════════════════════════════════════════════════════════════════════════════

class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        try:    return float(obj)
        except: pass
        try:    return int(obj)
        except: pass
        return super().default(obj)


# ══════════════════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class GameLogger:
    def __init__(self, gid):
        self.gid       = gid
        self.log_file  = os.path.join(LOG_DIR, f'game_{gid}.json')
        self.hist_file = os.path.join(LOG_DIR, 'all_games_history.jsonl')
        self.events    = []
        self.meta      = {}

    def set_board(self, red, blue, assassin, neutral):
        self.meta = {
            'game_id': self.gid,
            'started': datetime.datetime.now().isoformat(),
            'board':   {'red': list(red), 'blue': list(blue),
                        'assassin': assassin, 'neutral': list(neutral)}
        }
        self._ev('BOARD_SETUP', self.meta['board'])

    def clue(self, team, clue, count, targets, score):
        self._ev('SPYMASTER_CLUE', {
            'team': team, 'clue': clue, 'count': count,
            'targets': targets, 'score': round(float(score), 4)
        })

    def guess(self, team, word, result):
        self._ev('PLAYER_GUESS', {'team': team, 'word': word, 'result': result})

    def turn_end(self, team, reason):
        self._ev('TURN_END', {'team': team, 'reason': reason})

    def game_over(self, winner, reason):
        self._ev('GAME_OVER', {'winner': winner, 'reason': reason})
        record = {
            **self.meta,
            'ended':  datetime.datetime.now().isoformat(),
            'winner': winner,
            'reason': reason,
            'events': self.events
        }
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, cls=SafeEncoder)
        with open(self.hist_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, cls=SafeEncoder) + '\n')
        print(f'\n{GREEN("✓")} Log saved → {self.log_file}')

    def _ev(self, kind, data):
        self.events.append({
            'time':  datetime.datetime.now().isoformat(),
            'event': kind,
            'data':  data
        })


# ══════════════════════════════════════════════════════════════════════════════
# AI SPYMASTER
# ══════════════════════════════════════════════════════════════════════════════

class AISpymaster:
    """
    Wraps ReasoningEngine for one team.
    Tries to find a clue covering 3, then 2, then 1 target word.
    Tracks used clues to avoid repetition.
    """

    def __init__(self, engine: ReasoningEngine, profile=None):
        self.engine     = engine
        self.used_clues = []
        self.profile = profile

    def get_clue(self, team_words, opponent_words, assassin_word, neutral_words):
        """
        team_words : list of remaining words for this team
        bad_words  : list of opponent words + assassin + neutral
        Returns dict with keys: clue, count, targets, score — or None
        """
        for size in [3, 2, 1]:
            combos = list(itertools.combinations(team_words, size))
            random.shuffle(combos)
            for subset in combos[:40]:
                results = self.engine.find_clues(
                    list(subset), opponent_words, assassin_word, neutral_words,
                    used_clues=self.used_clues, top_n=1
                )
                if results:
                    best = results[0]
                    self.used_clues.append(best['clue'])
                    return {
                        'clue':    best['clue'].upper(),
                        'count':   len(best['targets']),
                        'targets': [t.upper() for t in best['targets']],
                        'score':   best['score'],
                        'logic':   best.get('logic', [])
                    }
        return None


# ══════════════════════════════════════════════════════════════════════════════
# GAME ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class CodenamesGame:

    def __init__(self):
        print()
        divider('═')
        print(BOLD('  Initialising V2 Codenames — loading knowledge graph...'))
        divider('═')

        # Reasoning engines/spymasters will be created per-game once player names
        # are entered at the start of each play() invocation.
        self.red_spy = None
        self.blue_spy = None

        # Load word list
        with open(WORD_LIST_PATH, 'r') as f:
            self.all_words = [w.strip().upper() for w in f if w.strip()]

        print(f'  {GREEN("✓")} Ready — {len(self.all_words)} words loaded.\n')

    # ── Board ─────────────────────────────────────────────────────────────────

    def _new_board(self):
        deck = list(self.all_words)
        random.shuffle(deck)
        board = deck[:25]
        red     = set(board[:9])
        blue    = set(board[9:17])
        assassin = board[17]
        neutral  = set(board[18:])
        shuffled = board[:]
        random.shuffle(shuffled)
        return red, blue, assassin, neutral, shuffled

    def _rem(self, words, revealed):
        return [w for w in words if w not in revealed]

    # ── Spymaster turn ────────────────────────────────────────────────────────

    def _spymaster_turn(self, team, team_words, opp_words,
                        assassin, neutral, revealed, logger):
        spy       = self.red_spy if team == 'RED' else self.blue_spy
        remaining = self._rem(list(team_words), revealed)
        opp       = self._rem(list(opp_words), revealed)
        ass       =  assassin
        neut   = self._rem(list(neutral), revealed)

        print()
        divider('═')
        print(team_color(team, f'  🕵  {team} SPYMASTER is thinking...'))
        divider('═')

        if spy.profile:
            raw_weights = spy.profile.give_weights()
            # Clamp each weight to a minimum of 0.05 so over-decayed profiles
            # don't silently return None and starve the engine of all candidates
            from utils.constants import DEFAULT_RELATION_WEIGHTS
            clamped = {
                k: max(0.05, float(v)) if v is not None else DEFAULT_RELATION_WEIGHTS.get(k, 1.0)
                for k, v in raw_weights.items()
            }
            spy.engine.update_relation_weights(clamped)

        result = spy.get_clue(remaining, opp, ass, neut)

        if result is None:
            print(team_color(team, f'  {team} Spymaster has no clue — passing.'))
            logger.clue(team, 'PASS', 0, [], 0.0)
            return 'PASS', 0, None

        clue  = result['clue']
        count = result['count']
        logic = result.get('logic', [])

        print()
        divider()
        print(team_color(team, BOLD(f'  CLUE: "{clue}"    COUNT: {count}')))
        divider()
        print()

        logger.clue(team, clue, count, result['targets'], result['score'])
        return clue, count, logic

    # ── Operative turn ────────────────────────────────────────────────────────

    def _operative_turn(self, team, clue, count,
                        team_words, opp_words, assassin, neutral,
                        board, revealed, logger, red_w, blue_w, bonus=False):
        opp   = 'BLUE' if team == 'RED' else 'RED'
        max_g = count + 1 if bonus else count
        guesses = 0
        guessed_words = []

        print()
        print(team_color(team, BOLD(f'  ══  {team} OPERATIVE — YOUR TURN  ══')))
        print(f'  Clue: {BOLD(clue)}   Count: {BOLD(str(count))}   '
              f'(up to {BOLD(str(max_g))} guesses)')
        print(f'  Type {BOLD("PASS")} to end your turn early.')

        while True:
            # Win check
            if not self._rem(list(team_words), revealed):
                return 'win', guessed_words

            # Exhausted guesses
            if guesses >= max_g:
                logger.turn_end(team, 'max guesses reached')
                return 'continue', guessed_words

            # Show board once per guess opportunity
            print_board(board, red_w, blue_w, assassin, revealed)
            print_scores(
                len(self._rem(list(red_w),  revealed)),
                len(self._rem(list(blue_w), revealed))
            )

            raw = input(team_color(team, f'  [{team}] Guess: ')).strip().upper()

            # PASS
            if raw == 'PASS':
                logger.turn_end(team, 'player passed')
                print(f'  {YELLOW("⏭")}  Turn passed.')
                return 'continue', guessed_words

            # Validation — loop back without consuming a guess
            if raw not in board:
                print(f'  {YELLOW("?")}  "{raw}" is not on the board. Try again.')
                continue
            if raw in revealed:
                print(f'  {YELLOW("?")}  "{raw}" is already revealed. Try again.')
                continue

            # Valid guess — reveal and count
            revealed.add(raw)
            guesses += 1
            guessed_words.append(raw)

            if raw == assassin:
                print()
                print(RED(BOLD(f'  ☠   ASSASSIN HIT: {raw}!')))
                print(RED(BOLD(f'      {team} TEAM LOSES!')))
                logger.guess(team, raw, 'assassin')
                logger.turn_end(team, 'hit assassin')
                return 'assassin', guessed_words

            elif raw in team_words:
                print(f'  {GREEN("✓")}  {BOLD(raw)} — {team_color(team, "YOUR WORD!")}')
                logger.guess(team, raw, 'correct')
                # Immediate win check
                if not self._rem(list(team_words), revealed):
                    return 'win', guessed_words
                # More guesses available?
                if guesses < max_g:
                    again = input(f'  Keep guessing? ({max_g - guesses} left) [Y/n]: ').strip().lower()
                    if again == 'n':
                        logger.turn_end(team, 'player stopped')
                        return 'continue', guessed_words
                    # answered Y — loop back to show board and take next guess
                else:
                    print(f'  {YELLOW("⏭")}  Max guesses used. Turn ends.')
                    logger.turn_end(team, 'max guesses reached')
                    return 'continue', guessed_words

            elif raw in opp_words:
                print(f'  {YELLOW("✗")}  {BOLD(raw)} — {team_color(opp, "OPPONENT")} word! Turn ends.')
                logger.guess(team, raw, 'opponent')
                logger.turn_end(team, 'hit opponent word')
                return 'continue', guessed_words

            else:
                print(f'  {YELLOW("~")}  {BOLD(raw)} — {GRAY("NEUTRAL")}. Turn ends.')
                logger.guess(team, raw, 'neutral')
                logger.turn_end(team, 'hit neutral word')
                return 'continue', guessed_words

    # ── Main game loop ────────────────────────────────────────────────────────

    def play(self):
        gid    = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        logger = GameLogger(gid)

        # Title
        print()
        print(BOLD('╔' + '═' * 68 + '╗'))
        print(BOLD('║') + BOLD(f'{"  CODENAMES V2  —  Two Players vs AI Spymasters":^68}') + BOLD('║'))
        print(BOLD('║') + DIM(f'{"  Powered by ConceptNet Knowledge Graph":^68}') + BOLD('║'))
        print(BOLD('╚' + '═' * 68 + '╝'))
        print()

        # Prompt for player names at the start of each game
        red_player = input('  Enter RED player name (leave blank for DEFAULT): ').strip()
        if not red_player:
            red_player = None
            print()
        blue_player = input('  Enter BLUE player name (leave blank for DEFAULT): ').strip()
        if not blue_player:
            blue_player = None

        # Load profiles
        red_profile = UserProfile(red_player) if red_player else None
        blue_profile = UserProfile(blue_player) if blue_player else None

        # Create ReasoningEngine instances with profile names and spymasters
        engine_red = ReasoningEngine(relation_weights=red_profile.give_weights() if red_profile else None)
        engine_blue = ReasoningEngine(relation_weights=blue_profile.give_weights() if blue_profile else None)

        self.red_spy = AISpymaster(engine_red, profile=red_profile)
        self.blue_spy = AISpymaster(engine_blue, profile=blue_profile)

        input('  Press ENTER to generate a new board...')

        # Board
        red_w, blue_w, assassin, neutral, board = self._new_board()
        revealed = set()
        logger.set_board(red_w, blue_w, assassin, neutral)

        # Reset used clues for fresh game
        self.red_spy.used_clues  = []
        self.blue_spy.used_clues = []

        input('  Press ENTER to begin...')

        turn              = 'RED'
        winner            = None
        turn_count        = {'RED': 0, 'BLUE': 0}
        consecutive_passes = 0   # break infinite loop if both teams pass repeatedly

        # Per-game relation hit/miss accumulator for end-of-game summary
        # Structure: { 'RED': { 'RelationName': {'hits': int, 'misses': int, 'examples': [...]} } }
        relation_stats = {
            'RED':  {},
            'BLUE': {}
        }
        # Decision tree: one entry per turn per player
        # Each entry: { turn_num, clue, relations_used, words: [{word, relation, result}],
        #               weights_before, weights_after, pivoted }
        decision_tree = {'RED': [], 'BLUE': []}

        while True:
            opp = 'BLUE' if turn == 'RED' else 'RED'
            t_w = red_w  if turn == 'RED' else blue_w
            o_w = blue_w if turn == 'RED' else red_w

            # Win check at start of turn
            if not self._rem(list(t_w), revealed):
                winner = turn
                logger.game_over(winner, f'{winner} found all words')
                break

            # Spymaster
            # Snapshot weights before this turn for the decision tree
            _active_profile = red_profile if turn == 'RED' else blue_profile
            _weights_before = dict(_active_profile.give_weights()) if _active_profile else {}

            clue, count, logic = self._spymaster_turn(
                turn, t_w, o_w, assassin, neutral, revealed, logger
            )
            if clue == 'PASS':
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    # Both teams have passed — no valid clues exist, call it a draw
                    winner = 'NONE'
                    logger.game_over('NONE', 'both teams passed — no valid clues')
                    print()
                    print(YELLOW(BOLD('  ⚠   Both spymasters are out of clues — game ends in a draw.')))
                    break
                turn = opp
                continue
            consecutive_passes = 0  # reset on any successful clue

            # print(logic)

            input(f'  Press ENTER to guess as {team_color(turn, turn)} operative...')

            # Operative
            turn_count[turn] += 1
            bonus = turn_count[turn] > 1

            outcome, guessed_words = self._operative_turn(
                turn, clue, count,
                t_w, o_w, assassin, neutral,
                board, revealed, logger, red_w, blue_w, bonus=bonus
            )

            # Persist spymaster logic and guessed words to the player's profile,
            # then call update_weights() (implemented by user) between rounds.
            profile = red_profile if turn == 'RED' else blue_profile
            if profile is not None:
                profile.get_target_relations(logic)
                profile.get_guessed_words(guessed_words)
                profile.update_weights()

            # Record decision tree node for this turn
            if logic and clue != 'PASS':
                _weights_after  = dict(profile.give_weights()) if profile else {}
                guessed_set_dt  = {w.strip().upper() for w in guessed_words}

                # Which relation types did the AI use this turn?
                rels_used = {}
                word_results = []
                for entry in logic:
                    if not isinstance(entry, str) or '(' not in entry:
                        continue
                    i = entry.rfind('(')
                    w   = entry[:i].strip().upper()
                    rel = entry[i+1:-1].strip()
                    if not rel or not w:
                        continue
                    rels_used[rel] = rels_used.get(rel, 0) + 1
                    word_results.append({
                        'word':     w,
                        'relation': rel,
                        'result':   'hit' if w in guessed_set_dt else 'miss'
                    })

                # Mark as pivoted only when:
                # 1. The previous turn had at least one miss (something went wrong)
                # 2. The previous turn's dominant relation is completely absent now
                prev_turns = decision_tree[turn]
                pivoted = False
                if prev_turns:
                    prev_node = prev_turns[-1]
                    prev_rels = prev_node.get('relations_used', {})
                    prev_top  = max(prev_rels, key=prev_rels.get) if prev_rels else None
                    prev_had_miss = any(
                        w['result'] == 'miss' for w in prev_node.get('words', [])
                    )
                    pivoted = (
                        prev_had_miss and
                        prev_top is not None and
                        prev_top not in rels_used
                    )

                decision_tree[turn].append({
                    'turn_num':       turn_count[turn],
                    'clue':           clue,
                    'relations_used': rels_used,
                    'words':          word_results,
                    'weights_before': _weights_before,
                    'weights_after':  _weights_after,
                    'pivoted':        pivoted
                })

            # Accumulate relation hit/miss stats for end-of-game summary
            if logic:
                guessed_set = {w.strip().upper() for w in guessed_words}
                stats = relation_stats[turn]
                for entry in logic:
                    if not isinstance(entry, str) or '(' not in entry:
                        continue
                    i = entry.rfind('(')
                    word = entry[:i].strip().upper()
                    rel  = entry[i+1:-1].strip()
                    if not rel or not word:
                        continue
                    if rel not in stats:
                        stats[rel] = {'hits': 0, 'misses': 0, 'examples': []}
                    if word in guessed_set:
                        stats[rel]['hits'] += 1
                        if len(stats[rel]['examples']) < 3:
                            stats[rel]['examples'].append(
                                {'word': word, 'clue': clue, 'result': 'hit'}
                            )
                    else:
                        stats[rel]['misses'] += 1
                        if len(stats[rel]['examples']) < 3:
                            stats[rel]['examples'].append(
                                {'word': word, 'clue': clue, 'result': 'miss'}
                            )

            if outcome == 'assassin':
                winner = opp
                logger.game_over(winner, f'{turn} hit the assassin')
                break
            elif outcome == 'win':
                winner = turn
                logger.game_over(winner, f'{winner} found all words')
                break
            elif not self._rem(list(t_w), revealed):
                winner = turn
                logger.game_over(winner, f'{winner} found all words')
                break

            turn = opp

        # Final board reveal
        print_board(board, red_w, blue_w, assassin, revealed, reveal_all=True)
        print()
        print(BOLD('=' * 72))
        if winner and winner != 'NONE':
            print(team_color(winner, BOLD(f'  🏆   {winner} TEAM WINS!')))
        print(BOLD('=' * 72))
        print()

        # Print decision trees
        if red_player and decision_tree['RED']:
            print_decision_tree(red_player, decision_tree['RED'], 'RED')
        if blue_player and decision_tree['BLUE']:
            print_decision_tree(blue_player, decision_tree['BLUE'], 'BLUE')

        # Print end-of-game player summaries
        if red_player and relation_stats['RED']:
            print_player_summary(red_player, relation_stats['RED'], 'RED')
        if blue_player and relation_stats['BLUE']:
            print_player_summary(blue_player, relation_stats['BLUE'], 'BLUE')

        # Update and save player profiles at end of game
        try:
            if red_profile is not None:
                # increment games played and persist updated weights
                red_profile.increment_games_played()
                red_profile.save_profile_to_json()
        except Exception:
            pass

        try:
            if blue_profile is not None:
                blue_profile.increment_games_played()
                blue_profile.save_profile_to_json()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# DECISION TREE
# ══════════════════════════════════════════════════════════════════════════════

def print_decision_tree(player_name, turns, team):
    """Print a full per-player decision tree for the game."""
    color = RED if team == 'RED' else BLUE

    print()
    divider('═')
    print(color(BOLD(f'  🌳  {player_name.upper()} — DECISION TREE')))
    divider('═')

    if not turns:
        print(DIM('  No turns recorded.'))
        divider()
        return

    for node in turns:
        turn_num   = node['turn_num']
        clue       = node['clue']
        rels_used  = node['relations_used']
        words      = node['words']
        w_before   = node['weights_before']
        w_after    = node['weights_after']
        pivoted    = node['pivoted']

        # ── Turn header ──────────────────────────────────────────────────
        pivot_tag = f'  {YELLOW("↩  pivoted relation")}' if pivoted else ''
        rels_str  = ', '.join(f'{r}×{c}' for r, c in rels_used.items())
        print()
        print(color(BOLD(f'  TURN {turn_num}')) +
              f'  clue {BOLD(clue)}  ({rels_str}){pivot_tag}')

        # ── Word results as tree branches ────────────────────────────────
        for idx, wr in enumerate(words):
            is_last  = (idx == len(words) - 1)
            branch   = '└──' if is_last else '├──'
            rel      = wr['relation']
            word     = wr['word']
            result   = wr['result']

            hit_icon = GREEN('●') if result == 'hit' else YELLOW('○')
            outcome  = GREEN('guessed ✓') if result == 'hit' else YELLOW('missed  ✗')

            # Weight delta for this relation
            wb = w_before.get(rel)
            wa = w_after.get(rel)
            if wb is not None and wa is not None:
                wb_f, wa_f = float(wb), float(wa)
                if wa_f < wb_f:
                    delta = RED(f'[{rel}  {wb_f:.3f} → {wa_f:.3f} ↓ decayed]')
                else:
                    delta = GREEN(f'[{rel}  {wb_f:.3f} → {wa_f:.3f} — unchanged]')
            else:
                delta = DIM(f'[{rel}]')

            print(f'  {DIM(branch)} {hit_icon} {BOLD(word):20s} {outcome}   {delta}')

    print()
    divider()


# ══════════════════════════════════════════════════════════════════════════════
# END-OF-GAME SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_player_summary(player_name, stats, team):
    """Print a readable per-player relation summary after the game."""
    color = RED if team == 'RED' else BLUE

    print()
    divider('═')
    print(color(BOLD(f'  📊  {player_name.upper()} — YOUR CLUE PROFILE')))
    divider('═')

    if not stats:
        print(DIM('  Not enough data this game to build a profile.'))
        divider()
        return

    # Sort relations by total appearances
    sorted_rels = sorted(stats.items(), key=lambda x: x[1]['hits'] + x[1]['misses'], reverse=True)

    strong   = []  # hit rate >= 0.6
    weak     = []  # hit rate < 0.4
    moderate = []  # in between

    for rel, data in sorted_rels:
        total = data['hits'] + data['misses']
        if total == 0:
            continue
        rate = data['hits'] / total
        entry = (rel, data, rate, total)
        if rate >= 0.6:
            strong.append(entry)
        elif rate < 0.4:
            weak.append(entry)
        else:
            moderate.append(entry)

    if strong:
        print(f'  {GREEN("✓")} {BOLD("You respond well to:")}')
        for rel, data, rate, total in strong[:3]:
            pct = int(rate * 100)
            print(f'    {GREEN("█")} {BOLD(rel):20s}  {pct}% hit rate  ({data["hits"]}/{total} words guessed)')
            # Show best example
            hits = [e for e in data['examples'] if e['result'] == 'hit']
            if hits:
                ex = hits[0]
                print(DIM(f'       e.g. clue "{ex["clue"]}" → you found {ex["word"]}'))
        print()

    if weak:
        print(f'  {YELLOW("✗")} {BOLD("You struggle with:")}')
        for rel, data, rate, total in weak[:3]:
            pct = int(rate * 100)
            print(f'    {YELLOW("░")} {BOLD(rel):20s}  {pct}% hit rate  ({data["hits"]}/{total} words guessed)')
            # Show a miss example
            misses = [e for e in data['examples'] if e['result'] == 'miss']
            if misses:
                ex = misses[0]
                print(DIM(f'       e.g. clue "{ex["clue"]}" → missed {ex["word"]}'))
        print()

    if moderate:
        print(f'  {DIM("~")} {BOLD("Mixed results with:")}')
        for rel, data, rate, total in moderate[:2]:
            pct = int(rate * 100)
            print(f'    {DIM("▒")} {BOLD(rel):20s}  {pct}% hit rate  ({data["hits"]}/{total} words guessed)')
        print()

    divider()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    game = CodenamesGame()

    while True:
        game.play()
        again = input('  Play again? [y/N]: ').strip().lower()
        if again != 'y':
            print('\n  Thanks for playing!\n')
            break