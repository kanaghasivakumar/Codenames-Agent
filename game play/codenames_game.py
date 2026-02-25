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

# ── Path setup ────────────────────────────────────────────────────────────────
GAMEPLAY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(GAMEPLAY_DIR)

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))

from src.reasoning_engine import ReasoningEngine

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

def print_board(board, red_words, blue_words, assassin, revealed):
    """5×5 board. Unrevealed words are plain. Revealed are struck-through + coloured."""
    W = 11  # fits longest Codenames words; border chars kept plain to avoid ANSI width bugs

    def cell(word):
        w = word.upper()
        if w not in revealed:
            return BOLD(f'{w:^{W}}')
        if w == assassin.upper():
            return RED(STRIKE(f'{w:^{W}}'))
        if w in red_words:
            return RED(STRIKE(f'{w:^{W}}'))
        if w in blue_words:
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

    def __init__(self, engine: ReasoningEngine):
        self.engine     = engine
        self.used_clues = []

    def get_clue(self, team_words, bad_words):
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
                    list(subset), bad_words,
                    used_clues=self.used_clues, top_n=1
                )
                if results:
                    best = results[0]
                    self.used_clues.append(best['clue'])
                    return {
                        'clue':    best['clue'].upper(),
                        'count':   len(best['targets']),
                        'targets': [t.upper() for t in best['targets']],
                        'score':   best['score']
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

        engine = ReasoningEngine()

        # Two independent spymasters — separate used_clues history
        self.red_spy  = AISpymaster(engine)
        self.blue_spy = AISpymaster(engine)

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
        bad       = self._rem(list(opp_words), revealed) + [assassin] + self._rem(list(neutral), revealed)

        print()
        divider('═')
        print(team_color(team, f'  🕵  {team} SPYMASTER is thinking...'))
        divider('═')

        result = spy.get_clue(remaining, bad)

        if result is None:
            print(team_color(team, f'  {team} Spymaster has no clue — passing.'))
            logger.clue(team, 'PASS', 0, [], 0.0)
            return 'PASS', 0

        clue  = result['clue']
        count = result['count']

        print()
        divider()
        print(team_color(team, BOLD(f'  CLUE: "{clue}"    COUNT: {count}')))
        divider()
        print()

        logger.clue(team, clue, count, result['targets'], result['score'])
        return clue, count

    # ── Operative turn ────────────────────────────────────────────────────────

    def _operative_turn(self, team, clue, count,
                        team_words, opp_words, assassin, neutral,
                        board, revealed, logger, red_w, blue_w, bonus=False):
        opp   = 'BLUE' if team == 'RED' else 'RED'
        max_g = count + 1 if bonus else count
        guesses = 0

        print()
        print(team_color(team, BOLD(f'  ══  {team} OPERATIVE — YOUR TURN  ══')))
        print(f'  Clue: {BOLD(clue)}   Count: {BOLD(str(count))}   '
              f'(up to {BOLD(str(max_g))} guesses)')
        print(f'  Type {BOLD("PASS")} to end your turn early.')

        while True:
            # Win check
            if not self._rem(list(team_words), revealed):
                return 'win'

            # Exhausted guesses
            if guesses >= max_g:
                logger.turn_end(team, 'max guesses reached')
                return 'continue'

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
                return 'continue'

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

            if raw == assassin:
                print()
                print(RED(BOLD(f'  ☠   ASSASSIN HIT: {raw}!')))
                print(RED(BOLD(f'      {team} TEAM LOSES!')))
                logger.guess(team, raw, 'assassin')
                logger.turn_end(team, 'hit assassin')
                return 'assassin'

            elif raw in team_words:
                print(f'  {GREEN("✓")}  {BOLD(raw)} — {team_color(team, "YOUR WORD!")}')
                logger.guess(team, raw, 'correct')
                # Immediate win check
                if not self._rem(list(team_words), revealed):
                    return 'win'
                # More guesses available?
                if guesses < max_g:
                    again = input(f'  Keep guessing? ({max_g - guesses} left) [Y/n]: ').strip().lower()
                    if again == 'n':
                        logger.turn_end(team, 'player stopped')
                        return 'continue'
                    # answered Y — loop back to show board and take next guess
                else:
                    print(f'  {YELLOW("⏭")}  Max guesses used. Turn ends.')
                    logger.turn_end(team, 'max guesses reached')
                    return 'continue'

            elif raw in opp_words:
                print(f'  {YELLOW("✗")}  {BOLD(raw)} — {team_color(opp, "OPPONENT")} word! Turn ends.')
                logger.guess(team, raw, 'opponent')
                logger.turn_end(team, 'hit opponent word')
                return 'continue'

            else:
                print(f'  {YELLOW("~")}  {BOLD(raw)} — {GRAY("NEUTRAL")}. Turn ends.')
                logger.guess(team, raw, 'neutral')
                logger.turn_end(team, 'hit neutral word')
                return 'continue'

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
        input('  Press ENTER to generate a new board...')

        # Board
        red_w, blue_w, assassin, neutral, board = self._new_board()
        revealed = set()
        logger.set_board(red_w, blue_w, assassin, neutral)

        # Reset used clues for fresh game
        self.red_spy.used_clues  = []
        self.blue_spy.used_clues = []

        input('  Press ENTER to begin...')

        turn       = 'RED'
        winner     = None
        turn_count = {'RED': 0, 'BLUE': 0}

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
            clue, count = self._spymaster_turn(
                turn, t_w, o_w, assassin, neutral, revealed, logger
            )
            if clue == 'PASS':
                turn = opp
                continue

            input(f'  Press ENTER to guess as {team_color(turn, turn)} operative...')

            # Operative
            turn_count[turn] += 1
            bonus = turn_count[turn] > 1

            outcome = self._operative_turn(
                turn, clue, count,
                t_w, o_w, assassin, neutral,
                board, revealed, logger, red_w, blue_w, bonus=bonus
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
        print_board(board, red_w, blue_w, assassin, revealed)
        print()
        print(BOLD('=' * 72))
        print(team_color(winner, BOLD(f'  🏆   {winner} TEAM WINS!')))
        print(BOLD('=' * 72))
        print()


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