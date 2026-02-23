"""
Codenames Game — Human Operative vs AI Spymasters
==================================================
Run:  python "game play/codenames_game.py"
"""

import os
import sys
import random
import datetime
import json
import io
import contextlib

# ── Path setup ────────────────────────────────────────────────────────────────
GAMEPLAY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(GAMEPLAY_DIR)
AGENT_DIR    = os.path.join(PROJECT_DIR, 'codenames_agent')
SRC_DIR      = os.path.join(AGENT_DIR, 'src')

for p in (PROJECT_DIR, AGENT_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Pylance cannot resolve dynamic paths at edit-time — runtime import is fine.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "main_with_reasoning",
    os.path.join(AGENT_DIR, "main_with_reasoning.py")
)
_mod = _ilu.module_from_spec(_spec)   # type: ignore
_spec.loader.exec_module(_mod)        # type: ignore
CodenamesAgentWithReasoning = _mod.CodenamesAgentWithReasoning

# ── Log directory ─────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(GAMEPLAY_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL COLOURS
# ══════════════════════════════════════════════════════════════════════════════

def _c(code, text):  return f'\033[{code}m{text}\033[0m'
def RED(t):          return _c('91;1', t)
def BLUE(t):         return _c('94;1', t)
def GRAY(t):         return _c('90',   t)
def YELLOW(t):       return _c('93',   t)
def GREEN(t):        return _c('92;1', t)
def BOLD(t):         return _c('1',    t)
def STRIKE(t):       return _c('9',    t)
def DIM(t):          return _c('2',    t)

def team_color(team, text):
    return RED(text) if team == 'RED' else BLUE(text)


# ══════════════════════════════════════════════════════════════════════════════
# BOARD DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def print_board(board, red_words, blue_words, assassin, revealed):
    W = 13

    def cell(word):
        w = word.upper()
        if w not in revealed:
            return BOLD(f'{w:^{W}}')
        if w == assassin.upper():
            return RED(STRIKE(f'{w:^{W}}'))
        if w in [r.upper() for r in red_words]:
            return RED(STRIKE(f'{w:^{W}}'))
        if w in [b.upper() for b in blue_words]:
            return BLUE(STRIKE(f'{w:^{W}}'))
        return GRAY(STRIKE(f'{w:^{W}}'))

    border = '─' * (W * 5 + 6)
    print()
    print(BOLD('┌' + border + '┐'))
    for row in range(5):
        cells = [cell(board[row * 5 + col]) for col in range(5)]
        print(BOLD('│ ') + '  '.join(cells) + BOLD(' │'))
    print(BOLD('└' + border + '┘'))
    print()


def print_scores(red_rem, blue_rem):
    bar_r = RED ('█' * red_rem  + '░' * (9 - red_rem))
    bar_b = BLUE('█' * blue_rem + '░' * (8 - blue_rem))
    print(f'  {RED("RED")}  {bar_r} {BOLD(str(red_rem))} left   '
          f'{BLUE("BLUE")} {bar_b} {BOLD(str(blue_rem))} left')
    print()


def divider(char='─', w=72):
    print(DIM(char * w))


# ══════════════════════════════════════════════════════════════════════════════
# JSON ENCODER  — handles numpy float32 / float16 etc.
# ══════════════════════════════════════════════════════════════════════════════

class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return float(obj)
        except (TypeError, ValueError):
            pass
        try:
            return int(obj)
        except (TypeError, ValueError):
            pass
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
            'board':   {'red': red, 'blue': blue,
                        'assassin': assassin, 'neutral': neutral}
        }
        self._ev('BOARD_SETUP', self.meta['board'])

    def clue(self, team, clue, count, targets, source, score):
        self._ev('SPYMASTER_CLUE', {
            'team': team, 'clue': clue, 'count': count,
            'targets': targets, 'source': source,
            'score': round(float(score), 4)
        })

    def guess(self, team, word, result):
        self._ev('PLAYER_GUESS', {'team': team, 'word': word, 'result': result})

    def turn_end(self, team, reason):
        self._ev('TURN_END', {'team': team, 'reason': reason})

    def game_over(self, winner, reason):
        self._ev('GAME_OVER', {'winner': winner, 'reason': reason})
        record = {**self.meta,
                  'ended': datetime.datetime.now().isoformat(),
                  'winner': winner, 'reason': reason,
                  'events': self.events}
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, cls=SafeEncoder)
        with open(self.hist_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, cls=SafeEncoder) + '\n')
        print(f'\n{GREEN("✓")} Log saved → {self.log_file}')

    def _ev(self, kind, data):
        self.events.append({
            'time': datetime.datetime.now().isoformat(),
            'event': kind, 'data': data
        })


# ══════════════════════════════════════════════════════════════════════════════
# GAME ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class CodenamesGame:

    def __init__(self, red_agent, blue_agent):
        self.red_agent  = red_agent
        self.blue_agent = blue_agent

    def _up(self, lst):   return [w.upper() for w in lst]
    def _rem(self, words, revealed):
        return [w for w in self._up(words) if w not in revealed]

    # ── New board ─────────────────────────────────────────────────────────────

    def new_board(self):
        red, blue, assassin, neutral = self.red_agent.generate_board()
        board = self._up(red + blue + [assassin] + neutral)
        random.shuffle(board)
        return self._up(red), self._up(blue), assassin.upper(), self._up(neutral), board

    # ── AI Spymaster ──────────────────────────────────────────────────────────

    def spymaster_think(self, team, t_w, o_w, assassin, neutral, revealed, logger):
        rem_t = self._rem(t_w, revealed)
        rem_o = self._rem(o_w, revealed)

        print()
        divider('═')
        print(team_color(team, f'  🕵  {team} SPYMASTER is thinking...'))
        divider('═')

        # Suppress the agent's internal stage output
        agent = self.red_agent if team == 'RED' else self.blue_agent
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = agent.play_turn(
                red_team  = rem_t,
                blue_team = rem_o,
                assassin  = assassin,
                neutral   = neutral
            )

        if result is None:
            print(team_color(team, f'  {team} Spymaster passes.'))
            logger.clue(team, 'PASS', 0, [], 'none', 0.0)
            return 'PASS', 0, [], 'none', 0.0

        clue    = result['clue']
        count   = result['count']
        targets = result['targets']
        source  = result['source']
        score   = result['score']

        print()
        divider()
        print(team_color(team, BOLD(f'  CLUE: "{clue}"    COUNT: {count}')))
        print(DIM(f'  Targeting: {", ".join(targets)}  |  source: {source}  |  score: {float(score):.2f}'))
        divider()
        print()

        logger.clue(team, clue, count, targets, source, score)
        return clue, count, targets, source, score

    # ── Human operative ───────────────────────────────────────────────────────

    def operative_turn(self, team, clue, count,
                       t_w, o_w, assassin, neutral,
                       board, revealed, logger, bonus=False, source='vector'):
        opp   = 'BLUE' if team == 'RED' else 'RED'
        max_g = count + 1 if bonus else count
        guesses = 0

        print()
        print(team_color(team, BOLD(f'  ══  {team} OPERATIVE — YOUR TURN  ══')))
        print(f'  Clue: {BOLD(clue)}   Count: {BOLD(str(count))}   '
              f'(up to {BOLD(str(max_g))} guesses)')
        print(f'  Type {BOLD("PASS")} to end your turn early.')

        guessed_targets = []
        intended_targets = [w.upper() for w in t_w]

        while guesses < max_g:
            if not self._rem(t_w, revealed):
                return 'win'

            print_board(board, t_w, o_w, assassin, revealed)
            print_scores(len(self._rem(t_w, revealed)),
                         len(self._rem(o_w, revealed)))

            raw = input(team_color(team, f'  [{team}] Guess: ')).strip().upper()

            if raw == 'PASS':
                logger.turn_end(team, 'player passed')
                print(f'  {YELLOW("⏭")}  Turn passed.')
                # record outcome for profiling
                agent = self.red_agent if team == 'RED' else self.blue_agent
                try:
                    agent.record_outcome(clue, source, intended_targets, guessed_targets)
                except Exception:
                    pass
                return 'continue'

            if raw not in board:
                print(f'  {YELLOW("?")}  Not on the board. Try again.')
                continue
            if raw in revealed:
                print(f'  {YELLOW("?")}  Already revealed. Try again.')
                continue

            revealed.add(raw)
            guesses += 1

            if raw == assassin:
                print()
                print(RED(BOLD(f'  ☠   ASSASSIN HIT: {raw}!')))
                print(RED(BOLD(f'      {team} TEAM LOSES!')))
                logger.guess(team, raw, 'assassin')
                logger.turn_end(team, 'hit assassin')
                # record outcome
                agent = self.red_agent if team == 'RED' else self.blue_agent
                try:
                    agent.record_outcome(clue, source, intended_targets, guessed_targets)
                except Exception:
                    pass
                return 'assassin'

            elif raw in t_w:
                print(f'  {GREEN("✓")}  {BOLD(raw)} — {team_color(team, "YOUR WORD!")}')
                logger.guess(team, raw, 'correct')
                guessed_targets.append(raw)
                if not self._rem(t_w, revealed):
                    # record outcome (all found)
                    agent = self.red_agent if team == 'RED' else self.blue_agent
                    try:
                        agent.record_outcome(clue, source, intended_targets, guessed_targets)
                    except Exception:
                        pass
                    return 'win'
                if guesses < max_g:
                    again = input(f'  Keep guessing? ({max_g - guesses} left) [Y/n]: ').strip().lower()
                    if again == 'n':
                        logger.turn_end(team, 'player stopped')
                        # record outcome
                        agent = self.red_agent if team == 'RED' else self.blue_agent
                        try:
                            agent.record_outcome(clue, source, intended_targets, guessed_targets)
                        except Exception:
                            pass
                        return 'continue'
                else:
                    print(f'  {YELLOW("⏭")}  Max guesses used. Turn ends.')
                    logger.turn_end(team, 'max guesses reached')
                    # record outcome
                    agent = self.red_agent if team == 'RED' else self.blue_agent
                    try:
                        agent.record_outcome(clue, source, intended_targets, guessed_targets)
                    except Exception:
                        pass

            elif raw in o_w:
                print(f'  {YELLOW("✗")}  {BOLD(raw)} — {team_color(opp, "OPPONENT")} word! Turn ends.')
                logger.guess(team, raw, 'opponent')
                logger.turn_end(team, 'hit opponent word')
                # record outcome
                agent = self.red_agent if team == 'RED' else self.blue_agent
                try:
                    agent.record_outcome(clue, source, intended_targets, guessed_targets)
                except Exception:
                    pass
                return 'continue'

            else:
                print(f'  {YELLOW("~")}  {BOLD(raw)} — {GRAY("NEUTRAL")}. Turn ends.')
                logger.guess(team, raw, 'neutral')
                logger.turn_end(team, 'hit neutral word')
                # record outcome
                agent = self.red_agent if team == 'RED' else self.blue_agent
                try:
                    agent.record_outcome(clue, source, intended_targets, guessed_targets)
                except Exception:
                    pass
                return 'continue'

        return 'continue'

    # ── Main loop ─────────────────────────────────────────────────────────────

    def play(self):
        gid    = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        logger = GameLogger(gid)

        print()
        print(BOLD("╔" + "═" * 68 + "╗"))
        print(BOLD("║") + BOLD(f'{"  CODENAMES  —  Two Players vs AI Spymasters":^68}') + BOLD("║"))
        print(BOLD('╚' + '═' * 68 + '╝'))
        print()
        input('  Press ENTER to generate a new board...')

        red_w, blue_w, assassin, neutral, board = self.new_board()
        revealed = set()
        logger.set_board(red_w, blue_w, assassin, neutral)

        input('  Press ENTER to begin...')

        turn       = 'RED'
        winner     = None
        turn_count = {'RED': 0, 'BLUE': 0}   # track how many turns each team has had

        while True:
            opp = 'BLUE' if turn == 'RED' else 'RED'
            t_w = red_w  if turn == 'RED' else blue_w
            o_w = blue_w if turn == 'RED' else red_w

            if not self._rem(t_w, revealed):
                winner = turn
                logger.game_over(winner, f'{winner} found all words')
                break

            clue, count, _, source, score = self.spymaster_think(
                turn, t_w, o_w, assassin, neutral, revealed, logger
            )
            if clue == 'PASS':
                turn = opp
                continue

            input(f'  Press ENTER to guess as {team_color(turn, turn)} operative...')

            turn_count[turn] += 1
            bonus = turn_count[turn] > 1   # bonus guess only from 2nd turn onwards

            outcome = self.operative_turn(
                turn, clue, count,
                t_w, o_w, assassin, neutral,
                board, revealed, logger, bonus=bonus, source=source
            )

            if outcome == 'assassin':
                winner = opp
                logger.game_over(winner, f'{turn} hit the assassin')
                break
            if outcome == 'win':
                winner = turn
                logger.game_over(winner, f'{winner} found all words')
                break
            if not self._rem(t_w, revealed):
                winner = turn
                logger.game_over(winner, f'{winner} found all words')
                break

            turn = opp

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
    print('\nLoading AI Spymasters — please wait...')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        user_id = input('  Enter player ID (leave blank for "anon"): ').strip() or 'anon'
        red_agent  = CodenamesAgentWithReasoning(user_id=user_id)
        blue_agent = CodenamesAgentWithReasoning(user_id=user_id)
    print('  Ready!\n')
    game = CodenamesGame(red_agent, blue_agent)

    while True:
        game.play()
        again = input('  Play again? [y/N]: ').strip().lower()
        if again != 'y':
            print('\n  Thanks for playing!\n')
            break
