import time
from typing import Dict, List, Tuple


CRITERIA = ['学業成績', '基礎能力テスト', '実践経験', '推薦・評価', '志望動機・フィット']


def make_user_payload(decision: str = '一次通過') -> Dict:
    """Create a deterministic user decision/weights payload for /save_decision."""
    weights = {
        '学業成績': 30,
        '基礎能力テスト': 25,
        '実践経験': 20,
        '推薦・評価': 15,
        '志望動機・フィット': 10,
    }
    return {
        'decision': decision,
        'weights': weights,
        'reasoning': 'テスト: 自動化シナリオ',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime()),
        # Provide fixed participant opinions for deterministic behavior
        'participant_opinions': [
            {
                'decision': '見送り',
                'weights': {
                    '学業成績': 15,
                    '基礎能力テスト': 10,
                    '実践経験': 30,
                    '推薦・評価': 25,
                    '志望動機・フィット': 20,
                },
                'bot_id': 0,
            },
            {
                'decision': '見送り',
                'weights': {
                    '学業成績': 35,
                    '基礎能力テスト': 25,
                    '実践経験': 15,
                    '推薦・評価': 15,
                    '志望動機・フィット': 10,
                },
                'bot_id': 1,
            },
            {
                'decision': '見送り',
                'weights': {
                    '学業成績': 20,
                    '基礎能力テスト': 40,
                    '実践経験': 10,
                    '推薦・評価': 20,
                    '志望動機・フィット': 10,
                },
                'bot_id': 2,
            },
        ],
    }


def make_setup_payload(decision: str = '一次通過') -> Dict:
    """Create payload for /setup_chat."""
    return {
        'decision': decision,
        'weights': {
            '学業成績': 30,
            '基礎能力テスト': 25,
            '実践経験': 20,
            '推薦・評価': 15,
            '志望動機・フィット': 10,
        }
    }


def percentiles(samples: List[float], ps: Tuple[int, ...] = (50, 90, 95, 99)) -> Dict[int, float]:
    if not samples:
        return {p: 0.0 for p in ps}
    xs = sorted(samples)
    out = {}
    for p in ps:
        k = (len(xs) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(xs) - 1)
        if f == c:
            out[p] = xs[f]
        else:
            d0 = xs[f] * (c - k)
            d1 = xs[c] * (k - f)
            out[p] = d0 + d1
    return out

