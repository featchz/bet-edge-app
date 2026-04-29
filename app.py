"""
Bet Edge App — Backend
Fetches live NBA props from The Odds API, runs the edge model, serves results.

Run:
  pip install -r requirements.txt
  python app.py
  Open: http://localhost:5000
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional — env vars can be set directly
import math
import requests
from math import erf, sqrt
from datetime import datetime
import pytz
from flask import Flask, render_template, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

# ── Cache — refreshed on schedule, served instantly to all users ──────────
_cache = {
    "picks":      [],
    "last_fetch": None,
    "credits":    None,
}

def cache_is_fresh():
    """If we have any picks at all, serve them. Scheduler keeps it fresh."""
    return bool(_cache["picks"])

def refresh_cache():
    """
    Credit-efficient refresh — 3 API calls per refresh instead of 16.
    Uses game-level odds only (h2h + totals) for all sports.
    Free tier: 500 credits/month. At 4-hour intervals = ~6 refreshes/day x 3 calls = ~18/day = ~540/month.
    """
    raw = []
    raw += fetch_nba_picks()                          # Smart model: team scoring + stats
    raw += fetch_game_picks("baseball_mlb",   "MLB")  # Market comparison for now
    raw += fetch_game_picks("icehockey_nhl",  "NHL")  # Market comparison for now

    # Deduplicate — keep only the best-edge side per game+market
    seen = {}
    for p in raw:
        key = (p["player"], p["stat"], p.get("line", 0))
        if key not in seen or p["edge"] > seen[key]["edge"]:
            seen[key] = p
    picks = sorted(seen.values(), key=lambda x: x["edge"], reverse=True)

    _cache["picks"]      = picks[:50]
    _cache["last_fetch"] = datetime.now()
    print(f"[cache] refreshed — {len(_cache['picks'])} picks, {datetime.now().strftime('%I:%M %p')}")

# ── Config ──────────────────────────────────────────────────────────────
ODDS_API_KEY  = os.getenv("ODDS_API_KEY", "YOUR_KEY_HERE")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"
BALLDONTLIE_KEY  = os.getenv("BALLDONTLIE_KEY", "YOUR_KEY_HERE")

SPORTS = {
    "NBA":    "basketball_nba",
    "MLB":    "baseball_mlb",
    "NHL":    "icehockey_nhl",
    "Soccer": "soccer_usa_mls",
}

# ── Projection Weights (tune via backtesting) ────────────────────────────
WEIGHTS = {
    "season_avg":     0.20,
    "last_10":        0.25,
    "last_5":         0.30,
    "matchup_adj":    0.15,
    "external_model": 0.10,
}

EDGE_THRESHOLD       = 0.02   # Minimum edge for player props
GAME_EDGE_THRESHOLD  = 0.001  # Show all game picks, sort by best edge
STRONG_THRESHOLD     = 0.05   # Strong edge threshold


# ── Math helpers ─────────────────────────────────────────────────────────
def poisson_over(lam, line):
    k = int(line)
    return 1 - sum((math.exp(-lam) * (lam**i)) / math.factorial(i) for i in range(k + 1))

def normal_over(mean, std, line):
    z = (line - mean) / std
    return 0.5 * (1 - erf(z / sqrt(2)))

def american_to_implied(odds):
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)

def ev_calc(true_prob, odds):
    try:
        if odds < 0:
            win_amount = 100 / (abs(odds) / 100)
        else:
            win_amount = float(odds)
        return round((true_prob * win_amount) - ((1 - true_prob) * 100), 2)
    except Exception:
        return 0.0

def weighted_proj(season_avg, last_10, last_5, matchup_adj, external_model):
    return (
        WEIGHTS["season_avg"]     * season_avg +
        WEIGHTS["last_10"]        * last_10 +
        WEIGHTS["last_5"]         * last_5 +
        WEIGHTS["matchup_adj"]    * matchup_adj +
        WEIGHTS["external_model"] * external_model
    )


# ── Odds API calls ───────────────────────────────────────────────────────
def get_live_games(sport_key):
    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_player_props(sport_key, event_id):
    url = f"{ODDS_API_BASE}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "player_points,player_rebounds,player_assists,player_threes",
        "oddsFormat": "american",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── BallDontLie — Team scoring model ─────────────────────────────────────
_team_cache = {}  # cache team scoring so we don't re-fetch same team twice per refresh

def get_team_scoring(team_name):
    """
    Fetch a team's last 15 game scores from BallDontLie.
    Returns avg points scored, avg points allowed, and recent game totals.
    """
    if team_name in _team_cache:
        return _team_cache[team_name]

    headers = {"Authorization": BALLDONTLIE_KEY}
    try:
        # Find team ID
        r = requests.get(
            f"{BALLDONTLIE_BASE}/teams",
            params={"search": team_name},
            headers=headers, timeout=10
        )
        if r.status_code != 200:
            return None
        teams = r.json().get("data", [])
        if not teams:
            return None
        team_id = teams[0]["id"]

        # Get last 15 completed games
        r2 = requests.get(
            f"{BALLDONTLIE_BASE}/games",
            params={"team_ids[]": team_id, "seasons[]": 2025,
                    "per_page": 15, "postseason": "true"},
            headers=headers, timeout=10
        )
        games = r2.json().get("data", []) if r2.status_code == 200 else []

        # Fall back to regular season if playoffs thin
        if len(games) < 5:
            r3 = requests.get(
                f"{BALLDONTLIE_BASE}/games",
                params={"team_ids[]": team_id, "seasons[]": 2025, "per_page": 15},
                headers=headers, timeout=10
            )
            games = r3.json().get("data", []) if r3.status_code == 200 else []

        pts_scored, pts_allowed = [], []
        for g in games:
            if g.get("status") != "Final":
                continue
            if g["home_team"]["id"] == team_id:
                pts_scored.append(g["home_team_score"])
                pts_allowed.append(g["visitor_team_score"])
            else:
                pts_scored.append(g["visitor_team_score"])
                pts_allowed.append(g["home_team_score"])

        if len(pts_scored) < 3:
            return None

        result = {
            "avg_scored":  round(sum(pts_scored)  / len(pts_scored),  1),
            "avg_allowed": round(sum(pts_allowed) / len(pts_allowed), 1),
            "games":       len(pts_scored),
        }
        _team_cache[team_name] = result
        return result
    except Exception:
        return None


def project_nba_total(home_team, away_team):
    """
    Project NBA game total using both teams' scoring averages.
    Returns projection, confidence, and a human-readable reason string.
    """
    home = get_team_scoring(home_team)
    away = get_team_scoring(away_team)
    if not home or not away:
        return None

    # Projection = blend of each team's offense vs opponent defense
    home_proj = (home["avg_scored"] + away["avg_allowed"]) / 2
    away_proj  = (away["avg_scored"]  + home["avg_allowed"]) / 2
    total_proj = round(home_proj + away_proj, 1)

    reason = (
        f"Model projects {total_proj} pts · "
        f"{home_team.split()[-1]} avg {home['avg_scored']}/g scored, "
        f"{home['avg_allowed']}/g allowed · "
        f"{away_team.split()[-1]} avg {away['avg_scored']}/g scored, "
        f"{away['avg_allowed']}/g allowed"
    )
    return {"projection": total_proj, "reason": reason}


# ── BallDontLie stats ────────────────────────────────────────────────────
def get_player_stats(player_name):
    """Fetch season avg + recent game stats for a player from BallDontLie."""
    headers = {"Authorization": BALLDONTLIE_KEY}
    try:
        # Find player ID
        r = requests.get(
            f"{BALLDONTLIE_BASE}/players",
            params={"search": player_name, "per_page": 5},
            headers=headers, timeout=10
        )
        if r.status_code != 200:
            return None
        players = r.json().get("data", [])
        if not players:
            return None

        # Match by closest name
        player_id = None
        name_lower = player_name.lower()
        for p in players:
            full = f"{p['first_name']} {p['last_name']}".lower()
            if full == name_lower or name_lower in full:
                player_id = p["id"]
                break
        if not player_id:
            player_id = players[0]["id"]

        # Season averages
        r2 = requests.get(
            f"{BALLDONTLIE_BASE}/season_averages",
            params={"player_ids[]": player_id, "season": 2025},
            headers=headers, timeout=10
        )
        avgs = {}
        if r2.status_code == 200:
            data2 = r2.json().get("data", [])
            avgs = data2[0] if data2 else {}

        # Last 10 games stats
        r3 = requests.get(
            f"{BALLDONTLIE_BASE}/stats",
            params={"player_ids[]": player_id, "per_page": 10,
                    "seasons[]": 2025, "postseason": "true"},
            headers=headers, timeout=10
        )
        games = []
        if r3.status_code == 200:
            games = r3.json().get("data", [])

        # Also try regular season if postseason is thin
        if len(games) < 5:
            r4 = requests.get(
                f"{BALLDONTLIE_BASE}/stats",
                params={"player_ids[]": player_id, "per_page": 10, "seasons[]": 2025},
                headers=headers, timeout=10
            )
            if r4.status_code == 200:
                games = r4.json().get("data", [])

        def avg_stat(key, n=10):
            vals = [g.get(key) or 0 for g in games[:n] if g.get(key) is not None]
            return round(sum(vals) / len(vals), 1) if vals else None

        return {
            "season_pts":  avgs.get("pts"),
            "season_reb":  avgs.get("reb"),
            "season_ast":  avgs.get("ast"),
            "season_fg3m": avgs.get("fg3m"),
            "last_10_pts":  avg_stat("pts", 10),
            "last_10_reb":  avg_stat("reb", 10),
            "last_10_ast":  avg_stat("ast", 10),
            "last_10_fg3m": avg_stat("fg3m", 10),
            "last_5_pts":   avg_stat("pts", 5),
            "last_5_reb":   avg_stat("reb", 5),
            "last_5_ast":   avg_stat("ast", 5),
            "last_5_fg3m":  avg_stat("fg3m", 5),
        }
    except Exception:
        return None


# ── Edge analysis ────────────────────────────────────────────────────────
def analyze_prop(player, stat, line, over_odds, under_odds,
                 season_avg, last_10, last_5, matchup_adj=None, external=None,
                 use_poisson=True, std_dev=6.0):

    matchup_adj  = matchup_adj  or season_avg
    external     = external     or season_avg
    proj = weighted_proj(season_avg, last_10, last_5, matchup_adj, external)

    p_over  = poisson_over(proj, line) if use_poisson else normal_over(proj, std_dev, line)
    p_under = 1 - p_over

    imp_over  = american_to_implied(over_odds)
    imp_under = american_to_implied(under_odds)

    edge_over  = round(p_over  - imp_over,  4)
    edge_under = round(p_under - imp_under, 4)

    best_side = "OVER" if edge_over > edge_under else "UNDER"
    best_edge = max(edge_over, edge_under)
    best_odds = over_odds if best_side == "OVER" else under_odds
    best_ev   = ev_calc(p_over if best_side == "OVER" else p_under, best_odds)

    strength = "strong" if best_edge >= STRONG_THRESHOLD else ("value" if best_edge >= EDGE_THRESHOLD else "none")

    return {
        "player":      player,
        "stat":        stat,
        "line":        line,
        "projection":  round(proj, 1),
        "best_side":   best_side,
        "edge":        round(best_edge * 100, 1),   # as percentage
        "ev":          best_ev,
        "strength":    strength,
        "over_odds":   over_odds,
        "under_odds":  under_odds,
        "p_over":      round(p_over * 100, 1),
        "p_under":     round(p_under * 100, 1),
        "imp_over":    round(imp_over * 100, 1),
        "imp_under":   round(imp_under * 100, 1),
    }


# ── Demo data (used when API key not yet set) ────────────────────────────
def demo_picks():
    return [
        analyze_prop("Nikola Jokic", "Points",  27.5, -110, -110, 27.7, 26.1, 24.5, 25.0, 26.4, False, 6.0),
        analyze_prop("Jalen Brunson", "Assists",  6.5, -132,  100,  6.0,  7.1,  8.2,  7.5,  7.0, True),
        analyze_prop("Anthony Edwards", "Points", 28.5, -115, -105, 27.3, 28.8, 29.1, 27.5, 28.0, False, 6.5),
        analyze_prop("Karl-Anthony Towns", "Rebounds", 9.5, -118, -102, 9.1, 9.8, 10.2, 9.5, 9.7, True),
        analyze_prop("Shai Gilgeous-Alexander", "Points", 31.5, -110, -110, 32.1, 31.4, 30.8, 31.0, 31.2, False, 6.0),
        analyze_prop("Jalen Brunson", "Points", 25.5, -112, -108, 26.2, 25.8, 26.4, 25.5, 26.0, False, 5.5),
    ]


# ── Live pipeline ─────────────────────────────────────────────────────────
PROP_MARKETS = "player_points,player_rebounds,player_assists,player_threes"

STAT_MAP = {
    "player_points":    ("pts", False, 6.0),   # (bdl_key, use_poisson, std_dev)
    "player_rebounds":  ("reb", True,  None),
    "player_assists":   ("ast", True,  None),
    "player_threes":    ("fg3m", True, None),
}

def fetch_live_picks():
    """Full pipeline: Odds API → props → BallDontLie stats → edge model → picks."""
    picks = []

    # Step 1 — get today's NBA games
    try:
        r = requests.get(
            f"{ODDS_API_BASE}/sports/basketball_nba/odds",
            params={"apiKey": ODDS_API_KEY, "regions": "us",
                    "markets": "h2h", "oddsFormat": "american"},
            timeout=10
        )
        games = r.json() if r.status_code == 200 else []
    except Exception:
        return []

    if not games:
        return []

    # Step 2 — for each game, fetch player props
    for game in games[:6]:   # cap at 6 games to save API credits
        event_id   = game.get("id")
        home_team  = game.get("home_team", "")
        away_team  = game.get("away_team", "")

        try:
            r2 = requests.get(
                f"{ODDS_API_BASE}/sports/basketball_nba/events/{event_id}/odds",
                params={"apiKey": ODDS_API_KEY, "regions": "us",
                        "markets": PROP_MARKETS, "oddsFormat": "american"},
                timeout=10
            )
            if r2.status_code != 200:
                continue
            event_data = r2.json()
        except Exception:
            continue

        # Step 3 — parse each prop market
        bookmakers = event_data.get("bookmakers", [])
        if not bookmakers:
            continue

        # Aggregate lines across books — use best available odds
        props_by_player = {}   # key: (player, market)
        for book in bookmakers:
            for market in book.get("markets", []):
                mkt_key = market.get("key")
                if mkt_key not in STAT_MAP:
                    continue
                for outcome in market.get("outcomes", []):
                    player = outcome.get("description", "")
                    side   = outcome.get("name")      # "Over" or "Under"
                    price  = outcome.get("price", 0)
                    line   = outcome.get("point", 0)
                    if not player or not line:
                        continue
                    key = (player, mkt_key)
                    if key not in props_by_player:
                        props_by_player[key] = {"line": line, "over": None, "under": None,
                                                 "player": player, "market": mkt_key,
                                                 "matchup": f"{away_team} @ {home_team}"}
                    if side == "Over"  and (props_by_player[key]["over"]  is None or price > props_by_player[key]["over"]):
                        props_by_player[key]["over"]  = price
                        props_by_player[key]["line"]  = line
                    if side == "Under" and (props_by_player[key]["under"] is None or price > props_by_player[key]["under"]):
                        props_by_player[key]["under"] = price

        # Step 4 — run edge model on each prop
        for (player, mkt_key), prop in props_by_player.items():
            if prop["over"] is None or prop["under"] is None:
                continue

            stat_key, use_poisson, std_dev = STAT_MAP[mkt_key]
            stat_label = mkt_key.replace("player_", "").replace("_", " ").title()

            # Get player stats from BallDontLie
            stats = get_player_stats(player)
            if not stats:
                # No real stats — skip this pick entirely, don't generate fake edge
                continue
            s_avg = stats.get(f"season_{stat_key}") or 0
            l10   = stats.get(f"last_10_{stat_key}") or s_avg
            l5    = stats.get(f"last_5_{stat_key}")  or s_avg
            # Skip if all stats are zero — means data didn't load
            if s_avg == 0 and l10 == 0 and l5 == 0:
                continue

            result = analyze_prop(
                player=player,
                stat=stat_label,
                line=prop["line"],
                over_odds=prop["over"],
                under_odds=prop["under"],
                season_avg=s_avg,
                last_10=l10,
                last_5=l5,
                use_poisson=use_poisson,
                std_dev=std_dev or 6.0,
            )
            result["matchup"] = prop["matchup"]
            if result["strength"] != "none":
                picks.append(result)

    return picks


# ── NBA smart picks — team scoring model ─────────────────────────────────
def fetch_nba_picks():
    """
    NBA game total picks using real team scoring projections.
    Projects total from both teams' last 15 games, uses normal distribution
    to get true win probability. Much higher confidence than market comparison.
    """
    picks = []
    _team_cache.clear()  # fresh team data each refresh

    try:
        r = requests.get(
            f"{ODDS_API_BASE}/sports/basketball_nba/odds",
            params={"apiKey": ODDS_API_KEY, "regions": "us",
                    "markets": "totals", "oddsFormat": "american"},
            timeout=10
        )
        if r.status_code != 200:
            return []
        games = r.json()
    except Exception:
        return []

    NBA_TOTAL_STD = 12.0  # NBA game total standard deviation (~12 pts historically)

    for game in games:
        home  = game.get("home_team", "")
        away  = game.get("away_team", "")
        books = game.get("bookmakers", [])
        if not books:
            continue

        matchup = f"{away} @ {home}"

        # Get best available over/under odds across all books
        over_prices, under_prices, line = [], [], None
        for book in books:
            for mkt in book.get("markets", []):
                if mkt.get("key") != "totals":
                    continue
                for outcome in mkt.get("outcomes", []):
                    if outcome.get("name") == "Over":
                        over_prices.append(outcome.get("price", 0))
                        line = outcome.get("point", line)
                    elif outcome.get("name") == "Under":
                        under_prices.append(outcome.get("price", 0))

        if not over_prices or not under_prices or not line:
            continue

        best_over  = max(over_prices)
        best_under = max(under_prices)

        # Project total using team scoring model
        proj_data = project_nba_total(home, away)
        if not proj_data:
            # Fall back to market comparison if no BallDontLie data
            continue

        projection = proj_data["projection"]
        reason     = proj_data["reason"]

        # True win probability from normal distribution
        p_over  = normal_over(projection, NBA_TOTAL_STD, line)
        p_under = 1 - p_over

        # Market implied probability (best book, de-vigged)
        raw_imp_over  = sum(american_to_implied(p) for p in over_prices)  / len(over_prices)
        raw_imp_under = sum(american_to_implied(p) for p in under_prices) / len(under_prices)
        vig_total     = raw_imp_over + raw_imp_under
        mkt_over      = raw_imp_over  / vig_total
        mkt_under     = raw_imp_under / vig_total

        # Edge = model probability minus market probability
        edge_over  = round(p_over  - mkt_over,  4)
        edge_under = round(p_under - mkt_under, 4)

        best_side = "OVER" if edge_over > edge_under else "UNDER"
        best_edge = max(edge_over, edge_under)
        best_odds = best_over if best_side == "OVER" else best_under
        true_p    = p_over if best_side == "OVER" else p_under

        # Only show picks with meaningful model edge (>3%)
        if best_edge < 0.03:
            continue

        strength = "strong" if best_edge >= 0.08 else "value"

        picks.append({
            "player":      matchup,
            "stat":        "Game Total",
            "sport":       "NBA",
            "line":        line,
            "projection":  projection,
            "best_side":   best_side,
            "edge":        round(best_edge * 100, 1),
            "ev":          ev_calc(true_p, best_odds),
            "strength":    strength,
            "over_odds":   best_over,
            "under_odds":  best_under,
            "p_over":      round(p_over  * 100, 1),
            "p_under":     round(p_under * 100, 1),
            "imp_over":    round(mkt_over  * 100, 1),
            "imp_under":   round(mkt_under * 100, 1),
            "matchup":     matchup,
            "reason":      reason,
        })

    picks.sort(key=lambda x: x["edge"], reverse=True)
    return picks


# ── Game-level picks (moneyline + totals) for any sport ──────────────────
def fetch_game_picks(sport_key, sport_label):
    """
    Finds value from odds discrepancies across bookmakers.
    Returns moneyline and totals picks. Works for NBA, MLB, NHL.
    """
    picks = []
    try:
        r = requests.get(
            f"{ODDS_API_BASE}/sports/{sport_key}/odds",
            params={"apiKey": ODDS_API_KEY, "regions": "us",
                    "markets": "h2h,totals", "oddsFormat": "american"},
            timeout=10
        )
        if r.status_code != 200:
            return []
        games = r.json()
    except Exception:
        return []

    for game in games:
        home  = game.get("home_team", "")
        away  = game.get("away_team", "")
        books = game.get("bookmakers", [])
        if not books:
            continue

        matchup = f"{away} @ {home}"

        # ── Totals (Over/Under) ──
        totals = {}
        for book in books:
            for mkt in book.get("markets", []):
                if mkt.get("key") != "totals":
                    continue
                for outcome in mkt.get("outcomes", []):
                    side  = outcome.get("name")
                    price = outcome.get("price", 0)
                    line  = outcome.get("point", 0)
                    if side == "Over":
                        totals.setdefault("over_prices", []).append(price)
                        totals["line"] = line
                    elif side == "Under":
                        totals.setdefault("under_prices", []).append(price)

        if totals.get("over_prices") and totals.get("under_prices"):
            best_over  = max(totals["over_prices"])
            best_under = max(totals["under_prices"])
            # Average IMPLIED probabilities (not raw odds) for accurate consensus
            raw_cons_over  = sum(american_to_implied(p) for p in totals["over_prices"])  / len(totals["over_prices"])
            raw_cons_under = sum(american_to_implied(p) for p in totals["under_prices"]) / len(totals["under_prices"])
            # De-vig: normalize so true probabilities sum to 100%
            total_vig      = raw_cons_over + raw_cons_under
            true_over      = raw_cons_over  / total_vig
            true_under     = raw_cons_under / total_vig
            best_imp_over  = american_to_implied(best_over)
            best_imp_under = american_to_implied(best_under)
            # Edge = best book implied minus de-vigged true probability
            edge_over  = round(best_imp_over  - true_over,  4)
            edge_under = round(best_imp_under - true_under, 4)

            for side, edge_val, odds in [("OVER", edge_over, best_over), ("UNDER", edge_under, best_under)]:
                if edge_val < GAME_EDGE_THRESHOLD:
                    continue
                true_p = true_over if side == "OVER" else true_under
                picks.append({
                    "player":     matchup,
                    "stat":       "Game Total",
                    "sport":      sport_label,
                    "line":       totals["line"],
                    "projection": totals["line"],
                    "best_side":  side,
                    "edge":       round(edge_val * 100, 1),
                    "ev":         ev_calc(true_p, odds),
                    "strength":   "strong" if edge_val >= STRONG_THRESHOLD else "value",
                    "over_odds":  best_over,
                    "under_odds": best_under,
                    "p_over":     round(true_over  * 100, 1),
                    "p_under":    round(true_under * 100, 1),
                    "imp_over":   round(best_imp_over  * 100, 1),
                    "imp_under":  round(best_imp_under * 100, 1),
                    "matchup":    matchup,
                })

        # ── Moneyline (h2h) ──
        ml = {}
        for book in books:
            for mkt in book.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                for outcome in mkt.get("outcomes", []):
                    team  = outcome.get("name")
                    price = outcome.get("price", 0)
                    ml.setdefault(team, []).append(price)

        for team, prices in ml.items():
            if len(prices) < 2:
                continue
            best_price = max(prices)
            # Average implied probabilities (not raw odds)
            cons_imp   = sum(american_to_implied(p) for p in prices) / len(prices)
            best_imp   = american_to_implied(best_price)
            edge_val   = round(best_imp - cons_imp, 4)
            if edge_val < GAME_EDGE_THRESHOLD:
                continue
            picks.append({
                "player":    team,
                "stat":      "Moneyline",
                "sport":     sport_label,
                "line":      0,
                "projection": 0,
                "best_side": "WIN",
                "edge":      round(edge_val * 100, 1),
                "ev":        ev_calc(cons_imp, best_price),
                "strength":  "strong" if edge_val >= STRONG_THRESHOLD else "value",
                "over_odds": best_price,
                "under_odds": best_price,
                "p_over":    round(cons_imp * 100, 1),
                "p_under":   round((1 - cons_imp) * 100, 1),
                "imp_over":  round(best_imp * 100, 1),
                "imp_under": round((1 - best_imp) * 100, 1),
                "matchup":   matchup,
            })

    return picks


# ── MLB player props ──────────────────────────────────────────────────────
MLB_PROP_MARKETS = "batter_hits,batter_total_bases,batter_rbis,pitcher_strikeouts,batter_home_runs"

def fetch_mlb_prop_picks():
    """Fetch MLB player prop edges."""
    picks = []
    try:
        r = requests.get(
            f"{ODDS_API_BASE}/sports/baseball_mlb/odds",
            params={"apiKey": ODDS_API_KEY, "regions": "us",
                    "markets": "h2h", "oddsFormat": "american"},
            timeout=10
        )
        if r.status_code != 200:
            return []
        games = r.json()
    except Exception:
        return []

    for game in games[:5]:
        event_id = game.get("id")
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        try:
            r2 = requests.get(
                f"{ODDS_API_BASE}/sports/baseball_mlb/events/{event_id}/odds",
                params={"apiKey": ODDS_API_KEY, "regions": "us",
                        "markets": MLB_PROP_MARKETS, "oddsFormat": "american"},
                timeout=10
            )
            if r2.status_code != 200:
                continue
            event_data = r2.json()
        except Exception:
            continue

        bookmakers = event_data.get("bookmakers", [])
        props_by_player = {}
        for book in bookmakers:
            for market in book.get("markets", []):
                mkt_key = market.get("key")
                for outcome in market.get("outcomes", []):
                    player = outcome.get("description", "")
                    side   = outcome.get("name")
                    price  = outcome.get("price", 0)
                    line   = outcome.get("point", 0)
                    if not player or not line:
                        continue
                    key = (player, mkt_key)
                    if key not in props_by_player:
                        stat_label = mkt_key.replace("batter_", "").replace("pitcher_", "").replace("_", " ").title()
                        props_by_player[key] = {"line": line, "over_prices": [], "under_prices": [],
                                                 "player": player, "stat": stat_label,
                                                 "matchup": f"{away} @ {home}"}
                    if side == "Over":
                        props_by_player[key]["over_prices"].append(price)
                    if side == "Under":
                        props_by_player[key]["under_prices"].append(price)

        for (player, mkt_key), prop in props_by_player.items():
            if not prop.get("over_prices") or not prop.get("under_prices"):
                continue
            best_over  = max(prop["over_prices"])
            best_under = max(prop["under_prices"])
            # Average implied probabilities (not raw odds)
            raw_cons_over  = sum(american_to_implied(p) for p in prop["over_prices"])  / len(prop["over_prices"])
            raw_cons_under = sum(american_to_implied(p) for p in prop["under_prices"]) / len(prop["under_prices"])
            # De-vig: normalize to true probabilities
            total_vig      = raw_cons_over + raw_cons_under
            true_over      = raw_cons_over  / total_vig
            true_under     = raw_cons_under / total_vig
            best_imp_over  = american_to_implied(best_over)
            best_imp_under = american_to_implied(best_under)
            edge_over  = round(best_imp_over  - true_over,  4)
            edge_under = round(best_imp_under - true_under, 4)
            best_side  = "OVER" if edge_over > edge_under else "UNDER"
            best_edge  = max(edge_over, edge_under)
            if best_edge < GAME_EDGE_THRESHOLD:
                continue
            best_odds = best_over if best_side == "OVER" else best_under
            true_p    = true_over if best_side == "OVER" else true_under
            picks.append({
                "player":     player,
                "stat":       prop["stat"],
                "sport":      "MLB",
                "line":       prop["line"],
                "projection": prop["line"],
                "best_side":  best_side,
                "edge":       round(best_edge * 100, 1),
                "ev":         ev_calc(true_p, best_odds),
                "strength":   "strong" if best_edge >= STRONG_THRESHOLD else "value",
                "over_odds":  best_over,
                "under_odds": best_under,
                "p_over":     round(true_over  * 100, 1),
                "p_under":    round(true_under * 100, 1),
                "imp_over":   round(best_imp_over  * 100, 1),
                "imp_under":  round(best_imp_under * 100, 1),
                "matchup":    prop["matchup"],
            })
    return picks


# ── Odds-only fallback for NBA ────────────────────────────────────────────
def fetch_odds_only_picks():
    return fetch_game_picks("basketball_nba", "NBA")


# ── Routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/picks")
def api_picks():
    try:
        no_key = (ODDS_API_KEY == "YOUR_KEY_HERE")
        sport  = request.args.get("sport", "NBA").upper()

        if no_key:
            picks = [dict(p, sport="NBA") for p in demo_picks()]
        else:
            if not cache_is_fresh():
                refresh_cache()
            all_picks = _cache["picks"]
            if sport == "ALL":
                picks = all_picks
            else:
                picks = [p for p in all_picks if p.get("sport") == sport]

        picks = sorted(picks, key=lambda x: x["edge"], reverse=True)
        updated = _cache["last_fetch"].strftime("%I:%M %p PT") if _cache["last_fetch"] else "—"

        return jsonify({
            "date":         datetime.now().strftime("%B %d, %Y"),
            "sport":        sport,
            "picks":        picks,
            "demo_mode":    no_key,
            "updated":      updated,
            "next_refresh": "Refreshes at 4am, 8am, 12pm, 4pm, 6pm PT",
            "weights":      WEIGHTS,
            "live":         not no_key,
        })
    except Exception as e:
        import traceback
        print(f"[api_picks ERROR] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e), "picks": [], "date": "", "demo_mode": False,
                        "updated": "", "next_refresh": "", "weights": {}, "live": False}), 500

@app.route("/api/raw")
def raw():
    r = requests.get(
        f"{ODDS_API_BASE}/sports/baseball_mlb/odds",
        params={"apiKey": ODDS_API_KEY, "regions": "us",
                "markets": "h2h,totals", "oddsFormat": "american"},
        timeout=10
    )
    data = r.json()
    # If it's a list, show first game structure
    if isinstance(data, list) and data:
        g = data[0]
        return jsonify({
            "status": r.status_code,
            "credits": r.headers.get("x-requests-remaining"),
            "games_returned": len(data),
            "first_game": f"{g.get('away_team')} @ {g.get('home_team')}",
            "bookmakers_count": len(g.get("bookmakers", [])),
            "first_bookmaker": g.get("bookmakers", [{}])[0].get("title") if g.get("bookmakers") else "none",
            "markets_available": [m.get("key") for m in g.get("bookmakers", [{}])[0].get("markets", [])] if g.get("bookmakers") else [],
        })
    # Otherwise show the raw response (likely an error)
    return jsonify({"status": r.status_code, "response": data, "credits": r.headers.get("x-requests-remaining")})

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})

@app.route("/api/debug")
def debug():
    results = {}
    for label, key in [("NBA", "basketball_nba"), ("MLB", "baseball_mlb"), ("NHL", "icehockey_nhl")]:
        try:
            r = requests.get(
                f"{ODDS_API_BASE}/sports/{key}/odds",
                params={"apiKey": ODDS_API_KEY, "regions": "us",
                        "markets": "h2h,totals", "oddsFormat": "american"},
                timeout=10
            )
            results[label] = {
                "status":    r.status_code,
                "games":     len(r.json()) if r.status_code == 200 else 0,
                "error":     r.text[:200] if r.status_code != 200 else None,
                "credits":   r.headers.get("x-requests-remaining", "unknown"),
                "sample":    (r.json()[0].get("home_team") + " vs " + r.json()[0].get("away_team")) if r.status_code == 200 and r.json() else "no games",
            }
        except Exception as e:
            results[label] = {"status": "exception", "error": str(e)}
    results["key_loaded"] = ODDS_API_KEY != "YOUR_KEY_HERE"
    results["key_prefix"] = ODDS_API_KEY[:8] + "..." if ODDS_API_KEY != "YOUR_KEY_HERE" else "NOT SET"
    return jsonify(results)


# ── Scheduler — refreshes picks at fixed times PT ────────────────────────
def start_scheduler():
    PT = pytz.timezone("America/Los_Angeles")
    scheduler = BackgroundScheduler(timezone=PT)
    # Refresh at 4am, 8am, 12pm, 4pm, 6pm PT daily
    for hour in [4, 8, 12, 16, 18]:
        scheduler.add_job(refresh_cache, "cron", hour=hour, minute=0)
    scheduler.start()
    print("[scheduler] started — refreshing at 4am, 8am, 12pm, 4pm, 6pm PT")
    # Run once immediately on startup so picks are ready right away
    refresh_cache()

# Start scheduler (skip in debug reloader child process to avoid double-start)
import os
if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
    start_scheduler()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
