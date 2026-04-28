# Bet Edge App

Daily NBA picks dashboard powered by the edge model.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API keys
export ODDS_API_KEY="your_key_here"     # https://the-odds-api.com (free tier available)
export BALLDONTLIE_KEY="your_key_here"  # https://app.balldontlie.io (free tier available)

# 3. Run
python app.py

# 4. Open
http://localhost:5000
```

## Without API keys
The app runs in demo mode automatically — shows sample picks using hardcoded data so you can see the UI working before wiring up live data.

## How it works
1. Fetches live odds from The Odds API
2. Fetches player stats from BallDontLie
3. Runs weighted projection model (season avg + L10 + L5 + matchup + external)
4. Calculates true probability vs. implied probability
5. Surfaces picks where edge > 4% (strong > 8%)
6. Auto-refreshes every 5 minutes

## Deploying for users
To make this accessible to users online, deploy to any Python host:
- **Railway** — `railway up` (easiest, free tier)
- **Render** — connect GitHub repo, set env vars
- **Heroku** — `git push heroku main`
- **VPS** — run behind nginx with gunicorn

## Tuning the model
Edit `WEIGHTS` in `app.py` to adjust how much each input matters.
Backtest by comparing projections vs. actual outcomes over 100+ games.
