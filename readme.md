# SPY Stock Model Work
This application is meant to store some historical data for SPY and to build a predictive model which can reliably predict whether SPY will move up or down 0.5% first from open. In other words, if SPY opened at $100 would it hit $100.50 (up) first or would it hit $99.50 (down) first or be flat (never go up or down 0.5% from open).

This project is in early stages and I started it to get back into some tech stacks which I haven't used in a while. I built the initial setup (model training, basic GUI dashboard, backtesting) in one weekend as a challenge and have done small iterations, fixes, and improvements since then.

## Stack
Python, Postgres, Django, Docker, Alpaca-py (for stock data), FRED (for VIX volatility index and US10Y rates), FastAPI

The following is not yet up to date for a setup:

## Useful Commands
Access postgres in docker
`docker exec -it postgres psql -U {Your user}`

Access modeling app python in docker
`docker exec -it modeling_dev bash`

Start modeling python app (dev only)
`docker-compose run modeling-dev`

## Modeling App Build
Access modeling app via docker command
`docker-compose run modeling-dev`

Run command to build venv based on requirements.txt
