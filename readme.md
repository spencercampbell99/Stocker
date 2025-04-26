# SPY Stock Model Work
This application is meant to store some historical data for SPY and to build a predictive model which can reliably predict whether SPY will move $5 up or down from market open.

## Stack
Python, Postgres, Django, Docker, Alpaca-py (for stock data)

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