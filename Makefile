PROJECT_NAME=dunderbot

# Make sure everything is not referencing files, by making them .phony:
#.PHONY: all check format yapf mypy lint run build stop clean redo bash notebook nb it jupyter

## Runs 
run:
	@docker-compose up -d --force-recreate --build $(PROJECT_NAME)

## Rebuild riskapi_app container
build:
	@docker-compose build

## Stops application. Stops running container without removing them.
stop:
	@docker-compose stop

## Removes stopped service containers.
clean:
	@docker-compose down

## Stops, cleans, rebuilds, and runs container. Useful for when you want to restart everything after making changes to codebase and want to rebuild.
# redo: githook stop clean build run  # To use this command, we need to implement githook code at the top. May take a long time to build.
redo: stop clean build run

## Execs into the docker container to run `bash` commands.
bash:
	@docker exec -it $(PROJECT_NAME) bash

## Starts a jupyter notebook server and opens the web browser. Refresh the page, and add the token.
nb:
	@docker exec -it $(PROJECT_NAME) poetry run python -m webbrowser "http://0.0.0.0:8888"
	echo "Please refresh the browser page and add the token from below"
	@docker-compose exec $(PROJECT_NAME) poetry run jupyter notebook --ip=0.0.0.0 --allow-root --no-browser