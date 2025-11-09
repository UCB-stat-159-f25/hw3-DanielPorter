ENV_NAME=ligo

.PHONY: env html clean

env:
	@conda env update -n $(ENV_NAME) -f environment.yml || conda env create -n $(ENV_NAME) -f environment.yml
	@echo "Activate with: conda activate $(ENV_NAME)"

html:
	@npx myst build --html
	@echo "Built site at _build/html"

clean:
	@rm -rf figures/* audio/* _build
