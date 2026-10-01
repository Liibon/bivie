PY      := .venv/bin/python
PLAN    ?= plan.example.yaml
FILES   ?= tests/fixtures
OUT     ?= out/run

.PHONY: setup test profile size export replay replay-linux fixtures

setup:
	uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements.txt

fixtures:
	$(PY) tests/make_fixtures.py $(FILES)

test:
	$(PY) -m unittest discover tests
	go vet ./... && go test ./...

profile:
	$(PY) -m mm_sizer profile $(PLAN) $(FILES)

size:
	$(PY) -m mm_sizer size $(PLAN) -o $(OUT)

export:
	$(PY) -m mm_sizer export $(PLAN)

replay:
	go build -o bin/bivie-replay ./cmd/bivie-replay

# static binary to copy next to a GPU endpoint
replay-linux:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o bin/bivie-replay-linux-amd64 ./cmd/bivie-replay
