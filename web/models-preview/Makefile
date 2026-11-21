# Supported-models preview. Every target runs from this folder with the Go in mise.toml.
GO ?= go
GOLANGCI := $(GO) run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.15.0

.PHONY: run
run: ## Serve the page on port 3000 (every interface) from ../tensorfold_web/data/models.json (ARGS="-catalog URL")
	$(GO) run ./cmd/preview $(ARGS)

.PHONY: generate
generate: ## templ components and port mocks
	$(GO) tool templ generate -path ui
	$(GO) generate ./...

.PHONY: fmt
fmt: ## gofmt and templ fmt
	gofmt -w .
	$(GO) tool templ fmt ui

.PHONY: lint
lint: ## gofmt check, golangci-lint (layering, magic numbers, parallel tests)
	@test -z "$$(gofmt -l .)" || (gofmt -l . && echo 'gofmt: run make fmt' && exit 1)
	$(GOLANGCI) run ./...

.PHONY: test
test: ## Service and API boundary tests under the race detector
	$(GO) test -race ./...

.PHONY: check-generated
check-generated: generate ## Fail if templ or mock output is stale
	@git diff --exit-code -- '*.go' || (echo 'generated code is stale: run make generate and commit' && exit 1)
	@test -z "$$(git status --porcelain --untracked-files=all -- '*.go')" || (git status --short -- '*.go'; exit 1)

.PHONY: check
check: lint test check-generated ## Everything before a commit

.PHONY: build
build: ## Static binary in bin/
	CGO_ENABLED=0 $(GO) build -trimpath -ldflags='-s -w' -o bin/models-preview ./cmd/preview
