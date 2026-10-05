# tensorfold-dsv41-TP2 — TensorFold serving DeepSeek-V4.1-Flash EXL3 2.9bpw at TP=2 (aiai head + aiai2 worker).
# Run on the head. The worker gets this directory at the same path (`make sync`) and its own image build.
#
#   make env        # first time: .env from .env.example (fill TF_API_KEY), chmod 600
#   make image      # stage the source snapshot (SRC), build the image on BOTH nodes
#   make up         # refuses while vLLM / a dev server holds the GPUs or the port; worker first, then waits
#   make down / restart / wait / status / logs / logs-worker / smoke
#   make swap-in    # vLLM down, TensorFold up        make swap-out   # TensorFold down, vLLM up
#   make watch-install / watch-status / watch-once / watch-uninstall   # the watchdog timer (watchdog.sh)
#   make lease [LEASE_MIN=20]   # the watchdog stands down until then  make soak / stress / structured  (bench)
#
# vLLM and TensorFold cannot co-serve: each takes ~100 GiB a rank of the 121.7 GiB pool, and both bind :8888.
SHELL := /bin/bash

DIR          := $(CURDIR)
WORKER       ?= aiai2-ib
SRC          ?= /home/urtho/tensorfold
VLLM_DIR     ?= $(CURDIR)/../deepseek41flash-exl3-TP2
VLLM_NAMES   ?= dsv41-exl3-head|dsv41-exl3-worker|deepseek4flash-tp2-vllm-dspark-1
IMAGE_NAME   ?= tensorfold-dsv41
WAIT_TIMEOUT ?= 1800
# a rank's weights, caches and kept prompts take ~100 GiB; vLLM's recipe waits for 114 before booting
MEM_READY_GIB  ?= 112
MEM_READY_WAIT ?= 600
# each rank's RoCE devices: its rankN.env NCCL_IB_HCA (NCCL stripes over both halves of the 200G port)
hcas_of      = $(shell grep -E '^NCCL_IB_HCA=' $(DIR)/$(1) 2>/dev/null | tail -1 | cut -d= -f2- | sed 's/^[=^]*//; s/:[0-9]*//g; s/,/ /g')
HCAS0        ?= $(or $(call hcas_of,rank0.env),roceP2p1s0f0)
HCAS1        ?= $(or $(call hcas_of,rank1.env),roceP2p1s0f1)

env_of = $(shell grep -E '^$(1)=' $(DIR)/.env 2>/dev/null | tail -1 | cut -d= -f2-)
PORT         := $(or $(call env_of,PORT),8888)
API_KEY      := $(call env_of,TF_API_KEY)
SERVED_NAME  := $(or $(call env_of,SERVED_NAME),DeepSeek-v4.1-Flash-EXL3)
CACHE_DIR    := $(call env_of,CACHE_DIR)
PREPARED_DIR := $(call env_of,PREPARED_DIR)
# a stop waits this long for the ranks to exit (with TF_DSV41_DISK they write their kept prompts first)
STOP_GRACE_S := $(or $(call env_of,STOP_GRACE_S),$(if $(call env_of,TF_DSV41_DISK),150,30))
MODEL        := $(or $(call env_of,MODEL),dsv41/DeepSeek-V4.1-Flash-EXL3-2.9bpw)
FASTBOOT     := python -m tensorfold.families.deepseek_v41.cuda.fastboot
BASE_URL     := http://localhost:$(PORT)
COMPOSE0     := docker compose --env-file .env --env-file rank0.env
COMPOSE1     := docker compose --env-file .env --env-file rank1.env
# each device's RoCE v2 IPv4 GID index, for the log and the guard only: NCCL (NCCL_IB_ROCE_VERSION_NUM=2,
# NCCL_IB_ADDR_FAMILY=AF_INET in the compose file) and the RoCE all-gather pick it per device by type and address.
# The index moves: the kernel keeps a deleted GID's slot while a queue pair still uses it, so an address re-added
# under a live connection (the peer rebooting drops the link) lands one slot up, on one device and not the other.
GID_SCRIPT    = for d in $(1); do x=-; for i in $$(seq 0 15); do g=$$(cat /sys/class/infiniband/$$d/ports/1/gids/$$i 2>/dev/null); \
                t=$$(cat /sys/class/infiniband/$$d/ports/1/gid_attrs/types/$$i 2>/dev/null); \
                case "$$g:$$t" in *0000:ffff:*"RoCE v2") x=$$i; break;; esac; done; printf "%s=%s " $$d $$x; done

# up / down / restart / swap-in / swap-out stop and start the pair in order: never run their steps in parallel
# (a global MAKEFLAGS=-jN would race `restart: down up`)
.NOTPARALLEL:

# the watchdog's state (watchdog.sh): the lock, the stop marker, the lease, its logs
STATE        := $(or $(XDG_STATE_HOME),$(HOME)/.local/state)/tensorfold-dsv41
LEASE_MIN    ?= 20
UNITS        := tensorfold-dsv41-watchdog.service tensorfold-dsv41-watchdog.timer
# one up / down / restart at a time, the watchdog's heal included. A heal holds the lock and runs `make restart`
# with TF_DSV41_LOCKED=1, as does every nested call: taking the lock again there would wait forever for itself.
LOCK = $(if $(TF_DSV41_LOCKED),,flock -n -E 75 $(STATE)/lock env TF_DSV41_LOCKED=1)

.PHONY: help env stage image prebuild sync guard memwait up down restart _up _down _restart wait status logs \
        logs-worker smoke vllm-down vllm-up swap-in swap-out lease watch-install watch-uninstall watch-once \
        watch-status soak stress structured

help:
	@sed -n '1,14p' Makefile | sed 's/^# \{0,1\}//'

env:
	@if [ -f .env ]; then echo ".env exists"; else cp .env.example .env && chmod 600 .env && \
	  echo ".env written from .env.example: set TF_API_KEY (clients send it as a bearer token)"; fi

# -- image ----------------------------------------------------------------------------------------------------------
stage:
	@test -f $(SRC)/pyproject.toml || { echo "❌ no TensorFold source at SRC=$(SRC)"; exit 1; }
	@mkdir -p build/tensorfold
	rsync -a --delete --delete-excluded --exclude __pycache__ --exclude '*.pyc' \
	  --include /src/ --include '/src/**' --include /pyproject.toml --include /README.md --include '/LICENSE*' \
	  --include /THIRD_PARTY_NOTICES.md --include /LICENSES/ --include '/LICENSES/**' --exclude '*' \
	  $(SRC)/ build/tensorfold/
	@tar --sort=name --mtime=@0 --owner=0 --group=0 -cf - Dockerfile build/tensorfold | sha256sum | cut -c1-12 > build/REVISION
	@echo "source snapshot $$(cat build/REVISION)"

image: stage sync
	@rev=$$(cat build/REVISION); \
	build="docker build -q -t $(IMAGE_NAME):$$rev -t $(IMAGE_NAME):latest --build-arg TF_REVISION=$$rev -f Dockerfile build"; \
	echo "building $(IMAGE_NAME):$$rev on the head and $(WORKER)"; \
	( $$build > build/head.log 2>&1 ) & h=$$!; \
	( ssh $(WORKER) "cd $(DIR) && $$build" > build/worker.log 2>&1 ) & w=$$!; \
	rc=0; wait $$h || { rc=1; echo "❌ head build:"; tail -20 build/head.log; }; \
	wait $$w || { rc=1; echo "❌ worker build:"; tail -20 build/worker.log; }; \
	[ $$rc = 0 ] && echo "✅ $(IMAGE_NAME):$$rev (also :latest) on both nodes"; [ $$rc = 0 ] || exit $$rc
	@$(MAKE) --no-print-directory prebuild
	@$(MAKE) --no-print-directory prepare

# compile the CUDA extensions into CACHE_DIR before any model is loaded: compiling beside a loaded model (a few GiB
# free) can push a GB10 into the OOM killer. Refuses while a server holds the GPU; a no-op when the cache is warm.
prebuild:
	@test -n "$(CACHE_DIR)" || { echo "❌ CACHE_DIR is not set in .env"; exit 1; }
	@run="docker run --rm --gpus all -v $(CACHE_DIR):/root/.cache -v $(DIR)/prebuild.py:/prebuild.py:ro $(IMAGE_NAME):latest python /prebuild.py"; \
	if docker ps --format '{{.Names}}' | grep -qE '^tensorfold-dsv41-serve|$(VLLM_NAMES)'; then \
	  echo "prebuild: a server holds the GPU here, skipped (run make prebuild after make down)"; exit 0; fi; \
	echo "compiling CUDA extensions on the head and $(WORKER) (minutes on a new image)"; \
	( $$run 2>&1 | grep -E "extensions ready|Error|error" | tail -3 ) & h=$$!; \
	( ssh $(WORKER) "$$run" 2>&1 | grep -E "extensions ready|Error|error" | tail -3 ) & w=$$!; \
	wait $$h; wait $$w

# each rank's built weights into PREPARED_DIR on its node (fastboot): a start then reads them in ~15 s instead of
# building them from the checkpoint. Run through the compose service, so the image, env and mounts are the serving
# ones; refuses while a server holds the GPU (the build takes a rank's full memory); a no-op when current.
prepare:
	@test -n "$(PREPARED_DIR)" || { echo "❌ PREPARED_DIR is not set in .env"; exit 1; }
	@if docker ps --format '{{.Names}}' | grep -qE '^tensorfold-dsv41-serve|$(VLLM_NAMES)'; then \
	  echo "prepare: a server holds the GPU here, skipped (run make prepare after make down)"; exit 0; fi; \
	mkdir -p $(PREPARED_DIR); ssh $(WORKER) "mkdir -p $(PREPARED_DIR)"; \
	echo "preparing each rank's weights on its node (minutes when the folder is stale)"; \
	( $(COMPOSE0) run --rm --no-deps serve $(FASTBOOT) prepare /models/$(MODEL) --rank 0 --root /prepared 2>&1 \
	  | grep -E "prepared folder|built in|removed|Error|error" ) & h=$$!; \
	( ssh $(WORKER) "cd $(DIR) && $(COMPOSE1) run --rm --no-deps serve $(FASTBOOT) prepare /models/$(MODEL) --rank 1 --root /prepared" 2>&1 \
	  | grep -E "prepared folder|built in|removed|Error|error" ) & w=$$!; \
	wait $$h; wait $$w

sync:
	@test -f .env || { echo "❌ no .env: 'make env' first"; exit 1; }
	@ssh $(WORKER) "mkdir -p $(DIR)"
	@rsync -a --delete --exclude 'build/*.log' $(DIR)/ $(WORKER):$(DIR)/
	@test -z "$(CACHE_DIR)" || { mkdir -p $(CACHE_DIR); ssh $(WORKER) "mkdir -p $(CACHE_DIR)"; }

# -- run ------------------------------------------------------------------------------------------------------------
guard:
	@for n in head $(WORKER); do \
	  run() { if [ $$n = head ]; then bash -c "$$1"; else ssh $(WORKER) "$$1"; fi; }; \
	  if run "docker ps --format '{{.Names}}'" | grep -qxE '$(VLLM_NAMES)'; then \
	    echo "❌ $$n: vLLM is running — 'make swap-in' stops it first (no co-serving: memory + :$(PORT))"; exit 2; fi; \
	  if run "docker exec tf-dev pgrep -f 'tensorfold serve|tools/dsv41_'" >/dev/null 2>&1; then \
	    echo "❌ $$n: a TensorFold dev run holds the GPU in tf-dev — stop it first"; exit 2; fi; \
	  if ! run "grep -q '^ARMED=0' /etc/watchdog-memory.conf"; then \
	    echo "❌ $$n: the house mem-watchdog is ARMED (it hard-resets at its 1 GiB floor) — disarm it first"; exit 3; fi; \
	done
	@if ! docker ps --format '{{.Names}}' | grep -q '^tensorfold-dsv41-serve' && ss -Hltn "sport = :$(PORT)" | grep -q .; then \
	  echo "❌ something already listens on :$(PORT)"; exit 2; fi
	@if [ "$(call env_of,TF_CARVEOUT)" != 0 ]; then for n in head $(WORKER); do \
	  if [ $$n = head ]; then v=$$(sudo -n cat /sys/module/nvidia_drm/parameters/modeset 2>/dev/null); \
	  else v=$$(ssh -o BatchMode=yes $(WORKER) 'sudo -n cat /sys/module/nvidia_drm/parameters/modeset' 2>/dev/null); fi; \
	  [ "$$v" = Y ] || { echo "❌ $$n: nvidia_drm modeset='$$v' — the display carveout needs modeset=1 (or set TF_CARVEOUT=0)"; exit 4; }; \
	done; fi

# Free memory, then defragment it: GB10's GPU allocates large contiguous blocks of system memory, and after an image
# build or a long dev session the free pages are scattered, so every allocation of the start (weights, caches, graph
# pools) stalls in direct compaction and the instance runs slow until restarted (measured: a 104 GiB fragmented
# host loaded for > 35 min). compact_memory makes the blocks up front; the high-order free count is logged.
# GiB of free memory in blocks of >= 2 MiB (buddy orders >= 9 of 4 KiB pages: /proc/buddyinfo fields 14..)
BIG_FREE = awk '/Normal/{s=0; for (i=14; i<=NF; i++) s += $$i * 2^(i-5) / 262144; printf "%.0f", s}' /proc/buddyinfo
BIG_FREE_SSH = awk '/Normal/{s=0; for (i=14; i<=NF; i++) s += \$$i * 2^(i-5) / 262144; printf \"%.0f\", s}' /proc/buddyinfo

memwait:
	@sudo -n sysctl -q vm.drop_caches=3; ssh $(WORKER) 'sudo -n sysctl -q vm.drop_caches=3'; \
	sudo -n sysctl -q vm.compact_memory=1; ssh $(WORKER) 'sudo -n sysctl -q vm.compact_memory=1'; \
	echo "free memory in blocks of >= 2 MiB after compaction: head $$($(BIG_FREE)) GiB, $(WORKER) $$(ssh $(WORKER) "$(BIG_FREE_SSH)") GiB"; \
	end=$$(( $$(date +%s) + $(MEM_READY_WAIT) )); \
	while :; do \
	  h=$$(awk '/MemAvailable/{printf "%d", $$2/1048576}' /proc/meminfo); \
	  w=$$(ssh $(WORKER) "awk '/MemAvailable/{printf \"%d\", \$$2/1048576}' /proc/meminfo"); \
	  if [ "$$h" -ge $(MEM_READY_GIB) ] && [ "$$w" -ge $(MEM_READY_GIB) ]; then echo "✅ memory ready: head=$$h worker=$$w GiB"; break; fi; \
	  if [ $$(date +%s) -ge $$end ]; then echo "❌ MemAvailable head=$$h worker=$$w GiB < $(MEM_READY_GIB) after $(MEM_READY_WAIT)s"; exit 1; fi; \
	  echo "⏳ MemAvailable head=$$h worker=$$w GiB, waiting for >= $(MEM_READY_GIB)..."; sleep 15; \
	done

up down restart:
	@mkdir -p $(STATE)
	@$(LOCK) $(MAKE) --no-print-directory _$@ || { rc=$$?; [ $$rc != 75 ] || \
	  echo "❌ another up / down / restart (or a watchdog heal) holds $(STATE)/lock"; exit $$rc; }

# a deliberate start clears the stop marker once the guards pass: the watchdog guards the pair again, and a start
# that fails after that is healed like a crash (WATCH_HEAL=1)
_up: guard sync memwait
	@rm -f $(STATE)/stopped
	@docker image inspect $(IMAGE_NAME):latest >/dev/null 2>&1 || { echo "❌ no image: 'make image' first"; exit 1; }
	@h=$$($(call GID_SCRIPT,$(HCAS0))); w=$$(ssh $(WORKER) '$(call GID_SCRIPT,$(HCAS1))'); \
	echo "RoCE v2 GID index: head $$h worker $$w"; \
	case "$$h $$w" in *=-*) echo "❌ a RoCE device has no RoCE v2 IPv4 GID (no address on its interface?)"; exit 1;; esac; \
	ssh $(WORKER) "cd $(DIR) && $(COMPOSE1) up -d --force-recreate" && \
	$(COMPOSE0) up -d --force-recreate
	@$(MAKE) --no-print-directory wait

# a deliberate stop: the watchdog stands down until the next up, across reboots too (the marker keeps the boot id)
_down:
	@echo "$$(cat /proc/sys/kernel/random/boot_id) $$(date +%s)" > $(STATE)/stopped
	@( $(COMPOSE0) down --timeout $(STOP_GRACE_S) ) & ( ssh $(WORKER) "cd $(DIR) && $(COMPOSE1) down --timeout $(STOP_GRACE_S)" ) & wait

_restart: _down _up

wait:
	@end=$$(( $$(date +%s) + $(WAIT_TIMEOUT) )); \
	while :; do \
	  if curl -sf -m 5 $(BASE_URL)/health | grep -q '"ok": *true'; then \
	    echo "✅ serving at $(BASE_URL)/v1"; docker logs tensorfold-dsv41-serve-1 2>&1 | grep '^\[tensorfold\]' | tail -6; break; fi; \
	  for n in head $(WORKER); do \
	    if [ $$n = head ]; then s=$$(docker inspect -f '{{.State.Status}}' tensorfold-dsv41-serve-1 2>/dev/null); \
	    else s=$$(ssh $(WORKER) "docker inspect -f '{{.State.Status}}' tensorfold-dsv41-serve-1" 2>/dev/null); fi; \
	    if [ "$$s" != running ]; then echo "❌ $$n container is '$$s':"; \
	      if [ $$n = head ]; then docker logs --tail 30 tensorfold-dsv41-serve-1 2>&1; \
	      else ssh $(WORKER) "docker logs --tail 30 tensorfold-dsv41-serve-1" 2>&1; fi; exit 1; fi; \
	  done; \
	  if [ $$(date +%s) -ge $$end ]; then echo "❌ not serving after $(WAIT_TIMEOUT)s"; exit 1; fi; \
	  echo "⏳ $$(docker logs tensorfold-dsv41-serve-1 2>&1 | grep '^\[tensorfold\]' | tail -1 | cut -c1-110)"; sleep 20; \
	done

status:
	@echo "head:";   $(COMPOSE0) ps --format 'table {{.Name}}\t{{.Status}}\t{{.Image}}'
	@echo "worker:"; ssh $(WORKER) "cd $(DIR) && $(COMPOSE1) ps --format 'table {{.Name}}\t{{.Status}}\t{{.Image}}'"
	@curl -sf -m 5 $(BASE_URL)/health | jq -c '{ok, busy, requests_running, requests_total, context_length}' 2>/dev/null \
	  || echo "health: not answering on $(BASE_URL)"

logs:
	$(COMPOSE0) logs -f --tail 200

logs-worker:
	ssh -t $(WORKER) "cd $(DIR) && $(COMPOSE1) logs -f --tail 200"

smoke:
	@curl -sf -m 300 $(BASE_URL)/v1/chat/completions -H 'Content-Type: application/json' \
	  -H 'Authorization: Bearer $(API_KEY)' \
	  -d '{"model": "$(SERVED_NAME)", "max_tokens": 64, "messages": [{"role": "user", "content": "What is the capital of France? One word."}]}' \
	  | jq -c '{reply: .choices[0].message.content, finish: .choices[0].finish_reason, usage: .usage}'

# -- swapping with vLLM ---------------------------------------------------------------------------------------------
vllm-down:
	cd $(VLLM_DIR) && $(MAKE) down

vllm-up:
	cd $(VLLM_DIR) && $(MAKE) up

swap-in: vllm-down up

swap-out: down vllm-up

# -- watchdog (watchdog.sh, systemd/) ----------------------------------------------------------------------------------
# the lease: the watchdog stands down until it expires (its mtime is the expiry), e.g. while benchmarking
lease:
	@mkdir -p $(STATE) && touch -d "@$$(( $$(date +%s) + $(LEASE_MIN) * 60 ))" $(STATE)/lease
	@echo "watchdog lease until $$(date -d "@$$(stat -c %Y $(STATE)/lease)" '+%F %T')"

watch-once:
	@bash $(DIR)/watchdog.sh

# user units with WorkingDirectory=this directory; a user timer runs at boot only when the user lingers
watch-install:
	@mkdir -p $(HOME)/.config/systemd/user
	@for u in $(UNITS); do sed 's|@DIR@|$(DIR)|g' systemd/$$u > $(HOME)/.config/systemd/user/$$u; done
	systemctl --user daemon-reload
	systemctl --user enable --now tensorfold-dsv41-watchdog.timer
	@[ "$$(loginctl show-user $$USER -p Linger --value 2>/dev/null)" = yes ] || \
	  echo "⚠️  $$USER does not linger: the timer stops at logout and does not start at boot (sudo loginctl enable-linger $$USER)"
	@grep -qE '^WATCH_HEAL=1' .env 2>/dev/null || echo "WATCH_HEAL is not 1 in .env: the watchdog alerts but does not restart"

watch-uninstall:
	-systemctl --user disable --now tensorfold-dsv41-watchdog.timer
	@for u in $(UNITS); do rm -f $(HOME)/.config/systemd/user/$$u; done
	systemctl --user daemon-reload

watch-status:
	@systemctl --user --no-pager list-timers tensorfold-dsv41-watchdog.timer 2>/dev/null || true
	@echo "bad ticks: $$(cat $(STATE)/fails 2>/dev/null || echo 0)  last heal: $$(t=$$(cat $(STATE)/last_heal 2>/dev/null); \
	  [ -n "$$t" ] && date -d @$$t '+%F %T' || echo never)  stop marker: $$(cat $(STATE)/stopped 2>/dev/null || echo none)"
	@[ -f $(STATE)/lease ] && [ $$(stat -c %Y $(STATE)/lease) -gt $$(date +%s) ] && \
	  echo "lease until $$(date -d "@$$(stat -c %Y $(STATE)/lease)" '+%F %T')" || true
	@tail -n 8 $(STATE)/watch.log 2>/dev/null || echo "no watchdog log yet ($(STATE)/watch.log)"

# -- bench (tools/dsv41_{soak,stress,structured}.py from SRC): a lease first, so the watchdog leaves the run alone ----
TOOLS_DIR ?= $(SRC)/tools
MINUTES   ?= 30
STAMP      = $$(date +%F-%H%M)
# the key reaches the clients in their environment, never on a command line (visible to every user in ps)
soak stress structured: export TF_API_KEY := $(API_KEY)

soak:
	@mkdir -p results && $(MAKE) --no-print-directory lease LEASE_MIN=$$(( $(MINUTES) + 15 ))
	python3 $(TOOLS_DIR)/dsv41_soak.py --base $(BASE_URL) --model $(SERVED_NAME) --minutes $(MINUTES) \
	  --out results/soak-$(STAMP).json

stress:
	@mkdir -p results && $(MAKE) --no-print-directory lease LEASE_MIN=90
	python3 $(TOOLS_DIR)/dsv41_stress.py stress --base $(BASE_URL) --model $(SERVED_NAME) \
	  --out results/stress-$(STAMP).json

structured:
	@mkdir -p results && $(MAKE) --no-print-directory lease LEASE_MIN=60
	python3 $(TOOLS_DIR)/dsv41_structured.py --base $(BASE_URL) --model $(SERVED_NAME) \
	  --out results/structured-$(STAMP).json
