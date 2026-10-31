"""Replay a conversation on another configuration, then compare: the bridge from the Chat to the Bench.

A replay sends the conversation's user turns again, in order, to a configuration: the served engine with drafts on or
off, or another engine at an address. Each turn carries the original conversation's earlier turns (its own questions
and the original replies), so every replay answers the same thing at the same point and the turns compare one by one.
Replays are greedy (temperature 0) so that two of them can be compared reply by reply; a replay is kept as a
conversation of its own, linked to the one it replays.
"""

from __future__ import annotations

import contextlib
import statistics
import threading
import time

import httpx

from . import bench, compare, conversations
from . import settings as settings_mod


def label(target: dict) -> str:
    if target.get("kind") == "address":
        return f"{target.get('model')} on {target.get('label') or target.get('url')}"
    return f"{target.get('model') or 'the served model'}, drafts {'off' if target.get('drafts') is False else 'on'}"


class Replays:
    """Runs replays in a thread, one at a time; each turn's reply is saved as it arrives."""

    def __init__(self, chats, served) -> None:
        self.chats, self.served = chats, served          # served() -> (base URL, key, model, engine name)
        self.current: dict | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._starting = threading.Lock()                # the busy check and the worker's start, one request at a time

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, cid: str, target: dict) -> dict:
        with self._starting:
            return self._start(cid, target)

    def _start(self, cid: str, target: dict) -> dict:
        if self.busy():
            raise RuntimeError("a replay is already going; wait for it or stop it")
        src = self.chats.get(cid)
        if src is None:
            raise KeyError(cid)
        turns = [i for i, m in enumerate(src["messages"]) if m["role"] == "user"]
        if not turns:
            raise ValueError("this conversation has no question to replay")
        if uses_tools(src):
            raise ValueError("this conversation used tools or a reply format (the workshop): a replay sends questions "
                             "only, so it would measure another conversation; replay one without them")
        if target.get("kind") == "address":
            if not isinstance(target.get("url"), str) or not target.get("model"):
                raise ValueError("another engine needs its address and a model")
            # a user or password in the address would be kept in the replay's title, settings and exports
            url = settings_mod._engine_url(target["url"]).removesuffix("/v1")
            target = {**target, "url": url}
            base, key, model, engine = url, target.get("key") or None, target["model"], target.get("label") or url
        else:
            base, key, model, engine = self.served()
            if not model:
                raise RuntimeError("no model is being served; start one from Models")
        public = {k: v for k, v in target.items() if k != "key"}           # the key is used, never kept
        public.setdefault("model", model)
        settings = {**(src.get("settings") or {}), "replay_of": cid, "target": public, "engine": engine,
                    "temperature": 0.0}
        # a replay of a replay is titled after the first conversation, not "Replay: Replay: …"
        first, seen = src, {cid}
        while (up := (first.get("settings") or {}).get("replay_of")) and up not in seen and (o := self.chats.get(up)):
            first = o
            seen.add(up)
        doc = self.chats.create({"messages": [], "settings": settings,
                                 "title": f"Replay: {first['title'][:50]} ({label(public)})"[:200]})
        self.current = {"id": doc["id"], "of": cid, "done": 0, "total": len(turns), "status": "running"}
        self._cancel.clear()
        self._thread = threading.Thread(target=self._go, args=(doc["id"], doc["rev"], src, base, key, model, engine,
                                                                target),
                                        name="replay", daemon=True)
        self._thread.start()
        return {**doc, "progress": self.current}

    def _show(self, rep) -> None:
        """What the reply being written says so far, for a page that watches it (the last 20,000 characters)."""

        if self.current:
            self.current["live_text"] = rep.text[-20000:]
            self.current["live_reasoning"] = rep.reasoning[-20000:]

    def cancel(self) -> bool:
        if not self.busy():
            return False
        self._cancel.set()
        return True

    def _go(self, rid: str, rev: int, src: dict, base: str, key: str | None, model: str, engine: str,
            target: dict) -> None:
        """Each turn written on the revision the last one made: a page that edited the replay meanwhile wins, and
        the replay stops there and says so rather than erase the edit."""

        headers = {"Authorization": f"Bearer {key}"} if key else {}
        st = src.get("settings") or {}
        preset = st.get("preset") or {}
        out: list[dict] = []
        try:
            with httpx.Client(timeout=httpx.Timeout(600.0, connect=5.0)) as client:
                msgs = src["messages"]
                for i, m in enumerate(msgs):
                    if m["role"] != "user":
                        continue
                    if self._cancel.is_set():
                        self.current["status"] = "cancelled"
                        break
                    history = [{"role": x["role"], "content": x["content"]} for x in msgs[: i + 1]
                               if x["role"] in ("user", "assistant", "system")]
                    # the settings the original reply was written with, as the page recorded them on it; a reply from
                    # before that record falls back on the conversation's settings as they are now
                    reply = msgs[i + 1] if i + 1 < len(msgs) and msgs[i + 1].get("role") == "assistant" else {}
                    req = reply.get("request") if isinstance(reply.get("request"), dict) else None
                    system = req.get("system") if req else preset.get("system")
                    thinking = req.get("thinking") if req else st.get("thinking")
                    budget = req.get("budget") if req else (st.get("budget") if thinking != "off" else None)
                    max_tokens = req.get("max_tokens") if req else preset.get("max_tokens") or 2048
                    if system:
                        history = [{"role": "system", "content": system}, *history]
                    # without randomness, whatever the original's temperature: two greedy replies are comparable token
                    # for token, sampled ones are not
                    body = {"model": model, "messages": history, "temperature": 0.0, "top_p": 1.0, "stream": True,
                            "stream_options": {"include_usage": True}}
                    if max_tokens:
                        body["max_tokens"] = max_tokens
                    # of the sampling, only the stops still matter without randomness: they end the reply
                    stop = (req.get("stop") if req else preset.get("stop")) or []
                    if stop:
                        body["stop"] = stop
                    if thinking in ("on", "off"):
                        body["chat_template_kwargs"] = {"enable_thinking": thinking == "on"}
                    if thinking != "off" and budget:
                        body["thinking_budget"] = budget
                    if target.get("kind") != "address" and target.get("drafts") is False:
                        body["draft"] = False
                    self.current.update(turn=i, question=m["content"][:2000], live_text="", live_reasoning="")
                    rep = bench.stream_once(client, base + "/v1/chat/completions", headers, body, self._cancel,
                                            on_text=self._show)
                    mm = bench.measure(rep)
                    outcome = "error" if rep.error else "length" if rep.finish == "length" else "done"
                    out += [{"role": "user", "content": m["content"], "turn": i},
                            {"role": "assistant", "content": rep.text, "reasoning": rep.reasoning, "model": model,
                             "engine": engine, "turn": i, "outcome": outcome, "error": rep.error,
                             "thinking": thinking, "at": time.time(),
                             "request": {**(req or {}), "thinking": thinking, "budget": budget, "system": system or "",
                                         "max_tokens": max_tokens, "temperature": 0.0},
                             "stats": {"tokens": mm["tokens"], "tps": mm["tps"], "ttft": mm["ttft"],
                                       "prompt": mm["prompt_tokens"], "measured": "engine" if rep.usage else "counted",
                                       "timing": "engine" if mm["timing"] == "engine" else "page", "sha": mm["sha"],
                                       "drafts": (mm["runtime"] or {}).get("drafts"),
                                       "drafted": (mm["runtime"] or {}).get("drafted")}}]
                    rev = self.chats.update(rid, {"messages": out}, rev=rev)["rev"]
                    self.current["done"] += 1
                else:
                    self.current["status"] = "done"
        except conversations.Conflict:
            self.current["status"] = "stopped"
            self.current["error"] = "the replay was edited in a page meanwhile; it stopped there and kept the edit"
        except Exception as exc:  # noqa: BLE001 - the replay keeps what it has and says why it stopped
            self.current["status"], self.current["error"] = "failed", str(exc)[:300]
        with contextlib.suppress(conversations.Conflict, KeyError):     # an edit since keeps its place
            self.chats.update(rid, {"settings": {**(self.chats.get(rid) or {}).get("settings", {}),
                                                 "replay_status": self.current["status"]}}, rev=rev)


def _measured(stats: dict) -> dict:
    """A reply's figures, its token count and speed dropped when they were stream events counted by the page."""

    return {**stats, "tokens": None, "tps": None} if stats.get("measured") == "counted" else stats


def uses_tools(doc: dict) -> bool:
    """Whether a conversation leaned on the workshop (tools, a reply format, tool turns): a replay cannot rebuild it."""

    ws = (doc.get("settings") or {}).get("workshop") or {}
    if ws.get("tools") or ws.get("output") not in (None, "text"):
        return True
    # the settings may have changed since: what the turns themselves recorded says the workshop was used (a tool
    # turn, a reply checked as JSON, a request that carried tools or a reply format)
    return any(m.get("role") == "tool" or m.get("tool_calls") or m.get("json") is not None
               or any(k in (m.get("request") or {}) for k in ("tools", "response_format"))
               or (m.get("request") or {}).get("output") not in (None, "text")
               for m in doc.get("messages") or [])


def _same_tokens(ma: dict, mb: dict, asked_off: tuple[bool, bool]) -> bool | None:
    """The same tokens, a difference, or None when it cannot be said: a reply that failed, a digest missing, or a
    side written with drafts off that the engine did not confirm off."""

    sa, sb = ma.get("stats") or {}, mb.get("stats") or {}
    complete = ("done", "length")                 # an error, a stop, an interruption: the reply is not whole
    if ma.get("outcome", "done") not in complete or mb.get("outcome", "done") not in complete:
        return None
    if not sa.get("sha") or not sb.get("sha"):
        return None
    for asked, st in zip(asked_off, (sa, sb)):    # each side as it was asked: off confirmed off, on confirmed on
        evidence = {"drafts": st.get("drafts"), "drafted": st.get("drafted")}   # the Bench's rules, contradictions too
        if asked is True and not bench.drafts_off_confirmed(evidence):
            return None
        if asked is False and not bench.drafts_on_confirmed(evidence):
            return None
    return sa["sha"] == sb["sha"]


def compare_replies(a: dict, b: dict) -> dict:
    """Two conversations, turn by turn: the same question's replies side by side, speed and sameness tested."""

    # the same question pairs the replies, whether the conversation is a replay or the original
    ra, rb = _by_question(a), _by_question(b)
    # a side replayed with drafts off counts only where the engine confirmed them off
    # what each side was asked: drafts off (True), on (False), or nothing said (None: an original conversation)
    off_a, off_b = ({False: True, True: False}.get(((d.get("settings") or {}).get("target") or {}).get("drafts"))
                    for d in (a, b))
    rows = []
    for q, ma in ra.items():
        mb = rb.get(q)
        if mb is None:
            continue
        # a reply kept before tokens were told from stream events ("counted") has no token figure to compare
        sa, sb = (_measured(m.get("stats") or {}) for m in (ma, mb))
        ta, tb = sa.get("tps"), sb.get("tps")
        text, rank = q
        rows.append({"question": (text if rank == 0 else f"{text} ({rank + 1})")[:200], "a": {"text": ma.get("content", ""), "tps": ta, "ttft": sa.get("ttft"),
                                                "tokens": sa.get("tokens"), "sha": sa.get("sha"), "outcome": ma.get("outcome")},
                     "b": {"text": mb.get("content", ""), "tps": tb, "ttft": sb.get("ttft"), "tokens": sb.get("tokens"),
                           "sha": sb.get("sha"), "outcome": mb.get("outcome")},
                     "same_text": bool(ma.get("content")) and ma.get("content") == mb.get("content"),
                     "same_tokens": _same_tokens(ma, mb, (off_a, off_b)),
                     "change": (tb / ta - 1) if ta and tb else None})
    changes = [r["change"] for r in rows if r["change"] is not None]
    up, down = sum(1 for c in changes if c > 0), sum(1 for c in changes if c < 0)
    p = compare.binomial_two_sided(up, up + down)
    med = statistics.median(changes) if changes else None
    return {"a": {"id": a["id"], "title": a["title"], "settings": a.get("settings")},
            "b": {"id": b["id"], "title": b["title"], "settings": b.get("settings")}, "turns": rows,
            "speed": {"median_change": med, "b_faster": up, "b_slower": down, "p": p,
                      "within_noise": med is None or abs(med) < compare.SAME_SPEED or p >= compare.NOISE_P},
            "same_text": sum(1 for r in rows if r["same_text"]), "same_tokens": sum(1 for r in rows if r["same_tokens"]),
            "checked": sum(1 for r in rows if r["same_tokens"] is not None)}


def _by_question(doc: dict) -> dict[tuple[str, int], dict]:
    """Each question's reply (the last one given to it), keyed by the question's exact text and its rank among
    equals: a question that reads "q (2)" is not taken for the second "q"."""

    out: dict[tuple[str, int], dict] = {}
    seen: dict[str, int] = {}
    q = None
    for m in doc.get("messages", []):
        if m["role"] == "user":
            n = seen.get(m["content"], 0)
            seen[m["content"]] = n + 1
            q = (m["content"], n)
        elif m["role"] == "assistant" and q is not None:
            out[q] = m
    return out
