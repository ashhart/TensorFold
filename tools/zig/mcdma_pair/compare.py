#!/usr/bin/env python3
"""The #1594 gate's verdict from tf-mcdma-pair logs: both ranks' results equal the host reference and each other."""
import argparse
import glob
import json
import os
import sys

from compare_log import Verdict, as_float, find_logs, receipt

NAMED = {"PeerDown", "PeerTimeout", "Timeout"}
# tf-mcdma-pair's tests in the order a round runs them (Phase in zig/tests/mcdma_pair_rank.zig)
SUITE = ["barrier", "broadcast", "allreduce_bf16", "allreduce_f32", "alltoall", "gather", "pipes", "bulk", "signals",
         "write2signal", "writesignalfrom"]
STARTUP = {"connect", "link", "handshake"}
# MCDMA's reasons for a post that failed (libmcdma_fabric.c WRITE_FAILED .. FLUSH_FAILED, and peer_fail's own)
COMPLETION = ("a write failed", "a signal failed", "a write-and-signal failed", "a flush failed", "a link operation failed")
GOODBYE = "said goodbye"
LINK_UP = "link up"
MTU_LINE = "path MTU above active MTU"
LIBRARIES = {"libmcdma-fabric.dylib", "libmcdma-fabric.so"}
# the run scripts' pre-flight WARNs no gate passes with, by how each begins: another kext, relaxed ordering, the clock
GATE_WARNS = (("the loaded MCDMA kext is ", "not the kext the settings expect"),
              ("relaxed ordering is on ", "relaxed ordering"),
              ("the clock is ", "the clock"))
# what MCDMA's libmcdma-fabric links on each OS, as meta.txt's links lines show; loop_fabric and the stub link neither
VERBS = {"macos": "/usr/lib/librdma.dylib", "linux": "libibverbs.so"}
# arm tests whose steps wait for the other rank's and end with a flush: an exit or stop victim delivered every earlier
LOCKSTEP = {"allreduce_bf16", "pipes"}
# the SETTINGS fields a --same-as run shares with the run (the hashes depend on them; mtu and the link do not)
SAME_AS = ("tests", "rounds", "seed", "window", "staging", "bulk_mib", "signals")
# Settings' defaults in zig/tests/mcdma_pair_rank.zig, for a plan file that leaves a field out
DEFAULTS = {"rounds": 1, "duration_s": 0, "window_mib": 0, "staging_mib": 6, "bulk_mib": 9, "signals": 5000, "wait_s": 10,
            "timeout_s": 10, "connect_timeout_s": 120, "arm": "none", "arm_rank": 1, "arm_test": "allreduce_bf16",
            "arm_step": 3, "tests": ["all"], "expect": []}
MIB = 1 << 20


def dropped_ok(chosen, dropped, v, what="--label"):
    """--label keeps each rank's newest log: an older one of that host and tree is noted, any other fails (listed)."""
    by_rank = {lg.rank: lg for lg in chosen}
    foreign = []
    for lg in dropped:
        keep = by_rank.get(lg.rank)
        if keep is not None and lg.host() == keep.host() and lg.tree == keep.tree:
            v.note("%s left out an older %s run of rank %d: %s" % (what, lg.label(), lg.rank, os.path.relpath(lg.path)))
        else:
            foreign.append(lg)
    if foreign:
        v.fail("%s took %s, and left out %d log(s) of that label from another host or tree: %s" % (
            what, "; ".join(os.path.relpath(lg.path) for lg in chosen), len(foreign),
            "; ".join("rank %d on %s: %s" % (lg.rank, lg.host(), os.path.relpath(lg.path)) for lg in foreign)))


def library_ok(lg, v, stub):
    lib = os.path.basename(lg.library())
    if lib in LIBRARIES:
        # the name alone is no proof: a gate's library links the host's verbs (meta.txt's links lines)
        want = VERBS.get(lg.header.get("os"))
        links = [f[len("links "):] for f in lg.meta.get("facts", []) if f.startswith("links ")]
        good = want is not None and any(want in ln for ln in links)
        what = "%s: MCDMA's %s, linking %s (meta.txt's links: %s)" % (
            lg.name(), lib, want or "the verbs of os %s" % lg.header.get("os"),
            "; ".join(ln.split()[0] for ln in links if ln.split()) or "none")
        if good or not stub:
            v.check(good, what)
        else:
            v.relax("%s, which it does not (--stub)" % what)
    elif stub:
        v.relax("%s: %s, an offline stand-in (--stub)" % (lg.name(), lib))
    else:
        v.fail("%s: ran %s, not MCDMA's libmcdma-fabric: an offline stand-in is no gate (--stub for an offline proof)" % (lg.name(), lib))


def warns_ok(lg, v, allow):
    """This rank's pre-flight WARNs: each shown, and those no gate passes with failed (notes with --allow-warn)."""
    for f in lg.meta.get("facts", []):
        if not f.startswith("WARN "):
            continue
        what = f[len("WARN "):]
        why = next((w for start, w in GATE_WARNS if what.startswith(start)), None)
        if why is None:
            v.note("%s: pre-flight WARN %s" % (lg.name(), what))
        elif allow:
            v.note("%s: pre-flight WARN (%s; --allow-warn) %s" % (lg.name(), why, what))
        else:
            v.fail("%s: pre-flight WARN (%s: no gate without --allow-warn) %s" % (lg.name(), why, what))


def arm_of(logs):
    for lg in logs:
        if lg.config.get("arm"):
            return lg.config.get("arm"), int(lg.config.get("arm_rank", "1"))
    return "none", 1


def reason_ok(arm, r, wait_s):
    """Whether a survivor's named failure is the one this arm causes, by MCDMA's reason (RESULT mcdma=)."""
    err, why, after = r.get("error"), r.get("mcdma", ""), as_float(r.get("after"))
    if arm == "exit":
        return err in NAMED and (GOODBYE in why or any(c in why for c in COMPLETION)), \
            "MCDMA's goodbye, or a completion error from a post that met the closed end"
    if arm in ("stop", "hang"):
        return err == "PeerTimeout" and why.startswith(LINK_UP) and after is not None and after >= wait_s - 0.5, \
            "PeerTimeout after at least %g s with the link up" % (wait_s - 0.5)
    if arm == "kill":
        return err in ("PeerDown", "Timeout") or (err == "PeerTimeout" and why.startswith(LINK_UP)), \
            "PeerDown or Timeout, or PeerTimeout with the link up"
    return err in NAMED, "any of %s" % ", ".join(sorted(NAMED))


def named_ok(lg, v, who, within, arm=None):
    r = lg.result or {}
    if r.get("status") != "FAIL-BY-NAME":
        v.fail("%s: expected a failure by name (PeerDown, PeerTimeout or Timeout), got %s" % (who, r.get("status", "no RESULT")))
        return
    err = r.get("error", "?")
    after = as_float(r.get("after"))
    v.check(err in NAMED and r.get("expected") == "yes" and after is not None and after <= within,
            "%s: failed by name: error.%s in %s s (bound %g s) at %s round %s; mcdma: %s" % (
                who, err, r.get("after"), within, r.get("test"), r.get("round"), r.get("mcdma")))
    if arm is not None:
        wait_s = as_float(lg.config.get("wait_s")) or 10.0
        good, want = reason_ok(arm, r, wait_s)
        v.check(good, "%s: the %s arm's reason (%s): error.%s, mcdma \"%s\"" % (who, arm, want, err, r.get("mcdma")))
    v.check(lg.exit == 3, "%s: exit %s (3: failed by name, not a hang or a crash)" % (who, lg.exit))


def fail_time(lg):
    return as_float(next((f.get("t_unix") for f in reversed(lg.fails) if f.get("t_unix")), None))


def startup_ok(lg, v):
    r = lg.result or {}
    code = r.get("code")
    v.check(r.get("expected") == "yes" and r.get("status", "").startswith("FAIL") and r.get("test") in STARTUP,
            "%s: startup outcome %s error.%s at %s after %s s, expected=%s" % (
                lg.name(), r.get("status"), r.get("error"), r.get("test"), r.get("after"), r.get("expected")))
    v.check(code is not None and lg.exit is not None and str(lg.exit) == code,
            "%s: exit %s, as its RESULT says (%s)" % (lg.name(), lg.exit, code))


def mtu_refusal_na(lg, v):
    """G5a counts only on an ACTIVE port whose active_mtu is below the path MTU: the N/A reason, or None."""
    if not any(MTU_LINE in t for t in lg.expect_log):
        return None
    state, active = lg.fact("port_state"), (lg.fact("active_mtu") or "").split()
    path = as_float(lg.config.get("mtu"))
    if not v.check(state is not None and "ACTIVE" in state,
                   "%s: meta.txt shows the port ACTIVE (port_state %s), so MCDMA's line is the MTU refusal, not a port that is down" % (lg.name(), state)):
        return None
    mtu = as_float(active[0]) if active else None
    if mtu is None or path is None:
        return "meta.txt gives no active_mtu (run_linux.sh reads it with ibv_devinfo): cannot tell the path MTU %s exceeds it" % lg.config.get("mtu")
    if mtu >= path:
        return "active_mtu %d is not below the path MTU %d on this port: it cannot show the MTU refusal" % (mtu, path)
    v.ok("%s: active_mtu %d is below the path MTU %d" % (lg.name(), mtu, path))
    return None


def startup_run(lg, v):
    na = mtu_refusal_na(lg, v)
    if na:
        v.na = "%s: %s" % (lg.name(), na)
        v.note("N/A   %s" % v.na)
        return
    startup_ok(lg, v)
    expect_log_ok(lg, v)


def expect_log_ok(lg, v):
    for want in lg.expect_log:
        hit = next((t for t in lg.text if want in t), None)
        v.check(hit is not None, "%s: the log says \"%s\"%s" % (lg.name(), want, "" if hit is None else ": " + hit.strip()[:200]))


def refuse_log_ok(lg, v):
    for bad in lg.refuse_log:
        hit = next((t for t in lg.text if bad in t), None)
        v.check(hit is None, "%s: the log never says \"%s\"%s" % (lg.name(), bad, "" if hit is None else "; it does: " + hit.strip()[:200]))


def suite_of(lg):
    """The tests and rounds this rank's SETTINGS asked for: (tests, rounds), rounds None for a duration run."""
    kv = lg.settings_kv
    tests = [t for t in kv.get("tests", "").split(",") if t]
    try:
        rounds = int(kv.get("rounds", "0"))
        duration = int(kv.get("duration_s", "0"))
    except ValueError:
        return tests, 0
    return tests, (None if duration > 0 else rounds)


def full_suite_ok(lg, v, partial):
    """Every test of every round passed on this rank, and the tests are the whole suite (unless --partial)."""
    tests, rounds = suite_of(lg)
    if sorted(tests) != sorted(SUITE):
        v.partial = True
    if not partial:
        v.check(sorted(tests) == sorted(SUITE) and len(tests) == len(SUITE),
                "%s: the whole suite (%d tests)%s" % (lg.name(), len(SUITE), "" if sorted(tests) == sorted(SUITE) else
                                                     "; it ran only %s (--partial accepts that, but it is no gate)" % ",".join(tests)))
    if rounds is None:  # a duration run: whole rounds, as many as rank 0's clock allowed
        rounds = 1 + max([k[1] for k in lg.passes] or [-1])
    want = {(t, r) for t in tests for r in range(rounds)}
    missing = sorted(want - set(lg.passes), key=lambda k: (k[1], k[0]))
    v.check(rounds > 0 and not missing and (lg.hash_all or {}).get("tests") == str(len(want)),
            "%s: %d test run(s) of %d passed (%d round(s)), HASH over %s%s" % (
                lg.name(), len(want) - len(missing), len(want), rounds, (lg.hash_all or {}).get("tests", "none"),
                "" if not missing else "; missing %s" % missing[:5]))


def suite_index(test):
    return SUITE.index(test) if test in SUITE else -1


def plan_files(label, settings_dir):
    """The plan's files for `label`: --settings DIR/<label>/*.json, or the bundle's settings/*/<label>/*.json."""
    if settings_dir:
        return sorted(glob.glob(os.path.join(settings_dir, label, "*.json"))), True
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings")
    return sorted(glob.glob(os.path.join(here, "*", label, "*.json")) + glob.glob(os.path.join(here, label, "*.json"))), False


def plan_ok(lg, v, settings_dir, need=True):
    """This rank's CONFIG and SETTINGS lines against its plan file; need=False: a label without plan files is a note."""
    files, given = plan_files(lg.label(), settings_dir)
    if not files:
        if given and need:
            v.fail("%s: no plan settings under %s/%s" % (lg.name(), settings_dir, lg.label()))
        elif given:
            v.note("%s: not checked against a plan (no %s/%s/)" % (lg.name(), settings_dir, lg.label()))
        else:
            v.note("%s: not checked against a plan (no settings/*/%s/ next to compare.py; give --settings DIR)" % (lg.name(), lg.label()))
        return
    match = []
    for f in files:
        try:
            s = json.load(open(f))
            link = s["links"][0]
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            continue
        if s.get("rank") == lg.rank and link.get("device") == lg.config.get("device") and link.get("via") == lg.config.get("via"):
            match.append((f, s, link))
    if len(match) != 1:
        v.fail("%s: %s plan settings for rank %d, device %s, via %s among %s" % (
            lg.name(), "no" if not match else "%d" % len(match), lg.rank, lg.config.get("device"), lg.config.get("via"),
            ", ".join(os.path.relpath(f) for f in files)))
        return
    f, s, link = match[0]
    get = lambda k: s.get(k, DEFAULTS.get(k))  # noqa: E731
    want = {
        "rounds": str(get("rounds")),
        "duration_s": str(get("duration_s")),
        "staging": str(get("staging_mib") * MIB),
        "mtu": str(link.get("mtu", 4096)),
        "kind": link.get("kind") or "any",
        "gid": str(link.get("gid")),
        "connect_timeout_s": str(get("connect_timeout_s")),
        "wait_s": str(get("wait_s")),
        "timeout_s": str(get("timeout_s")),
        "arm": get("arm"),
        "arm_rank": str(get("arm_rank")),
        "arm_test": get("arm_test"),
        "arm_step": str(get("arm_step")),
        "expect": ",".join(get("expect")) or "-",
    }
    if get("window_mib"):
        want["window"] = str(get("window_mib") * MIB)
    have = dict(lg.config)
    if lg.settings is not None:  # a pair run's handshake line: what both ranks agreed to run
        tests = get("tests")
        want.update({"signals": str(get("signals")), "bulk_mib": str(get("bulk_mib")),
                     "tests": ",".join(SUITE if tests == ["all"] else tests)})
        have.update({k: lg.settings_kv.get(k) for k in ("signals", "bulk_mib", "tests")})
    diff = ["%s=%s (plan %s)" % (k, have.get(k), w) for k, w in want.items() if have.get(k) != w]
    # what makes a startup run prove its cause (settings.py writes them with {device} filled in): the log's lines, as is
    for k, got in (("expect_log", lg.expect_log), ("refuse_log", lg.refuse_log)):
        if got != s.get(k, []):
            diff.append("%s=%s (plan %s)" % (k, got, s.get(k, [])))
    v.check(not diff, "%s: the plan's settings (%s)%s" % (lg.name(), os.path.relpath(f), "" if not diff else "; differ: " + ", ".join(diff)))


def drop_time(paths, label, given):
    """G4's drop time: --dropped-at, else the <label>.dropped_at file the operator wrote into a results tree."""
    if given is not None:
        return given, "--dropped-at"
    found = {}
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                if "%s.dropped_at" % label in files:
                    f = os.path.join(root, "%s.dropped_at" % label)
                    t = as_float(open(f).read().strip())
                    if t is not None:
                        found.setdefault(t, f)
    if len(found) == 1:
        (t, f), = found.items()
        return t, os.path.relpath(f)
    if found:
        return None, "several drop times: %s" % ", ".join("%s in %s" % (t, os.path.relpath(f)) for t, f in sorted(found.items()))
    return None, None


def external_ok(a, b, v, within, skew, dropped, min_rounds):
    """G4: an operator dropped the link at T; both ranks failed by name shortly after, and not before a few rounds."""
    t, src = dropped
    if not v.check(t is not None, "the operator's drop time: %s" % (
            "%.0f (%s)" % (t, src) if t is not None else src or "none (date +%%s > results/%s.dropped_at on the host whose link "
                                                                 "drops, right after the sudo line, or --dropped-at T)" % a.label())):
        t = None
    for lg in (a, b):
        named_ok(lg, v, lg.name(), within, "external")
        tf = fail_time(lg)
        if t is not None:
            v.check(tf is not None and t - skew <= tf <= t + within + skew,
                    "%s: failed %s s after the drop (window -%g .. %g s)" % (
                        lg.name(), "?" if tf is None else "%.2f" % (tf - t), skew, within + skew))
        v.check(lg.rounds_done >= min_rounds, "%s: %d whole round(s) before the failure (at least %d)" % (lg.name(), lg.rounds_done, min_rounds))


def compare_pair(logs, v, args):
    by_rank = {lg.rank: lg for lg in logs}
    if sorted(by_rank) != [0, 1] or len(logs) != 2:
        v.fail("need one rank 0 and one rank 1 log, have ranks %s" % sorted(lg.rank for lg in logs))
        return
    a, b = by_rank[0], by_rank[1]
    if a.host() == b.host():
        if args.same_host:
            v.relax("both ranks on %s (--same-host: an offline proof, not the two-host gate)" % a.host())
        else:
            v.fail("both ranks report host %s: the gate is two hosts (--same-host for an offline proof on one)" % a.host())
    if a.run_id or b.run_id:
        v.check(a.run_id is not None and a.run_id == b.run_id,
                "one run: RUN id %s on rank 0, %s on rank 1" % (a.run_id or "missing", b.run_id or "missing"))
    expecting = [lg for lg in (a, b) if lg.config.get("expect", "-") not in ("-", "")]
    if expecting:
        # a negative run: both ranks expected the same named failure at startup, and neither found anything wrong
        v.check(a.config.get("expect") == b.config.get("expect"),
                "both ranks expect the same failure: %s on rank 0, %s on rank 1" % (a.config.get("expect"), b.config.get("expect")))
        for lg in (a, b):
            startup_ok(lg, v)
            r = lg.result or {}
            v.check(r.get("status") != "FAIL-CHECK" and not lg.mismatch,
                    "%s: no FAIL-CHECK or MISMATCH (RESULT %s error.%s%s)" % (lg.name(), r.get("status"), r.get("error"),
                                                                           "; MISMATCH %s" % lg.mismatch[0] if lg.mismatch else ""))
        return
    if not (a.run_id or b.run_id):
        v.fail("one run: neither log has a RUN id (the ranks never swapped settings, or a tf-mcdma-pair before the run id)")
    v.check(a.settings is not None and a.settings == b.settings, "both ranks ran the same settings (tests, sizes, bounds, arm)")
    arm, victim = arm_of(logs)
    if a.link and b.link:
        same = a.link.get("kind") == b.link.get("kind")
        v.check(same and (args.kind is None or a.link.get("kind") == args.kind),
                "link kind %s on rank 0, %s on rank 1%s" % (a.link.get("kind"), b.link.get("kind"), "" if args.kind is None else " (want %s)" % args.kind))
        v.check(a.link.get("mtu") == b.link.get("mtu"), "path MTU %s on both" % a.link.get("mtu"))
    else:
        v.fail("a rank never connected (no LINK line)")
    # the host reference on each rank
    for lg in (a, b):
        bad = [k for k, p in lg.passes.items() if p.get("hash") != p.get("ref")]
        v.check(not bad and not lg.mismatch,
                "%s: %d test(s) passed, each equal to the host reference%s" % (lg.name(), len(lg.passes), "" if not bad else "; differ: %s" % bad))
        for m in lg.mismatch[:5]:
            v.note("%s MISMATCH %s" % (lg.name(), m))
    # the two ends against each other
    both = sorted(set(a.passes) & set(b.passes), key=lambda k: (k[1], list(a.passes).index(k)))
    only = sorted(set(a.passes) ^ set(b.passes))
    diff = [k for k in both if (a.passes[k]["hash"], a.passes[k]["ref"]) != (b.passes[k]["hash"], b.passes[k]["ref"])]
    if arm == "none":
        v.check(not only, "the same tests passed on both ranks%s" % ("" if not only else "; only one: %s" % only))
    v.check(not diff and both, "%d test(s) give the same result hash on both ranks%s" % (len(both), "" if not diff else "; differ: %s" % diff))
    keys = set(a.bufs) & set(b.bufs)
    keys = {k for k in keys if (k[0], k[1]) in a.passes and (k[0], k[1]) in b.passes}
    bdiff = sorted(k for k in keys if a.bufs[k] != b.bufs[k])
    if keys:
        v.check(not bdiff, "%d result buffers hash the same on both ranks%s" % (len(keys), "" if not bdiff else "; first differences: %s" % bdiff[:5]))
    if arm == "none":
        for lg in (a, b):
            full_suite_ok(lg, v, args.partial)
        ha, hb = (a.hash_all or {}).get("all"), (b.hash_all or {}).get("all")
        v.check(ha is not None and ha == hb, "run hash %s on both ranks" % (ha or "missing"))
        for lg in (a, b):
            r = lg.result or {}
            v.check(r.get("status") == "PASS" and lg.exit == 0, "%s: RESULT %s, exit %s" % (lg.name(), r.get("status", "missing"), lg.exit))
        return
    if arm == "external":
        external_ok(a, b, v, args.within, args.skew, drop_time(args.results, a.label(), args.dropped_at), args.min_rounds)
        return
    survivor = 1 - victim
    vic, sur = by_rank[victim], by_rank[survivor]
    v.note("arm %s, victim rank %d at %s step %s" % (arm, victim, a.config.get("arm_test"), a.config.get("arm_step")))
    va = vic.arm or {}
    at_arm = va.get("test") == a.config.get("arm_test") and va.get("step") == a.config.get("arm_step") and va.get("round") == "0"
    if arm in ("exit", "stop"):
        r = vic.result or {}
        v.check(at_arm and r.get("status") == "ARM-DONE" and vic.exit == 0,
                "%s (victim): %s at %s step %s, then exit %s" % (vic.name(), arm, va.get("test"), va.get("step"), vic.exit))
    elif arm == "kill":
        v.check(at_arm and vic.exit in (137, -9) and vic.result is None,
                "%s (victim): SIGKILL at %s step %s, exit %s" % (vic.name(), va.get("test"), va.get("step"), vic.exit))
    elif arm == "hang":
        r = vic.result or {}
        v.check(at_arm and r.get("status") == "HANG" and r.get("error") == "WatchdogHang" and r.get("expected") == "yes" and vic.exit == 4,
                "%s (victim): its watchdog named the hang: error.%s at %s after %s s, exit %s" % (vic.name(), r.get("error"), r.get("test"), r.get("after"), vic.exit))
    named_ok(sur, v, "%s (survivor)" % sur.name(), args.within, arm)
    # lockstep: the survivor cannot have failed before the victim's arm test (every test ends at a barrier)
    sr = sur.result or {}
    v.check(sr.get("round") == "0" and suite_index(sr.get("test")) >= suite_index(va.get("test")) >= 0,
            "%s (survivor): failed in %s round %s, at or after the arm's %s" % (sur.name(), sr.get("test"), sr.get("round"), va.get("test")))
    # and in a lockstep test, not before the arm's step (the kill arm's signal storm is no lockstep: its test is enough)
    if arm in ("exit", "stop") and va.get("test") in LOCKSTEP and sr.get("test") == va.get("test"):
        step = as_float((sur.fails[-1] if sur.fails else {}).get("step"))
        want = as_float(a.config.get("arm_step"))
        v.check(step is not None and want is not None and step >= want,
                "%s (survivor): failed at %s step %s, at or after the arm's step %s (an earlier step: a message lost "
                "before the arm, not the victim's doing)" % (sur.name(), sr.get("test"), "?" if step is None else "%d" % step,
                                                             a.config.get("arm_step")))
    # and by the two wall clocks: after the victim armed, within its delay and the bound
    ta, tf = as_float(va.get("t_unix")), fail_time(sur)
    hi = (as_float(va.get("delay_ms")) or 0) / 1000 + args.within
    v.check(ta is not None and tf is not None and -args.skew <= tf - ta <= hi + args.skew,
            "%s (survivor): failed %s s after the victim's arm by the two hosts' wall clocks (window -%g .. %g s)" % (
                sur.name(), "?" if ta is None or tf is None else "%.2f" % (tf - ta), args.skew, hi + args.skew))


def same_as(logs, others, v):
    def tests(ls):
        out = {}
        for lg in ls:
            for k, p in lg.passes.items():
                out.setdefault(k, set()).add(p["hash"])
        return out
    mine, theirs = tests(logs), tests(others)
    common = sorted(set(mine) & set(theirs))
    missing = sorted(set(mine) - set(theirs))
    extra = sorted(set(theirs) - set(mine))
    diff = [k for k in common if mine[k] != theirs[k] or len(mine[k]) != 1 or len(theirs[k]) != 1]
    v.check(common and not diff and not missing and not extra,
            "--same-as: %d test(s) give the same hashes as %s%s%s%s" % (
                len(common), ", ".join(sorted({o.dir for o in others})), "" if not diff else "; differ: %s" % diff,
                "" if not missing else "; not in the other run: %s" % missing[:5],
                "" if not extra else "; only in the other run: %s" % extra[:5]))
    # the same suite by its settings, not only the tests both passed: fewer rounds or a smaller window pass those too
    ref = logs[0]
    bad = []
    for lg in logs + others:
        if lg.settings is None:
            bad.append("%s has no SETTINGS line" % lg.name())
            continue
        d = ["%s=%s (%s)" % (k, lg.settings_kv.get(k), ref.settings_kv.get(k)) for k in SAME_AS
             if lg.settings_kv.get(k) != ref.settings_kv.get(k)]
        if d:
            bad.append("%s: %s" % (lg.name(), ", ".join(d)))
    v.check(not bad, "--same-as: both runs' SETTINGS give the same %s%s" % (", ".join(SAME_AS), "" if not bad else
                                                                             "; differ from %s: %s" % (ref.name(), "; ".join(bad))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, epilog="The checks, verdicts and exit codes: README.md beside this script.")
    ap.add_argument("results", nargs="+")
    ap.add_argument("--kind", choices=["cx5", "tb5"])
    ap.add_argument("--same-as", nargs="+", default=[])
    ap.add_argument("--same-as-label", help="the --same-as run's label, when it is not --label's (PLAN's Thunderbolt line)")
    ap.add_argument("--receipt")
    ap.add_argument("--label")
    ap.add_argument("--settings", help="the plan's settings for this host pair (settings.py's OUTDIR/<pair>)")
    ap.add_argument("--within", type=float, default=12.0, help="a named failure's bound from its step's start, s (12)")
    ap.add_argument("--skew", type=float, default=2.0, help="the two hosts' clocks may differ by this much, s (2)")
    ap.add_argument("--dropped-at", type=float, help="G4: the unix time the operator dropped the link")
    ap.add_argument("--min-rounds", type=int, default=3, help="G4: whole rounds before the drop (3)")
    ap.add_argument("--partial", action="store_true", help="accept a run of fewer tests than the whole suite (no gate)")
    ap.add_argument("--same-host", action="store_true", help="accept both ranks on one host (an offline proof, no gate)")
    ap.add_argument("--stub", action="store_true", help="accept MCDMA's offline stub or loop_fabric (an offline proof, no gate)")
    ap.add_argument("--allow-warn", action="store_true", help="the pre-flight WARNs that fail the gate (kext, relaxed ordering, "
                    "clock) as notes")
    args = ap.parse_args()
    if args.settings and not os.path.isdir(args.settings):
        print("compare.py: --settings %s is not a directory" % args.settings, file=sys.stderr)
        return 2
    if args.same_as_label and not args.same_as:
        print("compare.py: --same-as-label without --same-as", file=sys.stderr)
        return 2
    logs, dropped = find_logs(args.results, args.label)
    if not logs:
        print("compare.py: no rank*.log%s under %s" % ("" if args.label is None else " labelled %s" % args.label, " ".join(args.results)), file=sys.stderr)
        return 2
    labels = sorted({lg.label() for lg in logs})
    v = Verdict()
    v.note("run %s: %s" % ("/".join(labels), "; ".join(lg.name() for lg in sorted(logs, key=lambda x: x.rank))))
    dropped_ok(logs, dropped, v)
    if args.label is None and len({lg.rank for lg in logs}) != len(logs):
        v.fail("several logs of one rank under %s: give --label, or one run's directories" % " ".join(args.results))
    for lg in logs:
        library_ok(lg, v, args.stub)
        warns_ok(lg, v, args.allow_warn)
        plan_ok(lg, v, args.settings)
    if len(logs) == 1:
        lg = logs[0]
        if lg.config.get("expect", "-") not in ("-", ""):
            startup_run(lg, v)
        else:
            v.fail("one rank only, and it expected no failure: give both ranks' results")
    else:
        compare_pair(logs, v, args)
        for lg in logs:
            expect_log_ok(lg, v)
    for lg in logs:
        refuse_log_ok(lg, v)
    if args.same_as:
        # the other run by its own label (PLAN's Thunderbolt line: the pair's G1), held to its own plan
        olabel = args.same_as_label or args.label
        others, odropped = find_logs(args.same_as, olabel)
        if not others:
            v.fail("--same-as: no rank*.log%s under %s" % ("" if olabel is None else " labelled %s" % olabel, " ".join(args.same_as)))
        else:
            dropped_ok(others, odropped, v, "--same-as-label" if args.same_as_label else "--same-as --label")
            for lg in others:
                plan_ok(lg, v, args.settings, need=False)
            same_as(logs, others, v)
    print("\n".join(v.lines))
    word = v.word()
    print({"PARTIAL": "GATE PARTIAL (passed, but fewer tests than the whole suite: not the gate)",
           "N/A": "GATE N/A (this host cannot show it: %s)" % v.na,
           "OFFLINE PASS": "OFFLINE PASS (not the gate)"}.get(word, "GATE " + word))
    if args.receipt:
        if os.path.dirname(args.receipt):
            os.makedirs(os.path.dirname(args.receipt), exist_ok=True)
        with open(args.receipt, "w") as f:
            f.write(receipt(logs, v))
    return {"PASS": 0, "FAIL": 1, "PARTIAL": 3, "N/A": 4, "OFFLINE PASS": 5}[word]


if __name__ == "__main__":
    sys.exit(main())
