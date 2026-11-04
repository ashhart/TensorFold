#!/usr/bin/env python3
"""Every hardware run of the #1594 gate as settings for both hosts, from one hosts file, and the plan that runs them."""
import json
import os
import shlex
import sys

RUNS = [
    # name, rank of host a (or None: each host alone), extra settings, hosts that run, description ({a}, {b}: hosts)
    ("G1", 0, {}, "ab", "{a} rank 0, {b} rank 1: every collective equals the host reference on both ends, hashes equal"),
    ("G2", 1, {}, "ab", "ranks swapped, {b} rank 0 and {a} rank 1: each host is root and non-root"),
    ("G3a", 0, {"arm": "kill", "arm_rank": 1, "arm_test": "signals", "arm_step": 2000, "rounds": 1}, "ab", "{b} SIGKILLs itself mid signal storm: {a} must fail by name within 12 s"),
    ("G3b", 0, {"arm": "exit", "arm_rank": 1, "arm_test": "allreduce_bf16", "arm_step": 3, "rounds": 1}, "ab", "{b} closes its endpoint mid all-reduce: {a} must fail by name with MCDMA's goodbye (or a completion error)"),
    ("G3d", 0, {"arm": "stop", "arm_rank": 1, "arm_test": "pipes", "arm_step": 40, "rounds": 1}, "ab", "{b} stops posting mid pipes, its endpoint open: {a} must fail with PeerTimeout at 10 s, the link up"),
    # expect_log and refuse_log: what MCDMA or mcdma.zig must and must not have logged ({device}: this host's device)
    ("G5a", None, {"mtu": 4096, "expect": ["OpenFailed"], "connect_timeout_s": 30, "expect_log": ["MCDMA status 4 DEVICE", "path MTU above active MTU"]}, "b", "{b} alone at path MTU 4096: OpenFailed, logged as DEVICE. It counts only when meta.txt shows the port ACTIVE at an active_mtu below 4096; compare says N/A at 4096 (read it first: ibv_devinfo -d {b_device} | grep active_mtu)"),
    ("G5b", None, {"gid": -1, "kind": "cx5", "expect": ["NeedGidIndex"], "connect_timeout_s": 30}, "ab", "each host alone, gid -1 on a ConnectX link: NeedGidIndex before anything opens"),
    ("G5c", None, {"gid": -1, "kind": None, "expect": ["OpenFailed"], "connect_timeout_s": 30, "expect_log": ["MCDMA status 4 DEVICE", "mcdma-fabric: {device}: gid -1"], "refuse_log": ["port not active"]}, "ab", "each host alone, gid -1 and no declared kind: OpenFailed with MCDMA's own '<device>: gid -1' (a port that is down fails it)"),
    ("G5d", None, {"expect": ["ConnectFailed"], "connect_timeout_s": 30, "expect_log": ["MCDMA status 6 TIMEOUT"]}, "a", "{a} alone, no peer within connect_timeout_s 30: ConnectFailed, logged as TIMEOUT"),
    ("G4", 0, {"arm": "external", "duration_s": 300, "rounds": 1}, "ab", "an operator drops {b}'s link mid-run (the operator's sudo step below): both must fail by name soon after the recorded drop time"),
    ("G3c", 0, {"arm": "kill", "arm_rank": 0, "arm_test": "signals", "arm_step": 2000, "rounds": 1}, "ab", "{a} SIGKILLs itself mid signal storm: {b} must fail by name. LAST of all{after}"),
]

LINK_KEYS = {"mtu", "gid", "kind"}
NEED = ("os", "library", "device", "gid", "via", "mac", "gid_addr", "netdev")
# a Mac's prep line runs MCDMA's restore-rdma.py, then MCDMA's native checker: their paths come from the hosts file only
NEED_MAC = ("restore_rdma", "native_check")
# the checker's provider, where MCDMA installs it
PROVIDER = "/usr/local/lib/rdma/libmcdma-rdmav34.so"
BUNDLE = "tf-mcdma-pair-bundle"
USAGE = "usage: settings.py HOSTS.json OUTDIR [--root DIR]   (README.md beside it: the hosts file and the runs)"
TGZ = "build/mcdma-pair/tf-mcdma-pair-bundle.tgz"  # bundle.sh's, from the TensorFold root


def receipt(label):
    """A PLAN compare line's --receipt: the verdict with each rank's host facts and pre-flight WARNs, into results/."""
    return " --receipt results/%s.receipt.md" % label


def settings(hosts, me, other, rank, port, extra, label):
    h, peer = hosts["hosts"][me], hosts["hosts"][other]
    link = {
        "peer": 1 - rank,
        "device": h["device"],
        "via": h["via"],
        "port": port,
        "peer_port": port,
        "name": hosts.get("name", "gate1594"),
        "gid": h["gid"],
        "mtu": hosts.get("mtu", 1024),
        "kind": hosts.get("kind", "cx5"),
    }
    s = {
        "rank": rank,
        "library": h["library"],
        "label": label,
        "note": "%s on %s; the other end is %s" % (label, me, other),
        "links": [link],
        "expect_kind": hosts.get("kind", "cx5"),
        "rounds": hosts.get("rounds", 3),
        "window_mib": hosts.get("window_mib", 96),
        "connect_timeout_s": hosts.get("connect_timeout_s", 120),
        # the run script's pre-flight: the static neighbour this host must hold for its peer, and (a Mac) its kext
        "peer_gid_addr": peer["gid_addr"],
        "peer_mac": peer["mac"],
    }
    # and the address the peer's via pins for this host, which this host's meeting interface must hold
    if peer["via"].partition("/")[2]:
        s["meeting_addr"] = peer["via"].partition("/")[2]
    if h.get("expected_kext_uuid"):
        s["expected_kext_uuid"] = h["expected_kext_uuid"]
    for k, v in extra.items():
        if isinstance(v, list):
            v = [x.format(device=h["device"]) for x in v]
        if k in LINK_KEYS:
            link[k] = v
            if k == "kind":
                s["expect_kind"] = v
        else:
            s[k] = v
    if s.get("expect_kind") is None:
        s.pop("expect_kind")
    if link.get("kind") is None:
        link.pop("kind")
    return s


def port_of(h, name):
    """A host with several ConnectX ports ("ports": netdev -> device, mac, gid_addr) runs on the one "netdev" names."""
    ports = h.get("ports")
    if not ports:
        return h
    if h.get("netdev") not in ports:
        sys.exit("settings.py: host %s: netdev %s is not one of its ports (%s)" % (name, h.get("netdev"), ", ".join(sorted(ports))))
    both = sorted(k for k in ports[h["netdev"]] if k in h)
    if both:
        sys.exit("settings.py: host %s gives %s both beside and under \"ports\": once, under ports" % (name, ", ".join(both)))
    return dict(h, **ports[h["netdev"]])


def check_hosts(hosts):
    names = list(hosts.get("hosts", {}))
    if len(names) != 2:
        sys.exit("settings.py: HOSTS.json names two hosts")
    for n in names:
        h = hosts["hosts"][n]
        missing = [k for k in NEED if k not in h]
        if missing:
            sys.exit("settings.py: host %s has no %s (gid too: 0 on a Mac, the RoCE v2 index on Linux; no default)" % (n, ", ".join(missing)))
        missing = [k for k in NEED_MAC if h["os"] == "macos" and not h.get(k)]
        if missing:
            sys.exit("settings.py: host %s is a Mac and has no %s: the path of MCDMA's restore-rdma.py and cx5-native-check on it "
                     "(no default)" % (n, ", ".join(missing)))
        if not isinstance(h["gid"], int) or h["gid"] < 0:
            sys.exit("settings.py: host %s: gid is a GID index, 0 or more, not %r" % (n, h["gid"]))
        if h["os"] == "macos" and h["gid"] != 0:
            sys.exit("settings.py: host %s is a Mac: MCDMA's kext has GID index 0 only, so gid is 0, not %d" % (n, h["gid"]))
    return names


def ssh(h, name):
    """A host's ssh and scp (options, target): "ssh_opts", default none, and "ssh", an alias, default its name."""
    return h.get("ssh_opts", ""), h.get("ssh", name)


def prep(hosts, me, other):
    """The operator's sudo line for `me`'s ConnectX port to reach `other`, then the read-back that shows it took."""
    h, p = hosts["hosts"][me], hosts["hosts"][other]
    q = shlex.quote
    if h["os"] == "macos":
        restore, checker = h["restore_rdma"], h["native_check"]
        return [
            "%s$ sudo /usr/bin/python3 %s --interface %s --expected-mac %s --peer-gid %s --peer-mac %s" % (
                me, restore, h["netdev"], h["mac"], p["gid_addr"], p["mac"]),
            "     (restore-rdma.py re-adds %s's own GID alias, brings %s up and adds one static neighbour with ndp -s; a"
            % (me, h["netdev"]),
            "     second peer's line adds its entry beside the first one, it does not replace it). If it stops at ndp -s with",
            "     'ndp: writing to routing socket: File exists', that peer's entry is already there (the read-back below",
            "     shows it), but restore-rdma.py skipped its last step, the GID check: run it by hand,",
            "%s$ %s --provider %s --require-gid" % (me, checker, PROVIDER),
            "%s$ ndp -an | grep -i %s      (shows %s, permanent)" % (me, q("%s%%%s" % (p["gid_addr"], h["netdev"])), p["mac"]),
            "%s$ ifconfig %s | head -1; ioreg -r -c MCDMACX5Native -l | grep -E '\"MCDMA(AddressInterface|PortActive)\"'" % (me, h["netdev"]),
            "     (flags UP and RUNNING; MCDMAPortActive Yes under MCDMAAddressInterface \"%s\")" % h["netdev"],
        ] + [line for n in sorted(h.get("ports", {})) if n != h["netdev"] for line in (
            # which port is cabled may be known only now: the other one's values, and how to switch
            "     This plan is for %s. If %s is the cabled port instead (MCDMAPortActive Yes under \"%s\" above," % (h["netdev"], n, n),
            "     ifconfig %s RUNNING): set %s's \"netdev\" to \"%s\" in every hosts file that names it" % (n, me, n),
            "     (the hosts files the line below passes bundle.sh; device %s, MAC %s and GID %s" % (h["ports"][n]["device"], h["ports"][n]["mac"], h["ports"][n]["gid_addr"]),
            "     come from its \"ports\"), rebuild the bundle with zig 0.17 (Homebrew's zig 0.15 is too old):",
            "%s$ %s" % (me, os.environ.get("TF_MCDMA_PAIR_REBUILD") or
                        "cd <the TensorFold root> && ZIG=<zig 0.17> TF_MCDMA_SRC=<the MCDMA tree, e725b45 or later> "
                        "tools/zig/mcdma_pair/bundle.sh <every hosts file>"),
            "     and start again from Copy with the new PLAN: every %s settings file and both hosts' prep lines change." % me,
            "     With neither port active, the cable or the switch port first: without a link every %s run but G5b fails" % me,
            "     at open.")]
    return [
        "%s$ sudo ip -6 neigh replace %s lladdr %s nud permanent dev %s" % (me, p["gid_addr"], p["mac"], h["netdev"]),
        "%s$ ip -6 neigh show to %s dev %s      (shows lladdr %s PERMANENT)" % (me, p["gid_addr"], h["netdev"], p["mac"]),
    ]


def main():
    args = sys.argv[1:]
    root = None
    if len(args) == 4 and args[2] == "--root":
        root = args[3]
        args = args[:2]
    if len(args) != 2:
        sys.exit(USAGE)
    given = json.load(open(args[0]))
    hosts = dict(given, hosts={n: port_of(h, n) for n, h in given.get("hosts", {}).items()})
    a, b = check_hosts(hosts)
    pair = hosts.get("pair", "%s-%s" % (a, b))
    prefix = hosts.get("label_prefix", "")
    out = os.path.join(args[1], pair)
    root = os.path.abspath(root or os.path.dirname(os.path.abspath(args[1])))
    rel = lambda p: os.path.relpath(os.path.abspath(p), root)  # noqa: E731
    port = int(hosts.get("port", 7610))
    kind = hosts.get("kind", "cx5")
    hb = hosts["hosts"][b]
    opts, target = ssh(hb, b)
    sshb = " ".join(x for x in ("ssh", opts, target) if x)
    scpb = " ".join(x for x in ("scp", opts) if x)
    free = hb.get("free") is True
    fresh = "~/tf-mcdma-pair-%s-<UTC stamp>/%s" % (pair, BUNDLE)
    plan = ["# The #1594 gate's hardware runs, pair %s: %s and %s (path MTU %s, kind %s, labels %sG1 ..)." % (
                pair, a, b, hosts.get("mtu", 1024), kind, prefix),
            "# Copy's two lines run on %s from the TensorFold root (where bundle.sh wrote the .tgz). Every line" % a,
            "# after them runs from the unpacked bundle, and its paths are relative to it:",
            "#   on %s: the fresh directory Copy's second line unpacks into and cds to (%s)" % (a, fresh),
            "#   on %s: ~/%s (reach it with: %s)" % (b, BUNDLE, sshb)]
    for n in (a, b):
        if hosts["hosts"][n].get("caution"):
            plan.append("# %s: %s" % (n, hosts["hosts"][n]["caution"]))
    stamp = "$(date -u +%Y%m%dT%H%M%SZ)"
    plan += [
        "#",
        # an earlier bundle on b moves aside (as bundle.sh moves earlier settings): its results never reach Collect
        "# Copy:    %s$ %s %s %s: && %s 'if [ -e %s ]; then mv %s %s.before-%s; fi && tar xzf %s.tgz'" % (
            a, scpb, TGZ, target, sshb, BUNDLE, BUNDLE, BUNDLE, stamp, BUNDLE),
        "#          %s$ d=~/tf-mcdma-pair-%s-%s && mkdir \"$d\" && tar -C \"$d\" -xzf %s && cd \"$d/%s\" && pwd" % (
            a, pair, stamp, TGZ, BUNDLE),
        "#          (%s's earlier bundle moves aside to %s.before-<stamp>; %s unpacks into a fresh directory, never the" % (b, BUNDLE, a),
        "#          build tree's bundle, whose results/ and proofs an earlier run left there)",
        "# Build:   on each host, in the bundle: ./build_mcdma.sh   (libmcdma-fabric into mcdma/build/)",
        "# Check:   optional, on each host: ./stub_proof.sh   (MCDMA's offline stub and loop_fabric on free loopback ports; it",
        "#          writes proof/, never results/, and labels its runs stub-* and loop-*, which no compare line below takes)",
        "# Prep:    the operator, with sudo, before the first run, and again after a link drop or a reboot; each with its read-back:",
    ]
    plan += ["#   " + line for line in prep(hosts, a, b) + prep(hosts, b, a)]
    plan += [
        "# Run:     each run's lines below, on each host, from the bundle; either end may start first.",
        "# Collect: on %s after each run (into results-%s/results/):" % (a, b),
        "#          mkdir -p results-%s && %s 'tar -C %s -cf - results' | tar -C results-%s -xf -" % (b, sshb, BUNDLE, b),
        "",
    ]
    sdir = rel(out)
    for i, (base, rank_a, extra, who, what) in enumerate(RUNS):
        if base == "G4" and not free:
            continue  # a link drop only on a box marked free, never on a host other work depends on
        name = prefix + base
        if base == "G5a" and isinstance(hb.get("active_mtu"), int) and hb["active_mtu"] >= extra["mtu"]:
            # MCDMA opens at a path MTU the port's active MTU reaches: no refusal to show, only compare.py's N/A
            plan.append("== %s: N/A on the %s (active_mtu %d)" % (name, b.upper(), hb["active_mtu"]))
            plan.append("  (the hosts file records %s's active_mtu %d, not below this run's path MTU %d: MCDMA opens, so there is" % (
                b, hb["active_mtu"], extra["mtu"]))
            plan.append("  no refusal to show. If ibv_devinfo -d %s | grep active_mtu now shows less, set it there and re-bundle.)" % hb["device"])
            plan.append("")
            continue
        os.makedirs(os.path.join(out, name), exist_ok=True)
        p = port + 10 * i
        after = " (a killed Mac process may leave its ConnectX port needing restore-rdma.py)"
        if base == "G3c" and not free:
            after += ", and only after the rehearsal's G3c passed with a free box"
        plan.append("== %s: %s" % (name, what.format(a=a, b=b, b_device=hb["device"], after=after)))
        for host in (a, b):
            if (host == a and "a" not in who) or (host == b and "b" not in who):
                continue
            other = b if host == a else a
            if rank_a is None:
                rank = 0
            else:
                rank = rank_a if host == a else 1 - rank_a
            s = settings(hosts, host, other, rank, p, extra, name)
            path = os.path.join(out, name, "%s.json" % host)
            with open(path, "w") as f:
                json.dump(s, f, indent=1)
                f.write("\n")
            script = hosts["hosts"][host].get("script", "./run_macos.sh" if hosts["hosts"][host].get("os") == "macos" else "./run_linux.sh")
            plan.append("  %s$ %s %s" % (host, script, rel(path)))
        if base == "G4":
            drop = "sudo ip link set %s down" if hb["os"] != "macos" else "sudo ifconfig %s down"
            plan.append("  drop, the operator on %s once both logs show at least 3 ROUND lines (about 60 s in), from the bundle:" % b)
            plan.append("  %s$ %s; date +%%s > results/%s.dropped_at" % (b, drop % hb["netdev"], name))
            plan.append("  then collect, and: ./compare.py --label %s --kind %s --settings %s results results-%s%s" % (name, kind, sdir, b, receipt(name)))
            plan.append("        (it reads results-%s/results/%s.dropped_at; both ranks must fail between 2 s before it and 14 s after)" % (b, name))
            up = "sudo ip link set %s up" if hb["os"] != "macos" else "sudo ifconfig %s up"
            plan.append("  after: %s$ %s, then %s's prep line and read-back again (a link down drops the permanent neighbour);" % (b, up % hb["netdev"], b))
            plan.append("         rerun %sG1's lines with --label %sG4-after on both hosts, and (a fresh run passes, with %sG1's hashes" % (prefix, prefix, prefix))
            plan.append("         and suite; %sG1's logs are held to their plan):" % prefix)
            plan.append("         ./compare.py --label %sG4-after --kind %s results results-%s --same-as results results-%s --same-as-label %sG1%s" % (
                prefix, kind, b, b, prefix, receipt(prefix + "G4-after")))
        elif rank_a is None:
            plan.append("  then, on each host that ran it: ./compare.py --label %s --settings %s results%s" % (name, sdir, receipt(name)))
            plan.append("        (the named failure it expected)")
        else:
            plan.append("  then: ./compare.py --label %s --kind %s --settings %s results results-%s%s" % (name, kind, sdir, b, receipt(name)))
        if base == "G3b":
            # MCDMA's goodbye is one UDP datagram over the meeting LAN: lost, the survivor times out with the link up
            plan.append("  (if its only FAIL is the exit arm's reason, PeerTimeout with mcdma \"link up\": MCDMA's one UDP BYE was")
            plan.append("  likely lost over Wi-Fi. If its only FAIL is the survivor failing at allreduce_bf16 step %d, one before the" % (extra["arm_step"] - 1))
            plan.append("  arm's, with mcdma \"said goodbye\" (compare.py calls it a message lost before the arm): the BYE arrived")
            plan.append("  before the survivor's flush of that step, a race no stress run has shown. Either is rare: rerun %s once" % name)
            plan.append("  before calling the transport broken.)")
        plan.append("")
    if not free:
        plan.append("No G4 here: %s is not marked free (a link drop would take down a host other work depends on). G4 runs in the" % b)
        plan.append("plan of a pair whose second host is free.")
        plan.append("")
    # its own label: the Thunderbolt runs have no plan files here, and G1's logs are held to G1's
    g1, tb = prefix + "G1", prefix + "tb-G1"
    plan.append("Thunderbolt unchanged: the same suite between two Thunderbolt 5 Macs gives the same per-test hashes as %s." % g1)
    plan.append("  Run %s's settings on them (%s/%s/: compare.py holds both runs to one suite, rounds and window)," % (g1, sdir, g1))
    plan.append("  each Mac with its own Thunderbolt link (device, via; kind and expect_kind tb5), and --label %s: a label" % tb)
    plan.append("  with no plan files, which compare.py notes. With their results trees as TB-RESULTS (%s's logs are held" % g1)
    plan.append("  to %s/%s/ again):" % (sdir, g1))
    plan.append("  ./compare.py --label %s --kind tb5 TB-RESULTS... --same-as results results-%s --same-as-label %s%s" % (tb, b, g1, receipt(tb)))
    with open(os.path.join(out, "PLAN.txt"), "w") as f:
        f.write("\n".join(plan) + "\n")
    with open(os.path.join(out, "hosts.json"), "w") as f:
        json.dump(given, f, indent=1)
        f.write("\n")
    print(os.path.join(out, "PLAN.txt"))


if __name__ == "__main__":
    main()
