//! tf-tp-check: the mixed-TP gate (#1236 rules 1, 2 and 4) over the dumps each rank writes (format in check.zig).
const std = @import("std");
const check = @import("tp_check.zig");
const check_file = @import("tp_check_file.zig");

const usage =
    \\usage: tf-tp-check reference RANK0.tpdump RANK1.tpdump ...   every collective equals the host reference (rule 1)
    \\       tf-tp-check repeat RUN_A.tpdump RUN_B.tpdump          one rank's two runs, byte for byte (rule 2)
    \\       tf-tp-check agree MIXED.tpdump SOLO.tpdump            divergence and greedy agreement vs one box (rule 4)
    \\exit 0: reference and repeat hold; agree's report is valid (rule 4 has no bound, so 0 is not a pass)
    \\exit 1: the rule does not hold, or agree cannot give valid numbers; 2: bad arguments or an unreadable dump
    \\
;

fn bad(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("tf-tp-check: " ++ fmt ++ "\n", args);
    std.process.exit(2);
}

fn load(a: std.mem.Allocator, io: std.Io, path: []const u8) check.Dump {
    const bytes = check_file.read(a, io, path) catch |e| bad("{s}: {s}", .{ path, @errorName(e) });
    return check.parse(a, bytes) catch |e| bad("{s}: {s}", .{ path, @errorName(e) });
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2 or std.mem.eql(u8, args[1], "--help") or std.mem.eql(u8, args[1], "-h")) {
        std.debug.print("{s}", .{usage});
        std.process.exit(if (args.len < 2) 2 else 0);
    }
    var buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(io, &buf);
    const w = &out.interface;
    const cmd = args[1];
    var ok = false;
    if (std.mem.eql(u8, cmd, "reference")) {
        if (args.len < 4) bad("reference needs a dump per rank, rank 0 first", .{});
        const dumps = try a.alloc(check.Dump, args.len - 2);
        for (args[2..], dumps) |p, *d| d.* = load(a, io, p);
        const r = try check.checkReference(a, dumps);
        try w.print("rule 1, host reference: {d} collectives, {d} receive buffers over {d} ranks", .{ r.calls, r.buffers, dumps.len });
        for (dumps) |d| try w.print(" [{s}]", .{d.header.label});
        try w.print("\n  {s}\n", .{dumps[0].header.provenance});
        if (r.first) |m| {
            try w.print("FAIL: {d} mismatches; first: seq {d}, layer {d}, rank {d}: {s}", .{ r.mismatches, m.seq, m.layer, m.rank, m.why });
            if (std.mem.eql(u8, m.why, "value")) try w.print(", element {d}: got 0x{x}, want 0x{x}", .{ m.element, m.got, m.want });
            if (std.mem.eql(u8, m.why, "length") or std.mem.eql(u8, m.why, "call count")) try w.print(": got {d}, want {d}", .{ m.got, m.want });
            try w.writeAll("\n");
        } else if (r.calls == 0) {
            try w.writeAll("FAIL: the dumps hold no collectives\n");
        } else {
            try w.writeAll("PASS: every receive buffer is bit-identical to the host reference\n");
            ok = true;
        }
    } else if (std.mem.eql(u8, cmd, "repeat")) {
        if (args.len != 4) bad("repeat needs two dumps of the same rank", .{});
        const x = load(a, io, args[2]);
        const y = load(a, io, args[3]);
        if (x.header.rank != y.header.rank) bad("the dumps are ranks {d} and {d}", .{ x.header.rank, y.header.rank });
        const r = check.checkRepeat(x, y);
        try w.print("rule 2, same bytes run after run: rank {d} [{s}], {d} collectives and {d} logits rows vs {d} and {d}\n", .{ x.header.rank, x.header.label, x.collectives.len, x.logits.len, y.collectives.len, y.logits.len });
        try w.print("  run A: {s}\n  run B: {s}\n", .{ x.header.provenance, y.header.provenance });
        if (check.repeatPassed(r)) {
            try w.print("PASS: all {d} records identical\n", .{r.records});
            ok = true;
        } else {
            try w.writeAll("FAIL:");
            if (!r.same_shape) try w.writeAll(" record counts differ;");
            if (r.first_collective) |i| try w.print(" first differing collective is record {d} (seq {d}, layer {d});", .{ i, x.collectives[i].seq, x.collectives[i].layer });
            if (r.first_logits) |i| try w.print(" first differing logits row is prompt {d} position {d};", .{ x.logits[i].prompt, x.logits[i].position });
            try w.writeAll("\n");
        }
    } else if (std.mem.eql(u8, cmd, "agree")) {
        if (args.len != 4) bad("agree needs the mixed run's dump, then one box's", .{});
        const mixed = load(a, io, args[2]);
        const solo = load(a, io, args[3]);
        var r = try check.checkAgree(a, mixed, solo);
        try w.print("rule 4, mixed [{s}] ({s}) vs one box [{s}]: {d} prompts, {d} positions\n", .{ mixed.header.label, @tagName(r.mode), solo.header.label, r.prompts, r.positions });
        try w.print("  mixed: {s}\n  solo:  {s}\n", .{ mixed.header.provenance, solo.header.provenance });
        switch (r.mode) {
            .greedy => {
                try w.print("greedy tokens equal before each prompt's first split: {d}; prompts that split: {d} of {d}\n", .{ r.greedy_same, r.splits.len, r.prompts });
                for (r.splits[0..@min(r.splits.len, 64)]) |sp| try w.print("  prompt {d}: first split at position {d}\n", .{ sp.prompt, sp.position });
                if (r.splits.len > 64) try w.print("  ... {d} more\n", .{r.splits.len - 64});
                try w.writeAll("divergence below covers each prompt up to its first split (same prefix); use a teacher-forced run for the rest\n");
            },
            .teacher_forced => {
                try w.print("tokens equal: {d}/{d}; fed the solo run's tokens: {s}\n", .{ r.greedy_same, r.positions, if (r.fed_ok == true) "yes (hash matches)" else "NO (fed_hash differs)" });
            },
        }
        try w.print("compared {d} positions: top-1 equal {d}; |logit diff| max {e:.3} mean {e:.3}; KL(one box || mixed) mean {e:.3} max {e:.3} nats", .{ r.compared, r.top1_same, r.max_abs, r.mean_abs, r.kl_mean, r.kl_max });
        if (r.skipped > 0) try w.print("; {d} positions skipped (vocab size or non-finite logits)", .{r.skipped});
        try w.writeAll("\n");
        if (r.missing > 0) {
            try w.print("{d} solo positions have no mixed record", .{r.missing});
            for (r.gaps[0..@min(r.gaps.len, 64)]) |g| try w.print("; prompt {d} stops at the gap at position {d}", .{ g.prompt, g.position });
            try w.writeAll("\n");
        }
        if (r.missing > 0) {
            try w.writeAll("FAIL: the mixed dump is missing positions the solo run has, so the report is incomplete\n");
        } else if (r.positions == 0) {
            try w.writeAll("FAIL: no (prompt, position) appears in both dumps\n");
        } else if (r.fed_ok == false) {
            try w.writeAll("FAIL: this teacher-forced run was not fed this solo run's tokens, so its divergence is not reported\n");
        } else ok = true;
    } else bad("unknown command {s}\n{s}", .{ cmd, usage });
    try w.flush();
    std.process.exit(if (ok) 0 else 1);
}
