//! GLM-5.3-Flash replies on the native engine: each prompt's greedy reply at every draft depth, its hash, the first
//! difference from depth 0 and from the prompt's reference tokens, and the speeds.
//! tf-glm-run MODEL_DIR PROMPTS_JSON ({"prompts": [{"name": ..., "ids": [...], "expect": [...]}]})
//! GLM_DEPTHS (default "0,3"), GLM_MAX (64), GLM_CAP (prompt + reply room, default 8192), GLM_RUNS (1),
//! GLM_OUT (write each prompt's first-depth reply as JSON), GLM_VS (compare replies with a GLM_OUT file),
//! GLM_CANCEL_TEST (cancel a reply mid-prompt, then the next fresh reply must equal the plain one),
//! GLM_LAYERS (the first N layers only, with the MTP layer and head), GLM_REF_STRICT (a reference difference fails),
//! GLM_EP (expert parallel: this Mac's link settings; run the same command on both Macs),
//! GLM_TRACE=NAME:STEPS:PATH (every call of NAME's plain reply, sublayer by sublayer, feeding its expected tokens),
//! GLM_FORCED (each prompt's teacher-forced agreement with its expected tokens, before the replies).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const glm = tf.glm;

const Collect = struct {
    gpa: std.mem.Allocator,
    toks: std.ArrayList(u32) = .empty,
    cancel_at: usize = std.math.maxInt(usize), // the cancel check that answers true (0 = the first)
    checks: usize = 0,

    fn prefilled(_: *anyopaque) void {}
    fn tokens(ctx: *anyopaque, t: []const u32) bool {
        const c: *Collect = @ptrCast(@alignCast(ctx));
        c.toks.appendSlice(c.gpa, t) catch {};
        return false;
    }
    fn cancelled(ctx: *anyopaque) bool {
        const c: *Collect = @ptrCast(@alignCast(ctx));
        c.checks += 1;
        return c.checks > c.cancel_at;
    }
};

fn env(name: [:0]const u8, default: []const u8) []const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else default;
}

fn hash(t: []const u32) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(t));
}

fn firstDiff(a: []const u32, b: []const u32) ?usize {
    for (0..@min(a.len, b.len)) |i| if (a[i] != b[i]) return i;
    return if (a.len == b.len) null else @min(a.len, b.len);
}

fn ints(gpa: std.mem.Allocator, v: std.json.Value) ![]u32 {
    const out = try gpa.alloc(u32, v.array.items.len);
    for (v.array.items, out) |x, *o| o.* = @intCast(x.integer);
    return out;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 3) {
        std.debug.print("usage: tf-glm-run MODEL_DIR PROMPTS_JSON\n", .{});
        std.process.exit(2);
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const max = try std.fmt.parseInt(usize, env("GLM_MAX", "64"), 10);
    const cap = try std.fmt.parseInt(u32, env("GLM_CAP", "8192"), 10);
    const runs = try std.fmt.parseInt(usize, env("GLM_RUNS", "1"), 10);
    var depths: std.ArrayList(usize) = .empty;
    var it = std.mem.tokenizeScalar(u8, env("GLM_DEPTHS", "0,3"), ',');
    while (it.next()) |d| try depths.append(arena, try std.fmt.parseInt(usize, d, 10));
    const pf = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(arena, "{s}", .{args[2]}, 0));
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, pf.bytes[0..pf.size], .{});
    const e = try glm.engine.Engine.load(gpa, args[1], cap);
    defer e.deinit();
    std.debug.print("loaded in {d:.1} s: {d} of {d} layers, routed experts {d}-{d}, {d:.1} GB of weights, MTP head {s}\n", .{ e.load_seconds, e.c.run, e.c.layers, e.c.own[0], e.c.own[1] - 1, @as(f64, @floatFromInt(e.w.bytes)) / 1e9, if (e.hasMtp()) "yes" else "no" });
    const strict = std.c.getenv("GLM_REF_STRICT") != null;
    const eos = e.c.eos[0..e.c.eos_n];
    if (std.c.getenv("GLM_CAPTURE")) |path| { // the first prompt's first window, sublayer by sublayer (glm_ref.py --capture)
        const first = doc.object.get("prompts").?.array.items[0];
        try e.capture(try ints(arena, first.object.get("ids").?), std.mem.span(path));
    }
    if (std.c.getenv("GLM_TRACE")) |spec| { // NAME:STEPS:PATH, glm_ref.py --trace NAME STEPS PATH's twin
        var parts = std.mem.splitScalar(u8, std.mem.span(spec), ':');
        const want = parts.next() orelse return error.BadTrace;
        const steps = try std.fmt.parseInt(usize, parts.next() orelse return error.BadTrace, 10);
        const out_path = parts.rest();
        for (doc.object.get("prompts").?.array.items) |p| if (std.mem.eql(u8, p.object.get("name").?.string, want)) {
            try e.trace(try ints(arena, p.object.get("ids").?), try ints(arena, p.object.get("expect").?), steps, out_path);
        };
    }
    if (std.c.getenv("GLM_FORCED") != null) for (doc.object.get("prompts").?.array.items) |p| {
        const want = try ints(arena, p.object.get("expect") orelse continue);
        const r = try e.forced(try ints(arena, p.object.get("ids").?), want);
        std.debug.print("{s} forced: {d} of {d} picks equal the reference's", .{ p.object.get("name").?.string, r.same, want.len });
        if (r.first) |i| std.debug.print(", first differs at {d}\n", .{i}) else std.debug.print("\n", .{});
    };
    var failures: usize = 0;
    const vs: ?std.json.Value = if (std.c.getenv("GLM_VS")) |path| blk: {
        const f = try mtl.MappedFile.open(path);
        break :blk try std.json.parseFromSliceLeaky(std.json.Value, arena, f.bytes[0..f.size], .{});
    } else null;
    var saved: std.ArrayList(u8) = .empty;
    try saved.appendSlice(arena, "{");
    for (doc.object.get("prompts").?.array.items) |p| {
        const name = p.object.get("name").?.string;
        const ids = try ints(arena, p.object.get("ids").?);
        const expect: ?[]u32 = if (p.object.get("expect")) |x| try ints(arena, x) else null;
        var plain: ?[]u32 = null;
        for (depths.items) |d| for (0..runs) |run| {
            var col: Collect = .{ .gpa = gpa };
            defer col.toks.deinit(gpa);
            const r = try e.generate(ids, max, eos, d, .{ .ctx = &col, .prefilled = Collect.prefilled, .tokens = Collect.tokens, .cancelled = Collect.cancelled });
            const toks = col.toks.items;
            const tps = @as(f64, @floatFromInt(toks.len -| 1)) / @max(r.decode_seconds, 1e-9);
            std.debug.print("{s} depth {d} run {d}: {d} prompt tokens in {d:.2} s ({d:.0} tok/s), {d} tokens at {d:.1} tok/s, {d} rounds, {d}/{d} drafts kept, hash {x:0>16}\n", .{ name, d, run, ids.len, r.prompt_seconds, @as(f64, @floatFromInt(ids.len)) / @max(r.prompt_seconds, 1e-9), toks.len, tps, r.rounds, r.accepted, r.drafted, hash(toks) });
            std.debug.print("  tokens: {any}\n", .{toks[0..@min(toks.len, 48)]});
            if (plain == null) plain = try arena.dupe(u32, toks) else if (firstDiff(plain.?, toks)) |at| {
                failures += 1;
                std.debug.print("  DIFFERS from depth {d} at token {d}\n", .{ depths.items[0], at });
            } else std.debug.print("  equal to depth {d}\n", .{depths.items[0]});
            if (vs) |other| if (other.object.get(name)) |theirs| {
                const want = try ints(arena, theirs);
                if (firstDiff(want, toks)) |at| {
                    std.debug.print("  vs GLM_VS: first token {s}, first difference at token {d} of {d}\n", .{ if (at == 0) "DIFFERS" else "equal", at, @min(want.len, toks.len) });
                } else std.debug.print("  vs GLM_VS: all {d} tokens equal\n", .{toks.len});
            };
            if (run == 0 and d == depths.items[0]) {
                if (saved.items.len > 1) try saved.append(arena, ',');
                try saved.print(arena, "\"{s}\":[", .{name});
                for (toks, 0..) |t, i| try saved.print(arena, "{s}{d}", .{ if (i > 0) "," else "", t });
                try saved.append(arena, ']');
            }
            if (expect) |want| {
                if (firstDiff(want[0..@min(want.len, toks.len)], toks[0..@min(want.len, toks.len)])) |at| {
                    if (strict) failures += 1;
                    std.debug.print("  reference: first difference at token {d} of {d}\n", .{ at, @min(want.len, toks.len) });
                } else std.debug.print("  reference: {d} of {d} tokens equal\n", .{ @min(want.len, toks.len), want.len });
            }
        };
        if (std.c.getenv("GLM_CANCEL_TEST") != null) { // a reply cancelled after its first prompt window, then a fresh one
            var cut: Collect = .{ .gpa = gpa, .cancel_at = 1 };
            defer cut.toks.deinit(gpa);
            const rc = try e.generate(ids, max, eos, depths.items[0], .{ .ctx = &cut, .prefilled = Collect.prefilled, .tokens = Collect.tokens, .cancelled = Collect.cancelled });
            var again: Collect = .{ .gpa = gpa };
            defer again.toks.deinit(gpa);
            _ = try e.generate(ids, max, eos, depths.items[0], .{ .ctx = &again, .prefilled = Collect.prefilled, .tokens = Collect.tokens, .cancelled = Collect.cancelled });
            const same = firstDiff(plain.?, again.toks.items) == null;
            if (!same) failures += 1;
            std.debug.print("  cancel test: cancelled reply ended {t} after {d} tokens; the next fresh reply {s} the plain one\n", .{ rc.reason, cut.toks.items.len, if (same) "equals" else "DIFFERS from" });
        }
    }
    try saved.append(arena, '}');
    if (std.c.getenv("GLM_OUT")) |path| {
        const file = std.c.fopen(path, "wb") orelse return error.OpenFailed;
        defer _ = std.c.fclose(file);
        if (std.c.fwrite(saved.items.ptr, 1, saved.items.len, file) != saved.items.len) return error.WriteFailed;
    }
    if (failures > 0) {
        std.debug.print("{d} replies differ from the plain ones{s}\n", .{ failures, if (strict) " or the references" else "" });
        std.process.exit(1);
    }
}
