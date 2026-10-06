//! GLM-5.3-Flash replies on the native engine at each draft depth: hashes, first differences from depth 0 and the reference, speeds.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const glm = tf.glm;

const usage =
    \\tf-glm-run MODEL_DIR PROMPTS_JSON ({"prompts": [{"name": ..., "ids": [...], "expect": [...]}]})
    \\GLM_DEPTHS (default "0,3"), GLM_MAX (64), GLM_CAP (prompt + reply room, default 8192), GLM_RUNS (1),
    \\GLM_OUT (write each prompt's first-depth reply as JSON), GLM_VS (compare replies with a GLM_OUT file),
    \\GLM_CANCEL_TEST (cancel a reply mid-prompt, then the next fresh reply must equal the plain one),
    \\GLM_LAYERS (the first N layers only, with the MTP layer and head), GLM_REF_STRICT (a reference difference fails),
    \\GLM_EP (expert parallel: this Mac's link settings; run the same command on both Macs),
    \\GLM_TRACE=NAME:STEPS:PATH (every call of NAME's plain reply, sublayer by sublayer, feeding its expected tokens),
    \\GLM_FORCED (each prompt's teacher-forced agreement with its expected tokens, before the replies),
    \\GLM_PROFILE=D,D (after the replies: a knock-out profile of a round at each depth, GLM_PROFILE_REPS times),
    \\GLM_LOGITS=PREFIX (write each prompt's last-row logits, bf16, to PREFIX.NAME.bf16 at its first depth's first run),
    \\GLM_LOGITS_VS=PREFIX (compare them with such a file: the largest difference against the bf16 step at the top logit),
    \\GLM_PROMPT_PROFILE=REPS (after the replies: each prompt's first chunk by class, each left out or alone),
    \\GLM_MARGINS=PREFIX (each emitted token's top-two logit margin, f32, to PREFIX.NAME.margins), GLM_MARGINS_VS=PREFIX
    \\(at the first token that differs from GLM_VS's reply, both runs' margins there),
    \\GLM_TRACE_LAST=PREFIX (each prompt's last row at every capture point, bf16, to PREFIX.NAME.trace: embedding, then
    \\each layer's attention input and output and MLP input and output, then the final norm),
    \\GLM_PROMPT=0 (prompts in 16-row decode windows), GLM_CHUNK=N (prompt chunks of N rows), GLM_COPY=N (copy drafts from
    \\N-token matches), GLM_RANKS=1 (where missed drafts' targets fell among the MTP head's choices).
;

const Collect = struct {
    gpa: std.mem.Allocator,
    toks: std.ArrayList(u32) = .empty,
    cancel_at: usize = std.math.maxInt(usize), // the cancel check that answers true (0 = the first)
    checks: usize = 0,
    logits_from: []const u16 = &.{}, // the engine's logits row, copied into `logits` once the prompt is done
    logits: []u16 = &.{},
    trace_from: []const u8 = &.{}, // the engine's last-row trace, copied into `trace` likewise
    trace: []u8 = &.{},

    fn prefilled(ctx: *anyopaque) void {
        const c: *Collect = @ptrCast(@alignCast(ctx));
        if (c.logits_from.len > 0) c.logits = c.gpa.dupe(u16, c.logits_from) catch &.{};
        if (c.trace_from.len > 0) c.trace = c.gpa.dupe(u8, c.trace_from) catch &.{};
    }
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

fn bf16(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

/// The last-row logits against another run's (`path`): the largest difference, the top logit and bf16's step there.
fn compareLogits(gpa: std.mem.Allocator, ours: []const u16, path: [:0]const u8) !void {
    const f = mtl.MappedFile.open(path) catch {
        std.debug.print("  logits vs GLM_LOGITS_VS: no file {s}\n", .{path});
        return;
    };
    defer f.deinit();
    if (f.size != ours.len * 2) return error.LogitsSize;
    const theirs = try gpa.dupe(u16, @as([*]const u16, @ptrCast(@alignCast(f.bytes)))[0..ours.len]);
    defer gpa.free(theirs);
    var worst: f64 = 0;
    var at: usize = 0;
    var top: f64 = 0;
    var arg = [2]usize{ 0, 0 };
    for (ours, theirs, 0..) |a, b, i| {
        const d = @abs(bf16(a) - bf16(b));
        if (d > worst) {
            worst = d;
            at = i;
        }
        top = @max(top, @abs(bf16(b)));
        if (bf16(a) > bf16(ours[arg[0]])) arg[0] = i;
        if (bf16(b) > bf16(theirs[arg[1]])) arg[1] = i;
    }
    const step = std.math.pow(f64, 2, @floor(std.math.log2(@max(top, 1e-30))) - 7);
    std.debug.print("  logits vs GLM_LOGITS_VS: max |diff| {d:.4} at token {d} ({d:.4} vs {d:.4}), {d:.2} bf16 steps at the top logit {d:.3}; argmax {d} vs {d}\n", .{ worst, at, bf16(ours[at]), bf16(theirs[at]), worst / step, top, arg[0], arg[1] });
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
        std.debug.print("usage: {s}\n", .{usage});
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
    const trace_prefix: ?[*:0]const u8 = std.c.getenv("GLM_TRACE_LAST");
    const trace_bytes = (2 + 4 * @as(usize, e.c.run)) * e.c.hidden * 2;
    if (trace_prefix != null) e.trace_last = try e.arena.buffer(trace_bytes);
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
            defer gpa.free(col.logits);
            defer gpa.free(col.trace);
            if (trace_prefix != null and run == 0 and d == depths.items[0]) col.trace_from = e.trace_last.?.addr()[0..trace_bytes];
            var margins: std.ArrayList(f32) = .empty;
            defer margins.deinit(gpa);
            const want_margins = run == 0 and d == depths.items[0] and (std.c.getenv("GLM_MARGINS") != null or std.c.getenv("GLM_MARGINS_VS") != null);
            e.margins = if (want_margins) &margins else null;
            defer e.margins = null;
            const want_logits = run == 0 and d == depths.items[0] and (std.c.getenv("GLM_LOGITS") != null or std.c.getenv("GLM_LOGITS_VS") != null);
            if (want_logits) col.logits_from = @as([*]const u16, @ptrCast(@alignCast(e.sc.logits.addr())))[0..e.c.vocab];
            const r = try e.generate(ids, max, eos, d, .{ .ctx = &col, .prefilled = Collect.prefilled, .tokens = Collect.tokens, .cancelled = Collect.cancelled });
            const toks = col.toks.items;
            const tps = @as(f64, @floatFromInt(toks.len -| 1)) / @max(r.decode_seconds, 1e-9);
            std.debug.print("{s} depth {d} run {d}: {d} prompt tokens in {d:.2} s ({d:.0} tok/s), {d} tokens at {d:.1} tok/s, {d} rounds, {d}/{d} drafts kept, hash {x:0>16}\n", .{ name, d, run, ids.len, r.prompt_seconds, @as(f64, @floatFromInt(ids.len)) / @max(r.prompt_seconds, 1e-9), toks.len, tps, r.rounds, r.accepted, r.drafted, hash(toks) });
            std.debug.print("  tokens: {any}\n", .{toks[0..@min(toks.len, 48)]});
            if (r.rounds > 0) {
                const rn: f64 = @floatFromInt(r.rounds);
                std.debug.print("  a round: {d:.2} ms wall, {d:.2} ms GPU, {d:.2} ms encoding, {d:.2} ms GPU idle between rounds; {d:.2} tokens\n", .{ r.decode_seconds * 1e3 / rn, r.gpu_seconds * 1e3 / rn, r.encode_seconds * 1e3 / rn, r.gap_seconds * 1e3 / @max(rn - 1, 1), @as(f64, @floatFromInt(toks.len -| 1)) / rn });
                if (r.copy_rounds > 0) std.debug.print("  copy drafts: {d} rounds, {d} drafts kept ({d:.2} a copy round)\n", .{ r.copy_rounds, r.copy_accepted, @as(f64, @floatFromInt(r.copy_accepted)) / @as(f64, @floatFromInt(r.copy_rounds)) });
                if (std.c.getenv("GLM_RANKS") != null) {
                    std.debug.print("  draft ranks (kept: all / miss at 2nd / 3rd / later):", .{});
                    for (r.draft_ranks, 0..) |row, m| if (row[0] + row[1] + row[2] + row[3] > 0) std.debug.print(" m{d} {d}/{d}/{d}/{d}", .{ m, row[0], row[1], row[2], row[3] });
                    std.debug.print("\n", .{});
                }
            }
            if (plain == null) plain = try arena.dupe(u32, toks) else if (firstDiff(plain.?, toks)) |at| {
                failures += 1;
                std.debug.print("  DIFFERS from depth {d} at token {d}\n", .{ depths.items[0], at });
            } else std.debug.print("  equal to depth {d}\n", .{depths.items[0]});
            if (want_margins) if (std.c.getenv("GLM_MARGINS")) |prefix| {
                const path = try std.fmt.allocPrintSentinel(arena, "{s}.{s}.margins", .{ prefix, name }, 0);
                const file = std.c.fopen(path, "wb") orelse return error.OpenFailed;
                defer _ = std.c.fclose(file);
                if (std.c.fwrite(std.mem.sliceAsBytes(margins.items).ptr, 4, margins.items.len, file) != margins.items.len) return error.WriteFailed;
            };
            if (vs) |other| if (other.object.get(name)) |theirs| {
                const want = try ints(arena, theirs);
                if (firstDiff(want, toks)) |at| {
                    std.debug.print("  vs GLM_VS: first token {s}, first difference at token {d} of {d}\n", .{ if (at == 0) "DIFFERS" else "equal", at, @min(want.len, toks.len) });
                    if (want_margins) if (std.c.getenv("GLM_MARGINS_VS")) |prefix| {
                        const path = try std.fmt.allocPrintSentinel(arena, "{s}.{s}.margins", .{ prefix, name }, 0);
                        if (mtl.MappedFile.open(path)) |f| {
                            defer f.deinit();
                            const th: []const f32 = @as([*]const f32, @ptrCast(@alignCast(f.bytes)))[0 .. f.size / 4];
                            const sorted = try arena.dupe(f32, th[0..@min(th.len, at)]);
                            std.mem.sort(f32, sorted, {}, std.sort.asc(f32));
                            std.debug.print("  top-two margin at token {d}: here {d:.4}, GLM_VS run {d:.4} (its tokens {d} vs {d}); GLM_VS run's margins before it: median {d:.3}, smallest {d:.4}\n", .{ at, if (at < margins.items.len) margins.items[at] else -1, if (at < th.len) th[at] else -1, toks[at], want[at], if (sorted.len > 0) sorted[sorted.len / 2] else -1, if (sorted.len > 0) sorted[0] else -1 });
                        } else |_| std.debug.print("  no margins file {s}\n", .{path});
                    };
                } else std.debug.print("  vs GLM_VS: all {d} tokens equal\n", .{toks.len});
            };
            if (col.trace.len > 0) {
                const path = try std.fmt.allocPrintSentinel(arena, "{s}.{s}.trace", .{ std.mem.span(trace_prefix.?), name }, 0);
                const file = std.c.fopen(path, "wb") orelse return error.OpenFailed;
                defer _ = std.c.fclose(file);
                if (std.c.fwrite(col.trace.ptr, 1, col.trace.len, file) != col.trace.len) return error.WriteFailed;
            }
            if (want_logits and col.logits.len == 0) return error.NoLogits;
            if (want_logits) if (std.c.getenv("GLM_LOGITS")) |prefix| {
                const path = try std.fmt.allocPrintSentinel(arena, "{s}.{s}.bf16", .{ prefix, name }, 0);
                const file = std.c.fopen(path, "wb") orelse return error.OpenFailed;
                defer _ = std.c.fclose(file);
                if (std.c.fwrite(std.mem.sliceAsBytes(col.logits).ptr, 2, col.logits.len, file) != col.logits.len) return error.WriteFailed;
            };
            if (want_logits) if (std.c.getenv("GLM_LOGITS_VS")) |prefix| try compareLogits(gpa, col.logits, try std.fmt.allocPrintSentinel(arena, "{s}.{s}.bf16", .{ prefix, name }, 0));
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
    if (std.c.getenv("GLM_PROFILE")) |spec| {
        const reps = try std.fmt.parseInt(usize, env("GLM_PROFILE_REPS", "12"), 10);
        var pit = std.mem.tokenizeScalar(u8, std.mem.span(spec), ',');
        const only = std.c.getenv("GLM_PROFILE_ONLY") != null; // each class alone instead of each left out
        const parts = std.c.getenv("GLM_PROFILE_PARTS") != null; // by launch instead of by class
        while (pit.next()) |dv| try e.profile(try std.fmt.parseInt(u32, dv, 10), reps, only, parts);
    }
    if (std.c.getenv("GLM_MM_CHECK") != null) try e.checkMatmul(try ints(arena, doc.object.get("prompts").?.array.items[0].object.get("ids").?));
    if (std.c.getenv("GLM_PROMPT_PROFILE")) |v| { // each prompt's first chunk by class (GLM_PROFILE_ONLY: each alone)
        const reps = try std.fmt.parseInt(usize, std.mem.span(v), 10);
        for (doc.object.get("prompts").?.array.items) |p| try e.profilePrompt(try ints(arena, p.object.get("ids").?), reps, std.c.getenv("GLM_PROFILE_ONLY") != null, std.c.getenv("GLM_PROFILE_PARTS") != null);
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
