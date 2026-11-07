//! The host head against tools/clef_head_golden.py's tiny fixture: FP64 to FP64, the BF16 emulation to torch BF16.
const std = @import("std");
const bf16_round = @import("bf16_round");
const load = @import("load.zig");
const ref = @import("reference.zig");
const tensors = @import("tensors.zig");
const tiny = @import("tiny.zig");
const config = @import("config.zig");

const fixture = @embedFile("fixtures/clef_tiny.json");

fn num(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => unreachable,
    };
}

fn bitsValue(v: std.json.Value) f64 {
    return bf16_round.value(@intCast(v.integer));
}

/// The fixture's record in the reference's shape.
fn record(a: std.mem.Allocator, doc: std.json.Value) !ref.Record {
    const qs = doc.object.get("questions").?.array.items;
    var spans: std.ArrayList([2]u32) = .empty;
    var owner: std.ArrayList(u32) = .empty;
    const qspans = try a.alloc([2]u32, qs.len);
    const types = try a.alloc(u8, qs.len);
    for (qs, qspans, types, 0..) |q, *s, *t, k| {
        const sp = q.object.get("span").?.array.items;
        s.* = .{ @intCast(sp[0].integer), @intCast(sp[1].integer) };
        t.* = @intCast(q.object.get("type").?.integer);
        for (q.object.get("options").?.array.items) |o| {
            try spans.append(a, .{ @intCast(o.array.items[0].integer), @intCast(o.array.items[1].integer) });
            try owner.append(a, @intCast(k));
        }
    }
    return .{ .ids = try tiny.ids(a), .question_spans = qspans, .question_types = types, .option_spans = spans.items, .option_question = owner.items };
}

/// The tiny head written as a checkpoint, loaded back through the loader, its logits in `mode`.
fn run(a: std.mem.Allocator, mode: ref.Mode, doc: std.json.Value) ![]f64 {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tiny.write(a, io, tmp.dir, tiny.cfg, .none);
    var p: config.Problem = .{};
    var h = try load.open(std.testing.allocator, io, try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path}), &p);
    defer h.close(io);
    const c: ref.Ctx = .{ .a = a, .mode = mode };
    return ref.logits(c, &h.weights, try tiny.hidden(a), try record(a, doc), .{ .bytes = try tiny.table(a), .rows = tiny.vocab });
}

test "joint schema reference: the Zig generator gives the golden tool's tensors" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, fixture, .{});
    try std.testing.expectEqual(@as(i64, tiny.cfg.hidden_size), doc.object.get("config").?.object.get("hidden_size").?.integer);
    var l: tensors.Listing = .{ .a = a };
    _ = try tensors.fill(a, tiny.cfg, &l);
    const sums = doc.object.get("sums").?.object;
    try std.testing.expectEqual(l.names.items.len, sums.count());
    for (l.names.items, l.shapes.items) |name, shape| {
        var n: usize = 1;
        for (shape) |d| n *= d;
        var s: f64 = 0;
        for (0..n) |i| s += tiny.value(name, shape, i);
        if (s != num(sums.get(name).?)) {
            std.debug.print("{s}: generated sum {d}, golden {d}\n", .{ name, s, num(sums.get(name).?) });
            return error.TestUnexpectedResult;
        }
    }
    const want_ids = doc.object.get("ids").?.array.items;
    for (want_ids, try tiny.ids(a)) |w, got| try std.testing.expectEqual(w.integer, got);
}

test "joint schema reference: FP64 logits equal torch's FP64 head, BF16 emulation within 4 BF16 steps of torch's BF16" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, fixture, .{});
    const want64 = doc.object.get("fp64").?.array.items;
    const want16 = doc.object.get("bf16").?.array.items;
    const got64 = try run(a, .fp64, doc);
    const got16 = try run(a, .bf16, doc);
    try std.testing.expectEqual(want64.len, got64.len);
    var e64: f64 = 0;
    var e16: f64 = 0;
    var top: f64 = 0;
    var torch_err: [2]f64 = .{ 0, 0 }; // worst and mean against FP64
    var emul_err: [2]f64 = .{ 0, 0 };
    for (want64, want16, got64, got16) |w64, w16, g64, g16| {
        e64 = @max(e64, @abs(num(w64) - g64));
        e16 = @max(e16, @abs(bitsValue(w16) - g16));
        top = @max(top, @abs(num(w64)));
        for ([_]*[2]f64{ &torch_err, &emul_err }, [_]f64{ bitsValue(w16), g16 }) |e, v| {
            e[0] = @max(e[0], @abs(v - num(w64)));
            e[1] += @abs(v - num(w64)) / @as(f64, @floatFromInt(got64.len));
        }
    }
    std.debug.print("joint schema reference: FP64 vs torch FP64 max {e}; BF16 emulation vs torch BF16 max {d:.4} ({d:.2} steps at the largest logit); " ++
        "against FP64 (worst, mean): torch BF16 {d:.4} {d:.4}, emulation {d:.4} {d:.4}\n", .{ e64, e16, e16 / step(top), torch_err[0], torch_err[1], emul_err[0], emul_err[1] });
    try std.testing.expect(e64 <= 1e-9);
    // nine logits are too few for gate 3's rule: within 4 BF16 steps at the largest logit's magnitude of torch's head
    try std.testing.expect(e16 <= 4 * step(top));
}

test "joint schema reference: both normalize forms at norms 0 to 1, and the scale clamp, match torch" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, fixture, .{});
    const cases = doc.object.get("near_zero").?.object;
    const W = tiny.cfg.width;
    const x = try a.alloc(f64, W);
    const y = try a.alloc(f64, W);
    const out = try a.alloc(f64, W);
    var ulps: usize = 0;
    for (cases.get("normalize").?.array.items) |c| {
        for (x, c.object.get("x").?.array.items) |*v, b| v.* = bitsValue(b);
        ref.normalize(.{ .a = a, .mode = .fp64 }, x, out);
        for (out, c.object.get("fp64").?.array.items) |g, w| try std.testing.expectApproxEqAbs(num(w), g, 1e-15);
        ref.normalize(.{ .a = a, .mode = .bf16 }, x, out);
        for (out, c.object.get("bf16").?.array.items) |g, w| ulps = @max(ulps, apart(g, bitsValue(w)));
    }
    for (cases.get("cosine").?.array.items) |c| {
        for (x, c.object.get("f").?.array.items) |*v, b| v.* = bitsValue(b);
        for (y, c.object.get("o").?.array.items) |*v, b| v.* = bitsValue(b);
        try std.testing.expectApproxEqAbs(num(c.object.get("fp64").?), ref.cosine(.{ .a = a, .mode = .fp64 }, x, y), 1e-15);
        ulps = @max(ulps, apart(ref.cosine(.{ .a = a, .mode = .bf16 }, x, y), bitsValue(c.object.get("bf16").?)));
    }
    // the scale clamp: exp(min(5, ln 100)) is 100 in FP64 and exp of BF16 ln 100 rounded in BF16
    const clamp = cases.get("clamp").?.object;
    try std.testing.expectApproxEqAbs(num(clamp.get("fp64").?), ref.cappedExp(.{ .a = a, .mode = .fp64 }, num(clamp.get("p").?)), 1e-12);
    try std.testing.expectEqual(bitsValue(clamp.get("bf16").?), ref.cappedExp(.{ .a = a, .mode = .bf16 }, num(clamp.get("p").?)));
    std.debug.print("joint schema reference: near-zero normalize and cosine, BF16 emulation within {d} BF16 ulp of torch\n", .{ulps});
    try std.testing.expect(ulps <= 1);
}

/// The BF16 step (one unit in the last place) at magnitude x.
fn step(x: f64) f64 {
    return std.math.ldexp(@as(f64, 1), std.math.frexp(x).exponent - 8);
}

/// BF16 steps between two BF16 values.
fn apart(x: f64, y: f64) usize {
    const p: i32 = bf16_round.bits(@floatCast(x));
    const q: i32 = bf16_round.bits(@floatCast(y));
    const sp: i32 = if (p & 0x8000 != 0) -(p & 0x7fff) else p;
    const sq: i32 = if (q & 0x8000 != 0) -(q & 0x7fff) else q;
    return @abs(sp - sq);
}
