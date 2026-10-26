//! tf-muse prompt: the prompt path's chunk plans against decode rows, byte for byte (docs/recipes/muse-spark.md).

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const muse = @import("muse_spark");
const Model = muse.forward.Model;
const Row = muse.forward.Row;

fn env(name: [*:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

fn argmax(xs: []const f32) i32 {
    var best: usize = 0;
    for (xs, 0..) |x, i| if (x > xs[best]) {
        best = i;
    };
    return @intCast(best);
}

/// Greedy decode rows on `slot` from position `pos` after the token `first`: n tokens into out.
fn greedy(model: *Model, s: cuda.Stream, slot: u32, pos0: usize, first: i32, out: []i32, buf_all: []f32) !void {
    const buf = buf_all[0..model.w.full.vocab];
    var tok = first;
    for (out, 0..) |*o, j| {
        o.* = tok;
        if (j + 1 == out.len) break;
        try model.forward(s, &.{.{ .slot = slot, .pos = @intCast(pos0 + j), .token = tok }}, buf, null);
        tok = argmax(buf);
    }
}

pub fn run(gpa: std.mem.Allocator, ar: std.mem.Allocator, io: std.Io, driver: *const cuda.Driver, args: []const [:0]const u8, rank: usize, world: usize, comm: ?tp.Collective) !void {
    var s = try cuda.Stream.init(driver, false);
    defer s.deinit();
    const context = env("MUSE_CONTEXT", 8192);
    const G = env("MUSE_GREEDY", 32);
    const max_layers: ?usize = if (env("MUSE_LAYERS", 0) > 0) env("MUSE_LAYERS", 0) else null;
    var w = try muse.load.load(gpa, io, driver, args[2], rank, world, .{ .max_layers = max_layers, .log = rank == 0 });
    defer w.deinit();
    var k3 = try muse.exl3.Kernels.load(driver);
    defer k3.deinit();
    var model = try Model.init(gpa, driver, &w, &k3, comm, context, muse.forward.DECODE_ROWS, 2, s);
    defer model.deinit();
    const pr = try muse.prompt.Prompt.init(&model, gpa, muse.prompt.CAP, muse.prompt.EWIN, s);
    defer pr.deinit();
    const V = w.full.vocab;
    if (rank == 0) std.debug.print("[prompt] {d} layers, chunks of {d} rows, experts window {d}, scratch {d:.2} GiB\n", .{ w.layers.len, pr.cap, pr.ewin, @as(f64, @floatFromInt(pr.bytes())) / (1 << 30) });
    const plans_text = if (std.c.getenv("MUSE_PROMPT_PLANS")) |v| std.mem.span(v) else "r;100,r;1,r;17,256,1000,r;513,700,r";
    const names_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(ar, &.{ args[3], "names.txt" }), ar, .limited(1 << 20));
    const buf = try gpa.alloc(f32, muse.forward.DECODE_ROWS * V);
    defer gpa.free(buf);
    const ref_logits = try gpa.alloc(f32, V);
    defer gpa.free(ref_logits);
    const got = try gpa.alloc(f32, V);
    defer gpa.free(got);
    const ref_g = try gpa.alloc(i32, G);
    defer gpa.free(ref_g);
    const got_g = try gpa.alloc(i32, G);
    defer gpa.free(got_g);
    var fails: usize = 0;
    var nit = std.mem.tokenizeAny(u8, names_text, " \r\n\t");
    while (nit.next()) |name| {
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(ar, "{s}/{s}.ids", .{ args[3], name }), ar, .limited(1 << 30));
        const T = raw.len / 4;
        if (T + G > context) return error.TextTooLong;
        const ids = try ar.alloc(i64, T);
        for (ids, 0..) |*x, j| x.* = std.mem.readInt(i32, raw[j * 4 ..][0..4], .little);
        // the reference: decode windows of 128 rows on slot 0
        var at: usize = 0;
        var rows: [muse.forward.DECODE_ROWS]Row = undefined;
        while (at < T) {
            const n = @min(muse.forward.DECODE_ROWS, T - at);
            for (0..n) |j| rows[j] = .{ .slot = 0, .pos = @intCast(at + j), .token = ids[at + j] };
            try model.forward(s, rows[0..n], buf[0 .. n * V], null);
            if (at + n == T) @memcpy(ref_logits, buf[(n - 1) * V ..][0..V]);
            at += n;
        }
        try greedy(&model, s, 0, T, argmax(ref_logits), ref_g, buf);
        var pit = std.mem.tokenizeScalar(u8, plans_text, ';');
        while (pit.next()) |plan| {
            var sizes: std.ArrayList(usize) = .empty;
            defer sizes.deinit(gpa);
            var cit = std.mem.tokenizeScalar(u8, plan, ',');
            var used: usize = 0;
            while (cit.next()) |c| {
                const n = if (std.mem.eql(u8, c, "r")) T - used else try std.fmt.parseInt(usize, c, 10);
                if (n == 0 or used + n > T) break;
                var left = n;
                while (left > 0) { // a size over the chunk cap splits into cap-row chunks
                    const k = @min(left, pr.cap);
                    try sizes.append(gpa, k);
                    left -= k;
                }
                used += n;
            }
            if (used != T) continue; // a plan longer than the text
            at = 0;
            for (sizes.items) |n| {
                try pr.chunk(s, 1, at, ids[at .. at + n]);
                at += n;
            }
            try model.logitsTo(s, 1, got);
            var maxd: f32 = 0;
            var equal = true;
            for (ref_logits, got) |a, b| {
                if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b))) equal = false;
                maxd = @max(maxd, @abs(a - b));
            }
            try greedy(&model, s, 1, T, argmax(got), got_g, buf);
            var same: usize = 0;
            for (ref_g, got_g) |a, b| same += @intFromBool(a == b);
            if (!equal or same != G) fails += 1;
            if (rank == 0) std.debug.print("PROMPT {s} T={d} chunks {any}: logits {s} (max |diff| {e:.3}), argmax {d}/{d}, greedy {d}/{d}\n", .{ name, T, sizes.items, if (equal) "byte-equal" else "DIFFER", maxd, argmax(got), argmax(ref_logits), same, G });
        }
    }
    if (rank == 0) std.debug.print("PROMPT {s}: {d} plan(s) differ from decode rows\n", .{ if (fails == 0) "PASS" else "FAIL", fails });
}
