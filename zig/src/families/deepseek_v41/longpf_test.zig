//! Host tests of long prompts on the prefill emitter (`zig build test-dsv41-longpf`): stream top-k and split blocks.

const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const block_prefill = @import("block_prefill.zig");
const buffers = @import("buffers.zig");
const check = @import("calls_check.zig");
const longpf = @import("longpf.zig");
const Config = @import("config.zig").Config;

const limit: i64 = 1 << 20;
const n: i64 = 2048;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    // x3gm's K2s present (the forward gate's widths read them from the pack): two widths a layer
    for (0..40) |L| try w.gm.put(a, @intCast(L), .{ (1 << 3) | (1 << 4), (1 << 3) | (1 << 4) });
    return w;
}

fn poolOf(split: bool) block.Pool {
    const pages = @divExact(limit, block.Pool.page);
    return .{ .comp_pages = if (split) @divExact(pages, 2) + 2 else pages + 1, .ik_pages = pages + 1, .pts = pages, .split = split };
}

fn named(c: calls.Call, name: []const u8) ?calls.Arg {
    for (c.args) |x| if (std.mem.eql(u8, x.name, name)) return x.arg;
    return null;
}

fn roleOf(x: calls.Arg) []const u8 {
    return switch (x) {
        .t => |t| switch (t.role) {
            .buf => |b| b,
            else => "",
        },
        else => "",
    };
}

fn emit(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, pool: ?block.Pool, start: i64) ![]calls.Call {
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    var o: block.Options = .{ .limit = limit, .rope_rows = limit + 64 };
    o.pool = pool;
    return block_prefill.emitPrefill(a, cfg, w, o, &backbone, n, start, true);
}

test "stream_topk's splits and plan (its docstring's 300K example)" {
    try testing.expectEqual(@as(i64, 1), longpf.splits(1));
    try testing.expectEqual(@as(i64, 1), longpf.splits(16384));
    try testing.expectEqual(@as(i64, 2), longpf.splits(16385));
    try testing.expectEqual(@as(i64, 19), longpf.splits(300_000));
    // 19 splits x 8 KB a row under 0.25 GiB: >= 1,600 rows a launch
    try testing.expectEqual(@as(i64, 1724), longpf.plan(4096, 300_000, 512));
    try testing.expectEqual(@as(i64, 2048), longpf.plan(2048, 8192, 512));
    try testing.expectEqual(@as(i64, 1024), longpf.plan(2048, 1 << 19, 512));
}

test "a 2,048-row segment at the 1M limit's end: the stream top-k on full layers, in stream_topk.plan's row blocks" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    const start = limit - n;
    for ([_]?block.Pool{ null, poolOf(false), poolOf(true) }) |pool| {
        const cs = try emit(a, &cfg, &w, pool, start);
        var streams: usize = 0;
        var merges: usize = 0;
        var layer_streams: [40]usize = @splat(0);
        var layer: usize = 0;
        for (cs) |c| {
            if (std.mem.eql(u8, c.name, "_stream")) {
                const ratio = named(c, "RATIO").?.i;
                const rows = c.grid[0];
                const nk = named(c, "NK").?.i;
                try testing.expectEqual(longpf.splits(nk), c.grid[1]);
                try testing.expectEqual(c.grid[1], named(c, "nsplit").?.i);
                try testing.expectEqual(longpf.plan(n, @divFloor(start + n, ratio), 512), rows);
                // the scratch under the budget, KEYS = BUF (positions mode), paged keys with the pool
                try testing.expect(rows * c.grid[1] * 2 * 512 * 8 <= longpf.budget);
                try testing.expectEqualStrings(roleOf(named(c, "BUF").?), roleOf(named(c, "KEYS").?));
                try testing.expectEqual(@as(i64, 0), named(c, "MODE").?.i);
                try testing.expectEqual(pool != null, named(c, "PT").? != .none);
                // two blocks: each its own position
                try testing.expectEqualStrings(if (rows >= n) "w.pos" else "L.ix.pos", roleOf(named(c, "POS").?)[0..if (rows >= n) 5 else 8]);
                streams += 1;
                layer_streams[layer] += 1;
            }
            if (std.mem.eql(u8, c.name, "glue.stream_merge")) {
                const buf = c.args[0].arg.t;
                try testing.expectEqualSlices(i64, &.{ buf.shape[0], 512 }, c.args[1].arg.t.shape);
                try testing.expectEqualSlices(i64, &.{ buf.shape[0], buf.shape[1] * 512 }, c.args[3].arg.t.shape);
                merges += 1;
            }
            // the materialised selection never runs on a streamed layer (no `_keys` there)
            if (std.mem.eql(u8, c.name, "_keys")) try testing.expectEqual(@as(usize, 0), layer_streams[layer]);
            if (c.begin == .layer) layer += 1;
        }
        // every full-mode layer without the candidates' scores streams, in two row blocks at 512K visible keys
        var want: usize = 0;
        for (0..40) |L| {
            const l: u32 = @intCast(L);
            if (cfg.mode(l) == .full and !cfg.isCandidateSource(l)) want += 1;
        }
        try testing.expect(want > 0);
        try testing.expectEqual(2 * want, streams);
        try testing.expectEqual(streams, merges);
        // the plan sizes the stream scratch within the budget
        var p: buffers.Plan = .{ .a = a };
        try p.add(cs);
        try testing.expect(p.sizes.get("L.ix.stream").? <= longpf.budget);
    }
}

test "below the stream threshold the segment is unchanged: no `_stream`" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    for ([_]i64{ 0, 2048, 4096 }) |start| {
        const cs = try emit(a, &cfg, &w, null, start);
        for (cs) |c| try testing.expect(!std.mem.eql(u8, c.name, "_stream") and !std.mem.eql(u8, c.name, "glue.stream_merge"));
    }
}

test "split KV: an exchanged attention in row blocks moves the row-indexed tensors only" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    const cs = try emit(a, &cfg, &w, poolOf(true), limit - n);
    var exchanged: usize = 0;
    var local: usize = 0;
    for (cs) |*c| {
        if (!std.mem.eql(u8, c.name, "_fused")) continue;
        if (!longpf.readsExchange(c)) {
            local += 1;
            continue;
        }
        exchanged += 1;
        const lo: i64 = 640;
        const hi: i64 = 1500;
        const b = try longpf.blockCall(a, c, lo, hi);
        try testing.expectEqual(hi - lo, b.grid[0]);
        try testing.expectEqual(c.grid[1], b.grid[1]);
        for (c.args, b.args) |x, y| {
            try testing.expectEqualStrings(x.name, y.name);
            switch (x.arg) {
                .t => |t| {
                    const u = y.arg.t;
                    const moved = u.offset - t.offset;
                    if (std.mem.eql(u8, x.name, "POS")) {
                        try testing.expectEqualStrings(longpf.pos_role, u.role.buf);
                    } else if (std.mem.eql(u8, x.name, "Q") or std.mem.eql(u8, x.name, "TOK") or std.mem.eql(u8, x.name, "CNT") or std.mem.eql(u8, x.name, "LO")) {
                        try testing.expectEqual(lo * t.stride[0] * @as(i64, @intCast(buffers.dtSize(t.dt))), moved);
                        try testing.expectEqual(hi - lo, u.shape[0]);
                    } else if (std.mem.eql(u8, x.name, "OUT")) {
                        // group-major rows of HG x 512
                        try testing.expectEqual(lo * t.shape[2] * 2, moved);
                    } else try testing.expectEqual(@as(i64, 0), moved);
                },
                // every scalar (OR, HG, the constexprs) as the whole launch's: the same AOT variant
                .i => |v| try testing.expectEqual(v, y.arg.i),
                .b => |v| try testing.expectEqual(v, y.arg.b),
                .f => |v| try testing.expectEqual(v, y.arg.f),
                else => {},
            }
        }
        try testing.expectEqual(n, named(b, "OR").?.i);
    }
    // the compressed layers attend the exchange, the SWA-only layers their ring
    try testing.expect(exchanged > 0 and local > 0);
}

test "TF_DSV41_STREAM_LAZY: no graphed program names the stream scratch; a long segment does" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var layers: [64]u32 = undefined;
    for (layers[0..cfg.layers], 0..) |*l, i| l.* = @intCast(i);
    for ([_]bool{ false, true }) |split| {
        const o: block.Options = .{ .limit = limit, .rope_rows = limit + 2048, .pool = poolOf(split) };
        // decode windows (one slot, every row bucket) and row windows at the slot's last position: graph-captured
        var r: i64 = 1;
        while (r <= 64) : (r += 1) {
            // (a one-slot window the emitter refuses at this position never runs: block.zig's stream guard)
            if (r <= 17) if (block.emit(a, &cfg, &w, o, layers[0..cfg.layers], r, limit - r, true)) |cs|
                try testing.expect(!longpf.names(cs, longpf.lazy_role))
            else |e|
                try testing.expectEqual(error.Unsupported, e);
            var ro = o;
            ro.rows = true;
            try testing.expect(!longpf.names(try block.emit(a, &cfg, &w, ro, layers[0..cfg.layers], r, limit - r, true), longpf.lazy_role));
        }
        // a prompt segment past the threshold streams
        const seg = try block_prefill.emitPrefill(a, &cfg, &w, o, layers[0..cfg.layers], n, limit - n, true);
        try testing.expect(longpf.names(seg, longpf.lazy_role));
        // and one below it does not (the scratch is never allocated for a short prompt)
        const short = try block_prefill.emitPrefill(a, &cfg, &w, o, layers[0..cfg.layers], n, 0, true);
        try testing.expect(!longpf.names(short, longpf.lazy_role));
    }
}
