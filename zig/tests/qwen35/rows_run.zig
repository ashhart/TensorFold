//! `rows <model dir> <ids.npy> [n] [streams] [keep]`: the window invariant layer by layer, naming the first difference.

const std = @import("std");
const hip = @import("hip");
const qwen35 = @import("qwen3_5");
const ids_file = @import("ids_file.zig");

const Engine = qwen35.engine.Engine;

/// Every layer's residual rows of one run: (layers + 1, n, hidden) bytes, the last slot the final rows.
const Sink = struct {
    e: *Engine,
    bytes: []u8,
    n: usize,
    /// The row the next capture lands at, and the real rows of the round (the forward may run padding rows past them).
    row: usize = 0,
    take: usize = 0,

    fn width(s: *const Sink) usize {
        return s.e.model().spec.hidden * s.e.model().act.size();
    }

    fn put(s: *Sink, slot: usize, x: hip.ops.Tensor, rows: usize) !void {
        try s.e.stream.synchronize();
        const w = s.width();
        const at = (slot * s.n + s.row) * w;
        try hip.runtime.check(s.e.driver.api.hipMemcpyDtoH(s.bytes[at..].ptr, @ptrFromInt(x.ptr), rows * w));
    }

    fn layer(ctx: *anyopaque, index: usize, x: hip.ops.Tensor, rows: usize) anyerror!void {
        const s: *Sink = @ptrCast(@alignCast(ctx));
        _ = rows;
        try s.put(index, x, s.take);
    }
};

/// One round of every stream's `tokens` from `pos[j]` of `caches[j]`, in at least `pad` rows, captured into `sink`.
fn round(e: *Engine, caches: []const *qwen35.state.Caches, pos: []const usize, tokens: []const []const u32, keep: usize, sink: *Sink, pad: usize) !void {
    var rows: [8]Engine.Rows = undefined;
    var reqs: [128]qwen35.draw.Request = undefined;
    var out: [128]u32 = undefined;
    var total: usize = 0;
    for (caches, pos, tokens, 0..) |c, p, t, j| {
        rows[j] = .{ .caches = c, .pos = p, .tokens = t };
        for (0..t.len) |i| reqs[total + i] = .{ .sampling = null, .position = p + i + 1 };
        total += t.len;
    }
    sink.take = total;
    e.round.trace = .{ .ctx = sink, .layer = Sink.layer };
    e.round.pad_to = pad;
    defer {
        e.round.trace = null;
        e.round.pad_to = 0;
    }
    const done = try e.verify(rows[0..caches.len], reqs[0..total], out[0..total]);
    try sink.put(e.model().spec.n_layers, done.hidden, total);
    for (tokens, 0..) |t, j| e.keep(j, @min(keep, t.len));
    try e.flush();
    try e.stream.synchronize();
}

/// Streams of `n` rows each, the first `keep` of each kept.
pub const Case = struct { streams: usize, n: usize, keep: usize, pad: usize = 0 };

/// Where a case's two runs first differ.
pub const Diff = struct { layer: usize, kind: []const u8, stream: usize, row: usize, column: usize };

/// Runs one case over prompts cut from `ids`; null when every layer's rows and the final rows are equal.
pub fn runCase(gpa: std.mem.Allocator, e: *Engine, ids: []const u32, c: Case) !?Diff {
    const n = c.n;
    const streams = c.streams;
    // the trace waits on the stream layer by layer, which a capture cannot
    const was = e.o.graphs;
    e.o.graphs = false;
    defer e.o.graphs = was;
    if (n < 1 or n > 16) return error.BadWindow;
    if (c.keep < 1 or c.keep > n) return error.BadKeep;
    if (streams < 1 or streams > 8 or ids.len <= n + streams or streams * n > 64) return error.PromptTooShort;
    const layers = e.model().spec.n_layers;
    const rows = streams * n;
    var serial: Sink = .{ .e = e, .bytes = undefined, .n = rows };
    var shared: Sink = .{ .e = e, .bytes = undefined, .n = rows };
    serial.bytes = try gpa.alloc(u8, (layers + 1) * rows * serial.width());
    defer gpa.free(serial.bytes);
    shared.bytes = try gpa.alloc(u8, (layers + 1) * rows * shared.width());
    defer gpa.free(shared.bytes);

    // stream j: the ids short by j, its prompt all but its last n tokens
    var alone: [8]qwen35.state.Caches = undefined;
    var together: [8]qwen35.state.Caches = undefined;
    var caches: [8]*qwen35.state.Caches = undefined;
    var pos: [8]usize = undefined;
    var tails: [8][]const u32 = undefined;
    var made: usize = 0;
    defer for (alone[0..made], together[0..made]) |*a1, *t1| {
        a1.deinit(gpa);
        t1.deinit(gpa);
    };
    for (0..streams) |j| {
        const end = ids.len - j;
        const prompt = ids[0 .. end - n];
        tails[j] = ids[end - n .. end];
        pos[j] = prompt.len;
        const req: qwen35.draw.Request = .{ .sampling = null, .position = prompt.len };
        alone[j] = try e.newCaches(ids.len + 8);
        together[j] = e.newCaches(ids.len + 8) catch |err| {
            alone[j].deinit(gpa);
            return err;
        };
        made += 1;
        _ = try e.prefill(&alone[j], prompt, 0, null, req, null);
        _ = try e.prefill(&together[j], prompt, 0, null, req, null);
        caches[j] = &together[j];
    }
    for (0..streams) |j| for (0..n) |i| {
        serial.row = j * n + i;
        const one = [1]*qwen35.state.Caches{&alone[j]};
        try round(e, &one, &.{pos[j] + i}, &.{tails[j][i..][0..1]}, 1, &serial, 0);
    };
    try round(e, caches[0..streams], pos[0..streams], tails[0..streams], c.keep, &shared, c.pad);
    // the rows past `keep` again, one at a time from the kept state, over the shared round's captures
    for (c.keep..n) |i| for (0..streams) |j| {
        shared.row = j * n + i;
        const one = [1]*qwen35.state.Caches{&together[j]};
        try round(e, &one, &.{pos[j] + i}, &.{tails[j][i..][0..1]}, 1, &shared, 0);
    };

    const w = serial.width();
    for (0..layers + 1) |slot| for (0..rows) |r| {
        const at = (slot * rows + r) * w;
        if (std.mem.indexOfDiff(u8, serial.bytes[at..][0..w], shared.bytes[at..][0..w])) |col| {
            const kind = if (slot == layers) "final" else if (e.model().spec.full(slot)) "full attention" else "linear attention";
            return .{ .layer = slot, .kind = kind, .stream = r / n, .row = r % n, .column = col / e.model().act.size() };
        }
    };
    return null;
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len < 2) return error.MissingArgument;
    const n = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 3;
    const streams = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 1;
    const keep = if (args.len > 4) try std.fmt.parseInt(usize, args[4], 10) else n;
    const ids = try ids_file.load(gpa, io, args[1]);
    defer gpa.free(ids);
    const e = try Engine.open(gpa, io, args[0], .{ .capacity = ids.len + 64, .prompt_rows = ids.len + 64, .streams = streams, .batch_rows = 128, .graphs = false, .policy = (try @import("group.zig").resolve("", 0)).policy });
    defer e.deinit();
    if (try runCase(gpa, e, ids, .{ .streams = streams, .n = n, .keep = keep })) |d| {
        std.debug.print("FAIL layer {d} ({s}), stream {d} row {d} of {d}: first difference at column {d}\n", .{ d.layer, d.kind, d.stream, d.row, n, d.column });
        return error.Mismatch;
    }
    std.debug.print("PASS {d} streams of {d} rows, {d} kept: every layer's residuals and the final rows equal each stream one row at a time\n", .{ streams, n, keep });
}
