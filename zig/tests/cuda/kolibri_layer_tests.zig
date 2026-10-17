//! Kolibri-1's layers on the real checkpoint: layer 0 against float64, the window's edge, the full layers' reach.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const kolibri1 = @import("kolibri1");
const check = @import("check.zig");
const ref = @import("kolibri_ref.zig");
const Gpu = check.Gpu;

const fwd = kolibri1.forward;

/// The first `layers` layers of the checkpoint, the shared kernels and `slots` streams' caches.
pub const Rig = struct {
    gpu: Gpu,
    w: kolibri1.weights.Weights,
    k: kolibri1.kernels.Kernels,
    s: cuda.Stream,
    m: fwd.Model,
    logits: cuda.DeviceBuffer,

    pub fn init(r: *Rig, gpu: Gpu, model: []const u8, count: ?usize, context: usize, slots: usize, rows: usize) !void {
        r.gpu = gpu;
        r.w = try kolibri1.weights.load(gpu.gpa, gpu.io, gpu.d, model, count);
        errdefer r.w.deinit();
        r.k = try kolibri1.kernels.Kernels.load(gpu.ctx);
        errdefer r.k.deinit();
        r.s = try cuda.Stream.init(gpu.d, false);
        errdefer r.s.deinit();
        r.m = try fwd.Model.init(gpu.gpa, gpu.d, r.s, &r.w, &r.k, context, slots, rows);
        errdefer r.m.deinit();
        r.logits = try cuda.DeviceBuffer.alloc(gpu.d, rows * r.w.config.vocab * 4);
    }

    pub fn deinit(r: *Rig) void {
        r.logits.free();
        r.m.deinit();
        r.s.deinit();
        r.k.deinit();
        r.w.deinit();
    }

    pub fn ops(r: *const Rig) kolibri1.kernels.Ops {
        return .{ .k = &r.k, .s = r.s };
    }

    /// The prompt in grid chunks into `slot`; the last row's fp32 logits (gpa-owned).
    pub fn prompt(r: *Rig, slot: usize, ids: []const u32) ![]f32 {
        var at: usize = 0;
        while (at < ids.len) : (at += fwd.prompt_rows) {
            const end = @min(ids.len, at + fwd.prompt_rows);
            try r.m.forward(r.ops(), &.{.{ .slot = slot, .p0 = at, .tokens = ids[at..end] }}, true, null, r.logits.ptr);
        }
        return r.read(1);
    }

    /// One decode row: `token` at position `p0` of `slot`; its logits.
    pub fn decode(r: *Rig, slot: usize, p0: usize, token: u32) ![]f32 {
        try r.m.forward(r.ops(), &.{.{ .slot = slot, .p0 = p0, .tokens = &.{token} }}, false, null, r.logits.ptr);
        return r.read(1);
    }

    pub fn read(r: *Rig, rows: usize) ![]f32 {
        try r.s.synchronize();
        const out = try r.gpu.gpa.alloc(f32, rows * r.w.config.vocab);
        try r.logits.download(0, std.mem.sliceAsBytes(out));
        return out;
    }
};

/// Prompt ids from a file of integers (commas or whitespace).
pub fn readIds(gpu: Gpu, path: []const u8) ![]u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 26));
    defer gpu.gpa.free(text);
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpu.gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \n\r\t[]");
    while (it.next()) |t| try out.append(gpu.gpa, try std.fmt.parseInt(u32, t, 10));
    return out.toOwnedSlice(gpu.gpa);
}

fn argmax(x: []const f32) usize {
    var best: usize = 0;
    for (x, 0..) |v, i| if (v > x[best]) {
        best = i;
    };
    return best;
}

fn same(a: []const f32, b: []const f32) bool {
    return std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}

/// Layer 0 and the head against float64 (routing bias, the shared expert, FP8 block scales): decode and prompt rows.
pub fn reference(gpu: Gpu, model: []const u8, ids_file: []const u8) !void {
    const ids = try readIds(gpu, ids_file);
    defer gpu.gpa.free(ids);
    var rig: Rig = undefined;
    try rig.init(gpu, model, 1, 4096, 2, fwd.prompt_rows);
    defer rig.deinit();
    var ck = try core.checkpoint.Checkpoint.openModel(gpu.gpa, gpu.io, model);
    defer ck.close();
    var map = try kolibri1.names.map(gpu.gpa, &ck, rig.w.config);
    defer map.deinit();
    var c = rig.w.config;
    c.layers = 1;
    var worst: f64 = 0;
    var bad: usize = 0;
    const n = 24;
    for (0..3) |case| {
        const toks = ids[case * 200 ..][0..n];
        var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
        defer arena.deinit();
        const want = try ref.lastLogits(arena.allocator(), &map, c, toks);
        const whole = try rig.prompt(0, toks);
        defer gpu.gpa.free(whole);
        const head = try rig.prompt(1, toks[0 .. n - 1]);
        gpu.gpa.free(head);
        const step = try rig.decode(1, n - 1, toks[n - 1]);
        defer gpu.gpa.free(step);
        var scale: f64 = 0;
        for (want) |v| scale = @max(scale, @abs(v));
        for ([_][]const f32{ whole, step }, [_][]const u8{ "prompt", "decode" }) |got, kind| {
            var most: f64 = 0;
            for (got, want) |g, w| most = @max(most, @abs(@as(f64, g) - w));
            const top = argmax(got) == std.mem.indexOfMax(f64, want);
            std.debug.print("RESULT kolibri1 reference case {d} {s}: max |logit - f64| {e:.3} of {d:.3}, top-1 {s}\n", .{ case, kind, most, scale, if (top) "equal" else "DIFFERS" });
            worst = @max(worst, most / scale);
            if (!top or most > 0.003 * scale) bad += 1; // clean rows stay near 0.1 %, a dropped bias or swapped scale passes 0.7 %
        }
    }
    try check.expect(bad == 0, "kolibri1 reference: {d} rows past 0.3% of the logits' scale or a different top-1", .{bad});
    check.pass("kolibri1 reference: layer 0 and the head within {d:.4} of the logits' scale of float64, top-1 equal", .{worst});
}

/// The window is 513 keys in prompt and decode rows (layer 0), and layer 4's full attention reaches key 0.
pub fn layers(gpu: Gpu, model: []const u8, ids_file: []const u8) !void {
    const ids = try readIds(gpu, ids_file);
    defer gpu.gpa.free(ids);
    var rig: Rig = undefined;
    try rig.init(gpu, model, 5, 4096, 2, fwd.prompt_rows);
    defer rig.deinit();
    const win = rig.w.config.window;
    const n = 1200;
    var base: [n]u32 = undefined;
    @memcpy(&base, ids[0..n]);
    // one layer: the last row (position n - 1) reads keys n - win through n - 1, nothing before
    var one: Rig = undefined;
    try one.init(gpu, model, 1, 4096, 2, fwd.prompt_rows);
    defer one.deinit();
    var bad: usize = 0;
    for ([_]usize{ n - 1 - win, n - win }, [_]bool{ false, true }) |at, moves| {
        var edit = base;
        edit[at] = if (edit[at] == 1000) 1001 else 1000;
        for ([_]bool{ true, false }) |prompt_rows| {
            const a = try lastRow(&one, &base, prompt_rows);
            defer gpu.gpa.free(a);
            const b = try lastRow(&one, &edit, prompt_rows);
            defer gpu.gpa.free(b);
            const changed = !same(a, b);
            std.debug.print("RESULT kolibri1 window: key {d} back {s} the last row ({s} row)\n", .{ n - 1 - at, if (changed) "moves" else "does not move", if (prompt_rows) "prompt" else "decode" });
            if (changed != moves) bad += 1;
        }
    }
    // five layers, four sliding: they reach 4 * (win - 1) keys back, so key 0 moves row 2,999 only through layer 4
    const far = 3000;
    const longer = try gpu.gpa.alloc(u32, far);
    defer gpu.gpa.free(longer);
    @memcpy(longer, ids[0..far]);
    const a = try lastRow(&rig, longer, true);
    defer gpu.gpa.free(a);
    longer[0] = if (longer[0] == 1000) 1001 else 1000;
    const b = try lastRow(&rig, longer, true);
    defer gpu.gpa.free(b);
    const reach = !same(a, b);
    std.debug.print("RESULT kolibri1 full layer: key 0 {s} row {d} through five layers (layer 4 full)\n", .{ if (reach) "moves" else "does not move", far - 1 });
    if (!reach) bad += 1;
    try check.expect(bad == 0, "kolibri1 layers: {d} window or full-layer checks wrong", .{bad});
    check.pass("kolibri1 layers: the window holds exactly {d} keys (prompt and decode rows), layer 4 reaches key 0", .{win});
}

/// The last row's logits of `ids`: all as prompt rows, or all but the last then the last as a decode row.
fn lastRow(r: *Rig, ids: []const u32, prompt_rows: bool) ![]f32 {
    if (prompt_rows) return r.prompt(0, ids);
    const head = try r.prompt(1, ids[0 .. ids.len - 1]);
    r.gpu.gpa.free(head);
    return r.decode(1, ids.len - 1, ids[ids.len - 1]);
}
