//! Dory on the real NVFP4 checkpoint: greedy tokens from a prompt, then verify windows of 2 to 16 rows bit-equal.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const dory = @import("dory");
const check = @import("check.zig");
const Gpu = check.Gpu;

const fwd = dory.cuda_model;

pub const Rig = struct {
    gpu: Gpu,
    w: dory.cuda_weights.Weights,
    k: fwd.Kernels,
    s: cuda.Stream,
    m: fwd.Model,

    pub fn init(r: *Rig, gpu: Gpu, model: []const u8, loops: usize, context: usize) !void {
        return r.initSlots(gpu, model, loops, context, 1);
    }

    pub fn initSlots(r: *Rig, gpu: Gpu, model: []const u8, loops: usize, context: usize, slots: usize) !void {
        r.gpu = gpu;
        r.w = try dory.cuda_weights.load(gpu.gpa, gpu.io, gpu.d, model, loops);
        errdefer r.w.deinit();
        r.k = try fwd.Kernels.load(gpu.ctx);
        errdefer r.k.deinit();
        r.s = try cuda.Stream.init(gpu.d, false);
        errdefer r.s.deinit();
        r.m = try fwd.Model.init(gpu.gpa, gpu.d, r.s, &r.w, &r.k, context, slots);
    }

    pub fn deinit(r: *Rig) void {
        r.m.deinit();
        r.s.deinit();
        r.k.deinit();
        r.w.deinit();
    }

    /// fp32 logits of `tokens` at p0.. (rows pick[0].. in order, a run), on the host; slot 0.
    pub fn rows(r: *Rig, tokens: []const u32, p0: usize, prompt: bool, pick: []const usize) ![]f32 {
        return r.slotRows(0, tokens, p0, prompt, pick[0], pick.len);
    }

    /// fp32 logits of rows [head0, head0 + heads) of `tokens` at p0.. in `slot`'s caches, on the host.
    pub fn slotRows(r: *Rig, slot: usize, tokens: []const u32, p0: usize, prompt: bool, head0: usize, heads: usize) ![]f32 {
        try r.m.forward(&r.k, r.s, &.{.{ .slot = slot, .p0 = p0, .tokens = tokens }}, prompt, head0, heads);
        try r.s.synchronize();
        const v = r.w.config.vocab;
        const out = try r.gpu.gpa.alloc(f32, heads * v);
        try (cuda.DeviceBuffer{ .d = r.gpu.d, .ptr = r.m.s.logits, .len = out.len * 4, .borrowed = true }).download(0, std.mem.sliceAsBytes(out));
        return out;
    }
};

fn argmax(x: []const f32) u32 {
    var best: usize = 0;
    for (x, 0..) |v, i| if (v > x[best]) {
        best = i;
    };
    return @intCast(best);
}

/// tf-cuda-test dory MODEL LOOPS N TEXT: N greedy tokens one row at a time, then every window width against them.
pub fn greedy(gpu: Gpu, model: []const u8, loops_text: []const u8, count_text: []const u8, text: []const u8) !void {
    const gpa = gpu.gpa;
    const loops = try std.fmt.parseInt(usize, loops_text, 10);
    const count = try std.fmt.parseInt(usize, count_text, 10);
    var tok = try core.tokenizer.loadTokenizer(gpu.io, gpa, model);
    defer tok.deinit();
    const ids = try tok.encode(gpa, text);
    defer gpa.free(ids);
    var prompt: std.ArrayList(u32) = .empty;
    defer prompt.deinit(gpa);
    if (ids.len == 0 or ids[0] != 1) try prompt.append(gpa, 1); // <s>
    try prompt.appendSlice(gpa, ids);
    var rig: Rig = undefined;
    try rig.init(gpu, model, loops, 4096);
    defer rig.deinit();
    const v = rig.w.config.vocab;
    std.debug.print("INFO dory: {d} loops, {d} prompt tokens, {d} weight bytes on the GPU\n", .{ loops, prompt.items.len, rig.w.bytes });
    const n0 = prompt.items.len;
    const t0 = check.now(gpu.io);
    const first = try rig.rows(prompt.items, 0, true, &.{n0 - 1});
    defer gpa.free(first);
    const plain = try gpa.alloc(u32, count);
    defer gpa.free(plain);
    const kept = try gpa.alloc(f32, count * v); // each decode row's logits
    defer gpa.free(kept);
    plain[0] = argmax(first);
    const t1 = check.now(gpu.io);
    for (1..count) |i| {
        const row = try rig.rows(plain[i - 1 .. i], n0 + i - 1, false, &.{0});
        defer gpa.free(row);
        @memcpy(kept[i * v ..][0..v], row);
        plain[i] = argmax(row);
    }
    const t2 = check.now(gpu.io);
    const said = try tok.decode(gpa, plain, false);
    defer gpa.free(said);
    std.debug.print("RESULT dory plain: {d} tokens, prompt {d:.1} ms, decode {d:.2} ms a token: \"{s}\"\n", .{ count, @as(f64, @floatFromInt(t1 - t0)) / 1e6, @as(f64, @floatFromInt(t2 - t1)) / 1e6 / @as(f64, @floatFromInt(count - 1)), said });
    std.debug.print("RESULT dory ids:", .{});
    for (plain) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});
    // a verify window of w rows: the last kept token and w - 1 drafts copied from the plain run
    var bad: usize = 0;
    var windows: usize = 0;
    for ([_]usize{ 2, 3, 5, 8, 16 }) |w| {
        gpa.free(try rig.rows(prompt.items, 0, true, &.{n0 - 1})); // the caches as the plain run had them
        var at: usize = 1;
        while (at + w <= count) : (at += w) {
            var pick: [fwd.window_rows]usize = undefined;
            for (pick[0..w], 0..) |*p, j| p.* = j;
            const got = try rig.rows(plain[at - 1 .. at - 1 + w], n0 + at - 1, false, pick[0..w]);
            defer gpa.free(got);
            windows += 1;
            for (0..w) |j| if (!std.mem.eql(u8, std.mem.sliceAsBytes(got[j * v ..][0..v]), std.mem.sliceAsBytes(kept[(at + j) * v ..][0..v]))) {
                bad += 1;
                if (bad < 4) std.debug.print("FAIL dory window {d} at {d} row {d}: logits differ from the plain row's (argmax {d} vs {d})\n", .{ w, at, j, argmax(got[j * v ..][0..v]), plain[at + j] });
            };
        }
    }
    try check.expect(bad == 0, "dory drafted == plain: {d} rows differ in {d} windows", .{ bad, windows });
    check.pass("dory drafted == plain: {d} windows of 2 to 16 rows, every row's logits bit-equal to the plain row's", .{windows});
    try seams(&rig, prompt.items, first);
}

/// The prompt cut in two chunks (bit-equal: per_loop rows are chunk-invariant), and fed as decode rows (lane GEMM).
fn seams(rig: *Rig, prompt: []const u32, whole: []const f32) !void {
    const gpa = rig.gpu.gpa;
    const n = prompt.len;
    if (n < 2) return;
    const cut = n / 2;
    gpa.free(try rig.rows(prompt[0..cut], 0, true, &.{cut - 1}));
    const two = try rig.rows(prompt[cut..], cut, true, &.{n - cut - 1});
    defer gpa.free(two);
    const same = std.mem.eql(u8, std.mem.sliceAsBytes(two), std.mem.sliceAsBytes(whole));
    try check.expect(same, "dory prompt cut at {d} of {d}: last-row logits differ from one chunk's", .{ cut, n });
    check.pass("dory prompt cut at {d} of {d}: last-row logits bit-equal to one chunk's", .{ cut, n });
    var last: []f32 = &.{};
    defer gpa.free(last);
    for (prompt, 0..) |t, i| {
        gpa.free(last);
        last = try rig.rows(&.{t}, i, false, &.{0});
    }
    var most: f32 = 0;
    var top: f32 = 0;
    for (last, whole) |a, b| {
        most = @max(most, @abs(a - b));
        top = @max(top, @abs(b));
    }
    try check.expect(argmax(last) == argmax(whole), "dory prompt as decode rows: first token {d}, the prompt GEMM's {d}", .{ argmax(last), argmax(whole) });
    check.pass("dory prompt as decode rows: the same first token, logits within {e} (largest {e})", .{ most, top });
}
