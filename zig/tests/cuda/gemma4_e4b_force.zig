//! Gemma 4 E4B on CUDA against a reference directory (gemma4_e4b_reference.py): its tokens forced, logits written.
const std = @import("std");
const cuda = @import("cuda");
const g4 = @import("gemma4_cuda");
const check = @import("check.zig");

const fwd = g4.forward;
const vocab = g4.config.vocab;

const Meta = struct { name: []const u8, prompt_ids: []const u32, generated_ids: []const u32 };

/// The last-row logits of a forward as fp32, appended to `rows`; its top token.
fn take(m: *fwd.Model, s: cuda.Stream, row: usize, bf: []u16, rows: *std.ArrayList(f32), gpa: std.mem.Allocator) !u32 {
    try s.synchronize();
    try m.d.check(m.d.api.cuMemcpyDtoH_v2(bf.ptr, m.s.logits + row * vocab * 2, vocab * 2), "cuMemcpyDtoH");
    var best: u32 = 0;
    var top: f32 = -std.math.inf(f32);
    for (bf, 0..) |b, i| {
        const v: f32 = @bitCast(@as(u32, b) << 16);
        try rows.append(gpa, v);
        if (v > top) {
            top = v;
            best = @intCast(i);
        }
    }
    return best;
}

fn npy(io: std.Io, path: []const u8, data: []const f32, steps: usize) !void {
    var head: [128]u8 = undefined;
    const dict = try std.fmt.bufPrint(&head, "{{'descr': '<f4', 'fortran_order': False, 'shape': ({d}, {d}), }}", .{ steps, vocab });
    var buf: [128]u8 = @splat(' ');
    @memcpy(buf[0..dict.len], dict);
    buf[127] = '\n';
    var f = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer f.close(io);
    var wb: [1 << 16]u8 = undefined;
    var w = f.writer(io, &wb);
    try w.interface.writeAll("\x93NUMPY\x01\x00");
    try w.interface.writeInt(u16, 118, .little);
    try w.interface.writeAll(buf[0..118]);
    try w.interface.writeAll(std.mem.sliceAsBytes(data));
    try w.interface.flush();
}

/// tf-cuda-test gemma4-e4b-force MODEL REF OUT NAME...: per prompt, prefill in 512-row chunks, then each forced token.
pub fn run(gpu: check.Gpu, rest: []const []const u8) !void {
    if (rest.len < 4) return error.MissingArgument;
    const gpa = gpu.gpa;
    var k = try g4.kernels.Kernels.load(gpu.ctx);
    defer k.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var w = try g4.weights.load(gpa, gpu.io, gpu.d, s, &k, rest[0], null, false);
    defer w.deinit();
    var m = try fwd.Model.init(gpa, gpu.d, s, &w, 8192, 1);
    defer m.deinit();
    const bf = try gpa.alloc(u16, vocab);
    defer gpa.free(bf);
    for (rest[3..]) |name| {
        const meta_path = try std.fmt.allocPrint(gpa, "{s}/{s}.json", .{ rest[1], name });
        defer gpa.free(meta_path);
        const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, meta_path, gpa, .limited(1 << 26));
        defer gpa.free(text);
        const meta = try std.json.parseFromSlice(Meta, gpa, text, .{ .ignore_unknown_fields = true });
        defer meta.deinit();
        const ids = meta.value.prompt_ids;
        const gen = meta.value.generated_ids;
        var rows: std.ArrayList(f32) = .empty;
        defer rows.deinit(gpa);
        var top: std.ArrayList(u32) = .empty;
        defer top.deinit(gpa);
        var at: usize = 0;
        const t0 = std.Io.Clock.awake.now(gpu.io);
        while (at < ids.len) : (at += fwd.prompt_rows) {
            const n = @min(fwd.prompt_rows, ids.len - at);
            const last = at + n == ids.len;
            try m.forward(&k, s, &.{.{ .slot = 0, .p0 = at, .tokens = ids[at..][0..n] }}, true, if (last) n - 1 else 0, @intFromBool(last));
        }
        try top.append(gpa, try take(&m, s, 0, bf, &rows, gpa));
        const t1 = std.Io.Clock.awake.now(gpu.io);
        for (gen[0 .. gen.len - 1], 0..) |t, i| {
            try m.forward(&k, s, &.{.{ .slot = 0, .p0 = ids.len + i, .tokens = &.{t} }}, false, 0, 1);
            try top.append(gpa, try take(&m, s, 0, bf, &rows, gpa));
        }
        const t2 = std.Io.Clock.awake.now(gpu.io);
        const out = try std.fmt.allocPrint(gpa, "{s}/{s}.logits.npy", .{ rest[2], name });
        defer gpa.free(out);
        try npy(gpu.io, out, rows.items, top.items.len);
        var same: usize = 0;
        for (top.items, gen) |a, b| same += @intFromBool(a == b);
        const pre_ms = @as(f64, @floatFromInt(t0.durationTo(t1).toNanoseconds())) / 1e6;
        const dec_ms = @as(f64, @floatFromInt(t1.durationTo(t2).toNanoseconds())) / 1e6;
        std.debug.print("E4B {s}: prompt {d}, forced {d}, top-1 equal {d}/{d}, prefill {d:.1} ms, decode {d:.2} ms/token (eager reads)\n", .{ name, ids.len, gen.len, same, gen.len, pre_ms, dec_ms / @as(f64, @floatFromInt(gen.len - 1)) });
    }
    check.pass("gemma4 e4b forced logits written to {s}", .{rest[2]});
}
