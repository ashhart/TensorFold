//! Developer probe of Parse model seams against the pinned HF example; metrics are not a qualification gate.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const processor = @import("processor.zig");
const Weights = @import("weights.zig").Weights;
const Compute = @import("compute.zig").Compute;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) {
        std.debug.print("usage: tf-nemotron-parse-test MODEL REFERENCE OUTPUT\n", .{});
        return error.InvalidArguments;
    }
    const a = init.gpa;
    const io = init.io;
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    const room = try ctx.memInfo();
    if (room.free < 5 * (1 << 30)) return error.InsufficientProbeMemory;
    std.debug.print("PROBE: free VRAM {d} MiB, allocations below 3 GiB\n", .{room.free >> 20});
    var s = try hip.Stream.init(&r);
    defer s.deinit();
    var l = try hip.Launcher.load(&r, .{}, hip.kernels.images);
    defer l.unload();
    l.wide = false; // use the same 16-row attention arithmetic for prefill, decode and verification
    const kernels = try hip.vision.Kernels.load(l.mods[0]);
    var w = try Weights.load(a, &r, io, args[1]);
    defer w.deinit();
    const image_path = try std.fs.path.join(a, &.{ args[2], "input.image" });
    defer a.free(image_path);
    const image = try std.Io.Dir.cwd().readFileAlloc(io, image_path, a, .limited(16 << 20));
    defer a.free(image);
    var pixels = try processor.prepare(io, a, image);
    defer pixels.deinit(a);
    var raw = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(pixels.pixels));
    defer raw.free();
    var scratch = try hip.Arena.init(&r, 1 << 30);
    defer scratch.deinit();
    var prefix = try hip.DeviceBuffer.alloc(&r, pixels.geometry.tokens() * 1024 * 2);
    defer prefix.free();
    try std.Io.Dir.cwd().createDirPath(io, args[3]);
    var tap: Probe = .{ .a = a, .io = io, .r = &r, .s = s, .reference = args[2], .output = args[3] };
    const c: Compute = .{ .ops = .{ .l = &l, .stream = s.handle, .bf16 = true, .arena = &scratch }, .kernels = kernels, .stream = s, .weights = &w, .tap = .{ .ctx = &tap, .record = Probe.record } };
    const input = try c.alloc(pixels.pixels.len, .bf16);
    try c.ops.cast(.{ .ptr = raw.base(), .kind = .f32 }, input, pixels.pixels.len);
    try @import("tower.zig").encode(c, input, pixels.geometry, .{ .ptr = prefix.base(), .kind = .bf16 });
    try s.synchronize();
    scratch.reset();
    var state = try @import("decoder.zig").State.init(&r, 16384);
    defer state.deinit();
    const ids_path = try std.fs.path.join(a, &.{ args[2], "input.input_ids.bin" });
    defer a.free(ids_path);
    const id_bytes = try std.Io.Dir.cwd().readFileAlloc(io, ids_path, a, .limited(1 << 20));
    defer a.free(id_bytes);
    if (id_bytes.len == 0 or id_bytes.len % 8 != 0) return error.BadReferenceIds;
    const ids = try a.alloc(i32, id_bytes.len / 8);
    defer a.free(ids);
    for (ids, 0..) |*id, i| {
        const value = std.mem.readInt(i64, id_bytes[i * 8 ..][0..8], .little);
        if (value < 0 or value >= 72512) return error.BadReferenceIds;
        id.* = @intCast(value);
    }
    var dev_ids = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(ids));
    defer dev_ids.free();
    const total = pixels.geometry.tokens() + ids.len;
    const embeddings = try c.alloc(total * 1024, .bf16);
    try c.ops.copyCols(.{ .ptr = prefix.base(), .kind = .bf16 }, 1024, 0, embeddings.ptr, pixels.geometry.tokens(), 1024);
    try kernels.embed(s, try w.get("decoder.embed_tokens.weight"), dev_ids.base(), .{ .ptr = embeddings.ptr + pixels.geometry.tokens() * 1024 * 2, .kind = .bf16 }, ids.len, 1024, 32);
    _ = try @import("decoder.zig").forward(c, &state, embeddings, total);
    const logits = try @import("decoder.zig").logits(c, try state.hiddenAt(total - 1), 1);
    try c.record("prefill.logits", logits, 1, 72512);
    var pick = try hip.DeviceBuffer.alloc(&r, 4);
    defer pick.free();
    try c.ops.argmaxRows(logits, 1, 72512, pick.base());
    try s.synchronize();
    var token: i32 = undefined;
    try pick.download(0, std.mem.asBytes(&token));
    std.debug.print("PREFILL first token {d}\n", .{token});
    const manifest_path = try std.fs.path.join(a, &.{ args[2], "manifest.json" });
    defer a.free(manifest_path);
    const manifest_bytes = try std.Io.Dir.cwd().readFileAlloc(io, manifest_path, a, .limited(4 << 20));
    defer a.free(manifest_bytes);
    const manifest = try std.json.parseFromSlice(std.json.Value, a, manifest_bytes, .{});
    defer manifest.deinit();
    const generated = manifest.value.object.get("new_ids").?.array.items;
    try Probe.record(&tap, "generation.logits.0", logits, 1, 72512);
    var dc = c;
    dc.tap = null;
    for (1..generated.len) |step| {
        var previous: i32 = @intCast(generated[step - 1].integer);
        if (previous < 0 or previous >= 72512) return error.BadReferenceIds;
        try dev_ids.upload(0, std.mem.asBytes(&previous));
        scratch.reset();
        const embedded = try @import("decoder.zig").embed(dc, dev_ids.base(), 1);
        const hidden = try @import("decoder.zig").forward(dc, &state, embedded, 1);
        const next_logits = try @import("decoder.zig").logits(dc, hidden, 1);
        var name: [64]u8 = undefined;
        try Probe.record(&tap, try std.fmt.bufPrint(&name, "generation.logits.{d}", .{step}), next_logits, 1, 72512);
    }
    std.debug.print("PROBE: {d} teacher-forced generation rows saved for HF logits-processor comparison\n", .{generated.len});

    try @import("model_check.zig").run(a, dc, &r, .{ .ptr = prefix.base(), .kind = .bf16 }, pixels.geometry.tokens(), ids, &state);

    std.debug.print("PROBE: image tokens {d}, scratch peak {d} MiB; model-seam metrics only\n", .{ pixels.geometry.tokens(), scratch.peak >> 20 });
}

const Probe = struct {
    a: std.mem.Allocator,
    io: std.Io,
    r: *const hip.Runtime,
    s: hip.Stream,
    reference: []const u8,
    output: []const u8,

    fn record(ctx: *anyopaque, name: []const u8, x: hip.ops.Tensor, rows: usize, width: usize) !void {
        const t: *Probe = @ptrCast(@alignCast(ctx));
        if (x.kind != .bf16) return error.UnexpectedProbeDtype;
        try t.s.synchronize();
        const length = rows * width;
        const got = try t.a.alloc(u16, length);
        defer t.a.free(got);
        try hip.runtime.check(t.r.api.hipMemcpyDtoH(std.mem.sliceAsBytes(got).ptr, @ptrFromInt(x.ptr), length * 2));
        const file = try std.fmt.allocPrint(t.a, "{s}.bin", .{name});
        defer t.a.free(file);
        const path = try std.fs.path.join(t.a, &.{ t.reference, file });
        defer t.a.free(path);
        const expected_all = try std.Io.Dir.cwd().readFileAlloc(t.io, path, t.a, .limited(128 << 20));
        defer t.a.free(expected_all);
        const expected = if (std.mem.eql(u8, name, "prefill.logits") and expected_all.len >= length * 2)
            expected_all[expected_all.len - length * 2 ..]
        else
            expected_all;
        if (expected.len != length * 2) return error.ProbeShapeMismatch;
        var equal: usize = 0;
        var worst: f64 = 0;
        var absolute: f64 = 0;
        for (0..rows) |row| {
            var square: f64 = 0;
            var norm: f64 = 0;
            for (0..width) |col| {
                const i = row * width + col;
                const bits = std.mem.readInt(u16, expected[i * 2 ..][0..2], .little);
                const want: f64 = core.bf16_round.value(bits);
                const value: f64 = core.bf16_round.value(got[i]);
                if (!std.math.isFinite(value) or !std.math.isFinite(want)) return error.NonFiniteProbe;
                equal += @intFromBool(got[i] == bits);
                square += (value - want) * (value - want);
                norm += want * want;
                absolute = @max(absolute, @abs(value - want));
            }
            worst = @max(worst, @sqrt(square / @max(norm, 1e-30)));
        }
        std.debug.print("{s}: {d}x{d}, equal {d}/{d}, max row relative L2 {d:.7}, max abs {d:.7}\n", .{ name, rows, width, equal, length, worst, absolute });
        const out_path = try std.fs.path.join(t.a, &.{ t.output, file });
        defer t.a.free(out_path);
        try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = out_path, .data = std.mem.sliceAsBytes(got) });
    }
};
