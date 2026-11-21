//! `audio-check MODEL WAV OUT [layers]`: one clip's every encoder stage to OUT (f32 safetensors, the dumps' names).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const nemo_audio = @import("nemo_audio");
const check = @import("check.zig");

const Stage = struct { cols: usize, rows: usize = 0, values: std.ArrayList(f32) = .empty };

const Collect = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stages: std.StringArrayHashMapUnmanaged(Stage) = .empty,

    fn put(ctx: *anyopaque, s: cuda.Stream, name: []const u8, at: u64, row0: usize, rows: usize, cols: usize, ld: usize, kind: nemo_audio.Kind) anyerror!void {
        const c: *Collect = @ptrCast(@alignCast(ctx));
        const got = try c.stages.getOrPut(c.gpa, name);
        if (!got.found_existing) {
            got.key_ptr.* = try c.gpa.dupe(u8, name);
            got.value_ptr.* = .{ .cols = cols };
        }
        const st = got.value_ptr;
        // a stage written once a call (row0 0 each time) runs on from the call before
        if (st.cols != cols or (st.rows != row0 and row0 != 0)) return error.StageRows;
        if (rows == 0) return;
        try s.synchronize();
        const size: usize = if (kind == .f32) 4 else 2;
        const raw = try c.gpa.alloc(u8, ((rows - 1) * ld + cols) * size);
        defer c.gpa.free(raw);
        try cuda.memory.download(c.d, at, raw);
        for (0..rows) |r| for (0..cols) |j| {
            const i = r * ld + j;
            const v: f32 = if (kind == .f32) std.mem.bytesAsValue(f32, raw[4 * i ..][0..4]).* else @bitCast(@as(u32, std.mem.readInt(u16, raw[2 * i ..][0..2], .little)) << 16);
            try st.values.append(c.gpa, v);
        };
        st.rows += rows;
    }

    fn deinit(c: *Collect) void {
        for (c.stages.keys(), c.stages.values()) |k, *v| {
            c.gpa.free(k);
            v.values.deinit(c.gpa);
        }
        c.stages.deinit(c.gpa);
    }
};

pub fn run(gpu: check.Gpu, model: []const u8, wav_path: []const u8, out_path: []const u8, layers: bool) !void {
    const a = gpu.gpa;
    const data = try std.Io.Dir.cwd().readFileAlloc(gpu.io, wav_path, a, .limited(1 << 31));
    defer a.free(data);
    const samples = try nemo_audio.wav.decode(a, data);
    defer a.free(samples);
    var enc = try nemo_audio.Encoder.init(a, gpu.io, gpu.d, model);
    defer enc.deinit();
    const rows = nemo_audio.Encoder.rows(samples.len);
    var out = try cuda.DeviceBuffer.alloc(gpu.d, rows * enc.c.output * 2);
    defer out.free();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    // one pass to warm the Linears' plans, then the timed one, then the one that writes every stage
    _ = try enc.encode(s, samples, out.ptr, null);
    try s.synchronize();
    const t0 = check.now(gpu.io);
    const written = try enc.encode(s, samples, out.ptr, null);
    try s.synchronize();
    const ms = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6;
    var c: Collect = .{ .gpa = a, .d = gpu.d };
    defer c.deinit();
    _ = try enc.encode(s, samples, out.ptr, .{ .ctx = &c, .put = Collect.put, .layers = layers });
    try s.synchronize();
    var tensors: std.ArrayList(core.shard_edit.New) = .empty;
    defer tensors.deinit(a);
    var shapes: std.ArrayList([2]usize) = .empty;
    defer shapes.deinit(a);
    try shapes.ensureTotalCapacity(a, c.stages.count());
    for (c.stages.keys(), c.stages.values()) |k, v| {
        shapes.appendAssumeCapacity(.{ v.rows, v.cols });
        try tensors.append(a, .{ .name = k, .dtype = .f32, .shape = &shapes.items[shapes.items.len - 1], .bytes = std.mem.sliceAsBytes(v.values.items) });
    }
    try core.shard_edit.create(a, out_path, &.{}, tensors.items);
    std.debug.print("RESULT audio-check {s}: {d} samples, {d} rows, {d} stages, encode {d:.1} ms\n", .{ wav_path, samples.len, written, tensors.items.len, ms });
    try check.expect(written == rows, "audio-check: {d} rows written, expected {d}", .{ written, rows });
}
