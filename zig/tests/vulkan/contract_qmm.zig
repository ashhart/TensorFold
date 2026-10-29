//! Vulkan Q4 affine fixture; its push ABI and staging discipline mirror affine_fixture.zig and HIP's fixture hooks.
const std = @import("std");
const vk = @import("vulkan");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const dispatch = vk.dispatch;
const group: usize = 64;
pub const local_size_x: u32 = 256;
pub const poison: u16 = 0x7fc5;
const timeout_ns: u64 = 10 * std.time.ns_per_s;

pub const Params = extern struct {
    count: u32 = 0,
    operation: u32 = 4,
    lhs_size: u32 = 0,
    rhs_size: u32 = 0,
    reduce_size: u32 = 64,
    output_size: u32 = 0,
    lhs_offset: u32 = 0,
    rhs_offset: u32 = 0,
    output_offset: u32 = 0,
    aux_size: u32 = 0,
    aux_offset: u32 = 0,
    matrix_m: u32 = 0,
    matrix_n: u32 = 0,
    matrix_k: u32 = 0,
    flags: u32 = 0,
    alpha: f32 = 1,
    beta: f32 = 0,
    dims: u32 = 0,
    shape: [4]u32 = .{ 1, 0, 0, 0 },
    in_strides: [4]u32 = .{ 0, 0, 0, 0 },
    out_strides: [4]u32 = .{ 0, 0, 0, 0 },
};
comptime {
    if (@sizeOf(Params) != 120) @compileError("qmm.comp push constants must stay 120 bytes");
}
fn depth(s: c.Shape) usize {
    return s.k + 2;
}
// The shapes the Qwen3.5-2B engine calls the projection at (K 2048 and 6144, N from the 16-wide gates up to the
// 248320-row head, M = 1 for decode and up to 32 rows for a prefill chunk), plus the small fixture shapes.
const served = [_]c.Shape{
    .{ .rows = 1, .n = 16, .k = 64 },      .{ .rows = 2, .n = 257, .k = 128 },
    .{ .rows = 1, .n = 16, .k = 2048 },    .{ .rows = 2, .n = 16, .k = 6144 },
    .{ .rows = 1, .n = 6144, .k = 2048 },  .{ .rows = 32, .n = 6144, .k = 2048 },
    .{ .rows = 32, .n = 2048, .k = 6144 }, .{ .rows = 1, .n = 248320, .k = 2048 },
};
pub const spec: c.Spec = .{
    .name = "vulkan qmm_bf16 affine q4 g64 transposed",
    .accum = .f32,
    .depth = depth,
    .out = .bf16,
    .tiles = .{ .{}, .{ .size = 256 }, .{ .size = 64, .step = 64 } },
    .served = &served,
    // gate 4 keeps the harness default: fp16 and bf16 sums must miss by 100x (no claim that output rounding hides them)
};

/// The shared gate harness expects a rig object; as in HIP, it exposes bytes while the fixture owns device buffers.
const Rig = struct {
    output: []const u8 = &.{},
    inputs: []const u8 = &.{},
    guards_ok: bool = false,
    pub fn outputBytes(r: Rig) []const u8 {
        return r.output;
    }
    pub fn inputBytes(r: Rig) []const u8 {
        return r.inputs;
    }
    pub fn intact(r: Rig) bool {
        return r.guards_ok;
    }
    pub fn overlap(_: Rig) bool {
        return false;
    }
};

// Caps on what prepare() will build on the host (codes are one byte each, so the largest served shape,
// the 248320-row head at K 2048, takes 0.5 GB of codes plus 0.25 GB of packed words). A served shape over a cap would be
// refused and fail gate 3; the caps exist only to bound the host allocation of the generated edge shapes.
const max_weights: usize = 520_000_000;
const max_outputs: usize = 300_000;

pub const Fixture = struct {
    a: std.mem.Allocator,
    dev: *const vk.Device,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    pipeline: *dispatch.Pipeline,
    shape: c.Shape = .{ .rows = 1, .n = 16, .k = 64 },
    affine: ?contract.affine.Affine = null,
    x: []u16 = &.{},
    w: []u32 = &.{},
    scales: []u16 = &.{},
    biases: []u16 = &.{},
    out: []u16 = &.{},
    want: []f64 = &.{},
    magnitude: []f64 = &.{},
    input_snapshot: []u8 = &.{},
    rig: Rig = .{},
    buffers: [5]vk.Buffer = undefined,
    made: usize = 0,
    bindings: [5]dispatch.Binding = undefined,
    /// Planted control: after the dispatch, overwrite this input buffer (0 x, 1 w, 2 scales, 3 biases) as a kernel that
    /// writes through its readonly input would; gate 2 must catch it now that the inputs are read back from the device.
    corrupt: ?usize = null,

    pub fn init(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, pipeline: *dispatch.Pipeline) Fixture {
        return .{ .a = a, .dev = dev, .cmds = cmds, .staging = staging, .pipeline = pipeline };
    }
    pub fn deinit(f: *Fixture) void {
        f.reset();
    }
    fn reset(f: *Fixture) void {
        while (f.made > 0) {
            f.made -= 1;
            f.buffers[f.made].deinit();
        }
        if (f.affine) |q| q.deinit();
        f.affine = null;
        for ([_][]u16{ f.x, f.scales, f.biases, f.out }) |b| if (b.len != 0) f.a.free(b);
        if (f.w.len != 0) f.a.free(f.w);
        for ([_][]f64{ f.want, f.magnitude }) |b| if (b.len != 0) f.a.free(b);
        if (f.input_snapshot.len != 0) f.a.free(f.input_snapshot);
        f.x = &.{};
        f.w = &.{};
        f.scales = &.{};
        f.biases = &.{};
        f.out = &.{};
        f.want = &.{};
        f.magnitude = &.{};
        f.input_snapshot = &.{};
        f.rig = .{};
    }
    // The table is a comptime constant: `&g.hooks(Fixture)` would point at a stack temporary that dies when kernel() returns.
    const vt = g.hooks(Fixture);
    const corrupt_names = [4][]const u8{ "control: qmm writes its activations", "control: qmm writes its weights", "control: qmm writes its scales", "control: qmm writes its biases" };
    pub fn kernel(f: *Fixture) g.Kernel {
        if (f.corrupt) |i| {
            var planted = spec;
            planted.name = corrupt_names[i];
            return .{ .spec = planted, .ctx = f, .vt = &vt, .control = .repeat };
        }
        return .{ .spec = spec, .ctx = f, .vt = &vt };
    }
    fn allocPoison(a: std.mem.Allocator, data_len: usize) ![]u16 {
        const extra: usize = if (data_len % 2 == 0) 16 else 17;
        const p = try a.alloc(u16, data_len + extra);
        @memset(p, poison);
        return p;
    }
    pub fn prepare(f: *Fixture, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k % group != 0 or shape.n * shape.k > max_weights or shape.rows * shape.n > max_outputs) return false;
        f.reset();
        f.shape = shape;
        var q = try contract.affine.Affine.init(f.a, shape, 4, group);
        errdefer q.deinit();
        q.form = .dequantized;
        try q.fill(switch (fill) {
            .random => .random,
            .nonfinite => .nonfinite,
            .stress => .stress,
        }, seed);
        const x = try allocPoison(f.a, shape.rows * shape.k);
        errdefer f.a.free(x);
        @memcpy(x[0..q.x.len], q.x);
        const words_len = shape.n * shape.k / 8;
        const w = try f.a.alloc(u32, words_len + 4);
        errdefer f.a.free(w);
        @memset(w, 0xffff_ffff);
        q.words(w[0..words_len]);
        const nparam = shape.n * (shape.k / group);
        const scales = try allocPoison(f.a, nparam);
        errdefer f.a.free(scales);
        @memcpy(scales[0..nparam], q.scales);
        const biases = try allocPoison(f.a, nparam);
        errdefer f.a.free(biases);
        @memcpy(biases[0..nparam], q.biases);
        const out = try allocPoison(f.a, shape.rows * shape.n);
        errdefer f.a.free(out);
        const want = try f.a.alloc(f64, shape.rows * shape.n);
        errdefer f.a.free(want);
        const mag = try f.a.alloc(f64, want.len);
        errdefer f.a.free(mag);
        try q.reference(want, mag);
        if (fill != .nonfinite) for (want, mag) |v, m| if (!std.math.isFinite(v) or !std.math.isFinite(m)) return error.NonFiniteReference;
        f.x = x;
        f.w = w;
        f.scales = scales;
        f.biases = biases;
        f.out = out;
        f.want = want;
        f.magnitude = mag;
        f.affine = q;
        q = undefined;
        const data = [_][]const u8{ std.mem.sliceAsBytes(f.x), std.mem.sliceAsBytes(f.w), std.mem.sliceAsBytes(f.scales), std.mem.sliceAsBytes(f.biases), std.mem.sliceAsBytes(f.out) };
        var input_total: usize = 0;
        for (data[0..4]) |bytes| input_total += bytes.len;
        f.input_snapshot = try f.a.alloc(u8, input_total);
        var pos: usize = 0;
        for (data[0..4]) |bytes| {
            @memcpy(f.input_snapshot[pos..][0..bytes.len], bytes);
            pos += bytes.len;
            f.buffers[f.made] = try vk.Buffer.init(f.dev, .device, bytes.len);
            const b = &f.buffers[f.made];
            f.made += 1;
            f.bindings[f.made - 1] = .{ .buffer = b, .len = bytes.len };
            try vk.orExit(vk.upload(f.cmds, b.*, 0, f.staging, bytes, timeout_ns));
        }
        f.buffers[f.made] = try vk.Buffer.init(f.dev, .device, data[4].len);
        f.bindings[4] = .{ .buffer = &f.buffers[f.made], .len = data[4].len };
        f.made += 1;
        try f.pipeline.bind(&f.bindings);
        f.rig = .{ .output = std.mem.sliceAsBytes(f.out), .inputs = f.input_snapshot, .guards_ok = true };
        return true;
    }
    pub fn run(f: *Fixture) !void {
        const s = f.shape;
        const p = Params{ .count = @intCast(s.rows * s.n), .output_size = @intCast(s.rows * s.n), .matrix_m = @intCast(s.rows), .matrix_n = @intCast(s.n), .matrix_k = @intCast(s.k) };
        try vk.orExit(vk.upload(f.cmds, f.buffers[4], 0, f.staging, std.mem.sliceAsBytes(f.out), timeout_ns));
        try vk.orExit(f.pipeline.run(f.cmds, std.mem.asBytes(&p), .{ @intCast((s.rows * s.n + local_size_x - 1) / local_size_x), 1, 1 }, timeout_ns));
        if (f.corrupt) |i| {
            var from: usize = 0;
            for (f.bindings[0..i]) |b| from += b.len;
            const len: usize = f.bindings[i].len;
            const bad = try f.a.dupe(u8, f.input_snapshot[from..][0..len]);
            defer f.a.free(bad);
            bad[len - 1] ^= 0x5a; // the last byte: a readback that stops one short misses it
            try vk.orExit(vk.upload(f.cmds, f.buffers[i], 0, f.staging, bad, timeout_ns));
        }
        try vk.orExit(vk.download(f.cmds, f.buffers[4], 0, f.staging, std.mem.sliceAsBytes(f.out), timeout_ns));
        // Gate 2 compares the inputs before and after the runs, so what it sees must be what the device holds now:
        // read the four input buffers back into the snapshot (the host copy taken at upload would make the check vacuous).
        var at: usize = 0;
        for (f.bindings[0..4]) |b| {
            try vk.orExit(vk.download(f.cmds, b.buffer.*, 0, f.staging, f.input_snapshot[at..][0..b.len], timeout_ns));
            at += b.len;
        }
        const data_count = s.rows * s.n;
        f.rig.guards_ok = true;
        for (f.out[data_count..]) |v| if (v != poison) {
            f.rig.guards_ok = false;
            break;
        };
        f.rig.output = std.mem.sliceAsBytes(f.out);
    }
    pub fn count(f: *Fixture) usize {
        return f.want.len;
    }
    pub fn reference(f: *Fixture, want: []f64, mag: []f64) !void {
        @memcpy(want, f.want);
        @memcpy(mag, f.magnitude);
    }
    pub fn simulate(f: *Fixture, accum: c.Accum, order: contract.rows.Order, out: []f64) !void {
        return f.affine.?.simulate(accum, order, out);
    }
};

/// One source shader variant, exercised through the same hook contract as HIP's planted controls.
pub const ControlKind = enum { qmm_fmax, atomic_sum, scribble_sum, drop_tail, fp16_accum };
fn controlGate(kind: ControlKind) g.Gate {
    return switch (kind) {
        .qmm_fmax => .nonfinite,
        .atomic_sum, .scribble_sum => .repeat,
        .drop_tail => .shapes,
        .fp16_accum => .sums,
    };
}
fn controlShape(kind: ControlKind) c.Shape {
    return switch (kind) {
        .qmm_fmax => .{ .rows = 5, .n = 1, .k = 4096 },
        .atomic_sum, .scribble_sum => .{ .rows = 8, .n = 1, .k = 4096 },
        .drop_tail => .{ .rows = 4, .n = 1, .k = 4097 },
        .fp16_accum => .{ .rows = 2, .n = 1, .k = 4096 },
    };
}
fn controlOp(kind: ControlKind) contract.rows.Op {
    return switch (kind) {
        .qmm_fmax => .max,
        .fp16_accum => .dot,
        else => .sum,
    };
}
fn controlDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 3;
}

pub const Control = struct {
    a: std.mem.Allocator,
    dev: *const vk.Device,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    pipeline: *dispatch.Pipeline,
    kind: ControlKind,
    shape: c.Shape,
    served: [1]c.Shape,
    rows: ?contract.rows.Rows = null,
    x_device: ?vk.Buffer = null,
    w_device: ?vk.Buffer = null,
    y_device: ?vk.Buffer = null,
    out: []u32 = &.{},
    want: []f64 = &.{},
    magnitude: []f64 = &.{},
    input_bytes: []u8 = &.{},
    rig: Rig = .{},

    pub fn init(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, pipeline: *dispatch.Pipeline, kind: ControlKind) Control {
        const shape = controlShape(kind);
        return .{ .a = a, .dev = dev, .cmds = cmds, .staging = staging, .pipeline = pipeline, .kind = kind, .shape = shape, .served = .{shape} };
    }
    pub fn deinit(f: *Control) void {
        f.reset();
    }
    fn reset(f: *Control) void {
        if (f.x_device) |*b| b.deinit();
        if (f.w_device) |*b| b.deinit();
        if (f.y_device) |*b| b.deinit();
        if (f.rows) |r| r.deinit();
        if (f.out.len != 0) f.a.free(f.out);
        if (f.want.len != 0) f.a.free(f.want);
        if (f.magnitude.len != 0) f.a.free(f.magnitude);
        if (f.input_bytes.len != 0) f.a.free(f.input_bytes);
        f.x_device = null;
        f.w_device = null;
        f.y_device = null;
        f.rows = null;
        f.out = &.{};
        f.want = &.{};
        f.magnitude = &.{};
        f.input_bytes = &.{};
        f.rig = .{};
    }
    pub fn kernel(f: *Control) g.Kernel {
        const spec_control: c.Spec = .{
            .name = switch (f.kind) {
                .qmm_fmax => "control: qmm fmax drops NaN",
                .atomic_sum => "control: atomic add row sum",
                .scribble_sum => "control: row sum scribbles input",
                .drop_tail => "control: row sum drops K tail",
                .fp16_accum => "control: dot accumulates in fp16",
            },
            .accum = if (f.kind == .qmm_fmax) .none else .f32,
            .depth = controlDepth,
            .out = .f32,
            .served = &f.served,
            .tiles = if (f.kind == .drop_tail) .{ .{}, .{}, .{ .size = 256 } } else .{ .{}, .{}, .{} },
            .sums = if (f.kind == .fp16_accum) .{ .f16 = 100, .bf16 = 100 } else .{},
        };
        return .{ .spec = spec_control, .ctx = f, .vt = &control_vt, .control = controlGate(f.kind) };
    }

    pub fn prepare(f: *Control, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        f.reset();
        if (shape.k == 0 or shape.rows == 0 or shape.k > 8192 or shape.rows > 64) return false;
        var rows = try contract.rows.Rows.init(f.a, controlOp(f.kind), shape);
        rows.fill(fill, seed);
        if (f.kind == .qmm_fmax) @memset(rows.w, c.toBf16(1));
        f.rows = rows;
        const n = rows.outputs();
        f.out = try f.a.alloc(u32, n + 1);
        @memset(f.out, 0x7fc5_a5a5);
        f.want = try f.a.alloc(f64, n);
        f.magnitude = try f.a.alloc(f64, n);
        try rows.reference(f.want, f.magnitude);
        const x_bytes = std.mem.sliceAsBytes(rows.x);
        const w_bytes = std.mem.sliceAsBytes(rows.w);
        const padded_w = try f.a.alloc(u8, (w_bytes.len + 3) & ~@as(usize, 3));
        defer f.a.free(padded_w);
        @memset(padded_w, 0);
        @memcpy(padded_w[0..w_bytes.len], w_bytes);
        f.input_bytes = try f.a.alloc(u8, x_bytes.len + padded_w.len);
        @memcpy(f.input_bytes[0..x_bytes.len], x_bytes);
        @memcpy(f.input_bytes[x_bytes.len..], padded_w);
        f.x_device = try vk.Buffer.init(f.dev, .device, x_bytes.len);
        f.w_device = try vk.Buffer.init(f.dev, .device, padded_w.len);
        f.y_device = try vk.Buffer.init(f.dev, .device, std.mem.sliceAsBytes(f.out).len);
        try vk.orExit(vk.upload(f.cmds, f.x_device.?, 0, f.staging, x_bytes, timeout_ns));
        try vk.orExit(vk.upload(f.cmds, f.w_device.?, 0, f.staging, padded_w, timeout_ns));
        const binds = [_]dispatch.Binding{
            .{ .buffer = &f.x_device.?, .len = x_bytes.len },
            .{ .buffer = &f.w_device.?, .len = padded_w.len },
            .{ .buffer = &f.y_device.?, .len = std.mem.sliceAsBytes(f.out).len },
        };
        try f.pipeline.bind(&binds);
        f.rig = .{ .output = std.mem.sliceAsBytes(f.out), .inputs = f.input_bytes, .guards_ok = true };
        return true;
    }
    pub fn run(f: *Control) !void {
        const r = f.rows.?;
        const data = std.mem.sliceAsBytes(f.out[0..r.outputs()]);
        @memset(data, 0);
        f.out[r.outputs()] = 0x7fc5_a5a5;
        try vk.orExit(vk.upload(f.cmds, f.y_device.?, 0, f.staging, std.mem.sliceAsBytes(f.out), timeout_ns));
        const push = .{ .rows = @as(u32, @intCast(r.shape.rows)), .k = @as(u32, @intCast(r.shape.k)) };
        const groups = if (f.kind == .atomic_sum)
            [3]u32{ @intCast((r.shape.k + 255) / 256), @intCast(r.shape.rows), 1 }
        else
            [3]u32{ @intCast(r.shape.rows), 1, 1 };
        try vk.orExit(f.pipeline.run(f.cmds, std.mem.asBytes(&push), groups, timeout_ns));
        try vk.orExit(vk.download(f.cmds, f.y_device.?, 0, f.staging, std.mem.sliceAsBytes(f.out), timeout_ns));
        const x_bytes = std.mem.sliceAsBytes(r.x);
        try vk.orExit(vk.download(f.cmds, f.x_device.?, 0, f.staging, @constCast(x_bytes), timeout_ns));
        @memcpy(f.input_bytes[0..x_bytes.len], x_bytes);
        f.rig.output = std.mem.sliceAsBytes(f.out);
        f.rig.inputs = f.input_bytes;
        f.rig.guards_ok = f.out[r.outputs()] == 0x7fc5_a5a5;
    }
    pub fn count(f: *Control) usize {
        return f.want.len;
    }
    pub fn reference(f: *Control, want: []f64, mag: []f64) !void {
        @memcpy(want, f.want);
        @memcpy(mag, f.magnitude);
    }
    pub fn simulate(f: *Control, accum: c.Accum, order: contract.rows.Order, out: []f64) !void {
        return f.rows.?.simulate(accum, order, out);
    }
};

test "planted Vulkan contract controls bind to the intended gate" {
    for (std.enums.values(ControlKind), [_]g.Gate{ .nonfinite, .repeat, .repeat, .shapes, .sums }) |kind, gate|
        try std.testing.expectEqual(gate, controlGate(kind));
}
test "Vulkan bf16 fixture uses raw u16 storage and guards are aligned for storage bindings" {
    const values = try Fixture.allocPoison(std.testing.allocator, 3);
    defer std.testing.allocator.free(values);
    try std.testing.expectEqual(@as(usize, 20), values.len);
    try std.testing.expectEqual(poison, values[0]);
    try std.testing.expectEqual(@as(usize, 2), @sizeOf(u16));
}

// Comptime constant for the same reason as Fixture.vt: the Kernel keeps a pointer to it.
const control_vt = g.hooks(Control);
