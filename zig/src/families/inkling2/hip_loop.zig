//! Inkling 2's loop kernels on HIP (zig/kernels/hip/inkling2/loop.hip): their widths, shapes, checks and launches.
const std = @import("std");
const hip = @import("hip");
const config = @import("config.zig");

/// The hidden widths loop.hip is built at: Inkling 2 Mini's 2,304 and tiny-inkling2-mini's 256.
pub const widths = [_]u32{ 2304, 256 };
pub const max_hidden = std.mem.max(u32, &widths);
/// Each width's symbol suffix in the image: Mini's kernels carry none.
const suffixes = [widths.len][:0]const u8{ "", "_w256" };

/// One of loop.hip's compile-time widths.
pub const Width = enum(u1) {
    w2304,
    w256,

    pub fn hidden(w: Width) u32 {
        return widths[@backingInt(w)];
    }

    /// The width built at `h`, or null.
    pub fn of(h: u32) ?Width {
        for (widths, 0..) |x, i| if (x == h) return @fromBackingInt(@intCast(i));
        return null;
    }
};

/// Four message roles (q, k, v, mlp) a core layer, the gate's rank 64.
pub const roles = 4;
pub const rank = 64;
/// Gates a launch (q, k and v share one), a down block's threads and ranks, an up block's columns and rows.
pub const max_gates = 3;
pub const down_threads = 256;
pub const down_ranks = 8;
pub const up_cols = 64;
pub const up_rows = 4;
/// Threads a message block, which holds one row of one message: wave 0 reduces bank 0, wave 1 bank 1.
pub const message_threads = 64;
/// Most rows a call takes: the CUDA engine's 2,048-row prompt chunk.
pub const max_rows = 2048;
/// Threads a math probe block (tf_ink2_loop_math).
pub const math_threads = 256;
/// rho's eps in the contract: float64 1e-6 narrowed to f32, as both checkpoints' rms_norm_eps parse.
pub const eps: f32 = @floatCast(@as(f64, 1e-6));

/// binary32(1 / sqrt(hidden)), rounded once from the float64 quotient: Mini's 1/48, the tiny checkpoint's 1/16.
pub fn queryScale(hidden: usize) f32 {
    return @floatCast(1 / @sqrt(@as(f64, @floatFromInt(hidden))));
}

/// A checkpoint's loop as the launches serve it: the width's kernels, its core layer count and rho's eps.
pub const Geometry = struct {
    width: Width,
    core_layers: u32,
    eps: f32,

    pub fn hidden(g: Geometry) u32 {
        return g.width.hidden();
    }
};

/// A checkpoint's loop geometry from its parsed text config, refusing by name, with `why` set, what the kernels lack.
pub fn geometry(t: *const config.Text, why: *config.Why) error{ Inkling2LoopRequired, UnsupportedLoopHidden, UnsupportedLoopGateRank }!Geometry {
    const lp = t.loop orelse {
        why.set("the HIP loop kernels need text_config.loop_config, which this checkpoint lacks", .{});
        return error.Inkling2LoopRequired;
    };
    const w = Width.of(t.hidden) orelse {
        why.set("hidden_size {d} has no HIP loop kernels: loop.hip is built at hidden {d} and {d} only", .{ t.hidden, widths[0], widths[1] });
        return error.UnsupportedLoopHidden;
    };
    if (lp.gate_rank != rank) {
        why.set("loop_config gate_rank {d} is not the HIP loop kernels' fixed rank {d}", .{ lp.gate_rank, rank });
        return error.UnsupportedLoopGateRank;
    }
    return .{ .width = w, .core_layers = lp.core_end - lp.core_start, .eps = t.eps };
}

/// The probe's functions: the baselines' f32 tanh, exp and 1 / sqrt as the kernels run them, then the device's own.
pub const Math = enum(u8) { tanh, exp, inv_sqrt, device_tanh, device_exp, device_rsqrt };

/// A message launch: the served kernel, or a control (rsqrt, fused share, no +0 start, serial rho, reversed butterfly).
pub const Messages = enum { served, rsqrt, fma, signed_zero, serial_rho, butterfly_desc };

/// A gate launch pair: the served kernels, or a control in the up half (tanhf, a fused last step) or the down half.
pub const Gates = enum {
    served,
    tanhf,
    fma,
    serial_rho,
    butterfly_desc,
    unfused_sq,

    /// The down kernel (served, serial rho, butterfly reversed, unfused squares) and the up one (served, tanhf, fma).
    pub fn halves(g: Gates) [2]usize {
        return switch (g) {
            .served => .{ 0, 0 },
            .tanhf => .{ 0, 1 },
            .fma => .{ 0, 2 },
            .serial_rho => .{ 1, 0 },
            .butterfly_desc => .{ 2, 0 },
            .unfused_sq => .{ 3, 0 },
        };
    }
};

/// One width's launches: the messages, the gate down and up halves, each its served kernel and then its controls.
pub const Set = struct {
    messages: [6]hip.Function,
    gate_down: [4]hip.Function,
    gate_up: [3]hip.Function,
};

const message_names = [_][:0]const u8{ "tf_ink2_loop_messages", "tf_ink2_loop_ctl_rsqrt", "tf_ink2_loop_ctl_fma", "tf_ink2_loop_ctl_signed_zero", "tf_ink2_loop_ctl_serial_rho", "tf_ink2_loop_ctl_butterfly_desc" };
const down_names = [_][:0]const u8{ "tf_ink2_loop_gate_down", "tf_ink2_loop_ctl_down_serial_rho", "tf_ink2_loop_ctl_down_butterfly_desc", "tf_ink2_loop_ctl_unfused_sq" };
const up_names = [_][:0]const u8{ "tf_ink2_loop_gate_up", "tf_ink2_loop_ctl_tanhf", "tf_ink2_loop_ctl_gate_fma" };

/// The inkling2_loop image's launches (hip.kernels.Group.inkling2_loop): the math probe and every width's set.
pub const Kernels = struct {
    mod: hip.Module,
    math: hip.Function,
    sets: [widths.len]Set,

    pub fn load(r: *const hip.Runtime, image: []align(8) const u8) !Kernels {
        if (image.len == 0) return error.BuiltWithoutKernels;
        var mod = try hip.Module.load(r, image);
        errdefer mod.unload();
        var k: Kernels = .{ .mod = mod, .math = try mod.function("tf_ink2_loop_math"), .sets = undefined };
        inline for (&k.sets, suffixes) |*one, suffix| {
            inline for (&one.messages, message_names) |*f, name| f.* = try mod.function(name ++ suffix);
            inline for (&one.gate_down, down_names) |*f, name| f.* = try mod.function(name ++ suffix);
            inline for (&one.gate_up, up_names) |*f, name| f.* = try mod.function(name ++ suffix);
        }
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }

    fn at(k: *const Kernels, w: Width) *const Set {
        return &k.sets[@backingInt(w)];
    }
};

fn int(x: anytype) c_int {
    return @intCast(x);
}

fn go(f: hip.Function, s: hip.Stream, grid: hip.launch.Dim3, threads: u32, a: *hip.launch.Args) !void {
    try goBlock(f, s, grid, .{ .x = threads }, a);
}

fn goBlock(f: hip.Function, s: hip.Stream, grid: hip.launch.Dim3, block: hip.launch.Dim3, a: *hip.launch.Args) !void {
    try hip.launch.launch(f, .{ .grid = grid, .block = block }, s, a);
}

/// `count` f32 inputs through one of the probe's functions.
pub fn math(k: *const Kernels, s: hip.Stream, in: u64, out: u64, count: u32, op: Math) !void {
    if (count == 0 or count > std.math.maxInt(c_int)) return error.Inkling2LoopShape;
    var a: hip.launch.Args = .{};
    for ([_]u64{ in, out }) |v| try a.add(v);
    try a.add(int(count));
    try a.add(int(@backingInt(op)));
    try go(k.math, s, .{ .x = std.math.divCeil(u32, count, math_threads) catch unreachable }, math_threads, &a);
}

/// One call's messages: `rows` rows, `count` core layers from the first whose query rows qm and kv point at.
pub const MessageCall = struct {
    bank0: u64, // the previous pass's last finished block (Mini's B_12, the tiny checkpoint's B_6), bf16 [rows][hidden]
    bank1: u64, // the previous pass's partial block (Mini's P_18, the tiny checkpoint's P_12), bf16 [rows][hidden]
    qm: u64, // middle_cycle_cross_pass_queries, f32, from row 2 * first: [2 * count][hidden]
    kv: u64, // middle_cycle_cross_pass_kv_queries, likewise
    out: u64, // f32 [count][4][rows][hidden]: each role's rows one contiguous plane
    rows: u32,
    count: u32,
};

/// The cross-pass messages of `c.count` of `g`'s core layers under HIP-RMS-W32 and HIP-RETRIEVE-W32.
pub fn messages(k: *const Kernels, g: Geometry, s: hip.Stream, c: MessageCall) !void {
    return messagesAs(k, g, s, .served, c);
}

/// `messages` through a contract control's kernel (or the served one).
pub fn messagesAs(k: *const Kernels, g: Geometry, s: hip.Stream, which: Messages, c: MessageCall) !void {
    if (c.rows == 0 or c.rows > max_rows or c.count == 0 or c.count > g.core_layers) return error.Inkling2LoopShape;
    for ([_]u64{ c.bank0, c.bank1, c.qm, c.kv, c.out }) |p| if (p % 16 != 0) return error.Inkling2LoopAlign;
    var a: hip.launch.Args = .{};
    for ([_]u64{ c.bank0, c.bank1, c.qm, c.kv, c.out }) |v| try a.add(v);
    try a.add(int(c.rows));
    try a.add(queryScale(g.hidden()));
    try a.add(g.eps);
    const grid: hip.launch.Dim3 = .{ .x = c.rows, .y = c.count * roles };
    try go(k.at(g.width).messages[@backingInt(which)], s, grid, message_threads, &a);
}

/// One gate's device tensors (loop.hip's Gate): h may be out, the gate then writes over its own input.
pub const Gate = extern struct {
    h: u64, // the raw AR output, f32 [rows][hidden]
    m: u64, // its role's message, f32 [rows][hidden]
    down: u64, // bf16 [64][2 * hidden], h's half first
    forget: u64, // bf16 [hidden][64]
    gain: u64, // bf16 [hidden][64]
    zeta: u64, // bf16 [rows][64] scratch between the two launches
    out: u64, // f32 [rows][hidden]
};

/// The message gate of 1 to 3 gates over `rows` rows of `g`'s width: one down launch and one up launch for them all.
pub fn gates(k: *const Kernels, g: Geometry, s: hip.Stream, calls: []const Gate, rows: u32) !void {
    return gatesAs(k, g, s, .served, calls, rows);
}

/// `gates` through a contract control's down or up kernel (or the served pair).
pub fn gatesAs(k: *const Kernels, g: Geometry, s: hip.Stream, which: Gates, calls: []const Gate, rows: u32) !void {
    if (calls.len == 0 or calls.len > max_gates or rows == 0 or rows > max_rows) return error.Inkling2LoopShape;
    var all: [max_gates]Gate = @splat(calls[0]);
    for (calls, 0..) |x, i| {
        inline for (@typeInfo(Gate).@"struct".field_names) |name| if (@field(x, name) % 16 != 0) return error.Inkling2LoopAlign;
        all[i] = x;
    }
    const n: u32 = @intCast(calls.len);
    const down, const up = which.halves();
    var a: hip.launch.Args = .{};
    try a.add(all);
    try a.add(g.eps);
    const set = k.at(g.width);
    try go(set.gate_down[down], s, .{ .x = rows, .y = rank / down_ranks, .z = n }, down_threads, &a);
    var b: hip.launch.Args = .{};
    try b.add(all);
    try b.add(int(rows));
    const row_groups = std.math.divCeil(u32, rows, up_rows) catch unreachable;
    try goBlock(set.gate_up[up], s, .{ .x = row_groups, .y = g.hidden() / up_cols, .z = n }, .{ .x = up_cols, .y = up_rows }, &b);
}

test {
    _ = @import("hip_loop_test.zig");
}
