//! Owned recording Metal objects execute only the common scalar test body, never a Metal API.
const std = @import("std");
const Params = extern struct { rows: c_int, n: c_int, k: c_int, format: c_int, group: c_int, scale_kind: c_int, reserved0: c_int, reserved1: c_int, global_scale: f32 };
pub const Size = @import("src/metal/types.zig").Size;
extern fn tf_weight_only_cpu([*]const u16, [*]const u8, [*]const u8, [*]const u8, [*]u16, Params, c_int, c_int) void;
pub var live: usize = 0;
pub var fail_pipe: ?usize = null;
pub var calls: usize = 0;
pub var groups: Size = undefined;
pub var threads: Size = undefined;
pub var bindings: [5]View = undefined;
pub var params: Params = undefined;
pub var repetition: c_int = 0;
pub var selected: c_int = 0;
const HcParams = extern struct { rows: c_int, cols: c_int, streams: c_int, injection: c_int, repeat: c_int };
extern fn tf_hc_activation_cpu([*]const u16, [*]u16, c_int, c_int, c_int, c_int, c_int) void;
var hc_params: HcParams = undefined;
const Op = enum { projection, hc };
var selected_op: Op = .projection;
pub const View = struct { b: Buffer, off: usize };
pub const Buffer = struct {
    bytes: []u8,
    owner: usize = 1,
    pub fn deviceId(b: Buffer) usize {
        return b.owner;
    }
    pub fn gpuAddress(b: Buffer) u64 {
        return @intFromPtr(b.bytes.ptr);
    }
    pub fn length(b: Buffer) usize {
        return b.bytes.len;
    }
};
pub const Device = struct { id: usize = 1 };
pub const CompileOptions = struct {
    pub fn mlx() @This() {
        return .{};
    }
};
pub const Library = struct {
    pub fn fromSource(_: Device, text: []const u8, _: CompileOptions) !Library {
        if (std.mem.indexOf(u8, text, "TF_WEIGHT_ONLY_BODY") == null) return error.BadOwnedSource;
        return .{};
    }
    pub fn deinit(_: Library) void {}
};
pub const Pipeline = struct {
    variant: usize,
    op: Op,
    pub fn init(_: Device, _: Library, name: []const u8, _: bool) !Pipeline {
        const prefix = if (std.mem.startsWith(u8, name, "qwen4_hc_activation")) "qwen4_hc_activation" else "tf_weight_only_projection";
        if (!std.mem.startsWith(u8, name, prefix)) return error.BadOwnedSymbol;
        const suffixes = [_][]const u8{ "", "_bad_nan", "_bad_repeat", "_bad_tail", "_bad_sums" };
        for (suffixes, 0..) |suffix, i| {
            if (!std.mem.eql(u8, name[prefix.len..], suffix)) continue;
            if (fail_pipe == i) return error.PlantedPipelineFailure;
            live += 1;
            return .{ .variant = i, .op = if (std.mem.eql(u8, prefix, "qwen4_hc_activation")) .hc else .projection };
        }
        return error.BadOwnedSymbol;
    }
    pub fn maxThreads(p: Pipeline) usize {
        return if (p.op == .hc) 256 else 128;
    }
    pub fn simdWidth(_: Pipeline) usize {
        return 32;
    }
    pub fn deinit(_: Pipeline) void {
        std.debug.assert(live > 0);
        live -= 1;
    }
};
pub const ComputeEncoder = struct {
    owner: usize = 1,
    pub fn deviceId(e: ComputeEncoder) usize {
        return e.owner;
    }
    pub fn setPipeline(_: ComputeEncoder, p: Pipeline) void {
        calls += 1;
        selected = @intCast(p.variant);
        selected_op = p.op;
    }
    pub fn setBuffer(_: ComputeEncoder, b: Buffer, off: usize, i: usize) void {
        bindings[i] = .{ .b = b, .off = off };
        calls += 1;
    }
    pub fn setBytes(_: ComputeEncoder, bytes: []const u8, i: usize) void {
        if (selected_op == .hc and i == 2) hc_params = std.mem.bytesToValue(HcParams, bytes[0..@sizeOf(HcParams)]);
        if (i == 5) params = std.mem.bytesToValue(Params, bytes[0..@sizeOf(Params)]);
        if (i == 6) repetition = std.mem.bytesToValue(c_int, bytes[0..@sizeOf(c_int)]);
        calls += 1;
    }
    pub fn dispatchGroups(_: ComputeEncoder, g: Size, t: Size) void {
        groups = g;
        threads = t;
        calls += 1;
        if (selected_op == .hc) {
            tf_hc_activation_cpu(@ptrCast(@alignCast(bindings[0].b.bytes[bindings[0].off..].ptr)), @ptrCast(@alignCast(bindings[1].b.bytes[bindings[1].off..].ptr)), hc_params.rows * hc_params.cols, hc_params.streams, hc_params.injection, selected, hc_params.repeat);
            return;
        }
        tf_weight_only_cpu(@ptrCast(@alignCast(bindings[0].b.bytes[bindings[0].off..].ptr)), bindings[1].b.bytes[bindings[1].off..].ptr, bindings[2].b.bytes[bindings[2].off..].ptr, bindings[3].b.bytes[bindings[3].off..].ptr, @ptrCast(@alignCast(bindings[4].b.bytes[bindings[4].off..].ptr)), params, selected, repetition);
    }
    pub fn barrier(_: ComputeEncoder) void {
        calls += 1;
    }
};

pub const CommandSessionFor = @import("src/metal/command_session.zig").Session;
