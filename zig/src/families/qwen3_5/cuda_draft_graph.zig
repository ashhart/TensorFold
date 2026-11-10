//! DFlash block replay through the shared CUDA graph cache; host selection and head spans remain outside it.
const std = @import("std");
const cuda = @import("cuda");

pub const Cache = cuda.graph_cache.Cache(u16);
pub const Engine = cuda.graph_cache.CudaEngine;

pub fn enabled(value: ?[]const u8) !bool {
    const raw = value orelse return false;
    if (std.mem.eql(u8, raw, "0")) return false;
    if (std.mem.eql(u8, raw, "1")) return true;
    return error.BadDraftGraph;
}

pub fn init(d: anytype) !void {
    if (!try enabled(try @import("sys").environment.given("TF_DRAFT_GRAPH"))) return;
    if (!admitted(d.e.head, d.e.g)) return error.UnsupportedDraftGraph;
    d.graph_engine = .{ .d = d.e.d };
    d.graphs = Cache.init(d.gpa, d.graph_engine.engine(), .{ .max = 32, .min_keep = 0, .hold = true });
    d.graphs.?.name = "qwen-dflash-block";
}

// forward projects the target head after graph.run(encode), preserving its ordinary precision dispatch.
fn admitted(head: @import("cuda_layers.zig").Proj, geometry: anytype) bool {
    if (!@import("cuda_geometry.zig").dense27(geometry)) return false;
    return switch (head) {
        .a4 => true,
        .ql => |weight| switch (weight) {
            .nvfp4, .fp8, .fp8g, .bf16 => true,
            else => false,
        },
        else => false,
    };
}

/// Every pointer captured by the block lives in these two owned arenas or in the target's embedding tables.
fn fingerprint(d: anytype) u64 {
    return cuda.graph_cache.fingerprintOf(&.{
        d.scratch.ptr,                                 d.scratch.len,                         d.w.arena.ptr, d.w.arena.len,
        d.e.embed_w,                                   d.e.embed_s,                           d.e.embed_b,   d.c.block,
        if (d.bf16_embedding) |t| t.buffer.ptr else 0, if (d.w.bf16) |*p| p.arena.ptr else 0,
    });
}

pub fn key(streams: usize) !u16 {
    if (streams == 0 or streams > std.math.maxInt(u16)) return error.BadDraftGraph;
    return @intCast(streams);
}

pub fn run(d: anytype, streams: usize, comptime encode: anytype) !void {
    const k = try key(streams);
    if (d.graphs) |*cache| {
        const Call = struct {
            draft: @TypeOf(d),
            streams: usize,
            fn body(p: *anyopaque, _: cuda.Stream) anyerror!void {
                const call: *@This() = @ptrCast(@alignCast(p));
                try encode(call.draft, call.streams);
            }
        };
        var call: Call = .{ .draft = d, .streams = streams };
        _ = try cache.run(k, d.e.stream, fingerprint(d), .{ .ctx = &call, .run = Call.body });
    } else try encode(d, streams);
}

test "DFlash graph keys separate active frame counts and reject empty or overflowing frames" {
    try std.testing.expect(try key(1) != try key(2));
    try std.testing.expect(try key(2) != try key(4));
    try std.testing.expectError(error.BadDraftGraph, key(0));
    try std.testing.expectError(error.BadDraftGraph, key(65536));
}

test "DFlash graph switch is explicit and refuses malformed settings" {
    try std.testing.expect(!try enabled(null));
    try std.testing.expect(!try enabled("0"));
    try std.testing.expect(try enabled("1"));
    for ([_][]const u8{ "", "true", "2", "01" }) |v|
        try std.testing.expectError(error.BadDraftGraph, enabled(v));
}

test "DFlash graph admits linear heads projected outside capture on dense27 geometry" {
    const Proj = @import("cuda_layers.zig").Proj;
    const heads = [_]Proj{ .{ .a4 = undefined }, .{ .ql = .{ .nvfp4 = undefined } }, .{ .ql = .{ .fp8 = undefined } }, .{ .ql = .{ .fp8g = undefined } }, .{ .ql = .{ .bf16 = undefined } } };
    for (heads) |head| {
        try std.testing.expect(admitted(head, .{ .hidden = 5120, .layers = 64, .experts = 0 }));
        try std.testing.expect(!admitted(head, .{ .hidden = 2048, .layers = 64, .experts = 0 }));
        try std.testing.expect(!admitted(head, .{ .hidden = 5120, .layers = 63, .experts = 0 }));
        try std.testing.expect(!admitted(head, .{ .hidden = 5120, .layers = 64, .experts = 8 }));
    }
}

test "DFlash graph admission retains other head format refusals" {
    const Proj = @import("cuda_layers.zig").Proj;
    const heads = [_]Proj{
        .{ .an = undefined },              .{ .x3 = undefined },
        .{ .bf = undefined },              .{ .ql = .{ .affine4 = undefined } },
        .{ .ql = .{ .exl3 = undefined } }, .{ .ql = .{ .mxfp8 = undefined } },
        .{ .ql = .{ .q8_0 = undefined } }, .{ .ql = .{ .iq2xxs = undefined } },
    };
    for (heads) |head| try std.testing.expect(!admitted(head, .{ .hidden = 5120, .layers = 64, .experts = 0 }));
}
