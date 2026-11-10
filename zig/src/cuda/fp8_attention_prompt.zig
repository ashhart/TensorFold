//! Experimental prompt-only native E4M3 score multiplication; decode and default attention remain separate.
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");

pub const Mode = enum { disabled, single, residual };
pub const head_dim = 256;
pub const block_rows = 32;
pub const threads = 128;
pub const shared = [_]u32{ 33408, 50048 };
pub const max_optimized_capacity: u32 = 131072;

pub fn selected(raw: ?[]const u8) !Mode {
    const text = raw orelse return .disabled;
    if (std.mem.eql(u8, text, "0")) return .disabled;
    if (std.mem.eql(u8, text, "1")) return .single;
    if (std.mem.eql(u8, text, "2")) return .residual;
    return error.InvalidFp8AttentionMode;
}

pub const Shape = struct {
    rows: u32,
    prefix: u32,
    heads: u32,
    kv_heads: u32,
    capacity: u32,
    scale: f32,
    page_tokens: u32 = 0,
    page_bytes: u64 = 0,

    pub fn validate(s: Shape, paged: bool) !void {
        if (s.capacity > max_optimized_capacity) return error.UnsupportedFp8AttentionShape;
        try s.validateAddressing(paged);
    }

    /// Physical page capacity can include unused slack; retain the signed kernel ABI and logical extent checks.
    pub fn validateAddressing(s: Shape, paged: bool) !void {
        if (s.rows == 0 or s.rows > @import("qwen_prefill_config").max_rows or s.capacity > std.math.maxInt(i32) or s.prefix >= s.capacity or
            s.rows > s.capacity - s.prefix or s.kv_heads == 0 or s.kv_heads > 128 or
            s.heads == 0 or s.heads > 256 or
            s.heads % s.kv_heads != 0 or (s.heads / s.kv_heads) % 2 != 0 or
            !std.math.isFinite(s.scale) or s.scale <= 0)
            return error.UnsupportedFp8AttentionShape;
        const minimum_page = @as(u64, s.kv_heads) * s.page_tokens * (2 * head_dim + 8);
        if (paged and (s.page_tokens == 0 or s.page_tokens > 131072 or s.page_bytes < minimum_page or
            s.page_bytes > std.math.maxInt(i64) or s.page_bytes % 16 != 0))
            return error.UnsupportedFp8AttentionPages;
    }
};

pub const Input = struct {
    q: u64,
    k: u64,
    v: u64,
    k_scales: u64,
    v_scales: u64,
    output: u64,
    pages: u64 = 0,
    shape: Shape,

    pub fn validate(p: Input) !void {
        try p.shape.validate(p.pages != 0);
        try p.validatePointers();
    }

    pub fn validateAddressing(p: Input) !void {
        try p.shape.validateAddressing(p.pages != 0);
        try p.validatePointers();
    }

    fn validatePointers(p: Input) !void {
        if (p.q == 0 or p.q % 2 != 0 or p.output == 0 or p.output % 2 != 0 or p.output == p.q or
            p.k == 0 or p.k % 16 != 0 or p.v == 0 or p.v % 16 != 0 or
            p.k_scales == 0 or p.k_scales % 4 != 0 or p.v_scales == 0 or p.v_scales % 4 != 0 or
            p.pages % 4 != 0) return error.InvalidFp8AttentionPointer;
    }
};

pub const Native = struct {
    module: Module,
    whole: [2]Function,
    paged: [2]Function,

    pub fn init(d: *const Driver, image: []const u8, capability: u32) !Native {
        if (capability != 120 and capability != 121) return error.UnsupportedFp8AttentionDevice;
        var module = try Module.load(d, image);
        errdefer module.unload();
        const n: Native = .{
            .module = module,
            .whole = .{ try module.function("tf_fp8q_attention_whole"), try module.function("tf_fp8q_attention_whole_residual") },
            .paged = .{ try module.function("tf_fp8q_attention_paged"), try module.function("tf_fp8q_attention_paged_residual") },
        };
        for (n.whole, n.paged, shared) |whole, paged, bytes| {
            try whole.allowDynamicShared(bytes);
            try paged.allowDynamicShared(bytes);
        }
        return n;
    }

    pub fn deinit(n: *Native) void {
        n.module.unload();
    }

    /// Caller owns nonoverlapping query/output and cache views until this stream completes.
    pub fn run(n: *const Native, stream: Stream, mode: Mode, p: Input) !void {
        if (mode == .disabled) return error.Fp8AttentionDisabled;
        try p.validate();
        const index: usize = if (mode == .single) 0 else 1;
        var args: launch.Args = .{};
        for ([_]u64{ p.q, p.k, p.v, p.k_scales, p.v_scales, p.output }) |ptr| args.add(ptr);
        const s = p.shape;
        for ([_]u32{ s.prefix, s.rows, s.heads, s.kv_heads, s.heads / s.kv_heads }) |v|
            args.add(@as(i32, @intCast(v)));
        args.add(s.scale);
        args.add(@as(i32, @intCast(s.capacity)));
        if (p.pages != 0) {
            args.add(p.pages);
            args.add(@as(i64, @intCast(s.page_bytes)));
            args.add(@as(i32, @intCast(s.page_tokens)));
        }
        const config: launch.Config = .{
            .grid = .{ .x = (s.rows + block_rows - 1) / block_rows, .y = s.heads / 2 },
            .block = .{ .x = threads },
            .shared = shared[index],
        };
        try launch.launch(if (p.pages == 0) n.whole[index] else n.paged[index], config, stream, &args);
    }
};

test "native FP8 score mode stays opt-in and rejects unsupported geometry" {
    try std.testing.expectEqual(Mode.disabled, try selected(null));
    try std.testing.expectEqual(Mode.single, try selected("1"));
    try std.testing.expectEqual(Mode.residual, try selected("2"));
    try std.testing.expectError(error.InvalidFp8AttentionMode, selected("bad"));
    var s: Shape = .{ .rows = 33, .prefix = 131039, .heads = 24, .kv_heads = 4, .capacity = 131072, .scale = 1.0 / 16.0 };
    try s.validate(false);
    s.rows = 34;
    try std.testing.expectError(error.UnsupportedFp8AttentionShape, s.validate(false));
    s.rows = 1;
    s.heads = 12;
    try std.testing.expectError(error.UnsupportedFp8AttentionShape, s.validate(false));
}
