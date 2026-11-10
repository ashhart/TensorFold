//! Geometry-only prompt attention variants, each using the unchanged BF16-expanded cache operator body.
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");
pub const Input = @import("fp8_attention_prompt.zig").Input;
pub const Mode = enum(u3) {
    disabled,
    w4h2s1,
    w2h2s1,
    w4h1s1,
    w8h2s1,
    w4h2s2,
    compact_w4h2s1,
    streamed_w4h2s1,
};
pub const Variant = struct {
    name: []const u8,
    symbol_prefix: []const u8 = "tf_exact_",
    warps: u32,
    heads: u32,
    slots: u32,
    shared: u32,
};
pub const variants = [_]Variant{
    .{ .name = "w4h2s1", .warps = 4, .heads = 2, .slots = 1, .shared = 49152 },
    .{ .name = "w2h2s1", .warps = 2, .heads = 2, .slots = 1, .shared = 32768 },
    .{ .name = "w4h1s1", .warps = 4, .heads = 1, .slots = 1, .shared = 49152 },
    .{ .name = "w8h2s1", .warps = 8, .heads = 2, .slots = 1, .shared = 81920 },
    .{ .name = "w4h2s2", .warps = 4, .heads = 2, .slots = 2, .shared = 65536 },
    .{ .name = "w4h2s1", .symbol_prefix = "tf_compact_", .warps = 4, .heads = 2, .slots = 1, .shared = 49152 },
    .{ .name = "w4h2s1", .symbol_prefix = "tf_streamed_", .warps = 4, .heads = 2, .slots = 1, .shared = 49152 },
};
pub fn selected(raw: ?[]const u8) !Mode {
    const text = raw orelse return .disabled;
    if (text.len != 1 or text[0] < '0' or text[0] > '7') return error.InvalidExactAttentionMode;
    return @fromBackingInt(@intCast(text[0] - '0'));
}
pub fn geometry(mode: Mode) !Variant {
    if (mode == .disabled) return error.ExactAttentionDisabled;
    return variants[@backingInt(mode) - 1];
}

pub const Tuned = struct {
    module: Module,
    whole: [variants.len]Function,
    paged: [variants.len]Function,
    owns_module: bool = true,
    mode_mask: u8 = 0xfe,

    pub fn init(d: *const Driver, image: []const u8, capability: u32) !Tuned {
        if (capability != 120 and capability != 121) return error.UnsupportedExactAttentionDevice;
        var module = try Module.load(d, image);
        errdefer module.unload();
        var tuned: Tuned = .{ .module = module, .whole = undefined, .paged = undefined };
        inline for (variants, 0..) |v, i| {
            tuned.whole[i] = try module.function(v.symbol_prefix ++ v.name ++ "_whole");
            tuned.paged[i] = try module.function(v.symbol_prefix ++ v.name ++ "_paged");
            try tuned.whole[i].allowDynamicShared(v.shared);
            try tuned.paged[i].allowDynamicShared(v.shared);
        }
        return tuned;
    }

    /// Mode 6 shares the already-resident prefill image and never unloads the caller's module.
    pub fn borrowCompact(module: Module) !Tuned {
        var tuned: Tuned = .{ .module = module, .whole = undefined, .paged = undefined, .owns_module = false, .mode_mask = 1 << 6 };
        const index = @backingInt(Mode.compact_w4h2s1) - 1;
        tuned.whole[index] = try module.function("tf_compact_w4h2s1_whole");
        tuned.paged[index] = try module.function("tf_compact_w4h2s1_paged");
        try tuned.whole[index].allowDynamicShared(49152);
        try tuned.paged[index].allowDynamicShared(49152);
        return tuned;
    }

    /// The contract owns its compact-only image independently of the serving engine.
    pub fn initCompact(d: *const Driver, image: []const u8, capability: u32) !Tuned {
        if (capability != 120 and capability != 121) return error.UnsupportedExactAttentionDevice;
        var module = try Module.load(d, image);
        errdefer module.unload();
        var tuned = try borrowCompact(module);
        tuned.owns_module = true;
        return tuned;
    }

    pub fn supports(tuned: *const Tuned, mode: Mode) bool {
        return mode != .disabled and (tuned.mode_mask & (@as(u8, 1) << @backingInt(mode))) != 0;
    }

    pub fn deinit(tuned: *Tuned) void {
        if (tuned.owns_module) tuned.module.unload();
        tuned.* = undefined;
    }

    /// Borrowed views are nonoverlapping and remain live until the stream completes.
    pub fn run(tuned: *const Tuned, stream: Stream, mode: Mode, p: Input) !void {
        const v = try geometry(mode);
        if (!tuned.supports(mode)) return error.ExactAttentionUnavailable;
        try p.validate();
        const s = p.shape;
        const group = s.heads / s.kv_heads;
        if (group % v.heads != 0) return error.UnsupportedExactAttentionShape;
        var args: launch.Args = .{};
        for ([_]u64{ p.q, p.k, p.v, p.k_scales, p.v_scales, p.output }) |ptr| args.add(ptr);
        for ([_]u32{ s.prefix, s.rows, s.heads, s.kv_heads, group }) |n| args.add(@as(i32, @intCast(n)));
        args.add(s.scale);
        args.add(@as(i32, @intCast(s.capacity)));
        if (p.pages != 0) {
            args.add(p.pages);
            args.add(@as(i64, @intCast(s.page_bytes)));
            args.add(@as(i32, @intCast(s.page_tokens)));
        }
        const rows = 16 * (v.warps / v.heads);
        const config: launch.Config = .{
            .grid = .{ .x = (s.rows + rows - 1) / rows, .y = s.kv_heads * (group / v.heads) },
            .block = .{ .x = 32 * v.warps },
            .shared = v.shared,
        };
        const index = @backingInt(mode) - 1;
        try launch.launch(if (p.pages == 0) tuned.whole[index] else tuned.paged[index], config, stream, &args);
    }
};

test "exact-attention selection and geometry match the unchanged row body" {
    try std.testing.expectEqual(Mode.disabled, try selected(null));
    try std.testing.expectEqual(Mode.disabled, try selected("0"));
    try std.testing.expectEqual(Mode.w4h2s1, try selected("1"));
    try std.testing.expectEqual(Mode.w4h2s2, try selected("5"));
    try std.testing.expectEqual(Mode.compact_w4h2s1, try selected("6"));
    try std.testing.expectEqual(Mode.streamed_w4h2s1, try selected("7"));
    try std.testing.expectError(error.InvalidExactAttentionMode, selected("8"));
    try std.testing.expectError(error.InvalidExactAttentionMode, selected(""));
    try std.testing.expectError(error.ExactAttentionDisabled, geometry(.disabled));
    for (variants) |v| try std.testing.expectEqual(v.shared, (v.warps * 16 + v.slots * 32) * 256 * 2);
}

test "exact-attention compact borrow masks modes and never unloads the shared module" {
    const abi = @import("abi.zig");
    const Fake = struct {
        var loads: usize = 0;
        var unloads: usize = 0;
        var resolves: usize = 0;
        var attributes: usize = 0;
        var valid: bool = true;
        fn load(out: *abi.Module, _: ?*const anyopaque) callconv(.c) abi.Result {
            loads += 1;
            out.* = @ptrFromInt(0x1000);
            return abi.success;
        }
        fn unload(_: abi.Module) callconv(.c) abi.Result {
            unloads += 1;
            return abi.success;
        }
        fn function(out: *abi.Function, _: abi.Module, symbol: [*:0]const u8) callconv(.c) abi.Result {
            const name = std.mem.span(symbol);
            valid = valid and (std.mem.eql(u8, name, "tf_compact_w4h2s1_whole") or
                std.mem.eql(u8, name, "tf_compact_w4h2s1_paged"));
            resolves += 1;
            out.* = @ptrFromInt(0x2000);
            return abi.success;
        }
        fn attribute(_: abi.Function, key: abi.FunctionAttribute, value: c_int) callconv(.c) abi.Result {
            valid = valid and key == .max_dynamic_shared_size_bytes and value == 49152;
            attributes += 1;
            return abi.success;
        }
    };
    Fake.loads = 0;
    Fake.unloads = 0;
    Fake.resolves = 0;
    Fake.attributes = 0;
    Fake.valid = true;
    var driver: Driver = .{ .lib = undefined, .api = undefined };
    driver.api.cuModuleLoadData = Fake.load;
    driver.api.cuModuleUnload = Fake.unload;
    driver.api.cuModuleGetFunction = Fake.function;
    driver.api.cuFuncSetAttribute = Fake.attribute;
    var borrowed = try Tuned.borrowCompact(.{ .d = &driver, .handle = @ptrFromInt(0x1000) });
    try std.testing.expect(!borrowed.owns_module);
    try std.testing.expect(borrowed.supports(.compact_w4h2s1));
    for ([_]Mode{ .disabled, .w4h2s1, .w2h2s1, .w4h1s1, .w8h2s1, .w4h2s2, .streamed_w4h2s1 }) |mode|
        try std.testing.expect(!borrowed.supports(mode));
    try std.testing.expectError(error.ExactAttentionUnavailable, borrowed.run(undefined, .w4h2s1, undefined));
    borrowed.deinit();
    try std.testing.expectEqual(@as(usize, 0), Fake.unloads);
    const image: [8]u8 align(8) = @splat(0);
    var owned = try Tuned.initCompact(&driver, &image, 120);
    try std.testing.expect(owned.owns_module);
    owned.deinit();
    try std.testing.expectEqual(@as(usize, 1), Fake.loads);
    try std.testing.expectEqual(@as(usize, 1), Fake.unloads);
    try std.testing.expectEqual(@as(usize, 4), Fake.resolves);
    try std.testing.expectEqual(@as(usize, 4), Fake.attributes);
    try std.testing.expect(Fake.valid);
}
