//! Reference-conditioned semantic CFG; built-in voices remain unguided and disabled mode ignores the header.
const std = @import("std");
const fields = @import("fields.zig");
const model_text = @import("model_text.zig");
const drama_prompt = @import("drama_prompt.zig");

pub const flags = [_]@import("cli.zig").Flag{
    .{ .name = "--cfg-gamma", .native = @import("builtin").os.tag == .macos },
    .{ .name = "--cfg-allow-request-gamma", .kind = .store_true, .native = @import("builtin").os.tag == .macos },
    .{ .name = "--cfg-force-guided", .kind = .store_true, .native = @import("builtin").os.tag == .macos },
    .{ .name = "--cfg-batched", .kind = .store_true, .native = @import("builtin").os.tag == .macos },
};

pub const Options = struct { gamma: f32 = 1, request_gamma: bool = false, force_guided: bool = false };
pub const Flags = struct {
    gamma: f32 = 0,
    request_gamma: bool = false,
    force_guided: bool = false,
    batched: bool = false,

    pub fn options(f: Flags) ?Options {
        if (f.gamma == 0 and !f.request_gamma) return null;
        return .{ .gamma = if (f.gamma == 0) 1 else f.gamma, .request_gamma = f.request_gamma, .force_guided = f.force_guided };
    }

    pub fn set(f: *Flags, name: []const u8, text: []const u8) error{BadGamma}!bool {
        const is = std.mem.eql;
        if (is(u8, name, "--cfg-gamma")) {
            const g = fields.pyFloat(text) orelse return error.BadGamma;
            if (!(g == 0 or valid(g))) return error.BadGamma;
            f.gamma = @floatCast(g);
        } else if (is(u8, name, "--cfg-allow-request-gamma")) {
            f.request_gamma = true;
        } else if (is(u8, name, "--cfg-force-guided")) {
            f.force_guided = true;
        } else if (is(u8, name, "--cfg-batched")) {
            f.batched = true;
        } else return false;
        return true;
    }
};

pub fn withGuidance(f: Flags, request: @import("engine_api").Open) !@import("engine_api").Open {
    if (f.options() != null and !std.mem.eql(u8, request.model_type, "fish_omni")) return error.GuidanceRequiresDrama;
    var out = request;
    out.cfg_enabled = f.options() != null;
    out.cfg_batched = f.batched;
    return out;
}

pub fn valid(g: f64) bool {
    return g >= 1 and g <= 5;
}

pub fn gamma(options: Options, sent: ?[]const u8) ?f32 {
    if (!options.request_gamma) return options.gamma;
    const g = fields.pyFloat(sent orelse return options.gamma) orelse return null;
    return if (valid(g)) @floatCast(g) else null;
}

pub fn active(options: ?Options, g: f32, referenced: bool) bool {
    const o = options orelse return false;
    return referenced and (g != 1 or o.force_guided);
}

pub const Guide = struct { rows: []const [11]i32 = &.{}, gamma: f32 = 1 };

pub fn guide(a: std.mem.Allocator, text: model_text.Text, options: ?Options, g: f32, input: []const u8, referenced: bool) model_text.Error!Guide {
    if (!active(options, g, referenced)) return .{};
    const tokens = try drama_prompt.encode(a, text, input);
    const rows = drama_prompt.rows(a, tokens) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.InvalidToken => error.Template,
    };
    return .{ .rows = rows, .gamma = g };
}

test "CFG header validation and fixed policy match the CUDA recipe" {
    const fixed: Options = .{ .gamma = 1.5 };
    try std.testing.expectEqual(@as(?f32, 1.5), gamma(fixed, "nonsense"));
    const open: Options = .{ .gamma = 1.5, .request_gamma = true };
    try std.testing.expectEqual(@as(?f32, 1.5), gamma(open, null));
    for ([_][]const u8{ "1", " 1.5 ", "3", "5" }, [_]f32{ 1, 1.5, 3, 5 }) |sent, want|
        try std.testing.expectEqual(@as(?f32, want), gamma(open, sent));
    for ([_][]const u8{ "0", "0.99", "5.01", "nan", "inf", "", "2x" }) |bad|
        try std.testing.expectEqual(@as(?f32, null), gamma(open, bad));
}

test "CFG activation requires reference and configured policy including forced gamma one" {
    try std.testing.expect((Flags{}).options() == null);
    const options = (Flags{ .gamma = 1.5 }).options();
    try std.testing.expect(active(options, 1.5, true));
    try std.testing.expect(!active(options, 1.5, false));
    try std.testing.expect(!active(options, 1, true));
    try std.testing.expect(!active(null, 1.5, true));
    try std.testing.expect(active(.{ .force_guided = true }, 1, true));
    try std.testing.expect(!active(.{ .force_guided = true }, 1, false));
}

test "CFG native flags allocate guidance states only when enabled and refuse unsupported combinations" {
    const cli = @import("cli.zig");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var usage: cli.Usage = .{};
    if (@import("builtin").os.tag != .macos) {
        try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--cfg-gamma", "1.5" }, &usage));
        return;
    }
    const enabled = try cli.parse(a, &.{ "m", "--cfg-gamma", "1.5", "--cfg-batched" }, &usage);
    const request = try cli.request(a, "checkpoint", "fish_omni", enabled);
    try std.testing.expect(request.cfg_enabled and request.cfg_batched);
    try std.testing.expectEqual(@as(f32, 1.5), enabled.cfg.options().?.gamma);
    try std.testing.expectError(error.GuidanceRequiresDrama, cli.request(a, "checkpoint", "text", enabled));
    const off = try cli.parse(a, &.{ "m", "--cfg-gamma", "0" }, &usage);
    try std.testing.expect(!(try cli.request(a, "checkpoint", "fish_omni", off)).cfg_enabled);
    try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--cfg-batched" }, &usage));
    try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--cfg-gamma", "nan" }, &usage));
    try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--cfg-gamma", "6" }, &usage));
    try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--cfg-gamma", "1.5", "--cfg-every", "2" }, &usage));
}
