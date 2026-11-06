//! Named environment validation before model, driver and transport work; successful values keep their old semantics.
const std = @import("std");
const sys = @import("sys.zig");

pub const Error = error{InvalidEnvironment};
pub const Scope = enum { common, cuda, metal, xpu, slide, hooks };
pub const Kind = enum { u32, u64, size, percent, positive, device, segments, mib, bit_width, pocket, prompt, flag };
pub const Rule = struct {
    name: [:0]const u8,
    scope: Scope,
    kind: Kind,
    forms: []const u8,
    trim: []const u8 = "",
    first_digit: bool = false,
    empty_default: bool = false,
};

pub const rules = [_]Rule{
    .{ .name = "TENSORFOLD_VIDEO_TOKENS", .scope = .common, .kind = .u64, .forms = "a nonnegative whole video-token budget (default 16384; unsigned 64-bit count)" },
    .{ .name = "TENSORFOLD_MAX_PDF_PAGES", .scope = .common, .kind = .positive, .forms = "a whole number from 1 to 4294967295 (capped by --vision-max-images)" },
    .{ .name = "TENSORFOLD_PRESSURE_CACHE_PERCENT", .scope = .common, .kind = .percent, .trim = " \t", .forms = "a whole percentage from 0 to 100" },
    .{ .name = "TENSORFOLD_PREFIX_WAIT_MS", .scope = .common, .kind = .u32, .trim = " \t", .first_digit = true, .forms = "whole milliseconds from 0 to 4294967295 (0 disables waiting)" },
    .{ .name = "TENSORFOLD_RESTORE_PREFETCH", .scope = .common, .kind = .u64, .trim = " \t", .first_digit = true, .forms = "a whole unsigned 64-bit read count (0 disables reading ahead; capped at 65536)" },
    .{ .name = "TENSORFOLD_RESTORE_PREFETCH_GIB", .scope = .common, .kind = .u64, .trim = " \t", .first_digit = true, .forms = "a whole unsigned 64-bit GiB count (capped at 16777216)" },
    .{ .name = "TENSORFOLD_RESTORE_PREFETCH_MS", .scope = .common, .kind = .u64, .trim = " \t", .first_digit = true, .forms = "whole unsigned 64-bit milliseconds (capped at 1099511627776)" },
    .{ .name = "TF_CUDA_DEVICE", .scope = .cuda, .kind = .device, .trim = " ", .empty_default = true, .forms = "a whole GPU ordinal from 0 to 4294967295, or empty for the default" },
    .{ .name = "TF_CUDA_SEGMENTS", .scope = .cuda, .kind = .segments, .trim = " ", .empty_default = true, .forms = "a whole number from 1 to 4, or empty for the default" },
    .{ .name = "TF_CUDA_CARVEOUT", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_CUDA_CARVEOUT_MIB", .scope = .cuda, .kind = .mib, .empty_default = true, .forms = "a positive whole MiB count fitting a native byte size, or empty for 1792 MiB" },
    .{ .name = "TF_MTP_DRAFTS", .scope = .cuda, .kind = .u32, .forms = "a whole number from 0 to 4294967295 (clamped to the drafter's window)" },
    .{ .name = "TF_DRAFT_NODES", .scope = .cuda, .kind = .size, .forms = "a whole nonnegative native-size node count (capped by the drafter's window)" },
    .{ .name = "TF_QMMF_HALF_BELOW", .scope = .cuda, .kind = .size, .forms = "a whole nonnegative native-size block count (0 disables half tiles)" },
    .{ .name = "TF_ROCE_GID_INDEX", .scope = .cuda, .kind = .u32, .forms = "a whole GID index from 0 to 2147483647" },
    .{ .name = "TF_Q27_COPY_ROWS", .scope = .metal, .kind = .u32, .forms = "a whole number from 0 to 4294967295 (0 disables copied rows)" },
    .{ .name = "TF_Q27_DRAFT_SLOTS", .scope = .metal, .kind = .u32, .forms = "a whole number from 0 to 4294967295 (clamped to the current stream bounds)" },
    .{ .name = "TF_HEAD_GEOMETRY", .scope = .metal, .kind = .size, .forms = "a whole nonnegative native-size geometry count" },
    .{ .name = "TF_PROMPT_ROWS", .scope = .metal, .kind = .prompt, .forms = "a multiple of 16 from 128 to 2048 for native serving" },
    .{ .name = "TF_ANE_COLUMNS", .scope = .metal, .kind = .u32, .forms = "a whole column count from 0 to 4294967295 (0 disables the share)" },
    .{ .name = "XPU_A8_BITS", .scope = .xpu, .kind = .bit_width, .forms = "4 or 8" },
    .{ .name = "LFM3_FUSE", .scope = .xpu, .kind = .u32, .forms = "a whole fusion mask from 0 to 4294967295" },
    .{ .name = "TF_SLIDE_POCKET_KEY", .scope = .slide, .kind = .pocket, .forms = "auto, reply, subject or both" },
    .{ .name = "TF_SLIDE_FIT_GIB", .scope = .slide, .kind = .size, .forms = "a whole nonnegative native-size GiB count" },
    .{ .name = "TF_SLIDE_FORGET_FAIL", .scope = .hooks, .kind = .u32, .trim = " ", .forms = "a whole fact index from 0 to 4294967295 (0 disables the planted failure)" },
};

pub fn rule(name: []const u8) Rule {
    for (rules) |r| if (std.mem.eql(u8, name, r.name)) return r;
    @panic("unregistered strict environment parser");
}

/// Null means unset or the setting's existing accepted empty-default form, never a malformed value.
pub fn parse(name: []const u8, raw: ?[]const u8) Error!?u64 {
    const r = rule(name);
    const text = std.mem.trim(u8, raw orelse return null, r.trim);
    if (text.len == 0) return if (r.empty_default) null else error.InvalidEnvironment;
    if (r.first_digit and (text[0] < '0' or text[0] > '9')) return error.InvalidEnvironment;
    if (r.kind == .pocket) {
        for ([_][]const u8{ "auto", "reply", "subject", "both" }, 0..) |v, i|
            if (std.mem.eql(u8, text, v)) return i;
        return error.InvalidEnvironment;
    }
    if (r.kind == .flag or r.kind == .bit_width) {
        const a: []const u8 = if (r.kind == .flag) "0" else "4";
        const b: []const u8 = if (r.kind == .flag) "1" else "8";
        if (!std.mem.eql(u8, text, a) and !std.mem.eql(u8, text, b)) return error.InvalidEnvironment;
    }
    const n = std.fmt.parseInt(u64, text, 10) catch return error.InvalidEnvironment;
    const max: u64 = switch (r.kind) {
        .u32, .positive, .device => std.math.maxInt(u32),
        .percent => 100,
        .size, .mib => std.math.maxInt(usize),
        .segments => 4,
        else => std.math.maxInt(u64),
    };
    if (n > max) return error.InvalidEnvironment;
    if ((r.kind == .positive or r.kind == .segments or r.kind == .mib) and n == 0) return error.InvalidEnvironment;
    if (r.kind == .mib and n > std.math.maxInt(usize) / (1 << 20)) return error.InvalidEnvironment;
    if (r.kind == .prompt and (n < 128 or n > 2048 or n % 16 != 0)) return error.InvalidEnvironment;
    if (std.mem.eql(u8, name, "TF_ROCE_GID_INDEX") and n > std.math.maxInt(c_int)) return error.InvalidEnvironment;
    return n;
}

pub fn describe(a: std.mem.Allocator, name: []const u8, raw: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{s}=\"{f}\": expected {s}", .{ name, std.zig.fmtString(raw), rule(name).forms });
}

/// A direct family/tool entry point gets the same refusal even when it has no server problem buffer.
fn checked(name: [:0]const u8, raw: ?[]const u8) Error!?u64 {
    return parse(name, raw) catch {
        std.log.err("{s}=\"{f}\": expected {s}", .{ name, std.zig.fmtString(raw.?), rule(name).forms });
        return error.InvalidEnvironment;
    };
}

pub fn check(name: [:0]const u8, raw: ?[]const u8) Error!void {
    _ = try checked(name, raw);
}

pub fn read(comptime T: type, name: [:0]const u8, default: T) Error!T {
    const raw = sys.getenv(name);
    if (try checked(name, raw)) |n| return @intCast(n);
    return default;
}

/// Every CUDA entry point shares flag precedence, accepted text and diagnostics.
pub fn device(flag: ?u32, raw: ?[]const u8) Error!u32 {
    if (flag) |n| return n;
    return @intCast((try checked("TF_CUDA_DEVICE", raw)) orelse 0);
}

pub const Flags = struct {
    cuda: bool = false,
    metal: bool = false,
    xpu: bool = false,
    slide: bool = false,
    hooks: bool = false,
    device: ?u32 = null,
    segments: ?u32 = null,
    carveout: bool = false,

    fn includes(f: Flags, scope: Scope) bool {
        return switch (scope) {
            .common => true,
            .cuda => f.cuda,
            .metal => f.metal,
            .xpu => f.xpu,
            .slide => f.slide,
            .hooks => f.hooks,
        };
    }
};

/// The environment is injectable so the actual startup validator runs against host fixtures without a GPU.
pub fn refusal(a: std.mem.Allocator, environment: anytype, flags: Flags) !?[]const u8 {
    for (rules) |r| {
        if (!flags.includes(r.scope)) continue;
        if (std.mem.eql(u8, r.name, "TF_CUDA_DEVICE") and flags.device != null) continue;
        if (std.mem.eql(u8, r.name, "TF_CUDA_SEGMENTS") and flags.segments != null) continue;
        if (std.mem.eql(u8, r.name, "TF_CUDA_CARVEOUT") and flags.carveout) continue;
        if (std.mem.eql(u8, r.name, "TF_CUDA_CARVEOUT_MIB") and !flags.carveout and
            !std.mem.eql(u8, environment.get("TF_CUDA_CARVEOUT") orelse "0", "1")) continue;
        if (std.mem.eql(u8, r.name, "TF_DRAFT_NODES") and environment.get("TF_NO_TREES") != null) continue;
        const raw = environment.get(r.name) orelse continue;
        _ = parse(r.name, raw) catch return try describe(a, r.name, raw);
    }
    return null;
}

pub const Process = struct {
    pub fn get(_: Process, name: [:0]const u8) ?[]const u8 {
        return sys.getenv(name);
    }
};

test {
    _ = @import("environment_test.zig");
}
