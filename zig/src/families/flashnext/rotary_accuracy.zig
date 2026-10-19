//! A complete, source-matched finite-domain receipt; absent deployments never gain an inferred allowance.
const std = @import("std");
pub const domain = @import("rotary_domain.zig");
pub const Role = domain.Role;
pub const Mode = domain.Mode;
pub const schema = 1;
pub fn oracleSources() [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    for ([_][]const u8{ @embedFile("rotary_accuracy.zig"), @embedFile("rotary_domain.zig"), @embedFile("rotary_fixture.zig"), @embedFile("rotary_interval.zig"), @embedFile("rotary_config.zig"), @embedFile("window_config.zig"), @embedFile("vision_contract.zig"), @embedFile("../../metal/library.zig"), @embedFile("../../metal/types.zig") }) |bytes| hash.update(bytes);
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return digest;
}
pub const Identity = struct {
    compiler_service: []const u8,
    compiler_framework: []const u8,
    compiler_deployment: [64]u8,
    os_version: []const u8,
    os_build: []const u8,
    device_name: []const u8,
    device_generation: u32,
    language: usize,
    math_mode: u32,
    math_functions: u32,
    math: []const u8 = "safe/fast-functions/explicit-precise-rotary",
    oracle: []const u8 = @import("builtin").zig_version_string,
    sources: [4][64]u8,
    oracle_sources: [64]u8,
};
pub const Witness = struct { linear: u64 = 0, got: u32 = 0, reference: f64 = 0 };
pub const Maximum = struct {
    absolute: f64 = 0,
    ulps: f64 = 0,
    absolute_at: Witness = .{},
    ulps_at: Witness = .{},

    pub fn observe(m: *Maximum, linear: u64, got: f32, reference: f64) !void {
        if (!std.math.isFinite(got) or !std.math.isFinite(reference)) return error.NonfiniteRotarySweep;
        const absolute = @abs(@as(f64, got) - reference);
        const ulps = absolute / ulp(reference);
        const at: Witness = .{ .linear = linear, .got = @bitCast(got), .reference = reference };
        if (linear == 0 or absolute > m.absolute) {
            m.absolute = absolute;
            m.absolute_at = at;
        }
        if (linear == 0 or ulps > m.ulps) {
            m.ulps = ulps;
            m.ulps_at = at;
        }
    }
};
pub fn ulp(value: f64) f64 {
    const magnitude = @abs(value);
    if (magnitude < 0x1p-126) return 0x1p-149;
    const bits: u64 = @bitCast(magnitude);
    const exponent = (bits >> 52) & 2047;
    return @bitCast((exponent - 23) << 52);
}
pub const Norm = struct {
    width: usize,
    weight: u32,
    epsilon: u32,
    input: [256]u16 = @splat(0),
    output: [256]u16 = @splat(0),
};
pub const Row = struct {
    role: Role,
    mode: Mode,
    cases: u64 = 0,
    exp2_cases: u64 = 0,
    axis_cases: [3]u64 = @splat(0),
    table_bits: [35]u32 = @splat(0),
    frequency_bits: [32]u32 = @splat(0),
    exp2: Maximum = .{},
    phase: Maximum = .{},
    cosine: Maximum = .{},
    sine: Maximum = .{},
    coefficient_cosine: Maximum = .{},
    coefficient_sine: Maximum = .{},
    normalization: []const Norm = &.{},
};
pub const Receipt = struct {
    version: u32 = schema,
    complete: bool = false,
    identity: Identity,
    rows: []const Row,

    pub fn validate(r: Receipt, identity: Identity) !void {
        if (r.version != schema or !r.complete or r.rows.len != 10) return error.IncompleteRotarySweep;
        inline for (.{ "compiler_service", "compiler_framework", "os_version", "os_build", "device_name", "math", "oracle" }) |key| {
            if (@field(identity, key).len == 0 or !std.mem.eql(u8, @field(identity, key), @field(r.identity, key))) return error.UnrecordedRotaryCompiler;
        }
        if (r.identity.language != identity.language or r.identity.math_mode != identity.math_mode or r.identity.math_functions != identity.math_functions or r.identity.device_generation != identity.device_generation or
            !std.mem.eql(u8, &r.identity.compiler_deployment, &identity.compiler_deployment) or
            !std.mem.eql(u8, &r.identity.oracle_sources, &identity.oracle_sources) or
            !std.mem.eql(u8, std.mem.asBytes(&r.identity.sources), std.mem.asBytes(&identity.sources))) return error.UnrecordedRotaryCompiler;
        var seen: [5][2]bool = @splat(@splat(false));
        for (r.rows) |row| {
            const slot = &seen[@backingInt(row.role)][@backingInt(row.mode)];
            if (slot.* or row.cases != domain.count(row.role)) return error.IncompleteRotarySweep;
            if (row.exp2_cases != (if (row.mode == .normal) row.cases else 0)) return error.IncompleteRotarySweep;
            slot.* = true;
            const axes: [3]u64 = if (row.role == .mrope) .{ domain.maximum * 11, domain.maximum * 11, domain.maximum * 10 } else .{ row.cases, 0, 0 };
            if (!std.mem.eql(u64, &row.axis_cases, &axes)) return error.IncompleteRotarySweep;
            const stored = domain.table(row.mode);
            for (row.table_bits, stored) |bits, value| if (bits != @as(u32, @bitCast(value))) return error.UnrecordedRotaryTable;
            const wanted: usize = if (domain.stride(row.role) == 4) 17 else 51;
            if (row.normalization.len != wanted) return error.IncompleteRotarySweep;
            var norm_seen: [51]bool = @splat(false);
            for (row.normalization) |n| {
                if ((n.width != 128 and n.width != 256) or n.epsilon != @as(u32, @bitCast(@as(f32, 1e-5)))) return error.InvalidRotaryReceipt;
                const at = @import("rotary_fixture.zig").match(row.role, n.width, n.weight, n.input[0..n.width]) orelse return error.UnrecordedRotaryFixture;
                if (norm_seen[at]) return error.IncompleteRotarySweep;
                norm_seen[at] = true;
                for (n.output[0..n.width]) |bits| if ((bits & 0x7f80) == 0x7f80) return error.InvalidRotaryReceipt;
            }
            for (row.frequency_bits) |bits| {
                const frequency: f32 = @bitCast(bits);
                if (!(frequency > 0 and std.math.isFinite(frequency))) return error.InvalidRotaryReceipt;
            }
            if (row.mode == .yarn) {
                const table = domain.table(.yarn);
                for (row.frequency_bits, table[3..35]) |bits, value| if (bits != @as(u32, @bitCast(value))) return error.UnrecordedRotaryTable;
            }
            inline for (.{ "exp2", "phase", "cosine", "sine", "coefficient_cosine", "coefficient_sine" }) |key| {
                const m = @field(row, key);
                if (!(m.absolute >= 0 and m.ulps >= 0) or !std.math.isFinite(m.absolute) or !std.math.isFinite(m.ulps)) return error.InvalidRotaryReceipt;
                if (m.absolute_at.linear >= row.cases or m.ulps_at.linear >= row.cases) return error.InvalidRotaryReceipt;
            }
        }
    }

    pub fn find(r: *const Receipt, role: Role, rotary: []const f32) !*const Row {
        const mode = try domain.modeOf(rotary);
        for (r.rows) |*entry| if (entry.role == role and entry.mode == mode) {
            for (entry.table_bits, rotary) |bits, value| if (bits != @as(u32, @bitCast(value))) return error.UnrecordedRotaryTable;
            return entry;
        };
        return error.IncompleteRotarySweep;
    }
};
