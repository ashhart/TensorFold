//! Measured CUDA window prices in the shared build/chip/model cache; no request-time calibration.
const std = @import("std");
const lanes = @import("lanes");
const shared = lanes.cost_cache;
const profile = @import("cuda_startup_profile.zig");
pub const most = 64;
pub const Keys = struct { current: [32]u8, reference: [32]u8 };
pub const Measured = struct { ms: [most]f64 = @splat(0), phases: profile.Times = .{} };
pub const Result = struct { measured: Measured, hit: bool, retried: bool = false };
pub const Record = extern struct { schema: u32 = 1, rows: u32, key: [32]u8, ms: [most]f64, digest: [32]u8 };

pub fn keys(exe: []const u8, parts: []const []const u8) Keys {
    return .{ .current = shared.buildKey(exe, parts), .reference = shared.referenceKey(parts) };
}

pub fn valid(ms: []const f64) bool {
    if (ms.len == 0 or ms.len > most) return false;
    for (ms) |m| if (!(m > 0) or !std.math.isFinite(m)) return false;
    return true;
}

pub fn record(key: [32]u8, rows: u32, ms: [most]f64) Record {
    var value: Record = .{ .rows = rows, .key = key, .ms = ms, .digest = undefined };
    std.crypto.hash.sha2.Sha256.hash(std.mem.asBytes(&value)[0..@offsetOf(Record, "digest")], &value.digest, .{});
    return value;
}

pub fn accepted(value: Record, key: [32]u8, rows: u32) bool {
    if (value.schema != 1 or value.rows != rows or rows == 0 or rows > most or !std.mem.eql(u8, &value.key, &key)) return false;
    if (!valid(value.ms[0..rows])) return false;
    for (value.ms[rows..]) |m| if (@as(u64, @bitCast(m)) != 0) return false;
    const expected = record(key, rows, value.ms);
    return std.mem.eql(u8, &value.digest, &expected.digest);
}

pub fn resolve(a: std.mem.Allocator, io: std.Io, identity: ?Keys, rows: u32, root: ?[]const u8, sampler: anytype) !Result {
    if (rows == 0 or rows > most) return error.UnsupportedWindowCosts;
    const started = profile.now(io);
    if (identity) |id| if (shared.loadAt(Record, a, io, id.current, root)) |kept| {
        if (accepted(kept, id.current, rows)) return .{ .hit = true, .measured = .{ .ms = kept.ms, .phases = .{ .calibration = profile.elapsed(io, started) } } };
    };
    var measured = try sampler.run();
    if (!valid(measured.ms[0..rows])) return error.InvalidWindowMeasurements;
    @memset(measured.ms[rows..], 0);
    var retried = false;
    if (identity) |id| if (shared.loadAt(Record, a, io, id.reference, root)) |reference| {
        if (accepted(reference, id.reference, rows) and lanes.cost_rule.drifted(measured.ms[0..rows], reference.ms[0..rows])) {
            const again = try sampler.run();
            if (!valid(again.ms[0..rows])) return error.InvalidWindowMeasurements;
            for (measured.ms[0..rows], again.ms[0..rows]) |*m, next| m.* = @min(m.*, next);
            measured.phases.add(again.phases);
            retried = true;
        }
    };
    if (identity) |id| {
        shared.saveAt(Record, a, io, id.current, record(id.current, rows, measured.ms), root);
        shared.saveAt(Record, a, io, id.reference, record(id.reference, rows, measured.ms), root);
    }
    measured.phases.calibration = @max(0, profile.elapsed(io, started) - measured.phases.reserve - measured.phases.warmups - measured.phases.load);
    return .{ .measured = measured, .hit = false, .retried = retried };
}
