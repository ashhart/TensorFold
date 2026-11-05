//! A Qwen CUDA engine as one stage of the ring (RFC #2005 v1): its binding of fabric/stage's driver.
const std = @import("std");
const stage = @import("stage");
const E = @import("cuda_engine.zig");

const boot = stage.boot;
const plan = stage.plan;
const driver = stage.driver;

/// Every stage of one placement reports this; rank 0 refuses another. This file is the driver's binding `B`.
pub const version = "tensorfold qwen3_5-cuda stage-v1";

pub const Engine = E.Engine;
pub const Cache = E.Cache;
pub const Part = E.Part;
pub const backend: stage.payload.Backend = .cuda;
pub const most_streams = E.max_segs;

/// The ring's boundary for this engine: h and r as bf16 planes, no drafter taps (v1).
pub fn boundary(e: *const E.Engine) boot.Boundary {
    return .{ .hidden = @intCast(e.g.hidden), .row_dtype = .bf16, .planes = 2, .draft_width = 0 };
}

/// The shapes every stage runs: a prompt chunk and a round at most `max_rows` rows, `parallel` slots of `context`.
pub fn shapes(parallel: u32, context: u32) boot.Shapes {
    return .{ .prompt_chunk = E.max_rows, .max_rows = E.max_rows, .parallel = parallel, .context = context };
}

pub fn kvDtype(e: *const E.Engine) stage.payload.Dtype {
    return if (e.kv8) .f8e4m3 else .bf16;
}

/// A slot's cache, for this stage's own layers only.
pub fn open(gpa: std.mem.Allocator, e: *E.Engine, context: u32) !E.Cache {
    return E.Cache.init(gpa, e, context);
}

pub fn close(gpa: std.mem.Allocator, c: *E.Cache) void {
    c.deinit(gpa);
}

pub fn reset(e: *E.Engine, c: *E.Cache) !void {
    try c.reset(e);
}

pub fn position(c: *const E.Cache) u32 {
    return @intCast(c.len);
}

pub fn window(c: *const E.Cache) ?driver.Window {
    const w = c.last orelse return null;
    return .{ .start = @intCast(w.start), .rows = @intCast(w.rows) };
}

/// The last window kept to its first `accepted` rows, a prefix path.
pub fn rollback(e: *E.Engine, c: *E.Cache, accepted: u32) !void {
    var path: [E.most_window]u32 = undefined;
    for (path[0..accepted], 0..) |*r, i| r.* = @intCast(i);
    try c.keep(e, path[0..accepted]);
}

/// This stage's layers over the round's rows: a prompt chunk keeps its last row's logits, a window every row's.
pub fn forward(e: *E.Engine, parts: []const E.Part, ids: []const u32, mode: plan.Mode, hop: ?driver.Hop) !void {
    e.hop = if (hop) |h| .{ .ctx = h.ctx, .give = h.give, .take = h.take } else null; // the engine builds without stage
    defer e.hop = null;
    try e.forward(parts, ids, if (mode == .prompt) .last else .logits);
}

pub const Driver = driver.Driver(@This());
pub const Entry = driver.Entry;
pub const Slots = Driver.Slots;
pub const Lead = Driver.Lead;
pub const serve = Driver.serve;
pub const describe = Driver.describe;

/// The checkpoint's and layout's fingerprint: SHA-256 over config.json, the version and every rank's range.
pub fn fingerprint(config: []const u8, ranges: []const boot.Range) [32]u8 {
    return driver.fingerprint(version, config, ranges);
}

/// Boot on the ring, every rank a CUDA stage; the next round returned.
pub fn bootRing(gpa: std.mem.Allocator, ring: *stage.Ring, own: *const boot.Description, model: *const boot.Model, ranges: []const boot.Range) !u64 {
    return driver.bootRing(gpa, ring, own, model, ranges, .cuda);
}
