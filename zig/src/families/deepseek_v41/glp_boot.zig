//! CUDA additive GLPv3 admission and upload; paired-site concept and format: Matt Suiche/weightless.
const std = @import("std");
const quiet = @import("builtin").is_test;
const glp_plan = @import("glp_plan.zig");
const pf = @import("forward_prefill.zig");

/// Loaded policy metadata survives zero strength even though there is then no active GPU plan.
pub const Source = struct {
    plan: ?glp_plan.Plan,
    policy: [8]i64,
    width: usize,
    deepest: u32,

    pub fn init(gpa: std.mem.Allocator, given: anytype) !?Source {
        const g = given orelse return null;
        if (g.vector.layers.len == 0) return error.BadGlpLayers;
        return .{
            .plan = try glp_plan.Plan.init(gpa, g),
            .policy = glp_plan.policyWords(given),
            .width = g.vector.width,
            .deepest = g.vector.layers[g.vector.layers.len - 1].id,
        };
    }

    pub fn deinit(s: *Source, gpa: std.mem.Allocator) void {
        if (s.plan) |*p| p.deinit(gpa);
        s.* = undefined;
    }
};

pub fn init(m: anytype) void {
    m.glp = null;
    m.glp_policy = @splat(0);
}

/// Validate every loaded policy, clone active directions and agree on ranks before any weights load.
pub fn open(m: anytype, a: std.mem.Allocator, given: ?*const Source, layer_prefix: bool) !void {
    const policy: [8]i64 = if (given) |g| g.policy else @splat(0);
    if (given) |g| {
        if (g.width != m.cfg.hidden) return error.BadGlpWidth;
        if (g.deepest >= m.cfg.layers) return error.BadGlpLayers;
        if (g.plan) |*p| {
            if (!std.mem.eql(i64, &glp_plan.Plan.words(p), &policy)) return error.GlpMismatch;
            m.glp = try p.clone(a);
        }
    }
    m.glp_policy = policy;
    try @import("glp_agree.zig").agree(&m.session.plan, policy);
    if (m.glp) |*p| try admit(m, p, layer_prefix);
}

/// Direction rows count against admission before the KV pool is allocated.
pub fn resident(m: anytype) u64 {
    const g = m.glp orelse return 0;
    return std.mem.alignForward(u64, g.tableBytes(), 256);
}

/// The active path requires the whole backbone and a built GLP image.
pub fn admit(m: anytype, g: *const glp_plan.Plan, layer_prefix: bool) !void {
    try g.admit(m.cfg.layers, m.cfg.hidden);
    if (layer_prefix) {
        if (!quiet) std.log.err("glp: a layer prefix skips paired folds; steer the whole backbone", .{});
        return error.GlpUnsupported;
    }
    try m.kernels.loadGlp(&m.driver);
}

/// Paths entering after the fold, splitting its layer or tracing its old value are not steering paths.
pub fn refuseKnobs(m: anytype) !void {
    const why: ?[]const u8 = if (pf.state(&m.f).mode == .replay)
        "TF_DSV41_PREFILL=replay enters past the fold"
    else if (m.f.opts.pf_tbo)
        "TF_DSV41_PF_TBO splits the paired layer across micro-batches"
    else if (m.f.opts.trace)
        "the trace exposes an unsteered fold"
    else
        null;
    if (why) |text| {
        if (!quiet) std.log.err("glp: {s}; unset it to serve a vector", .{text});
        return error.GlpUnsupported;
    }
}

/// Upload the persistent table before execution or graph capture; no per-token staging or allocation.
pub fn upload(m: anytype, gpa: std.mem.Allocator, g: *const glp_plan.Plan) !void {
    const table = try g.table(gpa);
    defer gpa.free(table);
    try m.f.upload(m.runner.addressOf(glp_plan.dirs_role).?, table);
    std.log.scoped(.dsv41).info("glp: add at strength {d} on {d} layers, directions uploaded ({d} bytes)", .{ g.alpha, g.layers.len, table.len });
}

/// Bind disk sessions to every loaded policy, including zero, without touching absent-file identities.
pub fn checkpoint(io: std.Io, pack: anytype, cache: anytype, policy: [8]i64, enabled: bool) !?[32]u8 {
    if (!enabled) return null;
    return glp_plan.bindPolicy(try @import("prepared.zig").packIdentity(io, pack, cache), policy);
}
