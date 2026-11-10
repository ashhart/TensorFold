//! A load owns its checkpoints and upload leases until all users have finished; inference keeps only identities.
const std = @import("std");
const core = @import("core");
const geometry = @import("cuda_geometry.zig");
const alternate = @import("cuda_nvfp4_mlp.zig");
const weights = @import("cuda_weights.zig");

pub const Span = struct {
    first: usize = 0,
    count: usize,
    whole: usize,

    pub fn full(layers: usize) Span {
        return .{ .count = layers, .whole = layers };
    }
    pub fn isWhole(s: Span) bool {
        return s.first == 0 and s.count == s.whole;
    }
    pub fn validate(s: Span) !void {
        if (s.whole == 0 or s.count == 0 or s.first > s.whole or s.count > s.whole - s.first)
            return error.InvalidQwenLoadSpan;
    }
};

/// A foreign upload implementation may lend mappings; this contract never creates or silently substitutes one.
pub const MappedSource = struct {
    ctx: *anyopaque,
    release: *const fn (*anyopaque) void,
};

pub const Budgets = struct {
    weight_bytes: ?usize = null,
    stage_bytes: usize,
    scratch_bytes: usize = 0,

    pub fn arenaBytes(b: Budgets) !usize {
        return std.math.add(usize, b.weight_bytes orelse return error.UnresolvedQwenWeightBudget, b.scratch_bytes);
    }
};
pub const Options = struct { span: ?Span = null };

/// Whole-model ordinary loads are the current entry; partial-span algorithms belong to their own qualified owner.
pub fn admit(span: Span, has_alternate: bool, has_mapping: bool) !void {
    try span.validate();
    if (!span.isWhole()) return if (has_alternate) error.AlternateQwenNeedsWholeModel else error.UnqualifiedQwenLoadSpan;
    if (has_alternate and has_mapping) return error.AlternateQwenCannotMapPrimary;
}

pub const LoadPlan = struct {
    checkpoint: core.Checkpoint,
    source: ?alternate.Opened,
    geometry: geometry.Geometry,
    span: Span,
    budgets: Budgets,
    workspace: @import("cuda_workspace_plan.zig").WorkspacePlan,
    mapped: ?MappedSource = null,
    open: bool = true,

    pub fn create(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, options: Options) !LoadPlan {
        const g = try geometry.Geometry.read(gpa, io, dir);
        const span = options.span orelse Span.full(g.layers);
        if (span.whole != g.layers) return error.InvalidQwenLoadSpan;
        const selected = (try alternate.selection(try alternate.directory(), try @import("sys").environment.given("TF_QWEN_NVFP4_SCOPE"))) != null;
        try admit(span, selected, false);
        var ck = try core.Checkpoint.openModel(gpa, io, dir);
        errdefer core.checkpoint.cache.close(&ck);
        var source = try alternate.open(gpa, io, g, &ck, dir);
        errdefer if (source) |*s| s.deinit();
        const workspace = try @import("cuda_workspace_plan.zig").WorkspacePlan.forRows(g, @import("qwen_prefill_config").max_rows);
        return .{ .checkpoint = ck, .source = source, .geometry = g, .span = span, .workspace = workspace, .budgets = .{
            .weight_bytes = if (source) |s| s.plan.weight_bytes else null,
            .stage_bytes = if (source) |s| s.plan.stage_bytes else weights.stageBytes(g),
            .scratch_bytes = workspace.budget.bytes,
        } };
    }

    pub fn primary(p: *LoadPlan) *core.Checkpoint {
        std.debug.assert(p.open);
        return &p.checkpoint;
    }
    pub fn alternateSource(p: *LoadPlan) ?*alternate.Opened {
        std.debug.assert(p.open);
        return if (p.source) |*s| s else null;
    }

    /// Ownership transfers only on success. Existing current-main loads supply no mapped source.
    pub fn adoptMapped(p: *LoadPlan, mapping: MappedSource) !void {
        if (!p.open or p.mapped != null) return error.QwenLoadAlreadyBound;
        try admit(p.span, p.source != null, true);
        p.mapped = mapping;
    }
    pub fn finishMapped(p: *LoadPlan) void {
        if (p.mapped) |m| {
            p.mapped = null;
            m.release(m.ctx);
        }
    }

    /// The mapped implementation is intentionally not emulated by an ordinary checkpoint copy.
    pub fn requireCurrentUpload(p: *const LoadPlan) !void {
        if (!p.open) return error.QwenLoadClosed;
        if (p.mapped != null) return error.UnqualifiedQwenMappedUpload;
        try admit(p.span, p.source != null, false);
    }

    pub fn deinit(p: *LoadPlan) void {
        if (!p.open) return;
        p.finishMapped();
        if (p.source) |*s| s.deinit();
        core.checkpoint.cache.close(&p.checkpoint);
        p.open = false;
    }
};

/// Current-main callers retain the ordinary entry; optional load modes use the typed constructor explicitly.
pub fn initDefault(gpa: std.mem.Allocator, io: std.Io, ctx: *const @import("cuda").Context, dir: []const u8, kv8: bool, fit: ?@import("cuda_host_embed.zig").Fit) !*@import("cuda_engine.zig").Engine {
    return @import("cuda_engine.zig").Engine.initWithOptions(gpa, io, ctx, dir, kv8, fit, .{});
}
