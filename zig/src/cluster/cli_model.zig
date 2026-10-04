//! A cluster file's model planned over its nodes: inventories, the tensor table, the MLA split, budgets and admission.
const std = @import("std");
const node = @import("node.zig");
const config = @import("config.zig");
const launch = @import("launch.zig");
const model = @import("model.zig");
const checkpoint = @import("checkpoint.zig");
const estimate = @import("estimate.zig");
const plan = @import("plan.zig");
const budget = @import("budget.zig");
const admission = @import("admission.zig");
const traffic = @import("traffic.zig");

const Allocator = std.mem.Allocator;

pub const Source = struct {
    io: std.Io,
    a: Allocator,
    runner: launch.Runner,
    /// Probe every node over the runner (ssh, or saved probes); else use the file's declared chip and memory.
    live: bool,
    /// config.json and the Hub file list for planning before download (kimi_linear); glm_moe_dsa needs only the config.
    shape_path: ?[]const u8 = null,
    files_path: ?[]const u8 = null,
};

/// Inventories for `names`: probed, or from the file's declared chip, memory and GPU limit.
pub fn inventories(src: Source, c: config.Config, names: []const []const u8) ![]node.Inventory {
    const out = try src.a.alloc(node.Inventory, names.len);
    for (names, out) |name, *inv| {
        const n = c.nodes[c.find(name).?];
        if (src.live) {
            inv.* = try launch.inventory(src.a, src.runner, c, n, false);
            continue;
        }
        const memory = n.memory orelse return error.MemoryUnknown;
        const chip = n.chip orelse "";
        inv.* = .{ .id = node.idFrom(n.name), .name = .of(n.name), .backend = n.backend, .chip = .of(chip), .memory = memory, .gpu_limit = n.gpu_limit orelse node.gpuLimitFor(memory, 0), .bandwidth = node.bandwidthOf(chip), .unified = n.backend == .metal };
        if (std.mem.indexOf(u8, chip, "M3 Ultra") != null) inv.gpu_cores = 80;
    }
    return out;
}

pub const Planned = struct {
    shape: model.Shape,
    ck: checkpoint.Checkpoint,
    p: plan.Plan,
    needs: []budget.Need,
    fits: []budget.Fit,
    /// The MLA split was the file's, or admission chose it.
    chosen: bool,
};

/// The model's tensor table: its folder on this host, or its config (and Hub file list) before download.
fn table(src: Source, m: config.Model) !struct { model.Shape, checkpoint.Checkpoint } {
    const cwd = std.Io.Dir.cwd();
    if (src.shape_path) |path| {
        const shape = try model.fromConfig(src.a, try cwd.readFileAlloc(src.io, path, src.a, .limited(1 << 20)));
        if (std.mem.eql(u8, shape.family, "glm_moe_dsa")) return .{ shape, try estimate.glm(src.a, &shape) };
        const files_path = src.files_path orelse return error.FileListNeeded;
        const files = try estimate.manifest(src.a, try cwd.readFileAlloc(src.io, files_path, src.a, .limited(16 << 20)));
        if (!std.mem.eql(u8, shape.family, "kimi_linear")) return error.EstimateUnsupported;
        return .{ shape, try estimate.kimi(src.a, &shape, files) };
    }
    const dir = m.pathFor(m.nodes[0]) orelse return error.NoModelPath;
    const shape = try model.fromConfig(src.a, try cwd.readFileAlloc(src.io, try std.fs.path.join(src.a, &.{ dir, "config.json" }), src.a, .limited(1 << 20)));
    return .{ shape, try checkpoint.load(src.io, src.a, dir) };
}

/// Plan `m` over `invs`; an "auto" expert degree or MLA split takes admission's choice by fit, then round cost.
pub fn planModel(src: Source, m: config.Model, invs: []const node.Inventory) !Planned {
    const t = try table(src, m);
    const shape = t[0];
    const ck = t[1];
    const splits: []const plan.MlaSplit = if (m.mla) |x| &.{x} else if (shape.count(.mla) == 0) &.{.heads} else &.{ .heads, .streams };
    const expert_choices: []const u32 = if (m.layout.expert != 0 or shape.experts == 0) &.{m.layout.expert} else &.{ 0, 1 };
    var plans: std.ArrayList(plan.Plan) = .empty;
    var needs: std.ArrayList([]budget.Need) = .empty;
    for (expert_choices) |e| for (splits) |mla| {
        var opts = m.options();
        opts.layout.expert = e;
        opts.mla = mla;
        const p = plan.plan(src.a, ck, &shape, invs, opts) catch |err| switch (err) {
            error.BadLayout => continue,
            else => return err,
        };
        try plans.append(src.a, p);
        try needs.append(src.a, try budget.needs(src.a, &plans.items[plans.items.len - 1], ck, &shape));
    };
    if (plans.items.len == 0) return error.BadLayout;
    const cands = try src.a.alloc(admission.Candidate, plans.items.len);
    for (cands, plans.items, needs.items) |*c, *p, n| c.* = .{ .p = p, .ns = n };
    const i = admission.best(cands, &shape, traffic.tb5_polled);
    const p = plans.items[i];
    return .{ .shape = shape, .ck = ck, .p = p, .needs = needs.items[i], .fits = try budget.fits(src.a, &p, needs.items[i]), .chosen = plans.items.len > 1 };
}

/// What admission allows under the plan: streams at the model's context, context at its streams, rows and prompt rows a round.
pub fn summary(w: *std.Io.Writer, pl: *const Planned, m: config.Model) !void {
    const o = pl.p.opts;
    const most = admission.maxStreams(&pl.p, pl.needs, &pl.shape, o.context);
    const longest = admission.maxContext(&pl.p, pl.needs, &pl.shape, o.streams);
    const chunk = if (m.chunk > 0) m.chunk else admission.chunkRows(&pl.p, pl.needs, &pl.shape, o.streams, o.context, 0.5, traffic.tb5_polled);
    const experts = if (pl.shape.experts == 0) "no experts" else if (pl.p.layout.expert == 1) "experts split" else "experts spread";
    try w.print("admission: {s}, MLA by {s}{s}; {d} streams fit at {d}k context ({d} planned); {d} streams fit {d}k each; prompt rows a round {d} (+50% round time at {d} decode rows)\n", .{ experts, @tagName(o.mla), if (pl.chosen) " (chosen)" else "", most, o.context / 1024, o.streams, o.streams, longest / 1024, chunk, o.streams });
}
