//! The round loop's constants from the model's attributes and the engine's settings (Python _family_setup).
const std = @import("std");
const Allocator = std.mem.Allocator;
const Table = @import("table.zig").Table;
const alloc = @import("allocate.zig");
const Constants = @import("depth.zig").Constants;

pub const Cost = alloc.Cost;

/// The class default acceptance by depth (Python DraftDepth.depth_prior).
pub const default_prior = [_]f64{ 0.85, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };

/// What a backend reports about its loaded model: the attributes the Python engine reads at setup.
pub const Model = struct {
    exact_width: u32 = 1, // the widest window that reproduces one-token steps bit for bit
    first_copy_rows: u32 = 0, // a lone stream's first copy window (0: the exact width)
    gpu_tokens: bool = false, // one-token rounds may run a step ahead (pipelined)
    mtp: bool = false, // a draft head is loaded
    speculate: bool = false, // the head can draft from a verify's rows
    speculate_early: bool = true, // draft every row's first draft behind the verify
    draft_prior: []const f64 = &.{}, // the head's acceptance by depth until a stream has its own
    depth_rate: f64 = 0, // the newest round's weight in a depth's acceptance estimate (0: the core's)
    plain_guard: bool = false, // plain rounds compete with drafted depths
    drafts: u32 = 1, // head drafts a round at most
    window_costs: []const Cost = &.{}, // forward ms by exact window width (timed at load)
    mtp_step_ms: f64 = 0.0, // one chained head step (timed at load)
    streams_exact: bool = true, // shared forwards keep each stream's bits
    hidden_rows: bool = false, // the backend runs several streams' windows in one forward
    batch_rows: u32 = 0, // a shared forward's rows at most (0: 32)
    max_streams: u32 = 0, // a shared forward's streams at most (0: 32)
    shared_costs: []const Cost = &.{}, // shared forwards' ms by total rows (timed at load)
    draft_probabilities: bool = false, // the head gives each drafted node's chance of landing
    draft_streams: bool = false, // the head drafts every stream of a shared round in one batch
    head_trees: bool = false, // the head drafts trees of lanes (DraftRequest.lanes) and holds them
    lane_costs: []const Cost = &.{}, // a lone stream's window ms by rows, up to its widest tree (timed at load)
};

pub const Config = struct {
    family_width: u32,
    max_copy: u32,
    first_copy: u32,
    base_width: u32,
    family_mtp: bool,
    speculate_early: bool,
    pipelined: bool,
    plain_guard: bool,
    most_drafts: u32,
    family_streams: bool,
    batch_rows: u32,
    batch_streams: u32,
    node_probabilities: bool,
    fill_lanes: u32 = 0, // rows a window may grow to with suffix-match branches (0: off; lanes/fill.zig)
    fill_match: u32 = 8, // context tokens a suffix-match branch must match (shorter ones are coincidental phrases)
    fill_floor: f64 = 0.35, // the share of grafted rows a stream must keep to go on reading its chain back every round
    draft_streams: bool,
    head_trees: bool,
    head_lanes: u32 = 0, // a lone stream's lanes at most, the planner picking each round's (0: the depth rule's chain)
    head_depth: u32 = 16, // and their depth at most
    own_prices: bool = true, // the planner prices each pick from the stream's own measured rounds (seeded from the table)
    depth_prior: []f64,
    family_costs: Table,
    shared_costs: Table,
    lane_costs: Table, // window ms by rows to 64 (lanes/plan_lanes.zig)
    mtp_step_ms: f64,
    enter_match: i64 = 8, // matching tokens a copy needs (rejects coincidental indentation)
    copy_rate: f64 = 0.94, // a copied token's chance of landing in allocation
    k: Constants = .{},

    pub fn init(gpa: Allocator, m: Model, max_rows: u32, max_draft: u32) !Config {
        if (max_rows < 1) return error.BadMaxRows;
        const width: u32 = @max(1, @min(@max(m.exact_width, 1), max_rows));
        const max_copy = width - 1;
        const first_rows: i64 = if (m.first_copy_rows != 0) m.first_copy_rows else width;
        const first_copy: u32 = @intCast(@max(1, @min(@as(i64, max_copy), first_rows - 1)));
        const prior = if (m.draft_prior.len > 0) m.draft_prior else &default_prior;
        const streams = width >= 2 and m.streams_exact and m.hidden_rows;
        const batch_rows: u32 = if (m.batch_rows != 0) m.batch_rows else 32;
        var family_costs: Table = .{};
        errdefer family_costs.deinit(gpa);
        for (m.window_costs) |c| {
            if (c.width >= 1 and c.width <= width) try family_costs.put(gpa, c.width, c.ms);
        }
        // shared forwards: window costs then timed shared rows (a later table overrides), extended to batch_rows
        var timed: std.ArrayList(Cost) = .empty;
        defer timed.deinit(gpa);
        for ([_][]const Cost{ m.window_costs, m.shared_costs }) |table| {
            for (table) |c| {
                if (c.width < 1 or c.width > batch_rows) continue;
                for (timed.items) |*t| {
                    if (t.width == c.width) {
                        t.ms = c.ms;
                        break;
                    }
                } else try timed.append(gpa, c);
            }
        }
        const depth_prior = try gpa.dupe(f64, prior);
        errdefer gpa.free(depth_prior);
        return .{
            .family_width = width,
            .max_copy = max_copy,
            .first_copy = first_copy,
            .base_width = @min(width, first_copy + 1),
            .family_mtp = m.mtp and width >= 2 and m.speculate,
            .speculate_early = m.speculate_early,
            .pipelined = m.gpu_tokens,
            .plain_guard = m.plain_guard,
            .most_drafts = @min(m.drafts, @min(width - 1, max_draft)),
            .family_streams = streams,
            .batch_rows = batch_rows,
            .batch_streams = if (streams) (if (m.max_streams != 0) m.max_streams else 32) else 1,
            .node_probabilities = m.draft_probabilities,
            .draft_streams = m.draft_streams,
            .head_trees = m.head_trees,
            .depth_prior = depth_prior,
            .family_costs = family_costs,
            .shared_costs = try alloc.extendCosts(gpa, timed.items, batch_rows),
            .lane_costs = try alloc.extendCosts(gpa, m.lane_costs, 64),
            .mtp_step_ms = m.mtp_step_ms,
            .k = .{ .depth_rate = if (m.depth_rate > 0) m.depth_rate else (Constants{}).depth_rate },
        };
    }

    pub fn deinit(c: *Config, gpa: Allocator) void {
        gpa.free(c.depth_prior);
        c.family_costs.deinit(gpa);
        c.shared_costs.deinit(gpa);
        c.lane_costs.deinit(gpa);
    }
};

test "setup derives widths like Python" {
    const gpa = std.testing.allocator;
    var c = try Config.init(gpa, .{ .exact_width = 64, .first_copy_rows = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .drafts = 4, .hidden_rows = true, .batch_rows = 128, .max_streams = 64 }, 64, 63);
    defer c.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 15), c.first_copy);
    try std.testing.expectEqual(@as(u32, 16), c.base_width);
    try std.testing.expectEqual(@as(u32, 4), c.most_drafts);
    try std.testing.expect(c.family_mtp and c.family_streams);
    try std.testing.expectEqual(@as(f64, 0.15), c.k.depth_rate);
    var steady = try Config.init(gpa, .{ .exact_width = 16, .mtp = true, .speculate = true, .drafts = 4, .depth_rate = 0.08 }, 16, 15);
    defer steady.deinit(gpa);
    try std.testing.expectEqual(@as(f64, 0.08), steady.k.depth_rate);
}
