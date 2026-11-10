//! One deterministic startup receipt of requested knobs and the effective serving paths on each rank.
const std = @import("std");
const Allocator = std.mem.Allocator;
pub const Entry = struct { name: []const u8, value: []const u8 };
pub const Arm = struct {
    rank: u32,
    world: u32,
    gather: []const u8,
    hcas: u32,
    graphs_requested: bool,
    graphs_effective: bool,
    layer_hash: bool,
    pieces_requested: bool,
    floor_effective: bool,
    adaptive_rows_effective: bool,
    prefill_rows: u32,
    slots: u32,
    drafts: bool,
    sessions: bool,
    prefill: []const u8,
};

pub fn relevant(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "TF_DSV41_") or std.mem.startsWith(u8, name, "TF_TP_") or
        std.mem.startsWith(u8, name, "NCCL_") or std.mem.eql(u8, name, "TF_ROCE") or
        std.mem.eql(u8, name, "TF_COMM_BACKEND") or std.mem.eql(u8, name, "CUDA_VISIBLE_DEVICES");
}

pub fn entries(a: Allocator, env: *const std.process.Environ.Map) ![]const Entry {
    var out: std.ArrayList(Entry) = .empty;
    for (env.keys(), env.values()) |key, value| if (relevant(key)) try out.append(a, .{ .name = key, .value = value });
    std.mem.sort(Entry, out.items, {}, struct {
        fn less(_: void, x: Entry, y: Entry) bool {
            return std.mem.lessThan(u8, x.name, y.name);
        }
    }.less);
    return out.toOwnedSlice(a);
}

pub fn line(a: Allocator, arm: Arm, knobs: []const Entry) ![]const u8 {
    return std.json.Stringify.valueAlloc(a, .{ .version = 1, .arm = arm, .knobs = knobs }, .{});
}

/// Startup receipts and per-layer hashes are separate opt-ins; both default to no extra receipt.
pub fn requested(hashes: bool, raw: ?[]const u8) !bool {
    return (try @import("sys").environment.value(bool, "TF_STARTUP_RECEIPT", raw, false)) or hashes;
}

pub fn log(m: anytype, a: Allocator) !void {
    return logIf(m, a, @import("serve_knobs.zig").env("TF_STARTUP_RECEIPT"));
}

/// A disabled receipt returns before environment enumeration or reading any other model fields.
pub fn logIf(m: anytype, a: Allocator, receipt: ?[]const u8) !void {
    if (!try requested(m.f.layer_hash != null, receipt)) return;
    const environ: std.process.Environ = .{ .block = .{ .slice = std.mem.span(std.c.environ) } };
    var env = try environ.createMap(a);
    defer env.deinit();
    const f = &m.f;
    const get = @import("serve_knobs.zig").env;
    const graphs = try @import("sys").environment.value(bool, "TF_DSV41_GRAPHS", get("TF_DSV41_GRAPHS"), f.graphs_default);
    const pieces = try @TypeOf(m.*).piecesEnv();
    const floor = pieces and m.gt.own_prefill and m.gt.prompt_tail == .verify and enabled(get("TF_DSV41_PIECES_FLOOR"));
    const adaptive = floor and try @import("sys").environment.value(bool, "TF_DSV41_PREFILL_ADAPT", get("TF_DSV41_PREFILL_ADAPT"), true);
    var gather: []const u8 = if (f.opts.world == 1) "local" else "nccl";
    var hcas: u32 = 0;
    if (m.session.mail != null) gather = "mailbox+nccl";
    if (m.session.roce) |r| {
        gather = if (r.wire == .shm) "roce_shm+nccl" else "roce+nccl";
        hcas = switch (r.wire) {
            .verbs => |v| v.n_hca,
            .shm => 0,
        };
    }
    if (m.session.roce_multi) |r| {
        gather = if (r.wire == .shm) "roce_multi_shm+nccl" else "roce_multi+nccl";
        hcas = switch (r.wire) {
            .verbs => |v| v.n_hca,
            .shm => 0,
        };
    }
    const json_line = try line(a, .{
        .rank = m.tcfg.rank,
        .world = f.opts.world,
        .gather = gather,
        .hcas = hcas,
        .graphs_requested = graphs,
        .graphs_effective = graphs and !f.opts.trace,
        .layer_hash = f.layer_hash != null,
        .pieces_requested = pieces,
        .floor_effective = floor,
        .adaptive_rows_effective = adaptive,
        .prefill_rows = @intCast(@import("forward_prefill.zig").segmentRows(f)),
        .slots = m.slots,
        .drafts = m.pass != null,
        .sessions = m.sessions != null,
        .prefill = @tagName(@import("forward_prefill.zig").state(f).mode),
    }, try entries(a, &env));
    std.log.scoped(.dsv41).info("serving arm: {s}", .{json_line});
}

fn enabled(value: ?[]const u8) bool {
    return if (value) |v| std.mem.eql(u8, std.mem.trim(u8, v, " "), "1") else false;
}
