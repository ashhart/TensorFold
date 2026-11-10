//! Native opening uses the shared cache before acquiring any calibration-only storage or running a width sweep.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const E = @import("cuda_engine.zig");
const warm = @import("cuda_warm.zig");
const profile = @import("cuda_startup_profile.zig");
const costs = @import("cuda_window_costs.zig");
pub const file_identity = @import("cuda_cost_identity.zig");
pub const Kind = enum { dflash, mtp };

pub fn identify(l: anytype, dir: []const u8, kind: Kind) void {
    const started = profile.now(l.io);
    l.cost_kind = kind;
    l.cost_identity = file_identity.checkpoint(l.gpa, l.io, dir) catch null;
    l.startup.load += profile.elapsed(l.io, started);
}

pub fn identity(l: anytype, d: anytype, reserved: ?*E.Cache) !costs.Keys {
    if (l.e.learning != null or l.e.pockets.any() or l.e.features != 0) return error.NonTextCostProfile;
    const ctx = l.cost_context orelse return error.MissingCostContext;
    const stamp = l.cost_identity orelse return error.MissingDrafterIdentity;
    var name: [256]u8 = undefined;
    const gpu = try ctx.name(&name);
    var driver: [64]u8 = undefined;
    const release = try std.fmt.bufPrint(&driver, "sm{d}/sms{d}/driver{d}", .{ try ctx.capability(), try ctx.attribute(.multiprocessor_count), try ctx.d.version() });
    const alternate: ?@import("cuda_nvfp4_mlp.zig").Identity = if (@hasField(@TypeOf(l.e.*), "alternate")) l.e.alternate else null;
    const target = if (alternate) |source| source.primary_hash else try file_identity.checkpoint(l.gpa, l.io, l.e.dir);
    const mlp_source: ?[32]u8 = if (alternate) |source| source.source_hash else null;
    const g = l.e.g;
    const geometry = try std.json.Stringify.valueAlloc(l.gpa, .{ .pocket = g.pocket, .hidden = g.hidden, .intermediate = g.intermediate, .layers = g.layers, .vocab = g.vocab, .k_heads = g.k_heads, .v_heads = g.v_heads, .q_heads = g.q_heads, .kv_heads = g.kv_heads, .eps = g.eps, .tied = g.tied, .context = g.context, .linear = g.linear[0..g.layers], .bits = .{ .embed = g.bits.embed, .head = g.bits.head, .proj = g.bits.proj[0..g.layers] }, .experts = g.experts, .top_k = g.top_k, .expert_bits = g.expert_bits }, .{});
    defer l.gpa.free(geometry);
    const prefill_modes = if (comptime @hasField(@TypeOf(l.e.*), "prefill_paths")) l.e.prefill_paths.config.key() else 0;
    const fp8_tile = @tagName(try cuda.fp8_prefill_tiles.requested());
    const affine_prepared = try cuda.fp8_prefill.affine_packing.requested();
    const nvfp4_tma = if (l.e.mo) |m| if (m.native) |n| n.tma != null else false else false;
    const storage = storagePolicy(l.e);
    const policy = try std.json.Stringify.valueAlloc(l.gpa, .{ .capacity = if (reserved) |c| @max(c.cap, l.e.window_rows) else l.e.window_rows, .kv8 = l.e.kv8, .verify = l.e.verify_rows, .window = l.e.window_rows, .tap_capacity = storage.tap_capacity, .rec_budget = storage.rec_budget, .affine_warps = l.e.affine.warp_flags, .nvfp4_mlp = mlp_source, .nvfp4_scope = if (alternate) |source| @tagName(source.scope) else null, .nvfp4_kinds = if (alternate) |source| source.kinds else null, .native_prefill = if (l.e.mo) |m| m.native_requested else false, .fast_prefill = if (l.e.mo) |m| m.fast8 != null else false, .prefill_rows = l.prefill_rows, .cublas_prefill = if (l.e.mo) |m| m.vendor != null else false, .prefill_operator_modes = prefill_modes, .fp8_tile = fp8_tile, .affine_prepared = affine_prepared, .nvfp4_tma = nvfp4_tma, .prefill_graph = try @import("cuda_prefill_graph.zig").requested(), .prefill_capacity = @import("qwen_prefill_config").max_rows, .prefill_revision = 2 }, .{});
    defer l.gpa.free(policy);
    const taps = try std.json.Stringify.valueAlloc(l.gpa, l.e.taps[0..l.e.n_taps], .{});
    defer l.gpa.free(taps);
    const draft = try draftBytes(l.gpa, d);
    defer l.gpa.free(draft);
    const facts = try std.json.Stringify.valueAlloc(l.gpa, .{ .trees = d.drafter().facts().trees, .nodes = d.drafter().facts().tree_nodes }, .{});
    defer l.gpa.free(facts);
    const checkpoint = std.fmt.bytesToHex(stamp, .lower);
    const target_hex = std.fmt.bytesToHex(target, .lower);
    const parts = [_][]const u8{ "qwen35-cuda/wall-chain-greedy-v2", gpu, release, geometry, policy, taps, @tagName(l.cost_kind), draft, facts, &checkpoint, &target_hex };
    const trace = if (@import("sys").getenv("TF_COST_KEY_TRACE")) |v| std.mem.eql(u8, v, "1") else false;
    if (trace) {
        const names = [_][]const u8{ "profile", "gpu", "driver", "geometry", "policy", "target-taps", "draft-kind", "draft-config", "draft-facts", "draft-checkpoint", "target-checkpoint" };
        for (names, parts) |name_, bytes| tracePart(name_, bytes);
    }
    return .{ .current = try lanes.cost_cache.keyTracing(l.gpa, l.io, &parts, trace), .reference = lanes.cost_cache.referenceKey(&parts) };
}

pub fn measure(l: anytype, d: anytype, reserved: ?*E.Cache) !profile.Times {
    const started = profile.now(l.io);
    const id = identity(l, d, reserved) catch null;
    var phases: profile.Times = .{ .load = profile.elapsed(l.io, started) };
    phases.add(try resolve(l, d, reserved, id, null));
    return phases;
}

pub fn resolve(l: anytype, d: anytype, reserved: anytype, id: ?costs.Keys, root: ?[]const u8) !profile.Times {
    const Cache = std.meta.Child(@typeInfo(@TypeOf(reserved)).optional.child);
    var phases: profile.Times = .{};
    const Sampler = struct {
        l: @TypeOf(l),
        d: @TypeOf(d),
        reserved: ?*Cache,
        pub fn run(s: *@This()) !costs.Measured {
            var out: costs.Measured = .{};
            s.d.n_costs = 0;
            if (s.reserved) |cache| {
                out.phases = try warm.prepareTimed(s.l.gpa, s.l.io, s.l.e, s.d, cache, &s.l.warmed);
            } else {
                const allocating = profile.now(s.l.io);
                var cache = try Cache.init(s.l.gpa, s.l.e, s.l.e.window_rows);
                errdefer cache.deinit(s.l.gpa);
                out.phases.reserve += profile.elapsed(s.l.io, allocating);
                out.phases.add(try warm.prepareTimed(s.l.gpa, s.l.io, s.l.e, s.d, &cache, &s.l.warmed));
                const freeing = profile.now(s.l.io);
                cache.deinit(s.l.gpa);
                out.phases.reserve += profile.elapsed(s.l.io, freeing);
            }
            if (s.d.n_costs != s.l.e.window_rows) return error.IncompleteWindowMeasurements;
            for (s.d.costs[0..s.d.n_costs], 1..) |cost, rows| {
                if (cost.width != rows) return error.InvalidWindowMeasurements;
                out.ms[rows - 1] = cost.ms;
            }
            return out;
        }
    };
    var sampler: Sampler = .{ .l = l, .d = d, .reserved = reserved };
    var result = try costs.resolve(l.gpa, l.io, id, @intCast(l.e.window_rows), root, &sampler);
    if (result.hit) {
        if (reserved) |cache| {
            result.measured.phases.add(try warm.prepareMode(l.gpa, l.io, l.e, d, cache, &l.warmed, false));
            try checkKept(l, &sampler, &result, id, root, cache);
        } else {
            const allocating = profile.now(l.io);
            var cache = try Cache.init(l.gpa, l.e, l.e.window_rows);
            errdefer cache.deinit(l.gpa);
            result.measured.phases.reserve += profile.elapsed(l.io, allocating);
            result.measured.phases.add(try warm.prepareMode(l.gpa, l.io, l.e, d, &cache, &l.warmed, false));
            try checkKept(l, &sampler, &result, id, root, &cache);
            const freeing = profile.now(l.io);
            cache.deinit(l.gpa);
            result.measured.phases.reserve += profile.elapsed(l.io, freeing);
        }
    }
    d.n_costs = 0;
    for (result.measured.ms[0..l.e.window_rows], 1..) |ms, rows| d.setCost(@intCast(rows), ms);
    phases.add(result.measured.phases);
    std.log.info("CUDA window costs: {s} {d} widths{s}{s}", .{ if (result.hit) "cached" else "measured", l.e.window_rows, if (result.retried) " (reference drift retry)" else "", if (id == null) " (cache identity unavailable)" else "" });
    return phases;
}

/// A kept table stands while a quick probe of one width stays within drift; one that does not stand is measured again.
fn checkKept(l: anytype, sampler: anytype, result: *costs.Result, id: ?costs.Keys, root: ?[]const u8, cache: anytype) !void {
    const rows = l.e.window_rows;
    const at = rows / 2; // the middle width: its cost the window choice turns on
    if (at == 0) return;
    const probed = try warm.probeWidth(l.io, l.e, cache, at);
    if (lanes.cost_rule.entryHolds(result.measured.ms[at - 1], probed)) return;
    const again = try sampler.run();
    if (!costs.valid(again.ms[0..rows])) return error.InvalidWindowMeasurements;
    result.measured.ms = again.ms;
    result.measured.phases.add(again.phases);
    result.hit = false;
    result.retried = true;
    if (id) |k| {
        _ = lanes.cost_cache.saveAt(costs.Record, l.gpa, l.io, k.current, costs.record(k.current, @intCast(rows), result.measured.ms), root);
        _ = lanes.cost_cache.saveAt(costs.Record, l.gpa, l.io, k.reference, costs.record(k.reference, @intCast(rows), result.measured.ms), root);
    }
}

const StoragePolicy = struct {
    tap_capacity: usize,
    rec_budget: struct { streams: usize, limit: usize },
};

fn storagePolicy(e: anytype) StoragePolicy {
    const Engine = @TypeOf(e.*);
    return .{
        .tap_capacity = if (@hasField(Engine, "tap_capacity")) e.tap_capacity else 0,
        .rec_budget = .{
            .streams = if (@hasField(Engine, "rec_budget")) e.rec_budget.streams else 0,
            .limit = if (@hasField(Engine, "rec_budget")) e.rec_budget.limit else 0,
        },
    };
}

fn draftBytes(a: std.mem.Allocator, d: anytype) ![]u8 {
    if (!@hasField(@TypeOf(d.*), "c")) return std.json.Stringify.valueAlloc(a, .{ .depth = d.depth, .trees = d.trees, .capacity = d.capacity }, .{});
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    inline for (@typeInfo(@TypeOf(d.c)).@"struct".field_names) |field| {
        try json.objectField(field);
        if (comptime std.mem.eql(u8, field, "taps") and @hasField(@TypeOf(d.c), "n_taps")) {
            try json.write(d.c.taps[0..d.c.n_taps]);
        } else try json.write(@field(d.c, field));
    }
    if (comptime @hasField(@TypeOf(d.*), "w")) if (comptime @hasField(@TypeOf(d.w), "bf16")) {
        try json.objectField("bf16_mask");
        try json.write(if (d.w.bf16) |p| p.mask else @as(u8, 0));
    };
    try json.endObject();
    return out.toOwnedSlice();
}

fn tracePart(name: []const u8, bytes: []const u8) void {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    std.log.info("window cost key part {s} {s}", .{ name, hex[0..12] });
}

test "draft cost identity: loaded BF16 masks separate while environment-only changes do not" {
    const a = std.testing.allocator;
    const Mask = struct { mask: u8 };
    const Draft = struct {
        c: struct { block: u32 = 8, n_taps: usize = 1, taps: [2]u32 = .{ 7, 99 } } = .{},
        w: struct { bf16: ?Mask = null } = .{},
    };
    var d: Draft = .{};
    const zero = try draftBytes(a, &d);
    defer a.free(zero);
    try std.testing.expect(std.mem.indexOf(u8, zero, "\"bf16_mask\":0") != null);
    d.w.bf16 = .{ .mask = 1 };
    const one = try draftBytes(a, &d);
    defer a.free(one);
    try std.testing.expect(std.mem.indexOf(u8, one, "\"bf16_mask\":1") != null);
    try std.testing.expect(!std.mem.eql(u8, zero, one));
    const name = "TF_DRAFT_BF16";
    const before = if (std.c.getenv(name)) |value| try a.dupeSentinel(u8, std.mem.span(value), 0) else null;
    defer if (before) |value| {
        _ = @import("sys").env.setenv(name, value.ptr, 1);
        a.free(value);
    } else {
        _ = @import("sys").env.unsetenv(name);
    };
    for ([_][:0]const u8{ "0", "31", "not-a-mask" }) |raw| {
        try std.testing.expectEqual(@as(c_int, 0), @import("sys").env.setenv(name, raw, 1));
        const actual = try draftBytes(a, &d);
        defer a.free(actual);
        try std.testing.expectEqualStrings(one, actual);
    }
}

test "draft cost identity: MTP identity retains its existing fields and bytes" {
    const a = std.testing.allocator;
    var mtp: struct { depth: u32 = 3, trees: bool = false, capacity: usize = 131072 } = .{};
    const text = try draftBytes(a, &mtp);
    defer a.free(text);
    try std.testing.expectEqualStrings("{\"depth\":3,\"trees\":false,\"capacity\":131072}", text);
}

test "storage cost identity: loaded tap capacity and each recurrent bound invalidate both cache keys" {
    const a = std.testing.allocator;
    var e: E.Engine = undefined;
    e.tap_capacity = 2112;
    e.rec_budget = .{ .streams = 16, .limit = 2889864192 };
    const original = try std.json.Stringify.valueAlloc(a, storagePolicy(&e), .{});
    defer a.free(original);
    try std.testing.expectEqualStrings("{\"tap_capacity\":2112,\"rec_budget\":{\"streams\":16,\"limit\":2889864192}}", original);
    const before = costs.keys("binary", &.{original});
    for (0..3) |field| {
        switch (field) {
            0 => e.tap_capacity += 1,
            1 => e.rec_budget.streams += 1,
            2 => e.rec_budget.limit += 1,
            else => unreachable,
        }
        const changed = try std.json.Stringify.valueAlloc(a, storagePolicy(&e), .{});
        defer a.free(changed);
        const after = costs.keys("binary", &.{changed});
        try std.testing.expect(!std.mem.eql(u8, &before.current, &after.current));
        try std.testing.expect(!std.mem.eql(u8, &before.reference, &after.reference));
        e.tap_capacity = 2112;
        e.rec_budget = .{ .streams = 16, .limit = 2889864192 };
    }
}

test "storage cost identity: old engine mocks default to zero while unrelated policy does not alter loaded storage" {
    var empty: struct {} = .{};
    var old: struct { tap_capacity: usize = 0, rec_budget: struct { streams: usize = 0, limit: usize = 0 } = .{}, unrelated: bool = false } = .{};
    try std.testing.expectEqualDeep(storagePolicy(&empty), storagePolicy(&old));
    old.unrelated = true;
    try std.testing.expectEqualDeep(storagePolicy(&empty), storagePolicy(&old));
}
