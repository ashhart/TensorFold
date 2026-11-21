//! The release format's rank plan: FP8 and BF16 projections, MXFP4 experts, and the Engram tables left on disk.
const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const log = std.log.scoped(.dsv41);
const plan = @import("plan.zig");
const names = @import("names.zig");
const partition = @import("partition.zig");
const projection = @import("projection.zig");
const et = @import("engram_tables.zig");
const Cuts = @import("release_shard.zig").Cuts;
const mesh = @import("mesh");
const Split = @import("core").exl3_layout.Split;

const Builder = plan.Builder;
const Share = partition.Share;

fn whole(n: u32) Share {
    return .{ .lo = 0, .len = n };
}

fn markAs(b: *Builder, prefix: []const u8, comptime suffix: []const u8) !void {
    var nb: [192]u8 = undefined;
    try b.mark(try std.fmt.bufPrint(&nb, "{s}" ++ suffix, .{prefix}));
}

/// A release projection `prefix` of stored N x K `want` (0: unchecked), the rank's `rows` and `cols` (null: all).
pub fn group(b: *Builder, proj: names.Proj, prefix: []const u8, split: Split, rows: ?Share, cols: ?Share, want: [2]u32) !plan.GroupPlan {
    const r = try projection.releaseOf(b.pack, prefix);
    const nk: [2]u32 = switch (r) {
        .fp8_block => |f| .{ @intCast(f.n), @intCast(f.k) },
        .bf16 => |f| .{ @intCast(f.n), @intCast(f.k) },
        .mxfp4 => return error.UnexpectedTensor,
    };
    if ((want[0] != 0 and nk[0] != want[0]) or (want[1] != 0 and nk[1] != want[1])) {
        if (!quiet) log.warn("{s}: N={d} K={d}, expected N={d} K={d}", .{ prefix, nk[0], nk[1], want[0], want[1] });
        return error.UnexpectedShape;
    }
    const rs = rows orelse whole(nk[0]);
    const cs = cols orelse whole(nk[1]);
    if (rs.lo + rs.len > nk[0] or cs.lo + cs.len > nk[1] or rs.len == 0 or cs.len == 0) return error.UnexpectedShape;
    try markAs(b, prefix, ".weight");
    const form: plan.Form = switch (r) {
        .fp8_block => |f| blk: {
            for ([_]u32{ rs.lo, rs.len, cs.lo, cs.len }) |x| if (x % 32 != 0) return error.SplitNotOnBlocks;
            try markAs(b, prefix, ".scale");
            break :blk .{ .fp8_block = .{ .weight = f.weight, .scale = f.scale, .n = nk[0], .k = nk[1], .rows = rs, .cols = cs } };
        },
        .bf16 => |f| .{ .bf16 = .{ .weight = f.weight, .n = nk[0], .k = nk[1], .rows = rs, .cols = cs } },
        .mxfp4 => unreachable,
    };
    return .{ .proj = proj, .prefix = try b.dupe(prefix), .split = split, .form = form };
}

/// The head: BF16 [vocab][hidden], the rank's vocabulary rows (whole 128-row blocks, as the logits gather expects).
pub fn head(b: *Builder) !plan.GroupPlan {
    const v = (try Cuts.of(b.c, b.o.rank, b.o.world)).vocab;
    const g = try group(b, .wq_a, names.head.suffix, .col, v, null, .{ b.c.vocab, b.c.hidden });
    if (g.form != .bf16) return error.UnexpectedTensor;
    return g;
}

/// One routed projection's experts: every expert MXFP4 at the same shape, the rank's rows (w1, w3) or columns (w2).
fn experts(b: *Builder, block: []const u8, index: u32, proj: names.ExpertProj, share: Share) !plan.ExpertsPlan {
    const count = b.c.expertsOf(index).count;
    const kn = proj.kn(b.c);
    // expert degree 1: every rank holds every expert (an expert group would give each rank its own ids here)
    const ids = (try Cuts.at(b.c, mesh.tensorGroup(b.o.rank, b.o.world), mesh.tensorGroup(0, 1), count)).ids;
    for (0..count) |e| {
        var nb: [128]u8 = undefined;
        const prefix = try names.expertName(block, @intCast(e), proj, &nb);
        switch (try projection.releaseOf(b.pack, prefix)) {
            .mxfp4 => |f| if (f.k != kn[0] or f.n != kn[1]) return error.UnexpectedShape,
            else => return error.UnexpectedTensor,
        }
        try markAs(b, prefix, ".weight");
        try markAs(b, prefix, ".scale");
    }
    return .{ .proj = proj, .form = .{ .mxfp4 = .{
        .count = count,
        .ids = ids,
        .n = kn[1],
        .k = kn[0],
        .rows = if (proj == .w2) whole(kn[1]) else share,
        .cols = if (proj == .w2) share else whole(kn[0]),
    } } };
}

/// The Engram table of block `index`: its E4M3 rows and E8M0 scales checked, read from SSD by row, never uploaded.
fn engramTable(b: *Builder, index: u32) !void {
    const li = std.mem.indexOfScalar(u32, b.c.engram_layers.items(), index) orelse return error.NotEngram;
    _ = try et.Plan.fromPack(b.pack, index, b.c.engram_rows.items()[li]);
    var pb: [64]u8 = undefined;
    const prefix = try std.fmt.bufPrint(&pb, "layers.{d}.engram.embed", .{index});
    try markAs(b, prefix, ".weight");
    try markAs(b, prefix, ".scale");
}

/// A release projection's stored N x K: the name list's, and Engram's wkv from its geometry (the list leaves it 0).
fn wantOf(b: *const Builder, s: names.ReleaseSpec) [2]u32 {
    if (s.proj != .engram_wkv) return .{ s.n, s.k };
    return .{ (b.c.hc_mult + 1) * b.c.hidden, (b.c.engram_max_ngram - 1) * b.c.engram_heads * b.c.engram_head_dim };
}

const Cut = struct { split: Split, rows: ?Share = null, cols: ?Share = null };

/// How a rank holds a release projection: main's TP layout on the stored [N][K] (wo_a is planned a group at a time).
fn cutOf(b: *const Builder, s: names.ReleaseSpec, cut: Cuts) Cut {
    return switch (s.proj) {
        .wq_b => .{ .split = .col, .rows = cut.byGroups(s.n, b.c.o_groups) },
        .wo_b => .{ .split = .row, .cols = cut.byGroups(s.k, b.c.o_groups) },
        .shared_w1, .shared_w3 => .{ .split = .col, .rows = cut.width },
        .shared_w2 => .{ .split = .row, .cols = cut.width },
        else => .{ .split = .whole },
    };
}

/// The release block `index`: its projections, natives, experts, and its Engram table checked and left on disk.
pub fn layer(b: *Builder, index: u32) !plan.LayerPlan {
    var pb: [32]u8 = undefined;
    const block = try b.dupe(try names.blockPrefix(b.c, index, &pb));
    var hb: [96]u8 = undefined;
    const has_shared = b.pack.has(try std.fmt.bufPrint(&hb, "{s}.ffn.shared_experts.w1.weight", .{block}));
    const specs = try names.releaseBlocks(b.c, index, has_shared);
    const cut = try Cuts.of(b.c, b.o.rank, b.o.world);
    const own = cut.groups;
    const share = cut.width;
    var groups: std.ArrayList(plan.GroupPlan) = .empty;
    for (specs.items()) |s| {
        var nb: [128]u8 = undefined;
        const prefix = try std.fmt.bufPrint(&nb, "{s}.{s}", .{ block, s.suffix });
        switch (s.proj) {
            .engram_embed => try engramTable(b, index),
            // one plan a local output group: its o_lora rows of the stored [o_groups x o_lora][K]
            .wo_a => for (own.lo..own.lo + own.len) |g| {
                const rows: Share = .{ .lo = @intCast(g * b.c.o_lora), .len = b.c.o_lora };
                var x = try group(b, .wo_a, prefix, .col, rows, null, .{ b.c.o_groups * b.c.o_lora, s.k });
                x.slice = @intCast(g);
                try groups.append(b.a, x);
            },
            else => {
                const proj = std.meta.stringToEnum(names.Proj, @tagName(s.proj)).?;
                const how = cutOf(b, s, cut);
                try groups.append(b.a, try group(b, proj, prefix, how.split, how.rows, how.cols, wantOf(b, s)));
            },
        }
    }
    const nspecs = try names.blockNatives(b.c, index);
    const natives = try b.a.alloc(plan.NativePlan, nspecs.len);
    for (nspecs.items(), natives) |s, *v| {
        var nb: [128]u8 = undefined;
        v.* = try b.native(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ block, s.suffix }), s);
    }
    var out: plan.LayerPlan = .{ .index = index, .prefix = block, .mode = b.c.mode(index), .groups = groups.items, .natives = natives, .experts = undefined, .gate_up_mismatch = 0 };
    inline for (.{ .w1, .w2, .w3 }, 0..) |p, i| out.experts[i] = try experts(b, block, index, p, share);
    return out;
}
