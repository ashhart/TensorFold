//! DSpark's Markov chain on the GPU: shared-w2 multi-ask full-vocab mode and top-K single-TG resident mode.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const sources = @import("kernel_sources");

/// Whether the kernels take a Markov head's rank: 32, 64, 128 or 256 (1 to 8 values a lane of a simdgroup).
pub fn takes(rank: u32) bool {
    return rank == 32 or rank == 64 or rank == 128 or rank == 256;
}
/// Threadgroups an ask's full-vocab search is cut into.
pub const parts = 256;
/// Slices a row's logits are split into for top-K extraction.
pub const topk_slices = 16;
/// Maximum top-K candidates per row supported by the GPU top-K extractor and single-TG chain.
pub const max_topk = 256;
const simdgroups = 8;
/// Rows of a launch (the drafter's 16).
const R = 16;

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
const Args = extern struct { vocab: u32, rows: u32, step: u32, parts: u32, dim: u32, asks: u32, topk: u32 };

pub const Pipelines = struct {
    z: mtl.Pipeline,
    pick: mtl.Pipeline,
    hid: mtl.Pipeline,
    topk_part: mtl.Pipeline,
    topk_merge: mtl.Pipeline,
    chain_topk: mtl.Pipeline,

    /// The kernels of a head of `rank` (`takes`): shared-w2 multi-ask + 2-stage radix top-K + single-TG chain.
    pub fn load(device: mtl.Device, rank: u32) !Pipelines {
        if (!takes(rank)) return error.MarkovRank;
        const lib = try mtl.Library.fromSource(device, sources.dsv41.keyed ++ sources.dsv41.markov, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var zb: [32]u8 = undefined;
        var pb: [32]u8 = undefined;
        var cb: [40]u8 = undefined;
        const z = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&zb, "tf_dsv41_markov_z_{d}", .{rank / 32}), false);
        errdefer z.deinit();
        const pick = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&pb, "tf_dsv41_markov_pick_{d}", .{rank / 32}), false);
        errdefer pick.deinit();
        const hid = try mtl.Pipeline.init(device, lib, "tf_dsv41_markov_hid", false);
        errdefer hid.deinit();
        const topk_part = try mtl.Pipeline.init(device, lib, "tf_dsv41_markov_topk_part", false);
        errdefer topk_part.deinit();
        const topk_merge = try mtl.Pipeline.init(device, lib, "tf_dsv41_markov_topk_merge", false);
        errdefer topk_merge.deinit();
        const chain_topk = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&cb, "tf_dsv41_markov_chain_topk_{d}", .{rank / 32}), false);
        return .{ .z = z, .pick = pick, .hid = hid, .topk_part = topk_part, .topk_merge = topk_merge, .chain_topk = chain_topk };
    }

    pub fn deinit(p: *Pipelines) void {
        p.z.deinit();
        p.pick.deinit();
        p.hid.deinit();
        p.topk_part.deinit();
        p.topk_merge.deinit();
        p.chain_topk.deinit();
    }
};

/// The heads on the device in one buffer: markov_embed and markov_head (bf16 [vocab][rank]), conf_w (f32 [dim + rank]).
pub const Tables = struct {
    buf: mtl.Buffer,
    w2: usize,
    cw: usize,

    pub fn init(device: mtl.Device, w1: []const u16, w2: []const u16, cw: []const f32) !Tables {
        if (w1.len != w2.len) return error.MarkovShape;
        const at2 = std.mem.alignForward(usize, 2 * w1.len, 256);
        const atc = std.mem.alignForward(usize, at2 + 2 * w2.len, 256);
        const buf = try device.buffer(atc + 4 * cw.len, mtl.ResourceOptions.shared);
        const m = buf.contents();
        @memcpy(m[0 .. 2 * w1.len], std.mem.sliceAsBytes(w1));
        @memcpy(m[at2..][0 .. 2 * w2.len], std.mem.sliceAsBytes(w2));
        @memcpy(m[atc..][0 .. 4 * cw.len], std.mem.sliceAsBytes(cw));
        return .{ .buf = buf, .w2 = at2, .cw = atc };
    }

    pub fn deinit(t: *Tables) void {
        t.buf.deinit();
    }
};

/// Ask descriptor matching `markov.metal` TfMarkovAsk (40 bytes; pos0 stays at offset 8 for early accept).
pub const Ask = extern struct {
    seed: u64 = 0,
    pos0: u64 = 0,
    t: f32 = 1,
    sampled: u32 = 0,
    top_k: u32 = 0,
    top_p: f32 = 1,
    min_p: f32 = 0,
    pad: u32 = 0,
};

comptime {
    if (@offsetOf(Ask, "pos0") != 8 or @sizeOf(Ask) != 40) @compileError("Ask layout mismatch");
}

/// The chain's asks, results and scratch: drafts, confidences, hidden parts, full-vocab parts, and top-K candidates.
pub const Out = struct {
    buf: mtl.Buffer,

    const drafts_at = 0;
    const conf_at = 4 * R;
    const hid_at = 8 * R;
    pub const asks_at = 12 * R;
    const ps_at = asks_at + @sizeOf(Ask) * R;
    const pz_at = ps_at + 4 * R * parts;
    const pid_at = pz_at + 4 * R * parts;
    const cand_at = pid_at + 4 * R * parts;
    const base_at = cand_at + 4 * R * max_topk;
    const pval_at = base_at + 4 * R * max_topk;
    const ptid_at = pval_at + 4 * R * topk_slices * max_topk;
    const bytes = ptid_at + 4 * R * topk_slices * max_topk;

    pub fn init(device: mtl.Device) !Out {
        const buf = try device.buffer(bytes, mtl.ResourceOptions.shared);
        @memset(buf.contents()[0..bytes], 0);
        return .{ .buf = buf };
    }

    pub fn asks(o: *const Out) []Ask {
        return @alignCast(std.mem.bytesAsSlice(Ask, o.buf.contents()[asks_at..][0 .. @sizeOf(Ask) * R]));
    }

    pub fn deinit(o: *Out) void {
        o.buf.deinit();
    }

    pub fn drafts(o: *const Out, n: usize) []const u32 {
        return o.buf.slice(u32, R)[0..n];
    }

    pub fn conf(o: *const Out, n: usize) []const f32 {
        return @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, o.buf.contents()[conf_at..][0 .. 4 * R])))[0..n];
    }

    pub fn cand(o: *const Out, n: usize, k: usize) []const i32 {
        return @as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, o.buf.contents()[cand_at..][0 .. 4 * n * k])));
    }

    pub fn base(o: *const Out, n: usize, k: usize) []const f32 {
        return @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, o.buf.contents()[base_at..][0 .. 4 * n * k])));
    }
};

/// The GPU chain's kernels, the heads' tables on the device, its results, and top-K width (0: full-vocab only).
pub const Chain = struct {
    p: Pipelines,
    t: Tables,
    o: Out,
    topk: u32 = 0,

    pub fn init(device: mtl.Device, m: host.dspark.Markov, topk: u32) !Chain {
        var p = try Pipelines.load(device, m.rank);
        errdefer p.deinit();
        var t = try Tables.init(device, m.w1, m.w2, m.conf.?);
        errdefer t.deinit();
        return .{ .p = p, .t = t, .o = try Out.init(device), .topk = topk };
    }

    pub fn deinit(ch: *Chain) void {
        ch.p.deinit();
        ch.t.deinit();
        ch.o.deinit();
    }

    pub fn writeAsk(ch: *const Chain, j: usize, p: host.dspark.Params) void {
        ch.o.asks()[j] = .{
            .seed = p.seed,
            .pos0 = p.pos0,
            .t = @floatCast(@max(p.temperature, 1e-6)),
            .sampled = @intFromBool(p.sampled),
            .top_k = p.top_k,
            .top_p = @floatCast(p.top_p),
            .min_p = @floatCast(p.min_p),
        };
    }
};

/// Full-vocab chain of `asks` asks of `rows` drafts (shared-w2 across up to 4 asks per threadgroup).
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, t: *const Tables, o: *const Out, logits: Ref, hf: Ref, ids: Ref, vocab: u32, dim: u32, asks: u32, rows: u32) void {
    var a: Args = .{ .vocab = vocab, .rows = rows, .step = 0, .parts = parts, .dim = dim, .asks = asks, .topk = 0 };
    e.setPipeline(p.hid);
    e.setValue(a, 0);
    e.setBuffer(hf.buf, hf.off, 1);
    e.setBuffer(t.buf, t.cw, 2);
    e.setBuffer(o.buf, Out.hid_at, 3);
    e.dispatchGroups(mtl.Size.of(asks * rows, 1, 1), mtl.Size.of(1024, 1, 1));
    const y_groups = (asks + 3) / 4;
    for (0..rows) |i| {
        a.step = @intCast(i);
        e.barrier();
        e.setPipeline(p.z);
        e.setValue(a, 0);
        e.setBuffer(logits.buf, logits.off, 1);
        e.setBuffer(t.buf, 0, 2);
        e.setBuffer(t.buf, t.w2, 3);
        e.setBuffer(ids.buf, ids.off, 4);
        e.setBuffer(o.buf, Out.drafts_at, 5);
        e.setBuffer(o.buf, Out.ps_at, 6);
        e.setBuffer(o.buf, Out.pz_at, 7);
        e.setBuffer(o.buf, Out.pid_at, 8);
        e.setBuffer(o.buf, Out.asks_at, 9);
        e.dispatchGroups(mtl.Size.of(parts, y_groups, 1), mtl.Size.of(32 * simdgroups, 1, 1));
        e.barrier();
        e.setPipeline(p.pick);
        e.setValue(a, 0);
        e.setBuffer(o.buf, Out.ps_at, 1);
        e.setBuffer(o.buf, Out.pz_at, 2);
        e.setBuffer(o.buf, Out.pid_at, 3);
        e.setBuffer(t.buf, 0, 4);
        e.setBuffer(t.buf, t.cw, 5);
        e.setBuffer(o.buf, Out.hid_at, 6);
        e.setBuffer(ids.buf, ids.off, 7);
        e.setBuffer(o.buf, Out.drafts_at, 8);
        e.setBuffer(o.buf, Out.conf_at, 9);
        e.dispatchGroups(mtl.Size.of(asks, 1, 1), mtl.Size.of(32, 1, 1));
    }
}

/// Top-K GPU candidate extraction + single-TG resident Markov chain (all B steps in 1 dispatch + GPU cand/base).
pub fn encodeTopK(p: *const Pipelines, e: mtl.ComputeEncoder, t: *const Tables, o: *const Out, logits: Ref, hf: Ref, ids: Ref, vocab: u32, dim: u32, asks: u32, rows: u32, topk: u32) void {
    const total_rows = asks * rows;
    const a: Args = .{ .vocab = vocab, .rows = rows, .step = 0, .parts = parts, .dim = dim, .asks = asks, .topk = topk };
    e.setPipeline(p.hid);
    e.setValue(a, 0);
    e.setBuffer(hf.buf, hf.off, 1);
    e.setBuffer(t.buf, t.cw, 2);
    e.setBuffer(o.buf, Out.hid_at, 3);
    e.dispatchGroups(mtl.Size.of(total_rows, 1, 1), mtl.Size.of(1024, 1, 1));

    e.setPipeline(p.topk_part);
    e.setValue(a, 0);
    e.setBuffer(logits.buf, logits.off, 1);
    e.setBuffer(o.buf, Out.pval_at, 2);
    e.setBuffer(o.buf, Out.ptid_at, 3);
    e.dispatchGroups(mtl.Size.of(topk_slices, total_rows, 1), mtl.Size.of(256, 1, 1));

    e.barrier();
    e.setPipeline(p.topk_merge);
    e.setValue(a, 0);
    e.setBuffer(o.buf, Out.pval_at, 1);
    e.setBuffer(o.buf, Out.ptid_at, 2);
    e.setBuffer(o.buf, Out.cand_at, 3);
    e.setBuffer(o.buf, Out.base_at, 4);
    e.dispatchGroups(mtl.Size.of(1, total_rows, 1), mtl.Size.of(256, 1, 1));

    e.barrier();
    e.setPipeline(p.chain_topk);
    e.setValue(a, 0);
    e.setBuffer(o.buf, Out.cand_at, 1);
    e.setBuffer(o.buf, Out.base_at, 2);
    e.setBuffer(t.buf, 0, 3);
    e.setBuffer(t.buf, t.w2, 4);
    e.setBuffer(t.buf, t.cw, 5);
    e.setBuffer(o.buf, Out.hid_at, 6);
    e.setBuffer(ids.buf, ids.off, 7);
    e.setBuffer(o.buf, Out.drafts_at, 8);
    e.setBuffer(o.buf, Out.conf_at, 9);
    e.setBuffer(o.buf, Out.asks_at, 10);
    e.dispatchGroups(mtl.Size.of(asks, 1, 1), mtl.Size.of(256, 1, 1));
}
