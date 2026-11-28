//! DSpark's Markov chain on the GPU over the whole vocabulary: a pass's drafts and confidences behind its head.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

/// Whether the kernels take a Markov head's rank: 32, 64, 128 or 256 (1 to 8 values a lane of a simdgroup).
pub fn takes(rank: u32) bool {
    return rank == 32 or rank == 64 or rank == 128 or rank == 256;
}
/// Threadgroups an ask's search of the vocabulary is cut into.
pub const parts = 256;
const simdgroups = 8;
/// Rows of a launch (the drafter's 16).
const R = 16;

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
const Args = extern struct { vocab: u32, rows: u32, step: u32, parts: u32, dim: u32 };

pub const Pipelines = struct {
    z: mtl.Pipeline,
    pick: mtl.Pipeline,
    hid: mtl.Pipeline,

    /// The kernels of a head of `rank` (`takes`).
    pub fn load(device: mtl.Device, rank: u32) !Pipelines {
        if (!takes(rank)) return error.MarkovRank;
        const lib = try mtl.Library.fromSource(device, sources.dsv41.keyed ++ sources.dsv41.markov, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var zb: [32]u8 = undefined;
        var pb: [32]u8 = undefined;
        const z = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&zb, "tf_dsv41_markov_z_{d}", .{rank / 32}), false);
        errdefer z.deinit();
        const pick = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&pb, "tf_dsv41_markov_pick_{d}", .{rank / 32}), false);
        errdefer pick.deinit();
        return .{ .z = z, .pick = pick, .hid = try mtl.Pipeline.init(device, lib, "tf_dsv41_markov_hid", false) };
    }

    pub fn deinit(p: *Pipelines) void {
        p.z.deinit();
        p.pick.deinit();
        p.hid.deinit();
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

/// An ask's draws as the kernels read them: greedy, or keyed at positions pos0 + i (markov.metal's TfMarkovAsk).
pub const Ask = extern struct { seed: u64 = 0, pos0: u64 = 0, t: f32 = 1, sampled: u32 = 0 };

/// The chain's asks, results and scratch: drafts [asks][rows] u32, confidences and hidden parts f32, the parts' best.
pub const Out = struct {
    buf: mtl.Buffer,

    const drafts_at = 0;
    const conf_at = 4 * R;
    const hid_at = 8 * R;
    pub const asks_at = 12 * R;
    const ps_at = asks_at + @sizeOf(Ask) * R;
    const pz_at = ps_at + 4 * R * parts;
    const pid_at = pz_at + 4 * R * parts;
    const bytes = pid_at + 4 * R * parts;

    pub fn init(device: mtl.Device) !Out {
        const buf = try device.buffer(bytes, mtl.ResourceOptions.shared);
        @memset(buf.contents()[0..bytes], 0);
        return .{ .buf = buf };
    }

    /// Ask j's draws.
    pub fn asks(o: *const Out) []Ask {
        return @alignCast(std.mem.bytesAsSlice(Ask, o.buf.contents()[asks_at..][0 .. @sizeOf(Ask) * R]));
    }

    pub fn deinit(o: *Out) void {
        o.buf.deinit();
    }

    /// The drafts and confidences of the launch's first `n` rows (ask j's at j * rows).
    pub fn drafts(o: *const Out, n: usize) []const u32 {
        return o.buf.slice(u32, R)[0..n];
    }

    pub fn conf(o: *const Out, n: usize) []const f32 {
        return @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, o.buf.contents()[conf_at..][0 .. 4 * R])))[0..n];
    }
};

/// The chain of `asks` asks of `rows` drafts (each ask's anchor its first row of `ids`), barriers between the steps.
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, t: *const Tables, o: *const Out, logits: Ref, hf: Ref, ids: Ref, vocab: u32, dim: u32, asks: u32, rows: u32) void {
    var a: Args = .{ .vocab = vocab, .rows = rows, .step = 0, .parts = parts, .dim = dim };
    e.setPipeline(p.hid);
    e.setValue(a, 0);
    e.setBuffer(hf.buf, hf.off, 1);
    e.setBuffer(t.buf, t.cw, 2);
    e.setBuffer(o.buf, Out.hid_at, 3);
    e.dispatchGroups(mtl.Size.of(asks * rows, 1, 1), mtl.Size.of(1024, 1, 1));
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
        e.dispatchGroups(mtl.Size.of(parts, asks, 1), mtl.Size.of(32 * simdgroups, 1, 1));
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
