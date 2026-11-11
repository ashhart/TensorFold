//! DeepSeek V4.1's slot kernels under the contract: rows of several slots in one launch equal each slot's own launch.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const lay = @import("metal_dsv41_segs_lay.zig");

const c = contract.spec;
const g = contract.gates;
const dm = tf.dsv41_metal;
const cp = dm.compress;
const fu = dm.fused;
const at = dm.attn;
const D = 512;

pub const Which = enum { kv, pool, rt, pack, attn, attn_pf };

const names = [_][]const u8{
    "dsv41 fused kv segs tf_dsv41_f_kv_segs",
    "dsv41 compress pool segs tf_dsv41_c_pool2_segs",
    "dsv41 compress fp4 round trip segs tf_dsv41_c_fp4rt_segs",
    "dsv41 compress fp4 pack segs tf_dsv41_c_fp4pack_segs",
    "dsv41 attn decode segs bf16 tf_dsv41_attn_dec_segs_bf16",
    "dsv41 attn decode segs prefetch bf16 tf_dsv41_attn_dec_segs_pf_bf16",
};
const replaces = [_][]const u8{
    "tf_dsv41_f_kv a slot",
    "tf_dsv41_c_pool2 and two tf_dsv41_g_scatter a slot",
    "tf_dsv41_c_fp4rt a slot",
    "tf_dsv41_c_fp4pack a slot",
    "tf_dsv41_attn_dec_bf16 or _pf_bf16 a slot, by its rows",
    "tf_dsv41_attn_dec_bf16 or _pf_bf16 a slot, by its rows",
};

fn none(_: c.Shape) usize {
    return 0;
}

/// rows: the launch's rows; k: its slots; n: attention heads (otherwise unused).
const served = [_]c.Shape{
    .{ .rows = 4, .n = 4, .k = 2 },  .{ .rows = 1, .n = 4, .k = 1 },   .{ .rows = 2, .n = 4, .k = 2 },
    .{ .rows = 4, .n = 4, .k = 4 },  .{ .rows = 16, .n = 4, .k = 1 },  .{ .rows = 16, .n = 4, .k = 2 },
    .{ .rows = 16, .n = 4, .k = 4 }, .{ .rows = 17, .n = 4, .k = 2 },  .{ .rows = 17, .n = 4, .k = 4 },
    .{ .rows = 5, .n = 64, .k = 3 }, .{ .rows = 16, .n = 64, .k = 4 }, .{ .rows = 2, .n = 4, .k = 1 },
};

pub fn spec(w: Which, name: []const u8) c.Spec {
    return .{ .name = name, .accum = .none, .depth = none, .out = .raw, .served = &served, .replaces = replaces[@backingInt(w)], .nan = .{ .stated = "every byte, NaN and infinity alike, is the slot's own launch's on the same inputs and state" }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "the per-slot kernel's body and its sums, unchanged" } };
}

const vt = g.hooks(Probe);

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    which: Which,
    ctl: u8,
    name: []const u8,
    /// the kernels under test (TF_CTL as registered) and the clean per-slot ones they replace
    mine: lay.Pipes,
    clean: lay.Pipes,
    lay: lay.Lay = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, w: Which, ctl: u8, name: []const u8) !Probe {
        var clean = try lay.Pipes.load(a, device, 0);
        errdefer clean.deinit();
        const mine = if (ctl == 0) clean else try lay.Pipes.load(a, device, ctl);
        return .{ .a = a, .arena = std.heap.ArenaAllocator.init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .which = w, .ctl = ctl, .name = name, .mine = mine, .clean = clean };
    }

    pub fn deinit(p: *Probe) void {
        p.arena.deinit();
        if (p.ctl != 0) p.mine.deinit();
        p.clean.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.which, p.name), .ctx = p, .vt = &vt, .control = if (p.ctl == 0) null else .shapes };
    }

    pub fn count(p: *Probe) usize {
        return p.lay.outBytes();
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k == 0 or shape.k > shape.rows or shape.k > lay.most or shape.rows > 2 * lay.rows_max) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        p.lay = try lay.Lay.make(p.arena.allocator(), &p.rig, @fromBackingInt(@intCast(@backingInt(p.which))), shape, fill == .nonfinite, seed);
        return true;
    }

    pub fn run(p: *Probe) !void {
        p.lay.restore(&p.rig);
        const pass = p.rig.begin();
        try p.launch(pass.e, &p.mine, p.lay.outs.items, true);
        try p.rig.finish(pass);
    }

    /// The slots' rows through one launch of the slot kernel, or (`merged` false) the per-slot kernel a slot.
    fn launch(p: *Probe, e: mtl.ComputeEncoder, pl: *const lay.Pipes, outs: []const lay.Out, merged: bool) !void {
        const L = &p.lay;
        const in = L.ins.items;
        const rows: u32 = @intCast(L.rows);
        if (merged) {
            const tab = try L.table(&p.rig, outs);
            for (outs) |o| e.useResource(o.ref.buf, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
            for (in) |r| e.useResource(r.buf, mtl.ResourceUsage.read);
            const t = cp.Ref{ .buf = tab.buf, .off = tab.off };
            const rs = cp.Ref{ .buf = in[lay.rowseg].buf, .off = in[lay.rowseg].off };
            switch (p.which) {
                .kv => try fu.kvSegs(&pl.fused, e, L.kvBind(outs, 0), .{ .buf = t.buf, .off = t.off }, .{ .buf = rs.buf, .off = rs.off }, rows, D, 1e-6),
                .pool => cp.pool2Segs(&pl.compress, e, L.whole(in[0]), L.whole(in[1]), L.whole(in[lay.pos]), L.whole(outs[0].ref), t, @intCast(L.slots), lay.keep, D),
                .rt => cp.fp4rtSegs(&pl.compress, e, L.whole(in[0]), L.whole(in[lay.dest]), t, rs, rows, D),
                .pack => cp.fp4packSegs(&pl.compress, e, L.whole(in[0]), t, rs, rows, lay.packArgs(L.whole(in[lay.dest]))),
                .attn, .attn_pf => try at.encodeDecodeSegs(&pl.attn, e, L.attnBind(outs, 0, 0), .{ .buf = t.buf, .off = t.off }, .{ .buf = rs.buf, .off = rs.off }, .{ .rows = rows, .heads = @intCast(L.heads) }, p.which == .attn_pf),
            }
            return;
        }
        for (0..L.slots) |j| {
            const r0 = L.base[j];
            const n: u32 = @intCast(L.size[j]);
            switch (p.which) {
                .kv => try fu.kv(&pl.fused, e, L.kvBind(outs, j), n, D, 1e-6),
                .pool => {
                    const st = L.state(outs, j);
                    cp.pool2(&pl.compress, e, L.plus(in[0], r0 * D * 4), L.plus(in[1], r0 * D * 4), L.whole(st[0]), L.plus(in[lay.pos], r0 * 4), L.plus(outs[0].ref, r0 * D * 2), n, D);
                    const gl = &pl.glue;
                    const iota = L.plus(in[lay.iota], r0 * 4);
                    dm.glue_tmp.scatter(gl, e, L.plus(in[0], r0 * D * 4), L.whole(st[1]), iota, n, D * 4, 1);
                    dm.glue_tmp.scatter(gl, e, L.plus(in[1], r0 * D * 4), L.plus(st[1], lay.keep * D * 4), iota, n, D * 4, 1);
                },
                .rt => cp.fp4rt(&pl.compress, e, L.plus(in[0], r0 * D * 2), L.whole(L.state(outs, j)[0]), L.plus(in[lay.dest], r0 * 4), n, D),
                .pack => {
                    const k = L.whole(L.state(outs, j)[0]);
                    cp.fp4pack(&pl.compress, e, L.plus(in[0], r0 * 128 * 2), k, k, n, lay.packArgs(L.plus(in[lay.dest], r0 * 4)));
                },
                .attn, .attn_pf => try at.encodeDecode(&pl.attn, e, L.attnBind(outs, j, r0), .{ .rows = n, .heads = @intCast(L.heads) }),
            }
        }
    }

    /// The clean per-slot kernels' bytes, a launch a slot, from the same inputs and starting state.
    pub fn replaced(p: *Probe) ![]const u8 {
        const twins = try p.lay.twins(&p.rig);
        const pass = p.rig.begin();
        try p.launch(pass.e, &p.clean, twins, false);
        try p.rig.finish(pass);
        return p.lay.gather(&p.rig, twins);
    }

    pub fn reference(_: *Probe, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Gate 1: the bytes equal the per-slot launches' with the planted values reaching some output.
    pub fn check(p: *Probe, gate: g.Gate) !?[]const u8 {
        const mine = try p.a.dupe(u8, try p.rig.outputBytes());
        defer p.a.free(mine);
        const want = try p.replaced();
        if (std.mem.indexOfDiff(u8, mine, want)) |i| return try std.fmt.bufPrint(&p.msg, "byte {d} differs from the per-slot launches'", .{i});
        if (gate == .nonfinite and (p.lay.plants == 0 or p.lay.nonfinite(&p.rig) == 0)) return try std.fmt.bufPrint(&p.msg, "{d} plants reach no output: no proof", .{p.lay.plants});
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Probe) = .empty,

    const E = struct { w: Which, ctl: u8, name: []const u8 };
    const list = [_]E{
        .{ .w = .kv, .ctl = 0, .name = names[0] },
        .{ .w = .kv, .ctl = 6, .name = "dsv41 fused kv segs control: every row into the first slot's ring" },
        .{ .w = .pool, .ctl = 0, .name = names[1] },
        .{ .w = .pool, .ctl = 6, .name = "dsv41 compress pool segs control: every slot pools over the first slot's carry" },
        .{ .w = .rt, .ctl = 0, .name = names[2] },
        .{ .w = .rt, .ctl = 6, .name = "dsv41 compress fp4 round trip segs control: every row into the first slot's rows" },
        .{ .w = .pack, .ctl = 0, .name = names[3] },
        .{ .w = .pack, .ctl = 6, .name = "dsv41 compress fp4 pack segs control: every row into the first slot's keys" },
        .{ .w = .attn, .ctl = 0, .name = names[4] },
        .{ .w = .attn, .ctl = 6, .name = "dsv41 attn decode segs control: every row reads the first slot's keys" },
        .{ .w = .attn_pf, .ctl = 0, .name = names[5] },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Probe);
            p.* = Probe.init(a, device, queue, entry.w, entry.ctl, entry.name) catch |err| {
                a.destroy(p);
                return err;
            };
            try s.probes.append(a, p);
            try kernels.append(a, p.kernel());
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }
};
