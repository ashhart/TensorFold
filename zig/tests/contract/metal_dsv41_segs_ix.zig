//! DeepSeek V4.1's indexer slot kernels under the contract: several slots' scores and candidates in one launch.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const ix = tf.dsv41_metal.index;
const cand = tf.dsv41_metal.cand;

pub const Which = enum { scores, cand };
const names = [_][]const u8{ "dsv41 ix scores segs tf_dsv41_ix_scores_dec_segs", "dsv41 cand segs tf_cand_segs" };
const replaces = [_][]const u8{ "tf_dsv41_ix_scores_dec a slot (its keys, its start)", "tf_cand a slot (its keys; a slot with none skipped)" };

/// Slots a launch holds at most, each slot's first position, and the candidate blocks a row keeps.
const most = 4;
const firsts = [most]u32{ 700, 0, 1290, 33 };
const keep = 20;
const qrow = 2048;
const qsrow = 128;

fn none(_: c.Shape) usize {
    return 0;
}

/// rows: the launch's rows; k: its slots; n: 1 ratio 1, 2 ratio 2, 3 ratio 1 with a candidate mask.
const served = [_]c.Shape{
    .{ .rows = 4, .n = 1, .k = 2 },  .{ .rows = 1, .n = 1, .k = 1 },  .{ .rows = 2, .n = 2, .k = 2 },
    .{ .rows = 4, .n = 3, .k = 4 },  .{ .rows = 16, .n = 1, .k = 1 }, .{ .rows = 16, .n = 2, .k = 2 },
    .{ .rows = 16, .n = 3, .k = 4 }, .{ .rows = 17, .n = 1, .k = 2 }, .{ .rows = 17, .n = 2, .k = 4 },
    .{ .rows = 5, .n = 3, .k = 3 },  .{ .rows = 4, .n = 2, .k = 4 },
};

pub fn spec(w: Which, name: []const u8) c.Spec {
    return .{ .name = name, .accum = .none, .depth = none, .out = .raw, .served = &served, .replaces = replaces[@backingInt(w)], .nan = .{ .stated = "every byte, NaN and infinity alike, is the slot's own launch's on the same inputs" }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "the per-slot kernel's body and its sums, unchanged" } };
}

const Seg = tf.dsv41_metal.segs.Seg;

fn at(r: rig_.Ref, bytes: usize) ix.Ref {
    return .{ .buf = r.buf, .off = r.off + bytes };
}

fn ca(r: rig_.Ref, bytes: usize) cand.Ref {
    return .{ .buf = r.buf, .off = r.off + bytes };
}

const vt = g.hooks(Probe);

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    which: Which,
    ctl: u8,
    name: []const u8,
    ix_mine: ix.Pipelines,
    ix_clean: ix.Pipelines,
    cand_mine: mtl.Pipeline,
    cand_clean: mtl.Pipeline,
    cand_segs: mtl.Pipeline,
    rows: usize = 0,
    slots: usize = 0,
    ratio: u32 = 1,
    masked: bool = false,
    stride: u32 = 0,
    blocks: u32 = 0,
    size: [most]usize = undefined,
    base: [most]usize = undefined,
    keys: [most]u32 = undefined,
    plants: usize = 0,
    ins: [8]rig_.Ref = undefined, // q, qs, w, cm, vis, rowseg, scores (cand), then the slots' keys from `kslot`
    kslot: [most]rig_.Ref = undefined,
    out: [2]rig_.Ref = undefined, // scores; or keyed, mask
    olen: [2]usize = .{ 0, 0 },
    init_: [2][]u8 = undefined,
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, w: Which, ctl: u8, name: []const u8) !Probe {
        var p: Probe = .{ .a = a, .arena = std.heap.ArenaAllocator.init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .which = w, .ctl = ctl, .name = name, .ix_mine = undefined, .ix_clean = undefined, .cand_mine = undefined, .cand_clean = undefined, .cand_segs = undefined };
        p.ix_clean = try ix.Pipelines.load(device);
        errdefer p.ix_clean.deinit();
        const it = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, sources.dsv41.index });
        defer a.free(it);
        p.ix_mine = try ix.Pipelines.loadSource(device, it);
        errdefer p.ix_mine.deinit();
        p.cand_clean = try cand.load(device);
        errdefer p.cand_clean.deinit();
        const ct = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, sources.dsv41.cand });
        defer a.free(ct);
        p.cand_segs = try rig_.pipeline(device, ct, "tf_cand_segs");
        p.cand_mine = p.cand_segs;
        return p;
    }

    pub fn deinit(p: *Probe) void {
        p.arena.deinit();
        p.ix_mine.deinit();
        p.ix_clean.deinit();
        p.cand_clean.deinit();
        p.cand_segs.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.which, p.name), .ctx = p, .vt = &vt, .control = if (p.ctl == 0) null else .shapes };
    }

    pub fn count(p: *Probe) usize {
        return p.olen[0] + p.olen[1];
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k == 0 or shape.k > shape.rows or shape.k > most or shape.n == 0 or shape.n > 3) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const a = p.arena.allocator();
        p.rows = shape.rows;
        p.slots = shape.k;
        p.ratio = if (shape.n == 2) 2 else 1;
        p.masked = shape.n == 3;
        p.plants = 0;
        var b: usize = 0;
        var top: u32 = 1;
        for (0..p.slots) |j| {
            p.size[j] = p.rows / p.slots + @intFromBool(j < p.rows % p.slots);
            if (j == 1 and p.size[1] > 1) {
                p.size[1] -= 1;
                p.size[0] += 1;
            }
        }
        for (0..p.slots) |j| {
            p.base[j] = b;
            b += p.size[j];
            p.keys[j] = (firsts[j] + @as(u32, @intCast(p.size[j]))) / p.ratio;
            top = @max(top, p.keys[j]);
        }
        p.stride = top + 13;
        p.blocks = std.mem.alignForward(u32, (p.stride + 7) / 8, 4);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = p.rows;
        const nf = fill == .nonfinite;
        const q = try a.alloc(u8, n * qrow);
        rng.bytes(q);
        const qs = try a.alloc(u8, n * qsrow);
        for (qs) |*v| v.* = 120 + rng.uintLessThan(u8, 10);
        const w = try a.alloc(u16, n * 32);
        for (w) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        const cm = try a.alloc(u8, n * p.blocks);
        for (cm) |*v| v.* = @intFromBool(rng.uintLessThan(u8, 4) != 0);
        const vis = try a.alloc(i32, n);
        const rs = try a.alloc(i32, n);
        for (0..p.slots) |j| for (0..p.size[j]) |i| {
            vis[p.base[j] + i] = @intCast((firsts[j] + i + 1) / p.ratio);
            rs[p.base[j] + i] = @intCast(j);
        };
        // scores of the candidate kernel: few distinct values, so blocks tie; -inf past each slot's visible keys
        const sc = try a.alloc(f32, n * p.stride);
        for (sc, 0..) |*v, i| v.* = if (i % p.stride < @as(usize, @intCast(vis[i / p.stride]))) @as(f32, @floatFromInt(rng.uintLessThan(u32, 9))) * 0.25 - 1 else -std.math.inf(f32);
        if (nf and n > 1) {
            qs[1 * qsrow + 5] = 255; // an E8M0 NaN scale in row 1's head 1
            sc[0] = std.math.nan(f32);
            sc[1 * p.stride + 2] = std.math.inf(f32);
            p.plants = 3;
        }
        const raws = [_][]const u8{ q, qs, std.mem.sliceAsBytes(w), cm, std.mem.sliceAsBytes(vis), std.mem.sliceAsBytes(rs), std.mem.sliceAsBytes(sc) };
        for (raws, 0..) |r, i| p.ins[i] = try p.rig.input(r);
        for (0..p.slots) |j| {
            // as many keys as the widest slot, so a slot kernel reading another slot's keys stays in its buffer
            const k = try a.alloc(u8, @as(usize, top) * 68);
            rng.bytes(k);
            for (0..top) |r| for (64..68) |s| {
                k[r * 68 + s] = 122 + rng.uintLessThan(u8, 6);
            };
            if (nf and j == 0 and top > 3) k[3 * 68 + 65] = 255;
            p.kslot[j] = try p.rig.input(k);
        }
        p.olen = if (p.which == .scores) .{ n * p.stride * 4, 0 } else .{ n * p.blocks * 4, n * p.blocks };
        for (0..2) |i| {
            if (p.olen[i] == 0) continue;
            p.out[i] = try p.rig.output(p.olen[i]);
            p.init_[i] = try a.alloc(u8, p.olen[i]);
            @memset(p.init_[i], if (i == 1) 0x00 else 0xff);
        }
        return true;
    }

    fn table(p: *Probe) !rig_.Ref {
        const t = try p.rig.scratch(most * @sizeOf(Seg));
        @memset(p.rig.bytes(t, most * @sizeOf(Seg)), 0);
        const segs = std.mem.bytesAsSlice(Seg, p.rig.bytes(t, most * @sizeOf(Seg)));
        for (0..p.slots) |j| segs[j] = .{ .ikeys = p.kslot[j].buf.gpuAddress() + p.kslot[j].off, .base = @intCast(p.base[j]), .rows = @intCast(p.size[j]), .nkeys = p.keys[j], .start = firsts[j] + 1 };
        return t;
    }

    fn restore(p: *Probe, outs: []const rig_.Ref) void {
        for (0..2) |i| if (p.olen[i] > 0) @memcpy(p.rig.bytes(outs[i], p.olen[i]), p.init_[i]);
    }

    pub fn run(p: *Probe) !void {
        p.restore(&p.out);
        const t = try p.table();
        const pass = p.rig.begin();
        const e = pass.e;
        for (p.kslot[0..p.slots]) |k| e.useResource(k.buf, mtl.ResourceUsage.read);
        const rows: u32 = @intCast(p.rows);
        const rs = at(p.ins[5], 0);
        var top: u32 = 0;
        for (p.keys[0..p.slots]) |k| top = @max(top, k);
        switch (p.which) {
            .scores => if (top > 0) try ix.encodeDecodeSegs(&p.ix_mine, e, at(p.ins[0], 0), at(p.ins[1], 0), at(p.ins[2], 0), at(p.out[0], 0), at(p.ins[3], 0), at(t, 0), rs, p.stride, p.call(rows, top, 1)),
            .cand => cand.encodeSegs(p.cand_mine, e, ca(p.ins[6], 0), ca(p.ins[4], 0), ca(p.out[0], 0), ca(p.out[1], 0), ca(t, 0), ca(p.ins[5], 0), rows, p.stride, p.blocks, keep),
        }
        try p.rig.finish(pass);
    }

    fn call(p: *const Probe, rows: u32, keys: u32, start: u32) ix.Call {
        return .{ .rows = rows, .keys = keys, .start = start, .ratio = p.ratio, .mask_stride = if (p.masked) p.blocks else 0 };
    }

    /// The clean per-slot kernels a slot at a time through their launchers, each at its own keys and row stride.
    pub fn replaced(p: *Probe) ![]const u8 {
        return p.perSlot(&p.ix_clean, p.cand_clean);
    }

    fn perSlot(p: *Probe, ixp: *const ix.Pipelines, cp: mtl.Pipeline) ![]const u8 {
        const a = p.arena.allocator();
        var twin: [2]rig_.Ref = undefined;
        for (0..2) |i| if (p.olen[i] > 0) {
            twin[i] = try p.rig.scratch(p.olen[i]);
        };
        p.restore(&twin);
        var dense: [most]rig_.Ref = undefined;
        const pass = p.rig.begin();
        const e = pass.e;
        for (0..p.slots) |j| {
            const r0 = p.base[j];
            const n = p.size[j];
            const k = p.keys[j];
            if (k == 0) continue;
            dense[j] = try p.rig.scratch(n * k * 4);
            if (p.which == .scores) {
                const cm = if (p.masked) at(p.ins[3], r0 * p.blocks) else at(dense[j], 0);
                try ix.encodeDecode(ixp, e, at(p.ins[0], r0 * qrow), at(p.ins[1], r0 * qsrow), at(p.ins[2], r0 * 64), at(p.kslot[j], 0), at(dense[j], 0), cm, p.call(@intCast(n), k, firsts[j] + 1));
            } else {
                for (0..n) |i| @memcpy(p.rig.bytes(dense[j], n * k * 4)[i * k * 4 ..][0 .. k * 4], p.rig.bytes(p.ins[6], p.rows * p.stride * 4)[(r0 + i) * p.stride * 4 ..][0 .. k * 4]);
                cand.encode(cp, e, ca(dense[j], 0), ca(p.ins[4], r0 * 4), ca(twin[0], r0 * p.blocks * 4), ca(twin[1], r0 * p.blocks), @intCast(n), k, p.blocks, keep);
            }
        }
        try p.rig.finish(pass);
        if (p.which == .scores) for (0..p.slots) |j| {
            const k = p.keys[j];
            if (k == 0) continue;
            for (0..p.size[j]) |i| @memcpy(p.rig.bytes(twin[0], p.olen[0])[(p.base[j] + i) * p.stride * 4 ..][0 .. k * 4], p.rig.bytes(dense[j], p.size[j] * k * 4)[i * k * 4 ..][0 .. k * 4]);
        };
        var b: std.ArrayList(u8) = .empty;
        for (0..2) |i| if (p.olen[i] > 0) try b.appendSlice(a, p.rig.bytes(twin[i], p.olen[i]));
        return b.items;
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
        if (gate != .nonfinite) return null;
        var bad: usize = 0;
        if (p.which == .scores) {
            for (std.mem.bytesAsSlice(u32, p.rig.bytes(p.out[0], p.olen[0]))) |v| bad += @intFromBool(v != 0xffffffff and (v & 0x7fffffff) > 0x7f800000);
        } else bad = p.plants;
        if (p.plants == 0 or bad == 0) return try std.fmt.bufPrint(&p.msg, "{d} plants reach no output: no proof", .{p.plants});
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    probes: std.ArrayList(*Probe) = .empty,

    const E = struct { w: Which, ctl: u8, name: []const u8 };
    const list = [_]E{
        .{ .w = .scores, .ctl = 0, .name = names[0] },
        .{ .w = .scores, .ctl = 6, .name = "dsv41 ix scores segs control: every row scores the first slot's keys" },
        .{ .w = .cand, .ctl = 0, .name = names[1] },
        .{ .w = .cand, .ctl = 6, .name = "dsv41 cand segs control: every row takes the first slot's key count" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a, .device = device };
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

    /// At one stream the scorer and candidate kernels equal the base's kept bodies byte for byte.
    pub fn base(s: *Suite, seed: u64) !usize {
        var any = false;
        for (s.probes.items) |p| any = any or p.ctl == 0;
        if (!any) return 0;
        var old = try ix.Pipelines.load(s.device);
        defer old.deinit();
        const lay = @import("metal_dsv41_segs_lay.zig");
        try lay.swap(s.a, s.device, &old.dec, sources.dsv41.index, @embedFile("dsv41_base/index.metal"), "tf_base_dsv41_ix_scores_dec");
        var oc = try cand.load(s.device);
        defer oc.deinit();
        try lay.swap(s.a, s.device, &oc, sources.dsv41.cand, @embedFile("dsv41_base/cand.metal"), "tf_base_cand");
        var cases: usize = 0;
        var differ: usize = 0;
        for (s.probes.items) |p| {
            if (p.ctl != 0) continue;
            for (served, 0..) |shape, i| for ([_]g.Fill{ .random, .nonfinite }) |fill| {
                if (!try p.prepare(shape, fill, seed + i)) continue;
                const now = try s.a.dupe(u8, try p.perSlot(&p.ix_clean, p.cand_clean));
                defer s.a.free(now);
                cases += 1;
                if (std.mem.indexOfDiff(u8, now, try p.perSlot(&old, oc))) |at_byte| {
                    differ += 1;
                    std.debug.print("FAIL base equality {s} {d}x{d}x{d}: first difference at byte {d}\n", .{ p.name, shape.rows, shape.n, shape.k, at_byte });
                }
            };
        }
        std.debug.print("{s} base equality dsv41 indexer per-slot forms: {d} cases of the decode scores and candidate blocks byte-equal to the base's kept bodies, {d} differ\n", .{ if (differ == 0) "PASS" else "FAIL", cases, differ });
        return differ;
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
