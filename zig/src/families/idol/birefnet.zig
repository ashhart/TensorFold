//! BiRefNet on Metal in fp32 (Swin-L twice, ASPP-deformable decoder): the normalized picture to its mask logits.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("birefnet_kernels.zig");
const wts = @import("birefnet_weights.zig");
const dec = @import("birefnet_decoder.zig");
const pre = @import("preprocess.zig");
const Ref = ks.Ref;

const cfg = @import("birefnet_config.zig");

/// The network's input side, its two passes' patch grids, a deformable branch's pixels at once (birefnet_config.zig).
pub const size = cfg.size;
pub const grids = cfg.grids;
pub const chunk = cfg.chunk;

pub const Timing = struct { wall: f64, gpu: f64 };

pub const padded = cfg.padded;

/// Each scratch tensor's floats: the picture and encoder outputs persist; encoder and decoder temporaries share one.
const P = struct {
    const x = size * size * 3;
    const x1 = 256 * 256 * wts.channels[3];
    const x2 = 128 * 128 * wts.channels[2];
    const x3 = 64 * 64 * wts.channels[1];
    const x4 = 32 * 32 * wts.squeeze_in;
    const logits = size * size;
    const pooled = 64 + wts.aspp_planes;
    const xh = (size / 2) * (size / 2) * 3;
    const tokens = 256 * 256 * wts.embed;
    const slots = 264 * 264 * wts.embed;
    const big = 256 * 256 * 4 * wts.embed;
    const lin = [4]usize{ 32 * 32 * dec.level_in[0], 64 * 64 * dec.level_in[1], 128 * 128 * dec.level_in[2], 256 * 256 * dec.level_in[3] };
    const out = [4]usize{ 32 * 32 * wts.channels[1], 64 * 64 * wts.channels[2], 128 * 128 * wts.channels[3], 256 * 256 * wts.channels[3] / 2 };
    const a64 = 256 * 256 * wts.inter;
    const offs = 256 * 256 * 192;
    const cols = chunk * 49 * wts.inter;
    const cat = 256 * 256 * 5 * wts.aspp_planes;
    const gdt = 256 * 256 * wts.inter;
    const patches = 256 * 256 * 64;
    const q48 = size * size * 48;
};

fn bytes(n: usize) usize {
    return std.mem.alignForward(usize, n * 4, 256);
}

pub const Net = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    k: ks.Kernels,
    w: wts.Weights,
    scratch: mtl.Buffer,
    maps: mtl.Buffer,
    /// The windows' token maps: [pass][layer][shifted], slot to token or -1.
    map: [2][4][2]Ref,
    x: Ref,
    xs: [4]Ref,
    logits: Ref,
    pooled: Ref,
    xh: Ref,
    ta: Ref,
    tb: Ref,
    slots: Ref,
    big: Ref,
    att: Ref,
    lin: [4]Ref,
    out: [4]Ref,
    a64: Ref,
    offs: Ref,
    cols: Ref,
    cat: Ref,
    gdt: Ref,
    patches: Ref,
    q48: Ref,

    /// `path` (birefnet.safetensors from a speridlabs/idol snapshot).
    pub fn load(gpa: std.mem.Allocator, device: mtl.Device, path: [:0]const u8) !*Net {
        const n = try gpa.create(Net);
        errdefer gpa.destroy(n);
        n.* = undefined;
        n.gpa = gpa;
        n.device = device;
        n.queue = try device.queue();
        errdefer n.queue.deinit();
        n.k = try ks.Kernels.load(device);
        errdefer n.k.deinit();
        n.w = try wts.load(gpa, device, path);
        errdefer n.w.deinit();
        try n.buildMaps();
        errdefer n.maps.deinit();
        const persistent = [_]usize{ P.x, P.x1, P.x2, P.x3, P.x4, P.logits, P.pooled };
        const enc = [_]usize{ P.xh, P.tokens, P.tokens, P.slots, P.big, P.slots };
        const decs = P.lin ++ P.out ++ [_]usize{ P.a64, P.offs, P.cols, P.cat, P.gdt, P.patches };
        var at: usize = 0;
        for (persistent) |v| at += bytes(v);
        var e_end = at;
        for (enc) |v| e_end += bytes(v);
        var d_end = at;
        for (decs) |v| d_end += bytes(v);
        n.scratch = try device.buffer(@max(e_end, d_end), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        const views = [_]*Ref{ &n.x, &n.xs[0], &n.xs[1], &n.xs[2], &n.xs[3], &n.logits, &n.pooled };
        at = 0;
        for (views, persistent) |v, len| v.* = n.take(&at, len);
        const shared = at;
        for ([_]*Ref{ &n.xh, &n.ta, &n.tb, &n.slots, &n.big, &n.att }, enc) |v, len| v.* = n.take(&at, len);
        at = shared;
        n.q48 = .{ .buf = n.scratch, .off = at };
        std.debug.assert(bytes(P.lin[0]) + bytes(P.lin[1]) + bytes(P.lin[2]) + bytes(P.lin[3]) >= P.q48 * 4);
        for (&n.lin, P.lin) |*v, len| v.* = n.take(&at, len);
        for (&n.out, P.out) |*v, len| v.* = n.take(&at, len);
        for ([_]*Ref{ &n.a64, &n.offs, &n.cols, &n.cat, &n.gdt, &n.patches }, decs[8..]) |v, len| v.* = n.take(&at, len);
        return n;
    }

    fn take(n: *Net, at: *usize, len: usize) Ref {
        const r: Ref = .{ .buf = n.scratch, .off = at.* };
        at.* += bytes(len);
        return r;
    }

    pub fn deinit(n: *Net) void {
        n.scratch.deinit();
        n.maps.deinit();
        n.w.deinit();
        n.k.deinit();
        n.queue.deinit();
        n.gpa.destroy(n);
    }

    /// Each layer's window slots to tokens, both passes, plain and shifted by half a window (torch.roll by -6).
    fn buildMaps(n: *Net) !void {
        var total: usize = 0;
        for (grids) |g0| for (0..4) |i| {
            const hp = padded(g0 >> @intCast(i));
            total += 2 * std.mem.alignForward(usize, hp * hp, 64);
        };
        n.maps = try n.device.buffer(total * 4, mtl.ResourceOptions.shared);
        const all = @as([*]i32, @ptrCast(@alignCast(n.maps.contents())))[0..total];
        var at: usize = 0;
        for (grids, 0..) |g0, pass| for (0..4) |i| for (0..2) |s| {
            const h = g0 >> @intCast(i);
            const hp = padded(h);
            n.map[pass][i][s] = .{ .buf = n.maps, .off = at * 4 };
            fillMap(all[at..][0 .. hp * hp], h, if (s == 1) wts.window / 2 else 0);
            at += std.mem.alignForward(usize, hp * hp, 64);
        };
    }

    pub fn floats(r: Ref, len: usize) []f32 {
        return @as([*]f32, @ptrCast(@alignCast(r.addr())))[0..len];
    }

    /// The GPU work so far, waited for; its seconds on the GPU.
    pub fn finish(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !f64 {
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("idol birefnet: a pass failed on the GPU: {s}", .{msg});
            return error.GpuFailure;
        }
        return cb.gpuSeconds();
    }

    /// Mask logits [1024, 1024] of the normalized picture `input` [1024 * 1024, 3] (remove_background's tensor).
    pub fn forward(n: *Net, input: []const f32, logits: []f32) !Timing {
        if (input.len != P.x or logits.len != P.logits) return error.PictureSize;
        @memcpy(floats(n.x, P.x), input);
        const started = mtl.clock.seconds();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = n.queue.commandBuffer();
        const e = cb.compute(.serial);
        n.encode(e);
        dec.squeeze(n, e);
        dec.decode(n, e);
        const gpu = try finish(cb, e);
        @memcpy(logits, floats(n.logits, P.logits));
        return .{ .wall = mtl.clock.seconds() - started, .gpu = gpu };
    }

    /// remove_background's alpha [h, w] of the resized photo `im` (its RGB): the input, the logits, then the alpha.
    pub fn removeBackground(n: *Net, a: std.mem.Allocator, im: pre.Image, logits: []f32, alpha: []u8) !Timing {
        const x = try a.alloc(f32, P.x);
        defer a.free(x);
        try pre.input(a, im, x);
        const t = try n.forward(x, logits);
        try pre.alpha(a, logits, im.width, im.height, alpha);
        return t;
    }

    /// _forward_enc: both backbone passes, concatenated into x1, x2, x3 and the squeeze's input [32^2, 5,760].
    pub fn encode(n: *Net, e: mtl.ComputeEncoder) void {
        ks.resize(e, n.k.get("br_resize"), n.x, n.xh, .{ .h = size, .w = size }, .{ .h = size / 2, .w = size / 2 }, 3, 3, 0, 3, 0);
        for (0..2) |pass| {
            const side = @as(usize, size) >> @intCast(pass);
            const g0 = grids[pass];
            ks.tiles(e, n.k.get("br_tiles"), if (pass == 0) n.x else n.xh, n.slots, .{ .h = g0, .w = g0 }, side, 3, 4, 4, false, 64);
            n.conv(e, n.w.patch, n.slots, n.tb, .{ .rows = g0 * g0, .ldy = wts.embed });
            n.norm(e, n.tb, null, n.w.patch_norm, n.ta, g0 * g0, wts.embed, .rows, 0, wts.embed, 0);
            for (0..4) |i| {
                const h = g0 >> @intCast(i);
                const C = wts.widths[i];
                n.layer(e, pass, i, n.ta, n.tb);
                // the layer's features: norm_i, at the full grid's channels or resized to it beside them
                const full = grids[0] >> @intCast(i);
                const dst, const ld, const off = n.featureSlot(i);
                if (pass == 0) {
                    n.norm(e, n.ta, null, n.w.norms[i], dst, h * h, C, .rows, 0, ld, off);
                } else {
                    n.norm(e, n.ta, null, n.w.norms[i], n.att, h * h, C, .rows, 0, C, 0);
                    ks.resize(e, n.k.get("br_resize"), n.att, dst, .{ .h = h, .w = h }, .{ .h = full, .w = full }, C, C, 0, ld, off + C);
                }
                if (i < 3) std.mem.swap(Ref, &n.ta, &n.tb);
            }
        }
        // the squeeze's input: x1, x2 and x3 resized to the coarsest grid, then x4
        var col: usize = 0;
        for (0..3) |i| {
            const g = grids[0] >> @intCast(i);
            const C = 2 * wts.widths[i];
            ks.resize(e, n.k.get("br_resize"), n.xs[i], n.xs[3], .{ .h = g, .w = g }, .{ .h = 32, .w = 32 }, C, C, 0, wts.squeeze_in, col);
            col += C;
        }
    }

    /// Where layer i's features go (buffer, row length, column): x1 to x3 [grid^2, 2 C], x4 in the squeeze's input.
    fn featureSlot(n: *const Net, i: usize) struct { Ref, usize, usize } {
        if (i < 3) return .{ n.xs[i], 2 * wts.widths[i], 0 };
        return .{ n.xs[3], wts.squeeze_in, wts.squeeze_in - wts.channels[0] };
    }

    /// Swin layer i of `pass` on t [h * h, C] in place, then (i < 3) PatchMerging into down [h^2 / 4, 2 C].
    pub fn layer(n: *Net, e: mtl.ComputeEncoder, pass: usize, i: usize, t: Ref, down: Ref) void {
        const h = grids[pass] >> @intCast(i);
        const hp = padded(h);
        const C = wts.widths[i];
        const L = &n.w.layers[i];
        for (L.blocks[0..wts.depths[i]], 0..) |*b, j| {
            const shift: usize = if (j % 2 == 1) wts.window / 2 else 0;
            const map = n.map[pass][i][j % 2];
            const slots = hp * hp;
            n.norm(e, t, map, b.norm1, n.slots, slots, C, .map, 0, C, 0);
            n.conv(e, b.qkv, n.slots, n.big, .{ .rows = slots, .ldy = 3 * C });
            ks.window(e, n.k.get("br_window"), n.big, b.table, n.att, C, wts.heads[i], hp, hp, shift);
            n.convWith(e, b.proj, n.att, t, t, map, .{ .rows = slots, .ldy = C, .flags = ks.flag_add | ks.flag_map });
            n.norm(e, t, null, b.norm2, n.slots, h * h, C, .rows, 0, C, 0);
            n.conv(e, b.fc1, n.slots, n.big, .{ .rows = h * h, .ldy = 4 * C, .flags = ks.flag_gelu });
            n.convWith(e, b.fc2, n.big, t, t, null, .{ .rows = h * h, .ldy = C, .flags = ks.flag_add });
        }
        if (L.merge) |m| {
            n.norm(e, t, null, m, n.slots, h * h / 4, 4 * C, .merge, h, 4 * C, 0);
            n.conv(e, L.reduction.?, n.slots, down, .{ .rows = h * h / 4, .ldy = 2 * C });
        }
    }

    /// The shape fields `conv` fills from c.
    pub const Out = struct { rows: usize, cols: usize = 1, ldy: usize, yoff: usize = 0, flags: u32 = 0, row0: usize = 0, ldx: ?usize = null };

    /// y = c(x) with its batch norm when it has one (flags add bn), rows x cols at c's side.
    pub fn conv(n: *const Net, e: mtl.ComputeEncoder, c: wts.Conv, x: Ref, y: Ref, o: Out) void {
        n.convWith(e, c, x, null, y, null, o);
    }

    pub fn convWith(n: *const Net, e: mtl.ComputeEncoder, c: wts.Conv, x: Ref, r: ?Ref, y: Ref, map: ?Ref, o: Out) void {
        const flags = o.flags | (if (c.post != null) ks.flag_bn else 0);
        ks.conv(e, n.k.get("br_conv"), x, c.w, c.b, c.post, r, map, y, .{ .rows = o.rows, .cols = o.cols, .cin = c.cin, .ldx = o.ldx orelse c.cin, .cout = c.cout, .ldw = c.ldw, .k = c.k, .ldy = o.ldy, .yoff = o.yoff, .flags = flags, .row0 = o.row0 });
    }

    pub fn norm(n: *const Net, e: mtl.ComputeEncoder, x: Ref, map: ?Ref, nm: wts.Norm, y: Ref, rows: usize, width: usize, mode: ks.NormMode, src_w: usize, ldy: usize, yoff: usize) void {
        ks.norm(e, n.k.get("br_norm"), x, map, nm.g, nm.b, y, rows, width, mode, src_w, ldy, yoff, wts.ln_eps);
    }
};

pub const fillMap = cfg.fillMap;
