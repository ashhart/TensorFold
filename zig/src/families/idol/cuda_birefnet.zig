//! BiRefNet on CUDA in fp32: the normalized picture to its mask logits, as birefnet.zig does on Metal.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("birefnet_config.zig");
const ks = @import("cuda_birefnet_kernels.zig");
const wts = @import("cuda_birefnet_weights.zig");
const dec = @import("cuda_birefnet_decoder.zig");
const pre = @import("preprocess.zig");

const size = cfg.size;
const grids = cfg.grids;

pub const Timing = struct { wall: f64, gpu: f64 };

pub fn seconds() f64 {
    const c = @import("sys").c;
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

/// The GPUs this family admits: GB10 (sm_121) and the RTX 50 / PRO Blackwell cards (sm_120), others refused by name.
pub fn admitted(sm: u32) bool {
    return sm == 120 or sm == 121;
}

/// Each scratch tensor's floats: the picture and encoder outputs persist; encoder and decoder temporaries share one.
pub const P = struct {
    pub const x = size * size * 3;
    pub const x1 = 256 * 256 * cfg.channels[3];
    pub const x2 = 128 * 128 * cfg.channels[2];
    pub const x3 = 64 * 64 * cfg.channels[1];
    pub const x4 = 32 * 32 * cfg.squeeze_in;
    pub const logits = size * size;
    pub const pooled = 64 + cfg.aspp_planes;
    pub const xh = (size / 2) * (size / 2) * 3;
    pub const tokens = 256 * 256 * cfg.embed;
    pub const slots = 264 * 264 * cfg.embed;
    pub const big = 256 * 256 * 4 * cfg.embed;
    pub const lin = [4]usize{ 32 * 32 * cfg.level_in[0], 64 * 64 * cfg.level_in[1], 128 * 128 * cfg.level_in[2], 256 * 256 * cfg.level_in[3] };
    pub const out = [4]usize{ 32 * 32 * cfg.level_out[0], 64 * 64 * cfg.level_out[1], 128 * 128 * cfg.level_out[2], 256 * 256 * cfg.level_out[3] };
    pub const a64 = 256 * 256 * cfg.inter;
    pub const offs = 256 * 256 * 192;
    pub const cols = cfg.chunk * 49 * cfg.inter;
    pub const cat = 256 * 256 * 5 * cfg.aspp_planes;
    pub const gdt = 256 * 256 * cfg.inter;
    pub const patches = 256 * 256 * 64;
    pub const q48 = size * size * 48;
    pub const persistent = [_]usize{ x, x1, x2, x3, x4, logits, pooled };
    pub const enc = [_]usize{ xh, tokens, tokens, slots, big, slots };
    pub const decs = lin ++ out ++ [_]usize{ a64, offs, cols, cat, gdt, patches };
};

fn bytes(n: usize) usize {
    return std.mem.alignForward(usize, n * 4, 256);
}

fn sum(list: []const usize) usize {
    var t: usize = 0;
    for (list) |v| t += bytes(v);
    return t;
}

/// The scratch buffer's bytes: the persistent tensors, then the larger of the encoder's and the decoder's.
pub fn scratchBytes() usize {
    return sum(&P.persistent) + @max(sum(&P.enc), sum(&P.decs));
}

/// The window maps' i32s: [pass][layer][plain, shifted], each padded grid's slots aligned to 64.
pub fn mapCount() usize {
    var total: usize = 0;
    for (grids) |g0| for (0..4) |i| {
        const hp = cfg.padded(g0 >> @intCast(i));
        total += 2 * std.mem.alignForward(usize, hp * hp, 64);
    };
    return total;
}

pub const Net = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    k: ks.Kernels,
    w: wts.Weights,
    scratch: cuda.DeviceBuffer,
    maps: cuda.DeviceBuffer,
    marks: [2]cuda.Event,
    /// The windows' token maps: [pass][layer][shifted], slot to token or -1.
    map: [2][4][2]u64 = undefined,
    x: u64 = 0,
    xs: [4]u64 = undefined,
    logits: u64 = 0,
    pooled: u64 = 0,
    xh: u64 = 0,
    ta: u64 = 0,
    tb: u64 = 0,
    slots: u64 = 0,
    big: u64 = 0,
    att: u64 = 0,
    lin: [4]u64 = undefined,
    out: [4]u64 = undefined,
    a64: u64 = 0,
    offs: u64 = 0,
    cols: u64 = 0,
    cat: u64 = 0,
    gdt: u64 = 0,
    patches: u64 = 0,
    q48: u64 = 0,

    /// `path` (birefnet.safetensors from a speridlabs/idol snapshot) on `ctx`'s GPU, if it is one this family admits.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, path: []const u8) !*Net {
        const sm = try ctx.capability();
        if (!admitted(sm)) {
            var buf: [256]u8 = undefined;
            const name = ctx.name(&buf) catch "this GPU";
            std.log.err("idol birefnet: {s} (sm_{d}) is not a GPU this family admits; it runs on sm_120 and sm_121 alone", .{ name, sm });
            return error.UncheckedGpu;
        }
        const d = ctx.d;
        const n = try gpa.create(Net);
        errdefer gpa.destroy(n);
        n.* = .{ .gpa = gpa, .d = d, .stream = undefined, .k = undefined, .w = undefined, .scratch = undefined, .maps = undefined, .marks = undefined };
        n.stream = try cuda.Stream.init(d, false);
        errdefer n.stream.deinit();
        n.k = try ks.Kernels.load(d);
        errdefer n.k.unload();
        n.w = try wts.load(gpa, io, d, path);
        errdefer n.w.deinit();
        try n.buildMaps();
        errdefer n.maps.free();
        n.scratch = try cuda.DeviceBuffer.alloc(d, scratchBytes());
        errdefer n.scratch.free();
        n.marks[0] = try cuda.Event.init(d, true);
        errdefer n.marks[0].deinit();
        n.marks[1] = try cuda.Event.init(d, true);
        errdefer n.marks[1].deinit();
        var at: usize = n.scratch.ptr;
        for ([_]*u64{ &n.x, &n.xs[0], &n.xs[1], &n.xs[2], &n.xs[3], &n.logits, &n.pooled }, P.persistent) |v, len| v.* = take(&at, len);
        const shared = at;
        for ([_]*u64{ &n.xh, &n.ta, &n.tb, &n.slots, &n.big, &n.att }, P.enc) |v, len| v.* = take(&at, len);
        at = shared;
        // ipt_blk1's 48 channels at 1024^2 take the levels' inputs' room: the head runs after the last level
        n.q48 = at;
        std.debug.assert(bytes(P.lin[0]) + bytes(P.lin[1]) + bytes(P.lin[2]) + bytes(P.lin[3]) >= P.q48 * 4);
        for (&n.lin, P.lin) |*v, len| v.* = take(&at, len);
        for (&n.out, P.out) |*v, len| v.* = take(&at, len);
        for ([_]*u64{ &n.a64, &n.offs, &n.cols, &n.cat, &n.gdt, &n.patches }, P.decs[8..]) |v, len| v.* = take(&at, len);
        return n;
    }

    fn take(at: *usize, len: usize) u64 {
        const r = at.*;
        at.* += bytes(len);
        return r;
    }

    pub fn deinit(n: *Net) void {
        for (&n.marks) |*m| m.deinit();
        n.scratch.free();
        n.maps.free();
        n.w.deinit();
        n.k.unload();
        n.stream.deinit();
        n.gpa.destroy(n);
    }

    /// The device bytes the net holds: its weights, scratch and maps.
    pub fn deviceBytes(n: *const Net) usize {
        return n.w.buf.len + n.scratch.len + n.maps.len;
    }

    /// Each layer's window slots to tokens, both passes, plain and shifted by half a window (torch.roll by -6).
    fn buildMaps(n: *Net) !void {
        const all = try n.gpa.alloc(i32, mapCount());
        defer n.gpa.free(all);
        n.maps = try cuda.DeviceBuffer.alloc(n.d, all.len * 4);
        errdefer n.maps.free();
        var at: usize = 0;
        for (grids, 0..) |g0, pass| for (0..4) |i| for (0..2) |s| {
            const h = g0 >> @intCast(i);
            const hp = cfg.padded(h);
            n.map[pass][i][s] = n.maps.ptr + at * 4;
            cfg.fillMap(all[at..][0 .. hp * hp], h, if (s == 1) cfg.window / 2 else 0);
            at += std.mem.alignForward(usize, hp * hp, 64);
        };
        try n.maps.upload(0, std.mem.sliceAsBytes(all));
    }

    pub fn upload(n: *const Net, at: u64, data: []const f32) !void {
        const b = std.mem.sliceAsBytes(data);
        try n.d.check(n.d.api.cuMemcpyHtoDAsync_v2(at, b.ptr, b.len, n.stream.handle), "cuMemcpyHtoDAsync");
        try n.stream.synchronize();
    }

    pub fn download(n: *const Net, out: []f32, at: u64) !void {
        const b = std.mem.sliceAsBytes(out);
        try n.d.check(n.d.api.cuMemcpyDtoHAsync_v2(b.ptr, at, b.len, n.stream.handle), "cuMemcpyDtoHAsync");
        try n.stream.synchronize();
    }

    /// Mask logits [1024, 1024] of the normalized picture `input` [1024 * 1024, 3] (remove_background's tensor).
    pub fn forward(n: *Net, input: []const f32, logits: []f32) !Timing {
        if (input.len != P.x or logits.len != P.logits) return error.PictureSize;
        const started = seconds();
        try n.upload(n.x, input);
        try n.marks[0].record(n.stream);
        try n.encode();
        try dec.squeeze(n);
        try dec.decode(n);
        try n.marks[1].record(n.stream);
        try n.download(logits, n.logits);
        const gpu = @as(f64, try cuda.Event.elapsedMs(n.marks[0], n.marks[1])) / 1000;
        return .{ .wall = seconds() - started, .gpu = gpu };
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
    pub fn encode(n: *Net) !void {
        const k = &n.k;
        const s = n.stream;
        try ks.resize(k, s, n.x, n.xh, .{ .h = size, .w = size }, .{ .h = size / 2, .w = size / 2 }, 3, 3, 0, 3, 0);
        for (0..2) |pass| {
            const side = @as(usize, size) >> @intCast(pass);
            const g0 = grids[pass];
            try ks.tiles(k, s, if (pass == 0) n.x else n.xh, n.slots, .{ .h = g0, .w = g0 }, side, 3, 4, 4, false, 64);
            try n.conv(n.w.patch, n.slots, n.tb, .{ .rows = g0 * g0, .ldy = cfg.embed });
            try n.norm(n.tb, 0, n.w.patch_norm, n.ta, g0 * g0, cfg.embed, .rows, 0, cfg.embed, 0);
            for (0..4) |i| {
                const h = g0 >> @intCast(i);
                const C = cfg.widths[i];
                try n.layer(pass, i, n.ta, n.tb);
                // the layer's features: norm_i, at the full grid's channels or resized to it beside them
                const full = grids[0] >> @intCast(i);
                const dst, const ld, const off = n.featureSlot(i);
                if (pass == 0) {
                    try n.norm(n.ta, 0, n.w.norms[i], dst, h * h, C, .rows, 0, ld, off);
                } else {
                    try n.norm(n.ta, 0, n.w.norms[i], n.att, h * h, C, .rows, 0, C, 0);
                    try ks.resize(k, s, n.att, dst, .{ .h = h, .w = h }, .{ .h = full, .w = full }, C, C, 0, ld, off + C);
                }
                if (i < 3) std.mem.swap(u64, &n.ta, &n.tb);
            }
        }
        // the squeeze's input: x1, x2 and x3 resized to the coarsest grid, then x4
        var col: usize = 0;
        for (0..3) |i| {
            const g = grids[0] >> @intCast(i);
            const C = 2 * cfg.widths[i];
            try ks.resize(k, s, n.xs[i], n.xs[3], .{ .h = g, .w = g }, .{ .h = 32, .w = 32 }, C, C, 0, cfg.squeeze_in, col);
            col += C;
        }
    }

    /// Where layer i's features go (buffer, row length, column): x1 to x3 [grid^2, 2 C], x4 in the squeeze's input.
    fn featureSlot(n: *const Net, i: usize) struct { u64, usize, usize } {
        if (i < 3) return .{ n.xs[i], 2 * cfg.widths[i], 0 };
        return .{ n.xs[3], cfg.squeeze_in, cfg.squeeze_in - cfg.channels[0] };
    }

    /// Swin layer i of `pass` on t [h * h, C] in place, then (i < 3) PatchMerging into down [h^2 / 4, 2 C].
    pub fn layer(n: *Net, pass: usize, i: usize, t: u64, down: u64) !void {
        const h = grids[pass] >> @intCast(i);
        const hp = cfg.padded(h);
        const C = cfg.widths[i];
        const L = &n.w.layers[i];
        for (L.blocks[0..cfg.depths[i]], 0..) |*b, j| {
            const shift: usize = if (j % 2 == 1) cfg.window / 2 else 0;
            const map = n.map[pass][i][j % 2];
            const slots = hp * hp;
            try n.norm(t, map, b.norm1, n.slots, slots, C, .map, 0, C, 0);
            try n.conv(b.qkv, n.slots, n.big, .{ .rows = slots, .ldy = 3 * C });
            try ks.window(&n.k, n.stream, n.big, b.table, n.att, C, cfg.heads[i], hp, hp, shift);
            try n.convWith(b.proj, n.att, t, t, map, .{ .rows = slots, .ldy = C, .flags = ks.flag_add | ks.flag_map });
            try n.norm(t, 0, b.norm2, n.slots, h * h, C, .rows, 0, C, 0);
            try n.conv(b.fc1, n.slots, n.big, .{ .rows = h * h, .ldy = 4 * C, .flags = ks.flag_gelu });
            try n.convWith(b.fc2, n.big, t, t, 0, .{ .rows = h * h, .ldy = C, .flags = ks.flag_add });
        }
        if (L.merge) |m| {
            try n.norm(t, 0, m, n.slots, h * h / 4, 4 * C, .merge, h, 4 * C, 0);
            try n.conv(L.reduction.?, n.slots, down, .{ .rows = h * h / 4, .ldy = 2 * C });
        }
    }

    /// The shape fields `conv` fills from c.
    pub const Out = struct { rows: usize, cols: usize = 1, ldy: usize, yoff: usize = 0, flags: u32 = 0, row0: usize = 0, ldx: ?usize = null };

    /// y = c(x) with its batch norm when it has one (flags add bn), rows x cols at c's side.
    pub fn conv(n: *const Net, c: wts.Conv, x: u64, y: u64, o: Out) !void {
        try n.convWith(c, x, 0, y, 0, o);
    }

    /// conv with a residual r (flag_add) and a row map (flag_map); 0 where unused.
    pub fn convWith(n: *const Net, c: wts.Conv, x: u64, r: u64, y: u64, map: u64, o: Out) !void {
        const flags = o.flags | (if (c.post != 0) ks.flag_bn else 0);
        try ks.conv(&n.k, n.stream, x, c.w, c.b, c.post, r, map, y, .{ .rows = o.rows, .cols = o.cols, .cin = c.cin, .ldx = o.ldx orelse c.cin, .cout = c.cout, .ldw = c.ldw, .k = c.k, .ldy = o.ldy, .yoff = o.yoff, .flags = flags, .row0 = o.row0 });
    }

    pub fn norm(n: *const Net, x: u64, map: u64, nm: wts.Norm, y: u64, rows: usize, width: usize, mode: ks.NormMode, src_w: usize, ldy: usize, yoff: usize) !void {
        try ks.norm(&n.k, n.stream, x, map, nm.g, nm.b, y, rows, width, mode, src_w, ldy, yoff, cfg.ln_eps);
    }
};

test "scratch: the encoder's and decoder's temporaries share one region, under 4 GiB, offsets fit c_int indexing" {
    try std.testing.expect(scratchBytes() < 4 << 30);
    // ipt_blk1's 48 channels at 1024^2 fit in the levels' inputs
    try std.testing.expect(bytes(P.lin[0]) + bytes(P.lin[1]) + bytes(P.lin[2]) + bytes(P.lin[3]) >= P.q48 * 4);
    // the kernels index elements as int: every tensor stays under 2^31 elements
    for (P.persistent ++ P.enc ++ P.decs) |v| try std.testing.expect(v < std.math.maxInt(c_int));
    // padded grids 264, 132, 72, 36 then 132, 72, 36, 24, each plain and shifted, slots aligned to 64
    try std.testing.expectEqual(@as(usize, 2 * (69696 + 17472 + 5184 + 1344) + 2 * (17472 + 5184 + 1344 + 576)), mapCount());
}

test "admission: sm_120 and sm_121 alone" {
    try std.testing.expect(admitted(121) and admitted(120));
    for ([_]u32{ 80, 86, 89, 90, 100, 103, 122 }) |sm| try std.testing.expect(!admitted(sm));
}
