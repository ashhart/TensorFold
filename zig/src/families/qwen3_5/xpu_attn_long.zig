//! Long-context decode attention (flash-decoding split-K) and the bf16/q8/q4 KV cache modes (ARC_KV).
const std = @import("std");
const rt = @import("xpu").rt;

const spv_bf16 = @import("xpu").kernels.qwen_attn_long;
const spv_q8 = @import("xpu").kernels.qwen_attn_long_q8;
const spv_q4 = @import("xpu").kernels.qwen_attn_long_q4;
const spv_kvq = @import("xpu").kernels.qwen_kvq;
const spv_pfs = @import("xpu").kernels.qwen_attn_pfs;
const spv_pfs8 = @import("xpu").kernels.qwen_attn_pfs_q8;
const spv_pfs4 = @import("xpu").kernels.qwen_attn_pfs_q4;

pub const chunk: u32 = 1024;
pub const max_chunks: u32 = 128;
pub const max_ctx: u32 = chunk * max_chunks;

pub const Mode = enum { bf16, q8, q4 };

var mode_cache: ?Mode = null;
var override: ?Mode = null;

/// The MTP block's cache format may differ from the target's: its code runs with the override set (Mtp.step).
pub fn setOverride(m: ?Mode) void {
    override = m;
}

/// KV format of the MTP block (env MTP_KV = bf16 | q8 | q4; default: target's quantized mode, else q4 from 16384).
pub fn mtpMode(cap: u32) Mode {
    if (std.c.getenv("MTP_KV")) |v| {
        const s = std.mem.span(v);
        if (std.mem.eql(u8, s, "bf16")) return .bf16;
        if (std.mem.eql(u8, s, "q8")) return .q8;
        if (std.mem.eql(u8, s, "q4")) return .q4;
    }
    if (kvMode() != .bf16) return kvMode();
    return if (cap > 16384) .q4 else .bf16;
}

/// Picks the KV cache format before the model loads (`--kv`); it wins over the ARC_KV environment variable.
pub fn setKvMode(m: Mode) void {
    mode_cache = m;
}

/// The KV cache format (env ARC_KV, read once unless set).
pub fn kvMode() Mode {
    if (override) |m| return m;
    if (mode_cache) |m| return m;
    var m: Mode = .bf16;
    if (std.c.getenv("ARC_KV")) |v| {
        const s = std.mem.span(v);
        if (std.mem.eql(u8, s, "q8")) m = .q8 else if (std.mem.eql(u8, s, "q4")) m = .q4;
    }
    mode_cache = m;
    return m;
}

/// Bytes of one kv head of one position in a K or V cache.
pub fn recBytes(m: Mode) usize {
    return switch (m) {
        .bf16 => 512,
        .q8 => 272,
        .q4 => 144,
    };
}

/// Bytes of a K (or V) cache of one layer for `cap` positions.
pub fn cacheBytes(cap: u32) usize {
    return cacheBytesFor(kvMode(), cap);
}

pub fn cacheBytesFor(m: Mode, cap: u32) usize {
    return @as(usize, cap) * 4 * recBytes(m);
}

pub fn threshold() u32 {
    if (kvMode() != .bf16) return 0;
    return if (std.c.getenv("ATTN_LONG_T")) |v| (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 4096) else 4096;
}

pub const Long = struct {
    part: rt.Kernel,
    merge: rt.Kernel,

    pub fn init(r: *rt.Runtime, mode: Mode) !Long {
        const spv = switch (mode) {
            .bf16 => spv_bf16,
            .q8 => spv_q8,
            .q4 => spv_q4,
        };
        var m = try r.moduleWith(spv, "-cl-intel-256-GRF-per-thread");
        return .{ .part = try m.kernel("attn_long_partial", .{ 128, 1, 1 }), .merge = try m.kernel("attn_long_merge", .{ 256, 1, 1 }) };
    }

    /// Attention of `rows` rows, row z over len0 + z keys: q / out [row][24][256], qg [row][24][512]; po/pm/pl scratch.
    pub fn run(self: *Long, q: rt.Buffer, kc: rt.Buffer, vc: rt.Buffer, po: rt.Buffer, pm: rt.Buffer, pl: rt.Buffer, qg: rt.Buffer, out: rt.Buffer, len0: u32, rows: u32) !void {
        const nch = (len0 + rows - 1 + chunk - 1) / chunk;
        var p = &self.part;
        try p.setBuffer(0, q);
        try p.setBuffer(1, kc);
        try p.setBuffer(2, vc);
        try p.setBuffer(3, po);
        try p.setBuffer(4, pm);
        try p.setBuffer(5, pl);
        try p.setU32(6, len0);
        try p.setF32(7, 0.0625);
        try p.setU32(8, max_chunks);
        try p.launch(.{ 4 * rows, nch, 1 });
        var g = &self.merge;
        try g.setBuffer(0, po);
        try g.setBuffer(1, pm);
        try g.setBuffer(2, pl);
        try g.setBuffer(3, qg);
        try g.setBuffer(4, out);
        try g.setU32(5, len0);
        try g.setU32(6, max_chunks);
        try g.launch(.{ 24, rows, 1 });
    }
};

/// Prompt-window causal flash attention on the matrix engine (qwen_attn_pfs.cl); the cache must hold the rows' K / V.
pub const Pfs = struct {
    r: *rt.Runtime,
    mode: Mode,
    k: rt.Kernel,
    prep: rt.Kernel,
    expand: ?rt.Kernel,
    qb: rt.Buffer, // the queries as B-operand tiles (attn_pfs_prep)
    kx: ?rt.Buffer, // fp16 K' / V' of the key range being processed (kvq_expand)
    vx: ?rt.Buffer,
    st: ?rt.Buffer = null, // online-softmax state between key ranges, created when a window needs more than one
    max_rows: u32,
    /// Keys a range (a multiple of 128): the expanded scratch is 2 x range x 2 KB. Env PFS_RANGE.
    range: u32 = 16384,
    const state_bytes: usize = 8 * 1632 * 4; // a work-group

    pub fn init(r: *rt.Runtime, mode: Mode, max_rows: u32) !Pfs {
        var bytes: []const u8 = switch (mode) {
            .bf16 => spv_pfs,
            .q8 => spv_pfs8,
            .q4 => spv_pfs4,
        };
        if (std.c.getenv("PFS_SPV")) |path| { // experiments: a kernel build from a file
            const f = std.c.fopen(path, "rb") orelse return error.Unsupported;
            defer _ = std.c.fclose(f);
            const buf = try std.heap.page_allocator.alloc(u8, 1 << 22);
            bytes = buf[0..std.c.fread(buf.ptr, 1, buf.len, f)];
        }
        var m = try r.moduleWith(bytes, "-cl-intel-256-GRF-per-thread");
        const q = mode != .bf16;
        var range: u32 = 16384;
        if (std.c.getenv("PFS_RANGE")) |v| range = std.fmt.parseInt(u32, std.mem.span(v), 10) catch range;
        range = @max(128, range / 128 * 128);
        return .{
            .r = r,
            .mode = mode,
            .range = range,
            .k = try m.kernel("attn_prefill_s", .{ 128, 1, 1 }),
            .prep = try m.kernel("attn_pfs_prep", .{ 128, 1, 1 }),
            .expand = if (q) try m.kernel("kvq_expand", .{ 128, 1, 1 }) else null,
            .qb = try r.alloc(@as(usize, (max_rows + 7) / 8) * 8 * 24 * 256 * 2),
            .kx = if (q) try r.alloc(@as(usize, range) * 4 * 256 * 2) else null,
            .vx = if (q) try r.alloc(@as(usize, range) * 4 * 256 * 2) else null,
            .max_rows = max_rows,
        };
    }

    fn expandRange(self: *Pfs, src: rt.Buffer, dst: rt.Buffer, k0: u32, n: u32) !void {
        var e = &self.expand.?;
        try e.setBuffer(0, src);
        try e.setBuffer(1, dst);
        try e.setU32(2, k0);
        try e.launch(.{ n, 1, 1 }); // n work-groups of 128 items: 4 kv heads x 32 groups of 8 dims a key
    }

    pub fn run(self: *Pfs, q: rt.Buffer, kc: rt.Buffer, vc: rt.Buffer, qg: rt.Buffer, out: rt.Buffer, pos0: u32, rows: u32) !void {
        const tiles = (rows + 7) / 8;
        var p = &self.prep;
        try p.setBuffer(0, q);
        try p.setBuffer(1, self.qb);
        try p.setU32(2, rows);
        try p.launch(.{ tiles * 4 * 3 * 2048 / 128, 1, 1 });
        const kall = pos0 + rows;
        var k0: u32 = 0;
        while (k0 < kall) {
            const k1 = if (self.mode == .bf16) kall else @min(k0 + self.range, kall);
            if (self.mode != .bf16) {
                try self.expandRange(kc, self.kx.?, k0, k1 - k0);
                try self.expandRange(vc, self.vx.?, k0, k1 - k0);
            }
            const flags: u32 = (if (k0 == 0) @as(u32, 1) else 0) | (if (k1 == kall) @as(u32, 2) else 0);
            if (flags != 3 and self.st == null) self.st = try self.r.alloc(@as(usize, (self.max_rows + 7) / 8) * 4 * state_bytes);
            var k = &self.k;
            try k.setBuffer(0, self.qb);
            try k.setBuffer(1, self.kx orelse kc);
            try k.setBuffer(2, self.vx orelse vc);
            try k.setBuffer(3, qg);
            try k.setBuffer(4, out);
            try k.setBuffer(5, self.st orelse self.qb); // unused when the window fits one range
            try k.setU32(6, pos0);
            try k.setU32(7, rows);
            try k.setU32(8, if (self.mode == .bf16) 0 else k0); // the position of row 0 of the K / V rows
            try k.setU32(9, k0);
            try k.setU32(10, k1);
            try k.setU32(11, flags);
            try k.launch(.{ tiles, 4, 1 });
            k0 = k1;
        }
    }
};

/// Quantizes appended K / V rows (bf16 [rows][4][256]) into the cache records at positions pos0 ..
pub const Kvq = struct {
    k8: rt.Kernel,
    k4: rt.Kernel,

    pub fn init(r: *rt.Runtime) !Kvq {
        var m = try r.module(spv_kvq);
        return .{ .k8 = try m.kernel("kvq_quant8", .{ 32, 1, 1 }), .k4 = try m.kernel("kvq_quant4", .{ 32, 1, 1 }) };
    }

    pub fn append(self: *Kvq, mode: Mode, src: rt.Buffer, dst: rt.Buffer, pos0: u32, rows: u32) !void {
        var k = if (mode == .q8) &self.k8 else &self.k4;
        try k.setBuffer(0, src);
        try k.setBuffer(1, dst);
        try k.setU32(2, pos0);
        try k.launch(.{ rows, 1, 1 }); // work-groups of 32 items: (head, block) pairs of one row
    }
};

/// How many of `rows` rows (a prefix; row r has first_len + r keys) use the original kernels instead of the long one.
pub fn nOld(t: u32, first_len: u32, rows: u32) u32 {
    if (first_len > t) return 0;
    return @min(rows, t - first_len + 1);
}
