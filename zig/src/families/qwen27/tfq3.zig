//! Zero-copy mmap loader for Qwen27 TFQ3 (`pw_v3_tables.bin` + single or sharded `qwen27_tfq3*.tffpw`).
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const projection = @import("projection.zig");
const widths = @import("widths.zig");
pub const check = @import("tfq3_check.zig");

const Mapped = struct {
    bytes: []align(std.heap.page_size_min) u8,
    len: usize,

    fn file(m: Mapped) []const u8 {
        return m.bytes[0..m.len];
    }

    fn deinit(m: Mapped) void {
        std.posix.munmap(m.bytes);
    }
};

const Shard = struct {
    map: Mapped,
    scales: mtl.Buffer,
    weights: mtl.Buffer,
    metas: []check.TensorMeta,
};

const Entry = struct {
    shard: usize,
    meta_idx: usize,
};

pub const Found = struct {
    tfq3: projection.Tfq3,
    bits: u8,
    part: check.PairPart,

    pub fn linear(f: Found, n: usize, k: usize, sr_split: u32) projection.Linear {
        var t = f.tfq3;
        t.sr_split = sr_split;
        return .{ .words = t.words, .pairs = t.scales, .n = @intCast(n), .k = @intCast(k), .tile = 32, .slices = 1, .bits = f.bits, .group = widths.tfq3_group, .gn = n, .tfq3 = t };
    }
};

/// Concatenates one or more BF16 `[rows_i, k]` tensor byte slices into one owned GPU buffer for BF16 linear dispatch.
pub fn packBf16Linear(a: std.mem.Allocator, device: mtl.Device, owned: *std.ArrayList(mtl.Buffer), slices: []const []const u8, rows_total: usize, gn_total: usize, k: usize) !projection.Linear {
    const buf = try device.buffer(rows_total * k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    owned.append(a, buf) catch |err| {
        buf.deinit();
        return err;
    };
    const dst = buf.contents();
    var off: usize = 0;
    for (slices) |src| {
        @memcpy(dst[off..][0..src.len], src);
        off += src.len;
    }
    const ref_w: projection.Ref = .{ .buffer = buf };
    return .{ .words = ref_w, .pairs = ref_w, .n = @intCast(rows_total), .k = @intCast(k), .tile = 32, .slices = 1, .bits = widths.bf16.bits, .group = widths.bf16.group, .gn = gn_total, .bf16 = ref_w };
}

pub const Store = struct {
    allocator: std.mem.Allocator,
    tables_map: Mapped,
    tables_buf: mtl.Buffer,
    shards: std.ArrayList(Shard) = .empty,
    index: std.StringHashMapUnmanaged(Entry) = .empty,
    total_bytes: usize = 0,

    pub fn open(a: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !?Store {
        const names = try collectShards(a, io, dir);
        defer {
            for (names) |n| a.free(n);
            a.free(names);
        }
        const tbl_opt = try mapFile(dir, check.tables_file, check.tables_bytes);
        if (tbl_opt == null and names.len == 0) return null;
        if (tbl_opt == null or names.len == 0) {
            if (tbl_opt) |m| m.deinit();
            return error.UnsupportedQuantization;
        }
        const tbl_map = tbl_opt.?;
        errdefer tbl_map.deinit();
        try check.tables(tbl_map.file());
        const tbl_span = try check.span(tbl_map.len, 0, tbl_map.len, true);
        const tbl_buf = try view(device, tbl_map, tbl_span);
        errdefer tbl_buf.deinit();

        var store: Store = .{
            .allocator = a,
            .tables_map = tbl_map,
            .tables_buf = tbl_buf,
            .total_bytes = tbl_map.len,
        };
        errdefer store.deinitOwnedShards();

        for (names, 0..) |shard_name, si| {
            const m = (try mapFile(dir, shard_name, @sizeOf(check.Header))) orelse return error.Tfq3Truncated;
            errdefer m.deinit();
            const layout = try check.shard(m.file());
            const metas = try a.alloc(check.TensorMeta, layout.count);
            errdefer a.free(metas);
            @memcpy(std.mem.sliceAsBytes(metas), m.file()[layout.metas.off..][0..layout.metas.len]);
            const b_sc = try view(device, m, layout.scales);
            errdefer b_sc.deinit();
            const b_wt = try view(device, m, layout.weights);
            errdefer b_wt.deinit();
            try store.shards.append(a, .{ .map = m, .scales = b_sc, .weights = b_wt, .metas = metas });
            store.total_bytes += m.len;
            const stored = store.shards.items[si].metas;
            for (stored, 0..) |*meta, mi| {
                const key = check.canonicalModule(meta.nameSlice());
                const gop = try store.index.getOrPut(a, key);
                if (gop.found_existing) return error.Tfq3Overlap;
                gop.value_ptr.* = .{ .shard = si, .meta_idx = mi };
            }
        }
        return store;
    }

    /// Looks up `module` (canonical, e.g. `model.layers.0.mlp.gate_proj` or `lm_head`) without shape check.
    pub fn lookupBits(s: *const Store, module: []const u8, n: usize, k: usize) ?u8 {
        const e = s.index.get(check.canonicalModule(module)) orelse return null;
        const m = &s.shards.items[e.shard].metas[e.meta_idx];
        if (m.R != n or m.C != k) return null;
        return @intCast(m.b_bits);
    }

    /// Resolves a validated tensor handle ready to attach to `projection.Linear`.
    pub fn find(s: *const Store, module: []const u8, n: usize, k: usize) ?Found {
        const e = s.index.get(check.canonicalModule(module)) orelse return null;
        const sh = &s.shards.items[e.shard];
        const m = &sh.metas[e.meta_idx];
        if (m.R != n or m.C != k) return null;
        const bits: u8 = @intCast(m.b_bits);
        return .{
            .tfq3 = .{
                .words = .{ .buffer = sh.weights, .offset = @intCast(m.word_off * 4) },
                .scales = .{ .buffer = sh.scales, .offset = @as(usize, m.sc_off) * 2 },
                .tables = .{ .buffer = s.tables_buf, .offset = 0 },
                .sr_split = 0,
            },
            .bits = bits,
            .part = .{
                .shard = e.shard,
                .n = n,
                .k = k,
                .bits = bits,
                .word_off = m.word_off,
                .word_count = m.word_count,
                .sc_off = m.sc_off,
            },
        };
    }

    pub fn appendBuffers(s: *const Store, a: std.mem.Allocator, list: *std.ArrayList(mtl.Buffer)) !void {
        try list.append(a, s.tables_buf);
        for (s.shards.items) |sh| {
            try list.append(a, sh.scales);
            try list.append(a, sh.weights);
        }
    }

    fn deinitOwnedShards(s: *Store) void {
        s.index.deinit(s.allocator);
        for (s.shards.items) |sh| {
            sh.weights.deinit();
            sh.scales.deinit();
            s.allocator.free(sh.metas);
            sh.map.deinit();
        }
        s.shards.deinit(s.allocator);
    }

    pub fn deinit(s: *Store) void {
        s.deinitOwnedShards();
        s.tables_buf.deinit();
        s.tables_map.deinit();
    }
};

fn collectShards(a: std.mem.Allocator, io: std.Io, dir: []const u8) ![][]u8 {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return a.alloc([]u8, 0);
    defer d.close(io);
    var list: std.ArrayList([]u8) = .empty;
    errdefer {
        for (list.items) |n| a.free(n);
        list.deinit(a);
    }
    var it = d.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!check.isShardName(entry.name)) continue;
        try list.append(a, try a.dupe(u8, entry.name));
    }
    std.mem.sort([]u8, list.items, {}, struct {
        fn less(_: void, x: []u8, y: []u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    return list.toOwnedSlice(a);
}

fn mapFile(dir: []const u8, name: []const u8, least: usize) !?Mapped {
    var buf: [512]u8 = undefined;
    const s = std.fmt.bufPrint(buf[0 .. buf.len - 1], "{s}/{s}", .{ dir, name }) catch return error.NameTooLong;
    buf[s.len] = 0;
    const fd = platform.c.open(buf[0..s.len :0].ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = platform.c.close(fd);
    const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
    if (end < 0) return error.Tfq3Truncated;
    const len = std.math.cast(usize, end) orelse return error.Tfq3Overflow;
    if (len < least) return error.Tfq3Truncated;
    const room = try check.mappedLen(len);
    const bytes = try std.posix.mmap(
        null,
        std.mem.alignForward(usize, room, std.heap.pageSize()),
        .{ .READ = true, .WRITE = true },
        .{ .TYPE = .PRIVATE },
        fd,
        0,
    );
    return .{ .bytes = bytes, .len = len };
}

fn view(device: mtl.Device, m: Mapped, at: check.Span) !mtl.Buffer {
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const len = try check.mappedLen(@max(at.len, 16));
    if (comptime @hasDecl(mtl.Device, "bufferNoCopy")) {
        return device.bufferNoCopy(@ptrCast(m.bytes.ptr + at.off), len, opts);
    }
    return error.UnsupportedQuantization;
}
