//! exl3-ple2: an EXL3 pack's n-gram path on v1.1.0's Flash Next family (cuda_ple.Ngram with `format = .exl3`). 1.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("../../check.zig");
const st = core.safetensors;
const ple = flash.ple;
const x3 = flash.exl3_ple;
const Gpu = check.Gpu;

const py_ids_crc: u32 = 0xd04d5584;
const py_raw_crc: u32 = 0x2d57eafe;
const py_decoded_crc: u32 = 0xdcf520dc;

const streams = 4;
const hidden = 2560;
const wide = streams * hidden;
const taps = 4;
const dilation = 3;
const tail_rows = (taps - 1) * dilation;
const eps: f32 = 1e-6;
const prefix = "model.language_model.layers.1.ple.";

// ---- RSS ----------------------------------------------------------------------------------------------------------

const Rss = struct { total: u64 = 0, file: u64 = 0, anon: u64 = 0 }; // kB

fn rss() Rss {
    var buf: [1 << 14]u8 = undefined;
    const fd = std.c.open("/proc/self/status", .{ .ACCMODE = .RDONLY });
    if (fd < 0) return .{};
    defer _ = std.c.close(fd);
    var len: usize = 0;
    while (len < buf.len) {
        const n = std.c.read(fd, buf[len..].ptr, buf.len - len);
        if (n <= 0) break;
        len += @intCast(n);
    }
    var out: Rss = .{};
    var it = std.mem.splitScalar(u8, buf[0..len], '\n');
    while (it.next()) |line| {
        const fields = [_]struct { key: []const u8, dst: *u64 }{ .{ .key = "VmRSS:", .dst = &out.total }, .{ .key = "RssFile:", .dst = &out.file }, .{ .key = "RssAnon:", .dst = &out.anon } };
        for (fields) |f| {
            if (!std.mem.startsWith(u8, line, f.key)) continue;
            var parts = std.mem.tokenizeAny(u8, line[f.key.len..], " \tkB");
            if (parts.next()) |n| f.dst.* = std.fmt.parseInt(u64, n, 10) catch 0;
        }
    }
    return out;
}

fn showRss(what: []const u8) Rss {
    const r = rss();
    std.debug.print("RSS {s}: total {d} MiB, file-backed {d} MiB, anon {d} MiB\n", .{ what, r.total >> 10, r.file >> 10, r.anon >> 10 });
    return r;
}

// ---- the branch's ids, ported from exl3-flashnext-window-wip zig/src/families/flashnext/exl3_ple_ids.zig ----------

/// Row ids [tokens.len][16] after `history` (incremental form; a token equal to eos starts a segment after itself).
fn oursIds(h: ple.Hash, history: [2]i64, tokens: []const u32, out: [][16]u64) void {
    var seq: [3]i64 = undefined;
    @memcpy(seq[0..2], &history);
    var seg: usize = 0;
    for (history) |t| seg = if (t == h.eos) 0 else @min(seg + 1, 2);
    for (tokens, out[0..tokens.len]) |tok, *row| {
        seq[2] = tok;
        for (row, 0..) |*id, head| {
            var mixed: u64 = 0;
            for (0..2 + head / h.per) |q| {
                const t = if (seg >= q) seq[2 - q] else h.eos;
                const term = @as(u64, @bitCast(t)) *% @as(u64, @bitCast(h.mult[q]));
                mixed = if (q == 0) term else mixed ^ term;
            }
            id.* = @as(u64, @intCast(h.offsets[head])) + mixed % @as(u64, @intCast(h.sizes[head]));
        }
        seg = if (tok == h.eos) 0 else @min(seg + 1, 2);
        seq[0] = seq[1];
        seq[1] = seq[2];
    }
}

// ---- weights --------------------------------------------------------------------------------------------------------

const Index = struct {
    parsed: std.json.Parsed(std.json.Value),
    files: std.StringHashMapUnmanaged(st.File) = .empty,
    dir: []const u8,

    fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Index {
        const path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
        defer gpa.free(text);
        return .{ .parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{}), .dir = dir };
    }

    fn deinit(x: *Index, gpa: std.mem.Allocator, io: std.Io) void {
        var it = x.files.valueIterator();
        while (it.next()) |f| f.close(io);
        x.files.deinit(gpa);
        x.parsed.deinit();
    }

    fn get(x: *Index, gpa: std.mem.Allocator, io: std.Io, name: []const u8) !st.Tensor {
        const rel = (x.parsed.value.object.get("weight_map").?.object.get(name) orelse return error.MissingTensor).string;
        if (x.files.get(rel)) |f| return f.get(name) orelse error.MissingTensor;
        const path = try std.fs.path.join(gpa, &.{ x.dir, rel });
        defer gpa.free(path);
        const f = try st.File.open(gpa, io, path);
        try x.files.put(gpa, rel, f);
        return f.get(name) orelse error.MissingTensor;
    }
};

fn widen(gpa: std.mem.Allocator, t: st.Tensor, offset: f32) ![]f32 {
    const out = try gpa.alloc(f32, t.numel());
    for (out, 0..) |*o, i| {
        const w = std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little);
        const v: f32 = switch (t.dtype) {
            .f16 => @floatCast(@as(f16, @bitCast(w))),
            .bf16 => @bitCast(@as(u32, w) << 16),
            else => return error.UnexpectedTensor,
        };
        o.* = v + offset;
    }
    return out;
}

fn bf(x: f32) f32 {
    return @bitCast(@as(u32, ple.toBf16(x)) << 16);
}

fn bfBits(x: f32) u16 {
    return ple.toBf16(x);
}

fn bfVal(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

// ---- references of the injection (from exl3-flashnext-window-wip exl3_ple_cpu.zig) ---------------------------------

const Weights = struct {
    key: []const f32, // [wide][hidden]
    value: []const f32, // [hidden][hidden]
    norm_key: []const f32, // [wide], 1 + w
    norm_query: []const f32,
    norm_conv: []const f32,
    conv: []const f32, // [wide][taps], as stored (F16)
};

fn dot32(w: []const f32, x: []const f32) f32 {
    var acc: f32 = 0;
    for (w, x) |a, b| acc += a * b;
    return acc;
}

/// The keys and values the reference's F16 dense projections store (fp32 sums, bf16 outputs).
fn project(w: Weights, emb: []const f32, keys: []f32, vals: []f32) void {
    for (0..wide) |o| keys[o] = bf(dot32(w.key[o * hidden ..][0..hidden], emb));
    for (0..hidden) |o| vals[o] = bf(dot32(w.value[o * hidden ..][0..hidden], emb));
}

/// The reference's roundings (glue.py `_ple_gate` / `_ple_conv`) on the host, one token at a time.
const Branch = struct {
    w: Weights,
    conv_bf: []f32,
    gated: []f32,
    nrow: []f32,
    tail: []f32,

    fn init(gpa: std.mem.Allocator, w: Weights) !Branch {
        const conv_bf = try gpa.alloc(f32, w.conv.len);
        for (conv_bf, w.conv) |*d, s| d.* = bf(s);
        const tail = try gpa.alloc(f32, tail_rows * wide);
        @memset(tail, 0);
        return .{ .w = w, .conv_bf = conv_bf, .gated = try gpa.alloc(f32, wide), .nrow = try gpa.alloc(f32, wide), .tail = tail };
    }

    fn step(b: *Branch, keys: []const f32, vals: []const f32, h: []const f32, out: []f32) void {
        const w = b.w;
        const d: f32 = hidden;
        for (0..streams) |s| {
            const k = keys[s * hidden ..][0..hidden];
            const q = h[s * hidden ..][0..hidden];
            var kss: f32 = 0;
            var qss: f32 = 0;
            for (k, q) |a, c| {
                kss += a * a;
                qss += c * c;
            }
            const krinv = 1.0 / @sqrt(kss / d + eps);
            const qrinv = 1.0 / @sqrt(qss / d + eps);
            var dot: f32 = 0;
            for (0..hidden) |i| dot += bf(bf(k[i] * krinv * w.norm_key[s * hidden + i]) * bf(q[i] * qrinv * w.norm_query[s * hidden + i]));
            var gate = bf(dot / @sqrt(d));
            const mag = @sqrt(@max(@abs(gate), 1e-6));
            gate = bf(if (gate > 0) mag else if (gate < 0) -mag else 0);
            const sg = bf(1.0 / (1.0 + @exp(-gate)));
            var gss: f32 = 0;
            for (0..hidden) |i| {
                const g = bf(sg * vals[i]);
                b.gated[s * hidden + i] = g;
                gss += g * g;
            }
            const rinv = 1.0 / @sqrt(gss / d + eps);
            for (0..hidden) |i| b.nrow[s * hidden + i] = bf(b.gated[s * hidden + i] * rinv * w.norm_conv[s * hidden + i]);
        }
        for (0..wide) |e| {
            var acc: f32 = 0;
            for (0..taps) |t| {
                const at = t * dilation;
                acc += b.conv_bf[e * taps + t] * (if (at < tail_rows) b.tail[at * wide + e] else b.nrow[e]);
            }
            const cv = bf(acc);
            out[e] = bf(b.gated[e] + bf(cv / (1.0 + @exp(-cv))));
        }
        std.mem.copyForwards(f32, b.tail[0 .. (tail_rows - 1) * wide], b.tail[wide..]);
        @memcpy(b.tail[(tail_rows - 1) * wide ..], b.nrow);
    }
};

/// model.py `PLELayer` in float64 with no intermediate rounding (emb as decoded fp16, weights as stored).
const Oracle = struct {
    w: Weights,
    tail: []f64,
    keys: []f64,
    vals: []f64,
    gated: []f64,
    normed: []f64,

    fn init(gpa: std.mem.Allocator, w: Weights) !Oracle {
        const tail = try gpa.alloc(f64, tail_rows * wide);
        @memset(tail, 0);
        return .{ .w = w, .tail = tail, .keys = try gpa.alloc(f64, wide), .vals = try gpa.alloc(f64, hidden), .gated = try gpa.alloc(f64, wide), .normed = try gpa.alloc(f64, wide) };
    }

    fn centred(x: []const f64, scale: []const f32, out: []f64) void {
        var ss: f64 = 0;
        for (x) |v| ss += v * v;
        const r = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + @as(f64, eps));
        for (out, x, scale) |*o, v, s| o.* = v * r * s;
    }

    fn step(o: *Oracle, emb: []const f32, h: []const f32, out: []f64) void {
        const w = o.w;
        for (0..wide) |r| {
            var acc: f64 = 0;
            for (w.key[r * hidden ..][0..hidden], emb) |a, b| acc += @as(f64, a) * b;
            o.keys[r] = acc;
        }
        for (0..hidden) |r| {
            var acc: f64 = 0;
            for (w.value[r * hidden ..][0..hidden], emb) |a, b| acc += @as(f64, a) * b;
            o.vals[r] = acc;
        }
        var kn: [hidden]f64 = undefined;
        var qn: [hidden]f64 = undefined;
        var hs: [hidden]f64 = undefined;
        for (0..streams) |s| {
            centred(o.keys[s * hidden ..][0..hidden], w.norm_key[s * hidden ..][0..hidden], &kn);
            for (&hs, h[s * hidden ..][0..hidden]) |*d, v| d.* = v;
            centred(&hs, w.norm_query[s * hidden ..][0..hidden], &qn);
            var dot: f64 = 0;
            for (kn, qn) |a, b| dot += a * b;
            var gate = dot / @sqrt(@as(f64, hidden));
            const sgn: f64 = if (gate > 0) 1 else if (gate < 0) -1 else 0;
            gate = sgn * @sqrt(@max(@abs(gate), 1e-6));
            const sg = 1.0 / (1.0 + @exp(-gate));
            for (0..hidden) |i| o.gated[s * hidden + i] = sg * o.vals[i];
            centred(o.gated[s * hidden ..][0..hidden], w.norm_conv[s * hidden ..][0..hidden], o.normed[s * hidden ..][0..hidden]);
        }
        for (0..wide) |e| {
            var acc: f64 = 0;
            for (0..taps) |t| {
                const at = t * dilation;
                acc += @as(f64, w.conv[e * taps + t]) * (if (at < tail_rows) o.tail[at * wide + e] else o.normed[e]);
            }
            out[e] = o.gated[e] + acc / (1.0 + @exp(-acc));
        }
        std.mem.copyForwards(f64, o.tail[0 .. (tail_rows - 1) * wide], o.tail[wide..]);
        @memcpy(o.tail[(tail_rows - 1) * wide ..], o.normed);
    }
};

const Qbench = struct { rows: []const struct { input_ids: []const u32 } };

fn crc(bytes: []const u8) u32 {
    return std.hash.Crc32.hash(bytes);
}

fn bytesOf(comptime T: type, v: []const T) []const u8 {
    return std.mem.sliceAsBytes(v);
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    const dir = rest[0];
    const qrow: usize = if (rest.len > 1) try std.fmt.parseInt(usize, rest[1], 10) else 3;
    const count: usize = if (rest.len > 2) try std.fmt.parseInt(usize, rest[2], 10) else 24;
    var arena_state = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = gpu.io;
    const rss_start = showRss("start");

    const qpath = try std.fs.path.join(a, &.{ dir, "qbench_prompts.json" });
    const qtext = try std.Io.Dir.cwd().readFileAlloc(io, qpath, a, .limited(1 << 26));
    const parsed = try std.json.parseFromSlice(Qbench, a, qtext, .{ .ignore_unknown_fields = true });
    const qrows = parsed.value.rows;
    try check.expect(qrow < qrows.len and count <= qrows[qrow].input_ids.len, "qbench row {d} with {d} tokens", .{ qrow, count });
    const tokens = qrows[qrow].input_ids[0..count];

    // ---- the table through upstream's Ngram (mapping) -----------------------------------------------------------
    var g = try ple.Ngram.init(gpu.gpa, io, dir);
    defer g.deinit();
    try check.expect(g.format == .exl3, "Ngram.init took the EXL3 table", .{});
    const h = g.hash;
    const heads = h.heads;
    const dh = h.dh;
    std.debug.print("hash: n {d}, heads {d} x {d}, eos {d}, {d} shards, {d} rows; table {d}-bit, {d} bytes a row, read {t}, emb {t}\n", .{ h.n, heads, dh, h.eos, h.shards, h.starts[h.shards], g.x3.?.bits, g.x3.?.rowBytes(), g.x3.?.opts.read, g.x3.?.opts.emb });
    check.pass("Ngram.init on the pack: config-derived multipliers, head sizes and offsets equal the pack's shipped tensors; 128 shard row counts equal Hash.starts", .{});
    const rss_open = showRss("after Ngram.init (30.4 GiB table mapped, header only)");

    // ---- (1) ids ---------------------------------------------------------------------------------------------------
    const ids = try a.alloc(i64, count * heads);
    {
        var hist = [_]i64{ h.eos, h.eos };
        for (tokens, 0..) |tok, t| {
            try ple.lastIds(h, &hist, tok, ids[t * heads ..][0..heads]);
            hist = .{ hist[1], tok };
        }
        const got = crc(bytesOf(i64, ids));
        std.debug.print("ids[0][0..4] = {any}; ids crc32 {x:0>8} (numpy NGram.ids {x:0>8})\n", .{ ids[0..4], got, py_ids_crc });
        if (qrow == 3 and count == 24) try check.expect(got == py_ids_crc, "upstream lastIds crc32 {x:0>8} != numpy {x:0>8}", .{ got, py_ids_crc });
        // every token of every qbench prompt: upstream's lastIds against the branch's incremental ids.
        var compared: usize = 0;
        var eos_seen: usize = 0;
        for (qrows) |r| {
            const ours = try gpu.gpa.alloc([16]u64, r.input_ids.len);
            defer gpu.gpa.free(ours);
            oursIds(h, .{ h.eos, h.eos }, r.input_ids, ours);
            var hist2 = [_]i64{ h.eos, h.eos };
            for (r.input_ids, ours) |tok, want| {
                if (tok == h.eos) eos_seen += 1;
                var up: [16]i64 = undefined;
                try ple.lastIds(h, &hist2, tok, up[0..heads]);
                hist2 = .{ hist2[1], tok };
                for (up[0..heads], want[0..heads]) |u, o| {
                    if (u < 0 or @as(u64, @intCast(u)) != o) {
                        std.debug.print("FAIL ids differ at token {d}: upstream {any} ours {any}\n", .{ tok, up[0..heads], want[0..heads] });
                        return error.TestFailed;
                    }
                }
                compared += 1;
            }
        }
        check.pass("row ids: upstream Hash/lastIds byte-equal to the numpy NGram.ids crc{s} and to the branch's exl3_ple_ids on all {d} tokens of {d} qbench prompts ({d} eos inside)", .{ if (qrow == 3 and count == 24) "" else " (crc not checked: not row 3 x 24)", compared, qrows.len, eos_seen });
    }

    // ---- (2) raw rows, two readers; decoded rows through Ngram.rows, two readers ------------------------------------
    var pread_t = (try x3.Table.open(gpu.gpa, io, dir, h, .{ .read = .pread })).?;
    defer pread_t.destroy();
    const rb = g.x3.?.rowBytes();
    const raw_map = try a.alloc(u8, ids.len * rb);
    const raw_ssd = try a.alloc(u8, ids.len * rb);
    for (ids, 0..) |id, i| {
        const loc = try h.locate(id);
        try g.x3.?.raw(loc.shard, loc.local, raw_map[i * rb ..][0..rb]);
        try pread_t.raw(loc.shard, loc.local, raw_ssd[i * rb ..][0..rb]);
    }
    try check.sameBytes("rows: mapping vs pread", raw_ssd, raw_map);
    const raw_crc = crc(raw_map);
    std.debug.print("{d} rows x {d} bytes: raw crc32 {x:0>8} (numpy {x:0>8})\n", .{ ids.len, rb, raw_crc, py_raw_crc });
    if (qrow == 3 and count == 24) try check.expect(raw_crc == py_raw_crc, "raw rows crc32 {x:0>8} != numpy {x:0>8}", .{ raw_crc, py_raw_crc });
    const rss_map = showRss("after the mapped row reads");

    const w = g.width();
    const emb_map = try a.alloc(u16, count * w);
    const xs_map = try a.alloc(f32, count * w / 32);
    {
        var hist = [_]i64{ h.eos, h.eos };
        try g.rows(&hist, tokens, emb_map, xs_map);
    }
    const dec_crc = crc(bytesOf(u16, emb_map));
    var peak_emb: f32 = 0;
    for (emb_map) |v| peak_emb = @max(peak_emb, @abs(@as(f32, @floatCast(@as(f16, @bitCast(v))))));
    std.debug.print("Ngram.rows: {d} fp16 values, max |value| {d:.4}, crc32 {x:0>8} (numpy _ple_rows {x:0>8})\n", .{ emb_map.len, peak_emb, dec_crc, py_decoded_crc });
    if (qrow == 3 and count == 24) try check.expect(dec_crc == py_decoded_crc, "decoded crc32 {x:0>8} != numpy {x:0>8}", .{ dec_crc, py_decoded_crc });
    // the same call with the pread reader swapped in behind Ngram
    const emb_ssd = try a.alloc(u16, count * w);
    const xs_ssd = try a.alloc(f32, count * w / 32);
    {
        const mapped_t = g.x3.?;
        g.x3 = pread_t;
        defer g.x3 = mapped_t;
        var hist = [_]i64{ h.eos, h.eos };
        try g.rows(&hist, tokens, emb_ssd, xs_ssd);
    }
    try check.sameBytes("Ngram.rows: mapping vs pread", bytesOf(u16, emb_ssd), bytesOf(u16, emb_map));
    try check.sameBytes("Ngram.rows group sums: mapping vs pread", bytesOf(f32, xs_ssd), bytesOf(f32, xs_map));
    check.pass("rows: {d} ids, stored bytes and Ngram.rows fp16 values byte-identical through the mapping and through pread + hot cache{s}", .{ ids.len, if (qrow == 3 and count == 24) "; raw and decoded crc32 equal the numpy references" else "" });

    // ---- (3) hot cache over real text; RSS -------------------------------------------------------------------------
    {
        var fresh = (try x3.Table.open(gpu.gpa, io, dir, h, .{ .read = .pread })).?;
        defer fresh.destroy();
        const r0 = rss();
        var buf: [256]u8 = undefined;
        var ntok: usize = 0;
        for (qrows) |r| {
            var hist = [_]i64{ h.eos, h.eos };
            for (r.input_ids) |tok| {
                var row_ids: [16]i64 = undefined;
                try ple.lastIds(h, &hist, tok, row_ids[0..heads]);
                hist = .{ hist[1], tok };
                for (row_ids[0..heads]) |id| {
                    const loc = try h.locate(id);
                    try fresh.raw(loc.shard, loc.local, buf[0..rb]);
                }
                ntok += 1;
            }
        }
        const s = fresh.hotStats();
        const r1 = rss();
        std.debug.print("pread + 4096-row hot cache over all {d} qbench tokens: {d} lookups, {d} hits ({d:.2}%); RSS file-backed {d} -> {d} MiB, anon {d} -> {d} MiB\n", .{ ntok, s.lookups, s.lookups - s.misses, 100.0 * @as(f64, @floatFromInt(s.lookups - s.misses)) / @as(f64, @floatFromInt(s.lookups)), r0.file >> 10, r1.file >> 10, r0.anon >> 10, r1.anon >> 10 });
        try check.expect(r1.file <= r0.file + 16 * 1024, "pread reads grew file-backed RSS by {d} MiB", .{(r1.file - r0.file) >> 10});
    }
    const rss_rows = showRss("after rows (both readers) and the hot-cache pass");
    const table_mib = g.x3.?.file.?.map.memory.len >> 20;
    std.debug.print("table {d} MiB; file-backed RSS: start {d}, after open {d}, after mapped reads {d}, now {d} MiB\n", .{ table_mib, rss_start.file >> 10, rss_open.file >> 10, rss_map.file >> 10, rss_rows.file >> 10 });
    try check.expect(rss_map.file <= rss_open.file + 128 * 1024 and rss_rows.file <= 512 * 1024, "the table became resident", .{});
    check.pass("30.4 GiB table stays out of memory: file-backed RSS {d} MiB after the mapped reads (+{d} MiB), anon {d} MiB", .{ rss_rows.file >> 10, (rss_map.file - @min(rss_map.file, rss_open.file)) >> 10, rss_rows.anon >> 10 });

    // ---- (4) layer 1's injection through upstream's fn_ple_gate / fn_ple_conv ---------------------------------------
    try injection(gpu, a, dir, tokens, emb_map, h.eos);
}

fn injection(gpu: Gpu, a: std.mem.Allocator, dir: []const u8, tokens: []const u32, emb_bits: []const u16, eos: i64) !void {
    _ = eos;
    const io = gpu.io;
    const n = tokens.len;
    var index = try Index.open(gpu.gpa, io, dir);
    defer index.deinit(gpu.gpa, io);
    const key_t = try index.get(gpu.gpa, io, prefix ++ "key_proj.weight");
    const val_t = try index.get(gpu.gpa, io, prefix ++ "value_proj.weight");
    const conv_t = try index.get(gpu.gpa, io, prefix ++ "conv1d.weight");
    try check.expect(key_t.is(.f16, &.{ wide, hidden }) and val_t.is(.f16, &.{ hidden, hidden }) and conv_t.is(.f16, &.{ wide, 1, taps }), "PLE key/value/conv dtypes or shapes (want F16)", .{});
    var norms: [3][]f32 = undefined;
    for ([_][]const u8{ "norm_key", "norm_query", "norm_conv" }, 0..) |nm, i| {
        var nb: [96]u8 = undefined;
        const t = try index.get(gpu.gpa, io, try std.fmt.bufPrint(&nb, prefix ++ "{s}.weight", .{nm}));
        try check.expect(t.is(.bf16, &.{wide}), "{s} dtype/shape (want bf16 [10240])", .{nm});
        const raw = try widen(a, t, 0);
        var mean: f64 = 0;
        for (raw) |v| mean += v;
        mean /= wide;
        std.debug.print("{s} stored mean {d:.4} (gamma - 1 storage: scale 1 + w)\n", .{ nm, mean });
        try check.expect(@abs(mean) < 0.5, "{s} does not look gamma - 1", .{nm});
        norms[i] = try widen(a, t, 1.0);
    }
    const weights: Weights = .{ .key = try widen(a, key_t, 0), .value = try widen(a, val_t, 0), .norm_key = norms[0], .norm_query = norms[1], .norm_conv = norms[2], .conv = try widen(a, conv_t, 0) };
    // what upstream's loader (cuda_prompt_load.zig with_ple) would upload as CW: the raw bytes, read as bf16
    var raw_as_bf: usize = 0;
    for (0..weights.conv.len) |i| {
        const bits = std.mem.readInt(u16, conv_t.bytes[2 * i ..][0..2], .little);
        if (bfVal(bits) != bf(weights.conv[i])) raw_as_bf += 1;
    }
    std.debug.print("conv1d.weight is F16: {d} of {d} values change if its bytes are read as bf16 (upstream's loader does), so it is converted (RNE) as exl3.py does\n", .{ raw_as_bf, weights.conv.len });

    // inputs: the decoded rows (fp16), the layer input stand-in (token embedding tiled over 4 streams, bf16)
    const embed_t = try index.get(gpu.gpa, io, "model.language_model.embed_tokens.weight");
    try check.expect(embed_t.dtype == .bf16 and embed_t.dim(1) == hidden, "embed_tokens", .{});
    const emb = try a.alloc(f32, n * hidden);
    for (emb, emb_bits) |*e, b| e.* = @floatCast(@as(f16, @bitCast(b)));
    const hq = try a.alloc(f32, n * wide);
    const hq_bits = try a.alloc(u16, n * wide);
    for (tokens, 0..) |tok, t| for (0..streams) |s| for (0..hidden) |d| {
        const bits = std.mem.readInt(u16, embed_t.bytes[(@as(usize, tok) * hidden + d) * 2 ..][0..2], .little);
        hq_bits[t * wide + s * hidden + d] = bits;
        hq[t * wide + s * hidden + d] = bfVal(bits);
    };
    // the projections the reference's F16 dense stores (bf16)
    const keys = try a.alloc(f32, n * wide);
    const vals = try a.alloc(f32, n * hidden);
    for (0..n) |t| project(weights, emb[t * hidden ..][0..hidden], keys[t * wide ..][0..wide], vals[t * hidden ..][0..hidden]);
    const keys_bits = try a.alloc(u16, keys.len);
    for (keys_bits, keys) |*b, v| b.* = bfBits(v);
    const vals_bits = try a.alloc(u16, vals.len);
    for (vals_bits, vals) |*b, v| b.* = bfBits(v);
    const cw_bits = try a.alloc(u16, weights.conv.len);
    for (cw_bits, weights.conv) |*b, v| b.* = bfBits(v);

    // host references
    const want = try a.alloc(f64, n * wide);
    const emu = try a.alloc(f32, n * wide);
    {
        var oracle = try Oracle.init(a, weights);
        var branch = try Branch.init(a, weights);
        for (0..n) |t| {
            oracle.step(emb[t * hidden ..][0..hidden], hq[t * wide ..][0..wide], want[t * wide ..][0..wide]);
            branch.step(keys[t * wide ..][0..wide], vals[t * hidden ..][0..hidden], hq[t * wide ..][0..wide], emu[t * wide ..][0..wide]);
        }
    }

    // the device: upstream's glue kernels
    var glue = try flash.Glue.load(gpu.d);
    defer glue.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    const D = cuda.DeviceBuffer;
    var d_keys = try D.fromHost(gpu.d, bytesOf(u16, keys_bits));
    defer d_keys.free();
    var d_vals = try D.fromHost(gpu.d, bytesOf(u16, vals_bits));
    defer d_vals.free();
    var d_h = try D.fromHost(gpu.d, bytesOf(u16, hq_bits));
    defer d_h.free();
    var d_nk = try D.fromHost(gpu.d, bytesOf(f32, weights.norm_key));
    defer d_nk.free();
    var d_nq = try D.fromHost(gpu.d, bytesOf(f32, weights.norm_query));
    defer d_nq.free();
    var d_nc = try D.fromHost(gpu.d, bytesOf(f32, weights.norm_conv));
    defer d_nc.free();
    var d_cw = try D.fromHost(gpu.d, bytesOf(u16, cw_bits));
    defer d_cw.free();
    const zeros = try a.alloc(u8, n * wide * 2);
    @memset(zeros, 0);
    var d_zero = try D.fromHost(gpu.d, zeros); // conv's H: zero, so HOUT is the injection itself
    defer d_zero.free();
    var d_tail = try D.fromHost(gpu.d, zeros[0 .. tail_rows * wide * 2]);
    defer d_tail.free();
    var d_gated = try D.alloc(gpu.d, n * wide * 2);
    defer d_gated.free();
    var d_pss = try D.alloc(gpu.d, n * streams * 4);
    defer d_pss.free();
    var d_out = try D.alloc(gpu.d, n * wide * 2);
    defer d_out.free();
    var d_nrow = try D.alloc(gpu.d, n * wide * 2);
    defer d_nrow.free();
    try gpu.ctx.synchronize();

    // whole prompt: one launch each, the conv reading its own earlier rows
    try glue.pleGate(s, d_keys.ptr, d_vals.ptr, d_h.ptr, d_nk.ptr, d_nq.ptr, d_gated.ptr, d_pss.ptr, eps, n, hidden, streams);
    try glue.pleConv(s, d_gated.ptr, d_pss.ptr, d_nc.ptr, d_tail.ptr, d_cw.ptr, d_zero.ptr, d_out.ptr, d_nrow.ptr, eps, n, hidden, streams, taps, dilation);
    try s.synchronize();
    const prompt_bits = try a.alloc(u16, n * wide);
    try d_out.download(0, std.mem.sliceAsBytes(prompt_bits));

    // token by token: the tail carried on the host from each launch's NROW, as shiftWindow keeps it
    const tail = try a.alloc(u16, tail_rows * wide);
    @memset(tail, 0);
    const decode_bits = try a.alloc(u16, n * wide);
    const nrow = try a.alloc(u16, wide);
    for (0..n) |t| {
        try d_tail.upload(0, bytesOf(u16, tail));
        try gpu.ctx.synchronize();
        try glue.pleGate(s, d_keys.ptr + t * wide * 2, d_vals.ptr + t * hidden * 2, d_h.ptr + t * wide * 2, d_nk.ptr, d_nq.ptr, d_gated.ptr, d_pss.ptr, eps, 1, hidden, streams);
        try glue.pleConv(s, d_gated.ptr, d_pss.ptr, d_nc.ptr, d_tail.ptr, d_cw.ptr, d_zero.ptr, d_out.ptr, d_nrow.ptr, eps, 1, hidden, streams, taps, dilation);
        try s.synchronize();
        try d_out.download(0, std.mem.sliceAsBytes(decode_bits[t * wide ..][0..wide]));
        try d_nrow.download(0, std.mem.sliceAsBytes(nrow));
        std.mem.copyForwards(u16, tail[0 .. (tail_rows - 1) * wide], tail[wide..]);
        @memcpy(tail[(tail_rows - 1) * wide ..], nrow);
    }

    var worst: f64 = 0;
    var peak: f64 = 0;
    var worst_rel_rms: f64 = 0;
    var emu_worst: f64 = 0;
    var emu_off: usize = 0;
    var emu_vs_oracle: f64 = 0;
    for (0..n) |t| {
        var se: f64 = 0;
        var sw: f64 = 0;
        for (prompt_bits[t * wide ..][0..wide], want[t * wide ..][0..wide], emu[t * wide ..][0..wide]) |gb, o, e| {
            const gv: f64 = bfVal(gb);
            const err = gv - o;
            worst = @max(worst, @abs(err));
            peak = @max(peak, @abs(o));
            se += err * err;
            sw += o * o;
            emu_worst = @max(emu_worst, @abs(gv - e));
            if (gv != e) emu_off += 1;
            emu_vs_oracle = @max(emu_vs_oracle, @abs(@as(f64, e) - o));
        }
        worst_rel_rms = @max(worst_rel_rms, @sqrt(se / sw));
    }
    const same_modes = std.mem.eql(u16, prompt_bits, decode_bits);
    std.debug.print("injection over {d} tokens: GPU max |gpu - f64| {e:.3} at peak {e:.3} ({d:.3}% of peak), worst per-token rms rel {e:.3}\n", .{ n, worst, peak, 100 * worst / peak, worst_rel_rms });
    std.debug.print("  GPU vs the reference's roundings emulated on the host: max |diff| {e:.3}, {d} of {d} values differ; emulation vs f64 max {e:.3}\n", .{ emu_worst, emu_off, n * wide, emu_vs_oracle });
    std.debug.print("  whole-prompt launch vs token-by-token launches: {s}\n", .{if (same_modes) "bit-identical" else "DIFFER"});
    try check.expect(peak > 0 and worst <= 0.03 * peak and worst_rel_rms <= 0.02, "injection beyond bf16 rounding of the f64 oracle", .{});
    try check.expect(same_modes, "prompt and decode launches differ", .{});
    check.pass("layer-1 PLE injection through upstream's fn_ple_gate/fn_ple_conv on real EXL3 weights: max |err| {e:.2} <= 3% of peak {e:.2}, rms rel {e:.2} <= 2e-2; prompt == decode bits", .{ worst, peak, worst_rel_rms });
}
