//! GLM EXL3: admission, trellis widths, the loader's uploads (mixed widths, 4-bit bytes as upstream's) and scratch.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const x3 = @import("cuda_exl3.zig");
const modelopt = core.modelopt;
const Tensor = core.checkpoint.Tensor;
const Span = @import("cuda_shard.zig").Span;
const testing = std.testing;

test "GLM EXL3: Mia-AiLab's 4-bit block and a mixed block without bits pass; other widths and codebooks are refused by name" {
    const gpa = testing.allocator;
    const mia =
        \\{"quantization_config": {"quant_method": "exl3", "bits": 4, "codebook": "mcg", "head_bits": 16}}
    ;
    try testing.expect(x3.isExl3(gpa, mia));
    var why: modelopt.Why = .{};
    try x3.admit(gpa, mia, &why);
    const mixed = // the 3.5 bpw checkpoint's: no "bits", its average instead
        \\{"quantization_config": {"quant_method": "exl3", "avg_bits": 3.5039, "codebook": "mcg", "head_bits": 16, "scope": "glm53_routed_experts_only"}}
    ;
    try testing.expect(x3.isExl3(gpa, mixed));
    try x3.admit(gpa, mixed, &why);
    try x3.admit(gpa, "{\"quantization_config\": {\"quant_method\": \"exl3\", \"bits\": 3, \"codebook\": \"mcg\"}}", &why);
    try x3.admit(gpa, "{\"quantization_config\": {\"quant_method\": \"exl3\", \"bits\": 8.0, \"codebook\": \"mcg\"}}", &why);
    try x3.admit(gpa, "{\"quantization_config\": {\"quant_method\": \"exl3\", \"codebook\": \"mcg\"}}", &why);
    try testing.expect(!x3.isExl3(gpa, "{\"quantization_config\": {\"quant_method\": \"modelopt\"}}"));
    try testing.expect(!x3.isExl3(gpa, "{}"));
    try testing.expectError(error.UnsupportedQuant, x3.admit(gpa, "{\"quantization_config\": {\"quant_method\": \"exl3\", \"bits\": 9, \"codebook\": \"mcg\"}}", &why));
    try testing.expectEqualStrings("EXL3 at 9 bits: the CUDA family reads mcg experts of 1 to 8 whole bits", why.text());
    try testing.expectError(error.UnsupportedQuant, x3.admit(gpa, "{\"quantization_config\": {\"quant_method\": \"exl3\", \"bits\": 3.5, \"codebook\": \"mcg\"}}", &why));
    try testing.expectEqualStrings("EXL3 at 3.5 bits: the CUDA family reads mcg experts of 1 to 8 whole bits", why.text());
    try testing.expectError(error.UnsupportedQuant, x3.admit(gpa, "{\"quantization_config\": {\"quant_method\": \"exl3\", \"bits\": \"4\", \"codebook\": \"mcg\"}}", &why));
    try testing.expectError(error.UnsupportedQuant, x3.admit(gpa, "{\"quantization_config\": {\"quant_method\": \"exl3\", \"avg_bits\": 0.5, \"codebook\": \"mcg\"}}", &why));
    try testing.expectEqualStrings("EXL3 averaging 0.5 bits: the CUDA family reads mcg experts of 1 to 8 bits", why.text());
    try testing.expectError(error.UnsupportedQuant, x3.admit(gpa, "{\"text_config\": {\"quantization_config\": {\"quant_method\": \"exl3\", \"bits\": 4, \"codebook\": \"mul1\"}}}", &why));
    try testing.expectEqualStrings("EXL3 codebook \"mul1\": the CUDA family reads \"mcg\"", why.text());
    try testing.expectError(error.UnsupportedQuant, x3.admit(gpa, "{\"quantization_config\": {\"quant_method\": \"exl3\", \"avg_bits\": 3.5}}", &why));
    try testing.expectEqualStrings("EXL3 codebook null: the CUDA family reads \"mcg\"", why.text());
}

test "GLM EXL3: a trellis's last dimension gives whole-bit mcg widths from 1 to 8; half steps and wider are refused" {
    for ([_][2]usize{ .{ 16, 2 }, .{ 32, 4 }, .{ 48, 6 }, .{ 64, 8 }, .{ 80, 10 }, .{ 96, 12 }, .{ 112, 14 }, .{ 128, 16 } }) |c|
        try testing.expectEqual(@as(?u32, @as(u32, @intCast(c[1]))), x3.halves(c[0]));
    for ([_]usize{ 0, 8, 24, 40, 56, 50, 136, 144 }) |last| try testing.expectEqual(@as(?u32, null), x3.halves(last));
    try testing.expectEqual(@as(usize, 128), x3.tileBytes(8));
    try testing.expectEqual(@as(usize, 96), x3.tileBytes(6));
}

/// A checkpoint in memory and device uploads recorded on the host, as cuda_weights' loader is to `load`.
const Mock = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    tensors: std.StringHashMapUnmanaged(Tensor) = .empty,
    host: std.ArrayList(u8) = .empty,
    puts: std.ArrayList(Put) = .empty,
    name_buf: [192]u8 = undefined,
    last: []const u8 = "",
    next: u64 = 0x7f00_0000_0000,

    const Put = struct { name: []const u8, bytes: []const u8, ptr: u64 };

    fn init(gpa: std.mem.Allocator) Mock {
        return .{ .gpa = gpa, .arena = .init(gpa) };
    }
    fn deinit(m: *Mock) void {
        m.host.deinit(m.gpa);
        m.arena.deinit();
    }
    pub fn get(m: *Mock, comptime fmt: []const u8, args: anytype) !Tensor {
        m.last = try std.fmt.bufPrint(&m.name_buf, fmt, args);
        return m.tensors.get(m.last) orelse error.MissingTensor;
    }
    pub fn put(m: *Mock, comptime fmt: []const u8, args: anytype) !u64 {
        const a = m.arena.allocator();
        const ptr = m.next;
        m.next += std.mem.alignForward(u64, m.host.items.len + 1, 1 << 20);
        try m.puts.append(a, .{ .name = try std.fmt.allocPrint(a, fmt, args), .bytes = try a.dupe(u8, m.host.items), .ptr = ptr });
        m.host.clearRetainingCapacity();
        return ptr;
    }
    fn add(m: *Mock, name: []const u8, dtype: core.checkpoint.DType, shape: []const usize, bytes: []const u8) !void {
        const a = m.arena.allocator();
        var t: Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(0), .bytes = try a.dupe(u8, bytes) };
        @memcpy(t.shape[0..shape.len], shape);
        try m.tensors.put(a, try a.dupe(u8, name), t);
    }
    fn find(m: *const Mock, name: []const u8) ?Put {
        for (m.puts.items) |p| if (std.mem.eql(u8, p.name, name)) return p;
        return null;
    }
};

const projs = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };

/// Layer `i`'s experts: p of e at `k2s[p][e]` half-bits, random mcg words and scales; `shared`: one input scale.
fn synth(m: *Mock, rng: std.Random, i: u32, count: usize, dims: usize, inter: usize, k2s: [3][]const u32, shared: bool) !void {
    const a = m.arena.allocator();
    const rot = try a.alloc(u8, dims * 2);
    rng.bytes(rot);
    for (projs, 0..) |proj, p| {
        const k = if (p < 2) dims else inter;
        const n = if (p < 2) inter else dims;
        for (0..count) |e| {
            var buf: [128]u8 = undefined;
            const k2 = k2s[p][e];
            const words = try a.alloc(u8, k / 16 * (n / 16) * 16 * k2);
            rng.bytes(words);
            try m.add(try std.fmt.bufPrint(&buf, "layers.{d}.mlp.experts.{d}.{s}.trellis", .{ i, e, proj }), .i16, &.{ k / 16, n / 16, 8 * k2 }, words);
            var marker: [4]u8 = undefined;
            std.mem.writeInt(u32, &marker, core.exl3_load.markers[0].value, .little);
            try m.add(try std.fmt.bufPrint(&buf, "layers.{d}.mlp.experts.{d}.{s}.mcg", .{ i, e, proj }), .i32, &.{1}, &marker);
            const suh = try a.alloc(u8, k * 2);
            if (shared and p < 2) @memcpy(suh, rot) else rng.bytes(suh);
            try m.add(try std.fmt.bufPrint(&buf, "layers.{d}.mlp.experts.{d}.{s}.suh", .{ i, e, proj }), .f16, &.{k}, suh);
            const svh = try a.alloc(u8, n * 2);
            rng.bytes(svh);
            try m.add(try std.fmt.bufPrint(&buf, "layers.{d}.mlp.experts.{d}.{s}.svh", .{ i, e, proj }), .f16, &.{n}, svh);
        }
    }
}

/// The rank's tiles of projection `p`, tile by tile (gate/up: every tile row, `cut`'s cols; down: `cut`'s rows).
fn expectedTiles(m: *Mock, i: u32, count: usize, dims: usize, inter: usize, cut: Span, p: usize) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    const k = if (p < 2) dims else inter;
    const n = if (p < 2) inter else dims;
    for (0..count) |e| {
        var buf: [128]u8 = undefined;
        const t = m.tensors.get(try std.fmt.bufPrint(&buf, "layers.{d}.mlp.experts.{d}.{s}.trellis", .{ i, e, projs[p] })).?;
        const tile = t.dim(2) * 2;
        const rows: [2]usize = if (p < 2) .{ 0, k / 16 } else .{ cut.lo / 16, cut.hi / 16 };
        const cols: [2]usize = if (p < 2) .{ cut.lo / 16, cut.hi / 16 } else .{ 0, n / 16 };
        for (rows[0]..rows[1]) |r| for (cols[0]..cols[1]) |c|
            try out.appendSlice(m.arena.allocator(), t.bytes[(r * (n / 16) + c) * tile ..][0..tile]);
    }
    return out.items;
}

fn ptrsOf(bytes: []const u8) []align(1) const u64 {
    return std.mem.bytesAsSlice(u64, bytes);
}

/// Every projection's tiles, pointers and width table as `load` uploaded them, vs the checkpoint, tile by tile.
fn expectUploads(m: *Mock, w: x3.Experts, i: u32, count: usize, dims: usize, inter: usize, cut: Span, k2s: [3][]const u32) !void {
    for ([_][]const u8{ "gt", "ut", "dt" }, [_]u64{ w.gt, w.ut, w.dt }, 0..) |tw, ptrs_at, p| {
        var buf: [96]u8 = undefined;
        const words = m.find(try std.fmt.bufPrint(&buf, "layers.{d}.moe.exl3.{s}", .{ i, tw })).?;
        try testing.expectEqualSlices(u8, try expectedTiles(m, i, count, dims, inter, cut, p), words.bytes);
        const ptrs = m.find(try std.fmt.bufPrint(&buf, "layers.{d}.moe.exl3.{s}_ptrs", .{ i, tw })).?;
        try testing.expectEqual(ptrs.ptr, ptrs_at);
        const tiles = if (p < 2) dims / 16 * (cut.len() / 16) else cut.len() / 16 * (dims / 16);
        var at = words.ptr;
        for (ptrsOf(ptrs.bytes), 0..) |ptr, e| {
            try testing.expectEqual(at, ptr);
            at += tiles * 16 * k2s[p][e];
        }
        try testing.expectEqual(words.ptr + words.bytes.len, at);
        const table = for (m.puts.items) |put| {
            if (put.ptr == w.tables()[p]) break put;
        } else return error.NoTable;
        for (std.mem.bytesAsSlice(i32, table.bytes), k2s[p]) |got, want| try testing.expectEqual(@as(i32, @intCast(want)), got);
    }
}

test "GLM EXL3: a mixed layer (3-bit gate and up, 4-bit down) loads each width's tiles, pointers and tables on either rank" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(53);
    const count = 5;
    const dims = 256;
    const inter = 512;
    const gu: [count]u32 = @splat(6);
    const dn: [count]u32 = @splat(8);
    const k2s: [3][]const u32 = .{ &gu, &gu, &dn };
    for ([_]Span{ .{ .lo = 0, .hi = 256 }, .{ .lo = 256, .hi = 512 }, .{ .lo = 128, .hi = 384 } }) |cut| {
        var m = Mock.init(gpa);
        defer m.deinit();
        try synth(&m, prng.random(), 4, count, dims, inter, k2s, true);
        const w = try x3.load(&m, 4, count, dims, inter, cut);
        try expectUploads(&m, w, 4, count, dims, inter, cut, k2s);
        try testing.expectEqual(@as(u32, 6), w.uniform_gu);
        try testing.expectEqual(@as(u32, 8), w.uniform_d);
        try testing.expect(w.shx);
        try testing.expectEqual(w.k2, w.k2_up); // gate and up's widths agree: one table
        try testing.expect(w.k2 != w.k2_down);
        try testing.expect(m.find("layers.4.moe.exl3.k2_gate") != null and m.find("layers.4.moe.exl3.k2_down") != null);
        try testing.expect(m.find("layers.4.moe.exl3.k2_up") == null and m.find("layers.4.moe.exl3.k2") == null);
        const l = x3.layer(w, count, dims, cut.len(), 10);
        try testing.expectEqual(@as(u32, 0), l.uniform_k2); // gate/up and down differ: no one width
        try testing.expectEqual(@as(u32, 6), l.uniform_k2_gu);
        try testing.expectEqual(@as(u32, 8), l.uniform_k2_d);
        try testing.expect(l.gate_k2 == w.k2 and l.up_k2 == w.k2_up and l.down_k2 == w.k2_down);
    }
}

test "GLM EXL3: widths that differ by expert and projection keep their own offsets; no uniform width is claimed" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const count = 4;
    const g = [count]u32{ 6, 8, 2, 16 };
    const u = [count]u32{ 8, 8, 8, 8 };
    const d = [count]u32{ 4, 6, 8, 12 };
    const k2s: [3][]const u32 = .{ &g, &u, &d };
    var m = Mock.init(gpa);
    defer m.deinit();
    try synth(&m, prng.random(), 9, count, 256, 256, k2s, false);
    const cut: Span = .{ .lo = 128, .hi = 256 };
    const w = try x3.load(&m, 9, count, 256, 256, cut);
    try expectUploads(&m, w, 9, count, 256, 256, cut, k2s);
    try testing.expectEqual(@as(u32, 0), w.uniform_gu);
    try testing.expectEqual(@as(u32, 0), w.uniform_d);
    try testing.expect(!w.shx);
    try testing.expect(w.k2 != w.k2_up and w.k2_up != w.k2_down and w.k2 != w.k2_down);
    const l = x3.layer(w, count, 256, 128, 10);
    try testing.expect(l.uniform_k2 == 0 and l.uniform_k2_gu == 0 and l.uniform_k2_d == 0);
    try testing.expect(!cuda.exl3_experts.prompt.eligible(l));
}

test "GLM EXL3: a trellis at a width the kernels do not read, or of the wrong shape, is refused with its reason" {
    const bytes: [72 * 2 * 2 * 2]u8 = @splat(0);
    var t: Tensor = .{ .dtype = .i16, .rank = 3, .shape = .{ 2, 2, 40, 0, 0 }, .bytes = bytes[0 .. 2 * 2 * 40 * 2] };
    try testing.expect(x3.refusal(t, 32, 32) != null); // 2.5 bits: mul1's half steps
    t.shape[2] = 72; // 4.5 bits
    t.bytes = bytes[0 .. 2 * 2 * 72 * 2];
    try testing.expect(x3.refusal(t, 32, 32) != null);
    t.shape[2] = 64;
    t.bytes = bytes[0 .. 2 * 2 * 64 * 2];
    try testing.expectEqual(@as(?[]const u8, null), x3.refusal(t, 32, 32));
    try testing.expect(x3.refusal(t, 48, 32) != null); // tiles that do not cover K
    t.dtype = .i32;
    try testing.expect(x3.refusal(t, 32, 32) != null);
    t.dtype = .i16;
    t.bytes = bytes[0..10];
    try testing.expect(x3.refusal(t, 32, 32) != null);
}

/// Upstream's 4-bit loader (f350cd3e), verbatim but its result's type: a 4-bit layer uploads exactly these bytes.
const Upstream = struct { gt: u64, ut: u64, dt: u64, suh_g: u64, suh_u: u64, svh_g: u64, svh_u: u64, suh_d: u64, svh_d: u64, k2: u64, shx: bool };
fn upstream4(L: anytype, i: u32, count: usize, dims: usize, inter: usize, cut: Span) !Upstream {
    const tile = 128;
    if (dims % 128 != 0 or cut.lo % 128 != 0 or cut.hi % 128 != 0) return error.NotExl3Mcg4;
    const width = cut.len();
    var out: Upstream = undefined;
    out.shx = true;
    const Mat = struct { k: usize, n: usize, by_cols: bool };
    const mats = [_]Mat{
        .{ .k = dims, .n = inter, .by_cols = true },
        .{ .k = dims, .n = inter, .by_cols = true },
        .{ .k = inter, .n = dims, .by_cols = false },
    };
    var first_suh: ?[]const u8 = null;
    inline for (mats, .{ "gate_proj", "up_proj", "down_proj" }, .{ "gt", "ut", "dt" }, .{ "suh_g", "suh_u", "suh_d" }, .{ "svh_g", "svh_u", "svh_d" }) |m, proj, tw, su, sv| {
        const kt = (if (m.by_cols) m.k else width) / 16;
        const nt = (if (m.by_cols) width else m.n) / 16;
        for (0..count) |e| {
            const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".trellis", .{ i, e });
            if (t.dtype != .i16 or t.rank != 3 or t.dim(0) != m.k / 16 or t.dim(1) != m.n / 16 or t.dim(2) != 64) return error.NotExl3Mcg4;
            const marker = L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".mcg", .{ i, e }) catch return error.NotExl3Mcg4;
            if (marker.dtype != .i32 or marker.numel() != 1 or marker.bytes.len != 4 or
                std.mem.readInt(u32, marker.bytes[0..4], .little) != core.exl3_load.markers[0].value)
                return error.NotExl3Mcg4;
            if (m.by_cols) {
                const row = (m.n / 16) * tile;
                for (0..kt) |r| try L.host.appendSlice(L.gpa, t.bytes[r * row + (cut.lo / 16) * tile ..][0 .. nt * tile]);
            } else try L.host.appendSlice(L.gpa, t.bytes[(cut.lo / 16) * nt * tile ..][0 .. kt * nt * tile]);
        }
        const base = try L.put("layers.{d}.moe.exl3." ++ tw, .{i});
        for (0..count) |e| {
            const ptr = base + e * kt * nt * tile;
            try L.host.appendSlice(L.gpa, std.mem.asBytes(&ptr));
        }
        @field(out, tw) = try L.put("layers.{d}.moe.exl3." ++ tw ++ "_ptrs", .{i});
        for (0..count) |e| {
            const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".suh", .{ i, e });
            if (t.dtype != .f16 or t.numel() != m.k) return error.NotExl3Mcg4;
            const mine = if (m.by_cols) t.bytes else t.bytes[cut.lo * 2 .. cut.hi * 2];
            if (m.by_cols) {
                if (first_suh) |f| (if (!std.mem.eql(u8, f, mine)) {
                    out.shx = false;
                }) else first_suh = mine;
            }
            try L.host.appendSlice(L.gpa, mine);
        }
        @field(out, su) = try L.put("layers.{d}.moe.exl3." ++ su, .{i});
        for (0..count) |e| {
            const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".svh", .{ i, e });
            if (t.dtype != .f16 or t.numel() != m.n) return error.NotExl3Mcg4;
            try L.host.appendSlice(L.gpa, if (m.by_cols) t.bytes[cut.lo * 2 .. cut.hi * 2] else t.bytes);
        }
        @field(out, sv) = try L.put("layers.{d}.moe.exl3." ++ sv, .{i});
    }
    for (0..count) |_| {
        const bits: i32 = 8;
        try L.host.appendSlice(L.gpa, std.mem.asBytes(&bits));
    }
    out.k2 = try L.put("layers.{d}.moe.exl3.k2", .{i});
    return out;
}

test "GLM EXL3: a 4-bit layer uploads upstream's bytes, names and pointers, and keeps one width (8) for every path" {
    const gpa = testing.allocator;
    for ([_]bool{ true, false }, [_]Span{ .{ .lo = 256, .hi = 512 }, .{ .lo = 0, .hi = 512 } }) |shared, cut| {
        var prng = std.Random.DefaultPrng.init(4);
        const count = 6;
        const four: [count]u32 = @splat(8);
        var m = Mock.init(gpa);
        defer m.deinit();
        try synth(&m, prng.random(), 3, count, 256, 512, .{ &four, &four, &four }, shared);
        var old = Mock.init(gpa);
        defer old.deinit();
        old.tensors = m.tensors;
        const want = try upstream4(&old, 3, count, 256, 512, cut);
        const got = try x3.load(&m, 3, count, 256, 512, cut);
        try testing.expectEqual(old.puts.items.len, m.puts.items.len);
        for (old.puts.items, m.puts.items) |a, b| {
            try testing.expectEqualStrings(a.name, b.name);
            try testing.expectEqual(a.ptr, b.ptr);
            try testing.expectEqualSlices(u8, a.bytes, b.bytes);
        }
        inline for (.{ "gt", "ut", "dt", "suh_g", "suh_u", "svh_g", "svh_u", "suh_d", "svh_d", "k2", "shx" }) |f| try testing.expectEqual(@field(want, f), @field(got, f));
        try testing.expectEqual([3]u64{ want.k2, want.k2, want.k2 }, got.tables());
        try testing.expect(got.uniform_gu == 8 and got.uniform_d == 8);
        const l = x3.layer(got, count, 256, cut.len(), 10);
        try testing.expect(l.uniform_k2 == 8 and l.uniform_k2_gu == 8 and l.uniform_k2_d == 8);
        try testing.expect(l.shared_input == shared);
    }
}

test "GLM EXL3: scratch covers the path each layer takes: the prompt kernels' when they take it, else the grouped path's" {
    const dims = 4096;
    const width = 1024;
    const four: x3.Experts = .{ .gt = 0, .ut = 0, .dt = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .k2 = 0, .uniform_gu = 8, .uniform_d = 8, .shx = true };
    var mixed = four;
    mixed.uniform_gu = 6;
    var any = four;
    any.uniform_gu = 0;
    any.uniform_d = 0;
    const common = cuda.exl3_experts;
    for ([_]usize{ 1, 8, 200, 1024, 2048 }) |rows| {
        const n: usize = @min(rows, x3.chunk_rows);
        for ([_]x3.Experts{ four, mixed, any }) |w| for ([_]bool{ false, true }) |prefill| {
            const l = x3.layer(w, cuda.grouped.max_experts, dims, width, 0);
            const prompt = prefill and common.prompt.eligible(l);
            const need = x3.picksBytes(rows) + if (prompt) try common.prompt.bytes(l, n, x3.top_k) else try common.scratchBytes(l, n, x3.top_k);
            try testing.expectEqual(need, x3.scratchFor(rows, dims, width, prefill, w)[0]);
        };
        // a 4-bit layer's scratch is upstream's: picks, then the prompt kernels', or the grouped path's
        const upstream: common.Layer = .{ .gate = 0, .up = 0, .down = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .experts = cuda.grouped.max_experts, .dims = dims, .width = width, .codebook = .mcg, .act_mode = 0, .uniform_k2 = 8, .shared_input = true };
        try testing.expectEqual(x3.picksBytes(rows) + try common.prompt.bytes(upstream, n, x3.top_k), x3.scratchFor(rows, dims, width, true, four)[0]);
        try testing.expectEqual(x3.picksBytes(rows) + try common.scratchBytes(upstream, n, x3.top_k), x3.scratchFor(rows, dims, width, false, four)[0]);
        for ([_]bool{ false, true }) |prefill| try testing.expectEqual(x3.scratchFor(rows, dims, width, prefill, four), x3.scratchBytes(rows, dims, width, prefill, true));
        // buffers for every layer take the largest: a 4-bit layer's prompt arena grows to the mixed layer's grouped one
        var most: [4]usize = @splat(4);
        x3.mostScratch(&most, rows, dims, width, true, four);
        x3.mostScratch(&most, rows, dims, width, true, any);
        try testing.expectEqual(@max(x3.scratchFor(rows, dims, width, true, four)[0], x3.scratchFor(rows, dims, width, true, any)[0]), most[0]);
        try testing.expectEqual([3]usize{ 4, 4, 4 }, most[1..4].*);
    }
}
