//! Fused Q/K norm and RoPE against the four frozen T622 launches, comparing every downstream byte.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ct = @import("contract");
const rr = @import("rig.zig");
const q = tf.qwen27;
const at = q.attention;
const Ref = q.projection.Ref;
const D = 256;
const H = 24;
const HK = 4;
const KV = HK * D;
const Field = enum { nq, nk, query, key, out };
const Side = struct {
    buffers: [5]rr.Ref,
    cache: at.Cache,
    allocation: rr.Ref,
    binding: at.Binding,
    scratch: at.Scratch,
    fn get(s: *Side, f: Field) Ref {
        const r = s.buffers[@backingInt(f)];
        return .{ .buffer = r.buf, .offset = r.off };
    }
    fn deinit(s: *Side) void {
        s.binding.deinit();
        s.scratch.deinit();
    }
};
fn equal(name: []const u8, a: []const u8, b: []const u8) !void {
    if (std.mem.indexOfDiff(u8, a, b)) |i| {
        std.debug.print("QK_DIFF {s} byte={d} got={x} base={x}\n", .{ name, i, a[i], b[i] });
        return error.QkBytesDiffer;
    }
}
fn ref(r: rr.Ref) Ref {
    return .{ .buffer = r.buf, .offset = r.off };
}
fn bits(a: std.mem.Allocator, n: usize, rng: std.Random) ![]u16 {
    const out = try a.alloc(u16, n);
    for (out, 0..) |*v, i| v.* = if (i % 127 == 0) 0x8000 else if (i % 127 == 1) 1 else ct.spec.toBf16((rng.float(f32) * 2 - 1) * (if (i % 71 == 0) @as(f32, 20) else 1));
    return out;
}
const Case = struct { rows: u32, tree: bool = false, image: bool = false, prompt: bool = false };
const cases = [_]Case{ .{ .rows = 1 }, .{ .rows = 1, .tree = true, .image = true }, .{ .rows = 16 }, .{ .rows = 16, .tree = true }, .{ .rows = 16, .tree = true, .image = true }, .{ .rows = 2048, .prompt = true }, .{ .rows = 2048, .prompt = true, .image = true } };
fn runCase(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ops: [2]*at.Attention, units: bool, paged: bool, cs: Case, seed: u64) !void {
    var rig = rr.Rig{ .a = a, .device = device, .queue = queue };
    defer rig.deinit();
    const start: u32 = 519;
    const cap = start + cs.rows + 3;
    const rows = cs.rows;
    const geo = q.attention_plan.Geometry{ .shared_prefix = units };
    const limits = at.Limits{ .rows = rows, .streams = 1, .keys = cap, .shared_prefix = units, .geometry = geo };
    const parents = try a.alloc(i32, rows);
    defer a.free(parents);
    for (parents, 0..) |*p, i| p.* = if (i == 0) -1 else @intCast(if (cs.tree) (i - 1) / 2 else i - 1);
    var plan = try at.Plan.init(a, &.{.{ .parents = parents, .start = start, .capacity = cap }}, geo);
    defer plan.deinit();
    const qbytes = @as(usize, rows) * H * D * 2;
    const kvbytes = @as(usize, rows) * KV * 2;
    var rng_ = std.Random.DefaultPrng.init(seed);
    const rng = rng_.random();
    const qg_host = try bits(a, rows * H * D * 2, rng);
    defer a.free(qg_host);
    const mixed_host = try bits(a, rows * KV * 2, rng);
    defer a.free(mixed_host);
    const gains = try bits(a, 2 * D, rng);
    defer a.free(gains);
    const qg = try rig.input(std.mem.sliceAsBytes(qg_host));
    const mixed = try rig.input(std.mem.sliceAsBytes(mixed_host));
    const qgain = try rig.input(std.mem.sliceAsBytes(gains[0..D]));
    const kgain = try rig.input(std.mem.sliceAsBytes(gains[D..]));
    const positions = try a.alloc(i32, rows * 3);
    defer a.free(positions);
    for (0..3) |axis| for (0..rows) |row| {
        positions[axis * rows + row] = @intCast(start + row + (if (cs.image) axis * 1009 else 0));
    };
    const pos = try rig.input(std.mem.sliceAsBytes(positions));
    const flat_bytes = @as(usize, cap) * KV * 2;
    const count = at.pageCount(cap);
    const part = HK * 256 * D * 2;
    const page_bytes = 2 * part + 64;
    const ids = try a.alloc(u32, count);
    defer a.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(count - i - 1);
    const table = try rig.input(std.mem.sliceAsBytes(ids));
    var sides: [2]Side = undefined;
    var made: usize = 0;
    defer for (sides[0..made]) |*s| s.deinit();
    for (&sides, 0..) |*s, i| {
        const pages = paged and i == 1;
        const total = if (pages) count * page_bytes else flat_bytes * 2;
        s.allocation = try rig.scratch(total);
        @memset(rig.bytes(s.allocation, total), 0x3f);
        const atv = if (pages) part else flat_bytes;
        s.cache = .{ .keys = ref(s.allocation), .values = .{ .buffer = s.allocation.buf, .offset = s.allocation.off + atv }, .capacity = cap, .pages = if (pages) .{ .table = ref(table), .page_bytes = page_bytes } else null };
        for (&s.buffers, [_]usize{ qbytes, kvbytes, qbytes, kvbytes, qbytes }) |*r, n| r.* = try rig.scratch(n);
        s.binding = try at.Binding.init(device, limits);
        s.scratch = try at.Scratch.init(device, limits);
        made += 1;
        try s.binding.upload(plan, &.{s.cache});
        try s.scratch.upload(plan);
    }
    var saved: [2][5][]u8 = undefined;
    for (&saved) |*side| {
        for (side, [_]usize{ qbytes, kvbytes, qbytes, kvbytes, qbytes }) |*b, n| b.* = try a.alloc(u8, n);
    }
    defer for (saved) |side| for (side) |b| a.free(b);
    for (0..5) |repeat| {
        for (&sides, ops, 0..) |*s, op, i| {
            const pass = rig.begin();
            const e = pass.e;
            const p = at.Preprocess{ .qg = ref(qg), .mixed = ref(mixed), .q_gain = ref(qgain), .k_gain = ref(kgain), .norm_q = s.get(.nq), .norm_k = s.get(.nk), .query = s.get(.query), .rotated_key = s.get(.key), .positions = ref(pos), .rows = rows, .eps = 1e-6, .theta = 10000000, .h_end = if (cs.image) 12 else 0, .w_end = if (cs.image) 10 else 0, .plane = rows, .geometry = geo };
            if (i == 1 and repeat == 0) {
                var alias = p;
                alias.norm_k = ref(mixed);
                if (op.preprocess(e, alias)) return error.MixedAliasMissed else |err| if (err != error.AttentionAlias) return err;
                var short = p;
                short.mixed.offset = short.mixed.buffer.length();
                if (op.preprocess(e, short)) return error.MixedSpanMissed else |err| if (err != error.AttentionBuffer) return err;
            }
            if (i == 0) @import("q27_qk_reference.zig").encode(e, op.io[7], op.io[8], p) else try op.preprocess(e, p);
            try op.encode(e, &s.scratch, &s.binding, plan, s.get(.query), s.get(.key), ref(mixed), s.get(.out), &.{s.cache}, ref(qg), cs.prompt, false);
            try rig.finish(pass);
            for (s.buffers, saved[i], [_]usize{ qbytes, kvbytes, qbytes, kvbytes, qbytes }) |r, was, n| {
                if (repeat > 0) try equal("repeat", rig.bytes(r, n), was);
                @memcpy(was, rig.bytes(r, n));
            }
        }
        for (saved[0], saved[1], std.enums.values(Field)) |old, now, field| try equal(@tagName(field), now, old);
        const qa_bytes = @as(usize, HK) * plan.tiles * 16 * D * 2;
        try equal("qa", sides[1].scratch.get(.qa).buffer.contents()[0..qa_bytes], sides[0].scratch.get(.qa).buffer.contents()[0..qa_bytes]);
        for (0..HK) |head| for (0..cap) |row| for (0..2) |kv| {
            const flat = sides[0].allocation.off + kv * flat_bytes + (head * cap + row) * D * 2;
            const phys = if (paged) sides[1].allocation.off + kv * part + @as(usize, ids[row / 256]) * page_bytes + (head * 256 + row % 256) * D * 2 else flat;
            try equal("cache", sides[1].allocation.buf.contents()[phys..][0 .. D * 2], sides[0].allocation.buf.contents()[flat..][0 .. D * 2]);
        };
        if (!try rig.intact() or rig.overlap()) return error.MixedGuards;
    }
    // Keys-only prompt tails must produce the same complete caches too.
    if (cs.prompt) {
        const s = &sides[1];
        const cache_n = if (paged) count * page_bytes else flat_bytes * 2;
        const before_cache = try a.dupe(u8, rig.bytes(s.allocation, cache_n));
        defer a.free(before_cache);
        const pass = rig.begin();
        try ops[1].keysOnly(pass.e, &s.scratch, &s.binding, plan, .{ .qg = ref(qg), .mixed = ref(mixed), .q_gain = ref(qgain), .k_gain = ref(kgain), .norm_q = s.get(.nq), .norm_k = s.get(.nk), .query = s.get(.query), .rotated_key = s.get(.key), .positions = ref(pos), .rows = rows, .eps = 1e-6, .theta = 10000000, .h_end = if (cs.image) 12 else 0, .w_end = if (cs.image) 10 else 0, .plane = rows, .geometry = geo }, &.{s.cache});
        try rig.finish(pass);
        try equal("keys-only cache", rig.bytes(s.allocation, cache_n), before_cache);
        try equal("keys-only rotated_key", rig.bytes(s.buffers[@backingInt(Field.key)], kvbytes), saved[0][@backingInt(Field.key)]);
    }
    std.debug.print("QK_NODES units={} pages={} rows={d} tree={} image={} prompt={} repeats=5 equal\n", .{ units, paged, rows, cs.tree, cs.image, cs.prompt });
}
fn emit(a: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    for ([_]bool{ false, true }) |units| for ([_]bool{ false, true }) |pages| {
        const text = try q.simd_attention.grouped(a, units, true, 6, 4, pages);
        defer a.free(text);
        const path = try std.fmt.allocPrint(a, "{s}/attention-{s}-{s}.metal", .{ dir, if (units) "tensor" else "simd", if (pages) "paged" else "flat" });
        defer a.free(path);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text });
    };
}
pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 3 and std.mem.eql(u8, args[1], "--emit-sources")) return emit(a, init.io, args[2]);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const mutant = args.len == 2 and std.mem.eql(u8, args[1], "--mutant");
    var n: usize = 0;
    for ([_]bool{ false, true }) |units| for ([_]bool{ false, true }) |pages| {
        if (units and !device.tensorUnits()) continue;
        const raw = try q.simd_attention.grouped(a, units, true, 6, 4, false);
        defer a.free(raw);
        const text = try std.mem.concat(a, u8, &.{ raw, @embedFile("q27_qk_t622.metal") });
        defer a.free(text);
        const oldlib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer oldlib.deinit();
        var old = try at.Attention.init(device, oldlib);
        defer old.deinit();
        old.io[7].deinit();
        old.io[7] = try mtl.Pipeline.init(device, oldlib, "q27_head_norm_t622", false);
        old.io[8].deinit();
        old.io[8] = try mtl.Pipeline.init(device, oldlib, "q27_rope_t622", false);
        const source = try q.simd_attention.grouped(a, units, true, 6, 4, pages);
        defer a.free(source);
        const needle = "q27_round(value * inverse[0], 0)";
        const end = std.mem.indexOf(u8, source, "kernel void q27_qk_cache_rope(") orelse return error.MixedControlEdit;
        if (std.mem.count(u8, source[0..end], needle) != 1) return error.MixedControlEdit;
        const edited = try std.mem.replaceOwned(u8, a, source[0..end], needle, "(value * inverse[0])");
        defer a.free(edited);
        const body = if (mutant) try std.mem.concat(a, u8, &.{ edited, source[end..] }) else try a.dupe(u8, source);
        defer a.free(body);
        const lib = try mtl.Library.fromSource(device, body, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var now = try at.Attention.init(device, lib);
        defer now.deinit();
        if (!units) {
            old.units = false;
            now.units = false;
            if (old.prompt) |p| for (p) |x| x.deinit();
            if (now.prompt) |p| for (p) |x| x.deinit();
            old.prompt = try at.Attention.promptPipelines(device, oldlib, false);
            now.prompt = try at.Attention.promptPipelines(device, lib, false);
        }
        for (cases, 0..) |cs, i| {
            if (!units and cs.rows > 128) continue;
            try runCase(a, device, queue, .{ &old, &now }, units, pages, cs, 0x625000 + i);
            n += 1;
        }
    };
    std.debug.print("QK_NODES cases={d}, every query/rotated key/cache/qa byte equal, five repeats each\n", .{n});
}
