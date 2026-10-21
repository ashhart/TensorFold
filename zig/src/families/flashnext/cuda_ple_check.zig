//! The n-gram table's host checks: one token's rows dequantized on the host (the row path) and against the GPU gather.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const embed = @import("cuda_embed.zig");
const hc = @import("cuda_hc.zig");
const ple = @import("cuda_ple.zig");

const max_n = ple.max_n;
const max_heads = ple.max_heads;
const layer = ple.layer;
const Hash = ple.Hash;
const Open = ple.Open;
const Tensor = core.safetensors.Tensor;
const lastIds = ple.lastIds;
const toBf16 = ple.toBf16;
const sumGroups = ple.sumGroups;
const readI64 = ple.readI64;
const weightName = ple.weightName;
const partName = ple.partName;
const findOpen = ple.findOpen;
const copyRow = ple.copyRow;
const tableScale = ple.tableScale;
const readSpec = ple.readSpec;

/// Host dequant of one token's sixteen n-gram rows. `history` is the previous n - 1 ids.
pub fn embedding(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8, history: []const i64, token: u32, out: []u16, xs: []f32) !void {
    const cfg = try readSpec(gpa, io, model_dir);
    const hash = try Hash.init(.{
        .vocab = cfg.vocab,
        .ngram = cfg.ngram,
        .per = cfg.per,
        .base = cfg.base,
        .divisor = cfg.divisor,
        .shards = cfg.shards,
        .seed = cfg.seed,
        .eos = cfg.eos,
        .ple_dim = cfg.ple_dim,
        .ple_index = cfg.ple_index,
    });
    if (history.len != hash.n - 1 or out.len != hash.heads * hash.dh or xs.len != out.len / 32) return error.UnexpectedTensor;
    var ids: [max_heads]i64 = undefined;
    try lastIds(hash, history, @intCast(token), ids[0..hash.heads]);
    const index_path = try std.fs.path.join(gpa, &.{ model_dir, "model.safetensors.index.json" });
    defer gpa.free(index_path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const wm = try core.safetensors.index.weightMap(parsed.value);
    const scale = try tableScale(gpa, io, model_dir, wm);
    const heads = hash.heads;
    const dh = hash.dh;
    const words = try gpa.alloc(u8, heads * (dh / 8) * 4);
    defer gpa.free(words);
    const scales = try gpa.alloc(u8, heads * (dh / 32) * 2);
    defer gpa.free(scales);
    const biases = try gpa.alloc(u8, scales.len);
    defer gpa.free(biases);
    var opened: [max_heads]Open = undefined;
    var nopen: usize = 0;
    defer for (opened[0..nopen]) |*o| o.close(gpa);
    var key: [180]u8 = undefined;
    for (0..heads) |h| {
        const loc = try hash.locate(ids[h]);
        var nested = false;
        const mapped = if (wm.get(try weightName(&key, loc.shard, false))) |v|
            v.string
        else blk: {
            nested = true;
            break :blk (wm.get(try weightName(&key, loc.shard, true)) orelse return error.MissingTensor).string;
        };
        const path = try std.fs.path.join(gpa, &.{ model_dir, mapped });
        defer gpa.free(path);
        const file = findOpen(opened[0..nopen], path) orelse file: {
            opened[nopen] = try Open.load(gpa, io, path);
            nopen += 1;
            break :file &opened[nopen - 1];
        };
        const wb = (dh / 8) * 4;
        const gb = (dh / 32) * 2;
        try copyRow(file, loc.shard, nested, loc.local, dh, words[h * wb ..][0..wb], scales[h * gb ..][0..gb], biases[h * gb ..][0..gb]);
    }
    for (0..heads) |h| try embed.dequant(words, scales, biases, dh, h, out[h * dh ..][0..dh]);
    if (scale != 1) {
        for (out) |*v| {
            const f: f32 = @bitCast(@as(u32, v.*) << 16);
            v.* = toBf16(f * scale);
        }
    }
    sumGroups(out, xs);
}

/// One decode token. `ple_layer_ids` `[2]` is this layer. Returns 0 when the gathered rows match the host dequant.
pub fn table(comptime Tri: type, gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8, driver: *cuda.Driver, stream: *cuda.Stream, tri: Tri, token: u32) !u32 {
    const cfg = try readSpec(gpa, io, model_dir);
    const hash = try Hash.init(.{
        .vocab = cfg.vocab,
        .ngram = cfg.ngram,
        .per = cfg.per,
        .base = cfg.base,
        .divisor = cfg.divisor,
        .shards = cfg.shards,
        .seed = cfg.seed,
        .eos = cfg.eos,
        .ple_dim = cfg.ple_dim,
        .ple_index = cfg.ple_index,
    });
    var history: [max_n]i64 = undefined;
    for (0..hash.n - 1) |i| history[i] = hash.eos;
    var ids: [max_heads]i64 = undefined;
    try lastIds(hash, history[0 .. hash.n - 1], token, ids[0..hash.heads]);

    const index_path = try std.fs.path.join(gpa, &.{ model_dir, "model.safetensors.index.json" });
    defer gpa.free(index_path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const wm = try core.safetensors.index.weightMap(parsed.value);

    var cname: [160]u8 = undefined;
    const mult_name = try std.fmt.bufPrint(&cname, "language_model.model.layers.{d}.ple.ple_embedding.layer_multipliers", .{layer});
    const const_file = wm.get(mult_name) orelse return error.MissingTensor;
    const const_path = try std.fs.path.join(gpa, &.{ model_dir, const_file.string });
    defer gpa.free(const_path);
    var constants = try Open.load(gpa, io, const_path);
    defer constants.close(gpa);
    var got_m: [max_n]i64 = undefined;
    var got_o: [max_heads]i64 = undefined;
    var got_s: [max_heads]i64 = undefined;
    try readI64(try constants.tensor(mult_name, .i64), got_m[0..hash.n]);
    const off_name = try std.fmt.bufPrint(&cname, "language_model.model.layers.{d}.ple.ple_embedding.ngram_heads_offsets", .{layer});
    try readI64(try constants.tensor(off_name, .i64), got_o[0..hash.heads]);
    const size_name = try std.fmt.bufPrint(&cname, "language_model.model.layers.{d}.ple.ple_embedding.ngram_heads_vocab_sizes", .{layer});
    try readI64(try constants.tensor(size_name, .i64), got_s[0..hash.heads]);
    if (!std.mem.eql(i64, got_m[0..hash.n], hash.mult[0..hash.n]) or !std.mem.eql(i64, got_o[0..hash.heads], hash.offsets[0..hash.heads]) or !std.mem.eql(i64, got_s[0..hash.heads], hash.sizes[0..hash.heads])) {
        std.debug.print("ngram constants differ\n", .{});
        return 1;
    }
    const scale = try tableScale(gpa, io, model_dir, wm);

    const heads = hash.heads;
    const dh = hash.dh;
    const words = try gpa.alloc(u8, heads * (dh / 8) * 4);
    defer gpa.free(words);
    const scales = try gpa.alloc(u8, heads * (dh / 32) * 2);
    defer gpa.free(scales);
    const biases = try gpa.alloc(u8, scales.len);
    defer gpa.free(biases);
    var opened: [max_heads]Open = undefined;
    var nopen: usize = 0;
    defer for (opened[0..nopen]) |*o| o.close(gpa);
    var key: [180]u8 = undefined;
    for (0..heads) |h| {
        const loc = try hash.locate(ids[h]);
        var nested = false;
        const mapped = if (wm.get(try weightName(&key, loc.shard, false))) |v|
            v.string
        else blk: {
            nested = true;
            break :blk (wm.get(try weightName(&key, loc.shard, true)) orelse return error.MissingTensor).string;
        };
        const path = try std.fs.path.join(gpa, &.{ model_dir, mapped });
        defer gpa.free(path);
        const file = findOpen(opened[0..nopen], path) orelse file: {
            opened[nopen] = try Open.load(gpa, io, path);
            nopen += 1;
            break :file &opened[nopen - 1];
        };
        const wb = (dh / 8) * 4;
        const gb = (dh / 32) * 2;
        try copyRow(file, loc.shard, nested, loc.local, dh, words[h * wb ..][0..wb], scales[h * gb ..][0..gb], biases[h * gb ..][0..gb]);
    }

    const host = try gpa.alloc(u16, heads * dh);
    defer gpa.free(host);
    for (0..heads) |h| try embed.dequant(words, scales, biases, dh, h, host[h * dh ..][0..dh]);
    if (scale != 1) {
        for (host) |*v| {
            const f: f32 = @bitCast(@as(u32, v.*) << 16);
            v.* = toBf16(f * scale);
        }
    }
    const host_xs = try gpa.alloc(f32, heads * dh / 32);
    defer gpa.free(host_xs);
    sumGroups(host, host_xs);

    var w_b = try cuda.DeviceBuffer.fromHost(driver, words);
    defer w_b.free();
    var s_b = try cuda.DeviceBuffer.fromHost(driver, scales);
    defer s_b.free();
    var b_b = try cuda.DeviceBuffer.fromHost(driver, biases);
    defer b_b.free();
    var out_b = try cuda.DeviceBuffer.alloc(driver, host.len * 2);
    defer out_b.free();
    var xs_b = try cuda.DeviceBuffer.alloc(driver, host_xs.len * 4);
    defer xs_b.free();
    try tri.pleEmbed(w_b.ptr, s_b.ptr, b_b.ptr, out_b.ptr, xs_b.ptr, 1, heads, dh, scale);
    const got_bytes = try gpa.alloc(u8, host.len * 2);
    defer gpa.free(got_bytes);
    const xs_bytes = try gpa.alloc(u8, host_xs.len * 4);
    defer gpa.free(xs_bytes);
    try stream.synchronize();
    try out_b.download(0, got_bytes);
    try xs_b.download(0, xs_bytes);
    const gpu = try gpa.alloc(u16, host.len);
    defer gpa.free(gpu);
    for (gpu, 0..) |*o, i| o.* = std.mem.readInt(u16, got_bytes[2 * i ..][0..2], .little);
    var off: usize = 0;
    var steps: u32 = 0;
    var at: usize = 0;
    for (gpu, host, 0..) |a, b, i| {
        const d = hc.mixedSteps(a, b);
        if (d != 0) off += 1;
        if (d > steps) {
            steps = d;
            at = i;
        }
    }
    var xs_ulp: u32 = 0;
    var xs_at: usize = 0;
    for (host_xs, 0..) |want, i| {
        const got: f32 = @bitCast(std.mem.readInt(u32, xs_bytes[4 * i ..][0..4], .little));
        const dist = hc.ulps(got, want);
        if (dist > xs_ulp) {
            xs_ulp = dist;
            xs_at = i;
        }
    }
    std.debug.print("ngram token {d} heads {d} values {d} off {d} max_steps {d} at {d} got {x:0>4} host {x:0>4} xs_ulp {d} xs_at {d} head {x:0>4}\n", .{ token, heads, host.len, off, steps, at, gpu[at], host[at], xs_ulp, xs_at, gpu[0] });
    if (off != 0 or steps != 0 or xs_ulp != 0) return 1;
    return 0;
}
