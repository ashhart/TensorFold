//! Native dots.mocr intermediate checks against the retained Hugging Face example fixtures.
const std = @import("std");
const hip = @import("hip");
const family = @import("dots_mocr");
const core = @import("core");

pub const std_options: std.Options = .{ .signal_stack_size = null };

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len >= 2 and (std.mem.eql(u8, args[1], "text") or std.mem.eql(u8, args[1], "teacher"))) {
        if (args.len < 5 or args.len > 7) return error.BadArguments;
        const features = if (args.len >= 6 and !std.mem.eql(u8, args[5], "-")) args[5] else null;
        const cap = if (args.len == 7) try std.fmt.parseInt(usize, args[6], 10) else 64;
        try textCheck(init.gpa, init.io, args[2], args[3], args[4], features, cap, std.mem.eql(u8, args[1], "teacher"));
        return 0;
    }
    if (args.len != 5 and args.len != 7 and args.len != 8) {
        std.debug.print("usage: tf-dots-hip-check patch|block0|block|vision|merger CHECKPOINT INPUT.npy OUTPUT.npy [GRID_H GRID_W [LAYER]]\n", .{});
        return 2;
    }
    const stage: Stage = std.meta.stringToEnum(Stage, args[1]) orelse return error.BadStage;
    const gh = if (args.len >= 7) try std.fmt.parseInt(u32, args[5], 10) else 0;
    const gw = if (args.len >= 7) try std.fmt.parseInt(u32, args[6], 10) else 0;
    const layer = if (args.len == 8) try std.fmt.parseInt(u32, args[7], 10) else 0;
    try check(init.gpa, init.io, stage, args[2], args[3], args[4], gh, gw, layer);
    return 0;
}

const Stage = enum { patch, block0, block, vision, merger };

fn check(a: std.mem.Allocator, io: std.Io, stage: Stage, dir: []const u8, input: []const u8, output: []const u8, gh: u32, gw: u32, layer: u32) !void {
    const c = try family.config.Config.read(a, io, dir);
    const source = try std.Io.Dir.cwd().readFileAlloc(io, input, a, .limited(256 << 20));
    defer a.free(source);
    const pixels = try core.npy.parse(source);
    const input_width: usize = if (stage == .merger or stage == .block) c.vision.hidden else 588;
    if (!std.mem.eql(u8, pixels.descr, "<f4") or pixels.rank != 2 or pixels.shape[1] != input_width) return error.BadPatchFixture;
    const rows = pixels.shape[0];
    const scratch_bytes = if (stage == .patch) try family.graph.patchBytes(c, rows) else try family.vision.scratchBytes(c, rows);
    if (stage != .patch and @as(u64, gh) * gw != rows) return error.BadGrid;
    const positions = if (stage != .patch) try family.image.positions(a, gh, gw, c.vision.merge) else try a.alloc([2]u32, 0);
    defer a.free(positions);
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try hip.Stream.init(&r);
    defer stream.deinit();
    var l = try hip.Launcher.load(&r, .{}, hip.kernels.images);
    defer l.unload();
    var blas = try hip.blas.Blas.open(stream);
    defer blas.close();
    const memory = try ctx.memInfo();
    const reserve = scratch_bytes + pixels.data.len + (2 << 30);
    if (memory.free < reserve) return error.DotsWeightsDoNotFit;
    var w = try family.weights.Weights.load(a, &r, io, dir, c, memory.free - reserve);
    defer w.deinit();
    var arena = try hip.Arena.init(&r, scratch_bytes);
    defer arena.deinit();
    var dx = try hip.DeviceBuffer.fromHost(&r, pixels.data);
    defer dx.free();
    var dp = try hip.DeviceBuffer.fromHost(&r, if (positions.len == 0) &.{ 0, 0, 0, 0 } else std.mem.sliceAsBytes(positions));
    defer dp.free();
    const graph: family.graph.Graph = .{
        .c = c,
        .w = &w,
        .ops = .{ .l = &l, .bf16 = true, .stream = stream.handle, .arena = &arena },
        .blas = &blas,
    };
    errdefer stream.synchronize() catch {};
    const x: hip.ops.Tensor = .{ .ptr = dx.base(), .kind = .f32 };
    var projection: ?hip.ops.Tensor = null;
    var normalized: ?hip.ops.Tensor = null;
    const result = switch (stage) {
        .patch => blk: {
            const p = try graph.patch(x, rows);
            projection = p.projection;
            break :blk p.normalized;
        },
        .block0 => blk: {
            const out = try graph.tensor(rows * c.vision.hidden);
            const mark = arena.mark();
            const p = try graph.patch(x, rows);
            // Keep the normalized patch buffer alive while block0 consumes it.
            try graph.visionBlock(p.normalized, out, dp.base(), rows, 0);
            arena.release(mark);
            break :blk out;
        },
        .block => blk: {
            const out = try graph.tensor(rows * c.vision.hidden);
            const bf = try graph.tensor(rows * c.vision.hidden);
            try graph.ops.cast(x, bf, rows * c.vision.hidden);
            try graph.visionBlock(bf, out, dp.base(), rows, layer);
            break :blk out;
        },
        .vision => blk: {
            const result = try family.vision.stages(graph, x, dp.base(), rows);
            normalized = result.normalized;
            break :blk result.merged;
        },
        .merger => blk: {
            const bf = try graph.tensor(rows * c.vision.hidden);
            try graph.ops.cast(x, bf, rows * c.vision.hidden);
            break :blk try family.vision.merger(graph, bf, rows);
        },
    };
    try stream.synchronize();
    const result_rows = if (stage == .vision or stage == .merger) rows / 4 else rows;
    const host = try a.alloc(u8, result_rows * c.vision.hidden * 2);
    defer a.free(host);
    try hip.runtime.check(r.api.hipMemcpyDtoH(host.ptr, @ptrFromInt(result.ptr), host.len));
    try write16(a, io, output, host, result_rows, c.vision.hidden);
    if (projection) |p| {
        const raw = try std.fmt.allocPrint(a, "{s}.projection.npy", .{output});
        defer a.free(raw);
        try hip.runtime.check(r.api.hipMemcpyDtoH(host.ptr, @ptrFromInt(p.ptr), host.len));
        try write16(a, io, raw, host, rows, c.vision.hidden);
    }
    if (normalized) |p| {
        const norm_host = try a.alloc(u8, rows * c.vision.hidden * 2);
        defer a.free(norm_host);
        const raw = try std.fmt.allocPrint(a, "{s}.norm.npy", .{output});
        defer a.free(raw);
        try hip.runtime.check(r.api.hipMemcpyDtoH(norm_host.ptr, @ptrFromInt(p.ptr), norm_host.len));
        try write16(a, io, raw, norm_host, rows, c.vision.hidden);
    }
    std.debug.print("dots native {s}: {d} rows, {d} scratch bytes, {d} device bytes\n", .{ @tagName(stage), rows, arena.peak, hip.usage(false).device });
}

fn write16(a: std.mem.Allocator, io: std.Io, path: []const u8, data: []const u8, rows: usize, width: usize) !void {
    const text = try std.fmt.allocPrint(a, "{{'descr': '<u2', 'fortran_order': False, 'shape': ({d}, {d}), }}", .{ rows, width });
    defer a.free(text);
    const header = std.mem.alignForward(usize, 10 + text.len + 1, 64) - 10;
    const bytes = try a.alloc(u8, 10 + header + data.len);
    defer a.free(bytes);
    @memcpy(bytes[0..8], "\x93NUMPY\x01\x00");
    std.mem.writeInt(u16, bytes[8..10], @intCast(header), .little);
    @memset(bytes[10..][0..header], ' ');
    @memcpy(bytes[10..][0..text.len], text);
    bytes[10 + header - 1] = '\n';
    @memcpy(bytes[10 + header ..], data);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

fn textCheck(a: std.mem.Allocator, io: std.Io, dir: []const u8, reference: []const u8, prefix: []const u8, feature_path: ?[]const u8, cap: usize, teacher: bool) !void {
    const c = try family.config.Config.read(a, io, dir);
    const forced = if (teacher) try referenceIds(a, io, reference, c) else try a.alloc(u32, 0);
    defer a.free(forced);
    const decode_cap = if (teacher) @min(cap, forced.len) else cap;
    const ids_path = try std.fmt.allocPrint(a, "{s}/input-input_ids.npy", .{reference});
    defer a.free(ids_path);
    const ids_file = try std.Io.Dir.cwd().readFileAlloc(io, ids_path, a, .limited(4 << 20));
    defer a.free(ids_file);
    const ids_npy = try core.npy.parse(ids_file);
    if (ids_npy.rank != 2 or ids_npy.shape[0] != 1 or !std.mem.eql(u8, ids_npy.descr, "<i8")) return error.BadTokenFixture;
    const rows = ids_npy.shape[1];
    if (rows == 0 or rows > c.context) return error.DotsContext;
    const ids = try a.alloc(i32, rows);
    defer a.free(ids);
    const positions = try a.alloc(i32, rows);
    defer a.free(positions);
    var first = rows;
    var count: usize = 0;
    for (ids, positions, 0..) |*id, *pos, i| {
        const raw = std.mem.readInt(i64, ids_npy.data[i * 8 ..][0..8], .little);
        if (raw < 0 or raw >= c.vocab or raw == c.video_token) return error.BadTokenFixture;
        id.* = @intCast(raw);
        pos.* = @intCast(i);
        if (id.* == c.image_token) {
            if (count == 0) first = i else if (i != first + count) return error.NoncontiguousImageFixture;
            count += 1;
        }
    }
    if (count == 0) return error.BadImageFixture;
    const default_features = try std.fmt.allocPrint(a, "{s}/vision-merged.npy", .{reference});
    defer a.free(default_features);
    const features_file = try std.Io.Dir.cwd().readFileAlloc(io, feature_path orelse default_features, a, .limited(128 << 20));
    defer a.free(features_file);
    const features = try core.npy.parse(features_file);
    const f32_input = std.mem.eql(u8, features.descr, "<f4");
    if ((!f32_input and !std.mem.eql(u8, features.descr, "<u2")) or features.rank != 2 or
        features.shape[0] != count or features.shape[1] != c.hidden) return error.BadImageFixture;
    const scratch = try family.text.scratchBytes(c, rows, false);
    if (decode_cap == 0 or decode_cap > 24000 or rows > c.context - decode_cap) return error.DotsContext;
    const kv = try family.text.Cache.bytes(c, rows + decode_cap);
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try hip.Stream.init(&r);
    defer stream.deinit();
    var l = try hip.Launcher.load(&r, .{}, hip.kernels.images);
    defer l.unload();
    var blas = try hip.blas.Blas.open(stream);
    defer blas.close();
    const free = (try ctx.memInfo()).free;
    const reserve = scratch + kv + features.data.len + rows * 8 + (2 << 30);
    if (free < reserve) return error.DotsWeightsDoNotFit;
    var w = try family.weights.Weights.load(a, &r, io, dir, c, free - reserve);
    defer w.deinit();
    var cache = try family.text.Cache.init(&r, c, rows + decode_cap);
    defer cache.deinit();
    var arena = try hip.Arena.init(&r, scratch);
    defer arena.deinit();
    var di = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(ids));
    defer di.free();
    var dp = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(positions));
    defer dp.free();
    var df = try hip.DeviceBuffer.fromHost(&r, features.data);
    defer df.free();
    const graph: family.graph.Graph = .{ .c = c, .w = &w, .ops = .{
        .l = &l,
        .bf16 = true,
        .stream = stream.handle,
        .arena = &arena,
    }, .blas = &blas };
    errdefer stream.synchronize() catch {};
    const bf_features = if (f32_input) blk: {
        const out = try graph.tensor(count * c.hidden);
        try graph.ops.cast(.{ .ptr = df.base(), .kind = .f32 }, out, count * c.hidden);
        break :blk out;
    } else hip.ops.Tensor{ .ptr = df.base(), .kind = .bf16 };
    const input = try family.text.embed(graph, di.base(), rows, &.{.{ .first = first, .rows = count, .features = bf_features }});
    const mark = arena.mark();
    const block0 = try graph.tensor(rows * c.hidden);
    try family.text.block(graph, input, block0, dp.base(), rows, 0, 0, &cache);
    try stream.synchronize();
    try saveTensor(a, io, &r, prefix, ".block0.npy", block0, rows, c.hidden);
    arena.release(mark);
    const result = try family.text.forward(graph, input, dp.base(), rows, 0, &cache, false);
    try stream.synchronize();
    try saveTensor(a, io, &r, prefix, ".norm.npy", result.normalized, rows, c.hidden);
    try saveTensor(a, io, &r, prefix, ".logits.npy", result.logits, 1, c.vocab);
    var pick = try hip.DeviceBuffer.alloc(&r, 4);
    defer pick.free();
    const scores = try a.alloc(u8, decode_cap * c.vocab * 2);
    defer a.free(scores);
    const generated = try a.alloc(u32, decode_cap);
    defer a.free(generated);
    var generated_count: usize = 0;
    var logits = result.logits;
    var decode_graph = graph;
    decode_graph.ops.window = true;
    for (0..decode_cap) |step| {
        try graph.ops.argmaxRows(logits, 1, c.vocab, pick.base());
        try stream.synchronize();
        const row = scores[step * c.vocab * 2 ..][0 .. c.vocab * 2];
        try hip.runtime.check(r.api.hipMemcpyDtoH(row.ptr, @ptrFromInt(logits.ptr), row.len));
        var token_bytes: [4]u8 = undefined;
        try pick.download(0, &token_bytes);
        const token = std.mem.readInt(u32, &token_bytes, .little);
        if (token >= c.vocab) return error.BadGeneratedToken;
        generated[step] = token;
        generated_count += 1;
        const feed = if (teacher) forced[step] else token;
        if (c.isEos(feed) or step + 1 == decode_cap) break;
        arena.reset();
        std.mem.writeInt(u32, &token_bytes, feed, .little);
        try di.upload(0, &token_bytes);
        var position_bytes: [4]u8 = undefined;
        std.mem.writeInt(i32, &position_bytes, @intCast(rows + step), .little);
        try dp.upload(0, &position_bytes);
        const embedded = try family.text.embed(decode_graph, di.base(), 1, &.{});
        logits = (try family.text.forward(decode_graph, embedded, dp.base(), 1, rows + step, &cache, false)).logits;
        if ((step + 1) % 256 == 0) std.debug.print("dots native {s}: {d} score rows\n", .{ if (teacher) "teacher" else "greedy", step + 1 });
    }
    const score_path = try std.fmt.allocPrint(a, "{s}.scores.npy", .{prefix});
    defer a.free(score_path);
    try write16(a, io, score_path, scores[0 .. generated_count * c.vocab * 2], generated_count, c.vocab);
    const ids_path_out = try std.fmt.allocPrint(a, "{s}.ids.json", .{prefix});
    defer a.free(ids_path_out);
    const ids_json = try std.json.Stringify.valueAlloc(a, generated[0..generated_count], .{});
    defer a.free(ids_json);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = ids_path_out, .data = ids_json });
    std.debug.print("dots native text: {d} rows, {d} scratch bytes, {d} device bytes\n", .{ rows, arena.peak, hip.usage(false).device });
}

fn referenceIds(a: std.mem.Allocator, io: std.Io, dir: []const u8, c: family.config.Config) ![]u32 {
    const path = try std.fmt.allocPrint(a, "{s}/result.json", .{dir});
    defer a.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(4 << 20));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadReferenceIds;
    const array = parsed.value.object.get("output_ids") orelse return error.BadReferenceIds;
    if (array != .array or array.array.items.len == 0 or array.array.items.len > 24000) return error.BadReferenceIds;
    const out = try a.alloc(u32, array.array.items.len);
    errdefer a.free(out);
    for (array.array.items, out) |id, *dst| {
        if (id != .integer or id.integer < 0 or id.integer >= c.vocab) return error.BadReferenceIds;
        dst.* = @intCast(id.integer);
    }
    return out;
}

fn saveTensor(a: std.mem.Allocator, io: std.Io, r: *const hip.Runtime, prefix: []const u8, suffix: []const u8, tensor: hip.ops.Tensor, rows: usize, width: usize) !void {
    const host = try a.alloc(u8, rows * width * 2);
    defer a.free(host);
    try hip.runtime.check(r.api.hipMemcpyDtoH(host.ptr, @ptrFromInt(tensor.ptr), host.len));
    const path = try std.fmt.allocPrint(a, "{s}{s}", .{ prefix, suffix });
    defer a.free(path);
    try write16(a, io, path, host, rows, width);
}
