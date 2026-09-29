const std = @import("std");
const mx = @import("mlx.zig");
const safe = @import("safetensors.zig");
const a = std.heap.c_allocator;
const Pair = struct { []const u8, []const u8 };
const linears = [_]Pair{
    .{ "attn.wq_a", "attn.wq_a" },                                .{ "attn.wq_b", "attn.wq_b" },                              .{ "attn.wkv", "attn.wkv" },
    .{ "attn.wo_a", "attn.wo_a" },                                .{ "attn.wo_b", "attn.wo_b" },                              .{ "ffn.shared_experts.w1", "ffn.shared_experts.gate_proj" },
    .{ "ffn.shared_experts.w2", "ffn.shared_experts.down_proj" }, .{ "ffn.shared_experts.w3", "ffn.shared_experts.up_proj" },
};
const plain = [_]Pair{
    .{ "attn_norm.weight", "attn_norm.weight" },              .{ "ffn_norm.weight", "ffn_norm.weight" },
    .{ "attn.q_norm.weight", "attn.q_norm.weight" },          .{ "attn.kv_norm.weight", "attn.kv_norm.weight" },
    .{ "attn.attn_sink", "attn.attn_sink" },                  .{ "ffn.gate.weight", "ffn.gate.weight" },
    .{ "ffn.gate.bias", "ffn.gate.e_score_correction_bias" },
};
const hyper = [_]Pair{ .{ "hc_attn", "attn_hc" }, .{ "hc_ffn", "ffn_hc" }, .{ "hc_head", "hc_head" } };
const experts = [_]Pair{ .{ "w1", "gate_proj" }, .{ "w2", "down_proj" }, .{ "w3", "up_proj" } };
const Kind = enum { mtp, dspark };
const Entry = struct { file: usize, tensor: safe.Tensor };

const Source = struct {
    files: std.ArrayList(safe.File) = .empty,
    tensors: std.StringHashMap(Entry) = .init(a),
    fn deinit(s: *Source) void {
        s.tensors.deinit();
        for (s.files.items) |*file| file.deinit();
        s.files.deinit(a);
    }
    fn open(io: std.Io, paths: []const []const u8, prefix: []const u8) !Source {
        var s = Source{};
        errdefer s.deinit();
        for (paths) |path| {
            var file = try safe.File.open(a, io, path);
            s.files.append(a, file) catch |err| {
                file.deinit();
                return err;
            };
            var it = file.header.tensors.iterator();
            while (it.next()) |e| if (std.mem.startsWith(u8, e.key_ptr.*, prefix)) {
                const entry = try s.tensors.getOrPut(e.key_ptr.*);
                if (entry.found_existing) return error.DuplicateWeight;
                entry.value_ptr.* = .{ .file = s.files.items.len - 1, .tensor = e.value_ptr.* };
            };
        }
        if (s.tensors.count() == 0) return error.MissingDraftWeights;
        return s;
    }
    fn read(s: *const Source, scope: *mx.Scope, name: []const u8) !mx.Array {
        const entry = s.tensors.get(name) orelse {
            std.debug.print("Missing conversion tensor: {s}\n", .{name});
            return error.MissingWeight;
        };
        const dt: mx.c.mlx_dtype = switch (entry.tensor.dtype) {
            .U8, .I8, .F8_E4M3, .F8_E8M0 => mx.c.MLX_UINT8,
            .BF16 => mx.bf16,
            .F32 => mx.f32t,
            else => return error.InvalidConversionDType,
        };
        const bytes = try a.alloc(u8, @intCast(entry.tensor.len));
        defer a.free(bytes);
        const file = &s.files.items[entry.file];
        if (try file.file.readPositionalAll(file.io, bytes, file.data_offset + entry.tensor.offset) != bytes.len) return error.TruncatedSafetensors;
        return scope.data(bytes.ptr, entry.tensor.shape(), dt);
    }
};

fn view(s: *mx.Scope, x: mx.Array, dtype: mx.c.mlx_dtype) !mx.Array {
    var out = mx.c.mlx_array_new();
    const rc = mx.c.mlx_view(&out, x, dtype, mx.stream);
    return s.result(rc, out);
}

fn fp8Block(s: *mx.Scope, weight: mx.Array, scale: mx.Array) !mx.Array {
    if (mx.shape(weight).len != 2 or mx.shape(scale).len != 2 or mx.dtype(weight) != mx.c.MLX_UINT8 or mx.dtype(scale) != mx.c.MLX_UINT8) return error.InvalidFp8Block;
    const rows = mx.dim(weight, 0);
    const cols = mx.dim(weight, 1);
    if (rows <= 0 or cols <= 0 or @mod(cols, 64) != 0 or mx.dim(scale, 0) != @divTrunc(rows - 1, 128) + 1 or mx.dim(scale, 1) != @divTrunc(cols - 1, 128) + 1) return error.InvalidFp8Block;
    try mx.eval(scale);
    const count = mx.c.mlx_array_size(scale);
    const codes = mx.c.mlx_array_data_uint8(scale)[0..count];
    const bits = try a.alloc(u32, count);
    defer a.free(bits);
    for (bits, codes) |*out, code| out.* = if (code == 0) 0x400000 else @as(u32, code) << 23;
    var scales = try s.data(bits.ptr, mx.shape(scale), mx.f32t);
    for (0..2) |axis| {
        var expanded = mx.c.mlx_array_new();
        const rc = mx.c.mlx_repeat_axis(&expanded, scales, 128, @intCast(axis), mx.stream);
        scales = try s.result(rc, expanded);
        scales = try s.slice(scales, axis, 0, if (axis == 0) rows else cols);
    }
    var decoded = mx.c.mlx_array_new();
    const rc = mx.c.mlx_from_fp8(&decoded, weight, mx.f32t, mx.stream);
    return s.cast(try s.binary(mx.c.mlx_multiply, try s.result(rc, decoded), scales), mx.bf16);
}

fn insert(map: mx.c.mlx_map_string_to_array, name: []const u8, value: mx.Array) !void {
    const key = try std.fmt.allocPrintSentinel(a, "{s}", .{name}, 0);
    defer a.free(key);
    try mx.eval(value);
    try mx.check(mx.c.mlx_map_string_to_array_insert(map, key, value));
}

fn linear(source: *const Source, map: mx.c.mlx_map_string_to_array, src: []const u8, dst: []const u8) !void {
    var buffer: [1024]u8 = undefined;
    const name = try std.fmt.bufPrint(&buffer, "{s}.weight", .{src});
    if (!source.tensors.contains(name)) return;
    var s = mx.Scope{};
    defer s.deinit();
    const weight = try source.read(&s, name);
    const scale = try source.read(&s, try std.fmt.bufPrint(&buffer, "{s}.scale", .{src}));
    const decoded = try fp8Block(&s, weight, scale);
    var quant = mx.c.mlx_vector_array_new();
    defer _ = mx.c.mlx_vector_array_free(quant);
    try mx.check(mx.c.mlx_quantize(&quant, decoded, mx.opt(64), mx.opt(4), "affine", mx.empty, mx.stream));
    for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |suffix, index| {
        var value = mx.c.mlx_array_new();
        const rc = mx.c.mlx_vector_array_get(&value, quant, index);
        try insert(map, try std.fmt.bufPrint(&buffer, "{s}.{s}", .{ dst, suffix }), try s.result(rc, value));
    }
}

fn copy(source: *const Source, map: mx.c.mlx_map_string_to_array, src: []const u8, dst: []const u8, required: bool) !void {
    if (!required and !source.tensors.contains(src)) return;
    var s = mx.Scope{};
    defer s.deinit();
    try insert(map, dst, try source.read(&s, src));
}

fn block(source: *const Source, map: mx.c.mlx_map_string_to_array, src: []const u8, dst: []const u8, kind: Kind) !void {
    var from: [1024]u8 = undefined;
    var to: [1024]u8 = undefined;
    for (linears) |pair| try linear(source, map, try std.fmt.bufPrint(&from, "{s}.{s}", .{ src, pair[0] }), try std.fmt.bufPrint(&to, "{s}.{s}", .{ dst, pair[1] }));
    const extra_linears: []const []const u8 = if (kind == .mtp) &.{ "e_proj", "h_proj" } else &.{"main_proj"};
    for (extra_linears) |name| try linear(source, map, try std.fmt.bufPrint(&from, "{s}.{s}", .{ src, name }), try std.fmt.bufPrint(&to, "{s}.{s}", .{ dst, name }));
    for (plain) |pair| try copy(source, map, try std.fmt.bufPrint(&from, "{s}.{s}", .{ src, pair[0] }), try std.fmt.bufPrint(&to, "{s}.{s}", .{ dst, pair[1] }), false);
    const extra_plain: []const []const u8 = if (kind == .mtp) &.{ "enorm.weight", "hnorm.weight", "norm.weight" } else &.{ "main_norm.weight", "norm.weight", "markov_head.markov_w1.weight", "markov_head.markov_w2.weight" };
    for (extra_plain) |name| try copy(source, map, try std.fmt.bufPrint(&from, "{s}.{s}", .{ src, name }), try std.fmt.bufPrint(&to, "{s}.{s}", .{ dst, name }), false);
    for (hyper) |pair| {
        if (!source.tensors.contains(try std.fmt.bufPrint(&from, "{s}.{s}_fn", .{ src, pair[0] }))) continue;
        for ([_][]const u8{ "fn", "base", "scale" }) |part| try copy(source, map, try std.fmt.bufPrint(&from, "{s}.{s}_{s}", .{ src, pair[0], part }), try std.fmt.bufPrint(&to, "{s}.{s}.{s}", .{ dst, pair[1], part }), true);
    }
    const prefix = try std.fmt.bufPrint(&from, "{s}.ffn.experts.", .{src});
    var count: usize = 0;
    var keys = source.tensors.keyIterator();
    while (keys.next()) |name| if (std.mem.startsWith(u8, name.*, prefix) and std.mem.endsWith(u8, name.*, ".w1.weight")) {
        count += 1;
    };
    if (count == 0) return error.MissingExperts;
    for (experts) |pair| {
        var s = mx.Scope{};
        defer s.deinit();
        const codes = try a.alloc(mx.Array, count);
        defer a.free(codes);
        const scales = try a.alloc(mx.Array, count);
        defer a.free(scales);
        for (0..count) |e| {
            const weight = try source.read(&s, try std.fmt.bufPrint(&from, "{s}.ffn.experts.{d}.{s}.weight", .{ src, e, pair[0] }));
            scales[e] = try source.read(&s, try std.fmt.bufPrint(&from, "{s}.ffn.experts.{d}.{s}.scale", .{ src, e, pair[0] }));
            if (mx.shape(weight).len != 2 or mx.shape(scales[e]).len != 2 or mx.dtype(weight) != mx.c.MLX_UINT8 or mx.dtype(scales[e]) != mx.c.MLX_UINT8) return error.InvalidFp4Expert;
            if (mx.dim(weight, 0) <= 0 or mx.dim(weight, 1) <= 0 or @mod(mx.dim(weight, 1), 16) != 0 or mx.dim(scales[e], 0) != mx.dim(weight, 0) or mx.dim(scales[e], 1) != @divTrunc(mx.dim(weight, 1), 16)) return error.InvalidFp4Expert;
            codes[e] = try view(&s, weight, mx.c.MLX_UINT32);
        }
        try insert(map, try std.fmt.bufPrint(&to, "{s}.ffn.switch_mlp.{s}.weight", .{ dst, pair[1] }), try s.stack(codes, 0));
        try insert(map, try std.fmt.bufPrint(&to, "{s}.ffn.switch_mlp.{s}.scales", .{ dst, pair[1] }), try s.stack(scales, 0));
    }
}

fn save(io: std.Io, source: *const Source, map: mx.c.mlx_map_string_to_array, path: []const u8) !void {
    if (std.Io.Dir.cwd().statFile(io, path, .{})) |stat| {
        for (source.files.items) |file| if ((try file.file.stat(io)).inode == stat.inode) return error.OutputAliasesInput;
    } else |err| if (err != error.FileNotFound) return err;
    try std.Io.Dir.cwd().createDirPath(io, std.fs.path.dirname(path) orelse ".");
    const temp = try std.fmt.allocPrintSentinel(a, "{s}.partial.safetensors", .{path}, 0);
    defer a.free(temp);
    const reservation = try std.Io.Dir.cwd().createFile(io, temp, .{ .exclusive = true });
    reservation.close(io);
    defer std.Io.Dir.cwd().deleteFile(io, temp) catch {};
    const metadata = mx.c.mlx_map_string_to_string_new();
    defer _ = mx.c.mlx_map_string_to_string_free(metadata);
    try mx.check(mx.c.mlx_map_string_to_string_insert(metadata, "format", "mlx"));
    try mx.check(mx.c.mlx_save_safetensors(temp, map, metadata));
    try safe.validateFile(io, temp);
    try std.Io.Dir.cwd().rename(temp, .cwd(), path, io);
}

const DraftConfig = struct {
    path: []const u8,
    text: []const u8,
    fn deinit(c: DraftConfig) void {
        a.free(c.path);
        a.free(c.text);
    }
    fn write(c: DraftConfig, io: std.Io) !void {
        const temp = try std.fmt.allocPrint(a, "{s}.partial", .{c.path});
        defer a.free(temp);
        const file = try std.Io.Dir.cwd().createFile(io, temp, .{ .exclusive = true });
        defer file.close(io);
        defer std.Io.Dir.cwd().deleteFile(io, temp) catch {};
        try file.writeStreamingAll(io, c.text);
        try std.Io.Dir.cwd().rename(temp, .cwd(), c.path, io);
    }
};

fn config(io: std.Io, shard: []const u8, out: []const u8) !?DraftConfig {
    const path = try std.fs.path.join(a, &.{ std.fs.path.dirname(shard) orelse ".", "config.json" });
    defer a.free(path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 * 1024 * 1024)) catch |err| return if (err == error.FileNotFound) null else err;
    defer a.free(bytes);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidDraftConfig;
    var selected: std.json.ObjectMap = .empty;
    defer selected.deinit(a);
    var it = parsed.value.object.iterator();
    while (it.next()) |e| if (std.mem.startsWith(u8, e.key_ptr.*, "dspark_")) {
        try selected.put(a, e.key_ptr.*, e.value_ptr.*);
    };
    const dest = try std.fs.path.join(a, &.{ std.fs.path.dirname(out) orelse ".", "config.json" });
    errdefer a.free(dest);
    const origin = try std.Io.Dir.cwd().statFile(io, path, .{});
    if (std.Io.Dir.cwd().statFile(io, dest, .{})) |existing| {
        if (origin.inode == existing.inode) {
            a.free(dest);
            return null;
        }
    } else |err| if (err != error.FileNotFound) return err;
    const text = try std.json.Stringify.valueAlloc(a, std.json.Value{ .object = selected }, .{ .whitespace = .indent_2 });
    return .{ .path = dest, .text = text };
}

pub fn run(io: std.Io, arguments: []const []const u8) !void {
    var args = arguments;
    var layer: usize = 0;
    if (args.len >= 2 and std.mem.eql(u8, args[args.len - 2], "--layer")) {
        layer = try std.fmt.parseInt(usize, args[args.len - 1], 10);
        args = args[0 .. args.len - 2];
    }
    if (args.len < 3) return error.ExpectedDrafterKindShardsAndOutput;
    const kind = std.meta.stringToEnum(Kind, args[0]) orelse return error.InvalidDrafterKind;
    if (kind == .dspark and arguments.len != args.len) return error.LayerOnlyAppliesToMtp;
    const paths = args[1 .. args.len - 1];
    const output = args[args.len - 1];
    var buffer: [128]u8 = undefined;
    const prefix = if (kind == .mtp) try std.fmt.bufPrint(&buffer, "mtp.{d}.", .{layer}) else "mtp.";
    var source = try Source.open(io, paths, prefix);
    defer source.deinit();
    const draft_config = if (kind == .dspark) try config(io, paths[0], output) else null;
    defer if (draft_config) |value| value.deinit();
    try mx.init();
    defer mx.shutdown();
    const map = mx.c.mlx_map_string_to_array_new();
    defer _ = mx.c.mlx_map_string_to_array_free(map);
    if (kind == .mtp) {
        try block(&source, map, prefix[0 .. prefix.len - 1], "mtp", kind);
    } else {
        var layers = std.AutoHashMap(usize, void).init(a);
        defer layers.deinit();
        var keys = source.tensors.keyIterator();
        while (keys.next()) |name| {
            const end = std.mem.indexOfScalarPos(u8, name.*, 4, '.') orelse return error.InvalidDraftLayer;
            try layers.put(try std.fmt.parseInt(usize, name.*[4..end], 10), {});
        }
        const ids = try a.alloc(usize, layers.count());
        defer a.free(ids);
        var iter = layers.keyIterator();
        for (ids) |*id| id.* = iter.next().?.*;
        std.mem.sort(usize, ids, {}, std.sort.asc(usize));
        for (ids) |id| {
            var dest: [128]u8 = undefined;
            try block(&source, map, try std.fmt.bufPrint(&buffer, "mtp.{d}", .{id}), try std.fmt.bufPrint(&dest, "dspark.{d}", .{id}), kind);
        }
    }
    try save(io, &source, map, output);
    if (draft_config) |value| try value.write(io);
    std.debug.print("Wrote {s}\n", .{output});
}

fn compare(io: std.Io, expected: []const u8, actual: []const u8) !usize {
    var lhs = try safe.File.open(a, io, expected);
    defer lhs.deinit();
    var rhs = try safe.File.open(a, io, actual);
    defer rhs.deinit();
    if (lhs.header.tensors.count() != rhs.header.tensors.count()) return error.ConvertedTensorCountMismatch;
    var left: [65536]u8 = undefined;
    var right: [65536]u8 = undefined;
    var it = lhs.header.tensors.iterator();
    while (it.next()) |e| {
        const x = e.value_ptr.*;
        const y = rhs.header.tensors.get(e.key_ptr.*) orelse return error.ConvertedTensorMissing;
        if (x.dtype != y.dtype or !std.mem.eql(i32, x.shape(), y.shape()) or x.len != y.len) return error.ConvertedTensorShapeMismatch;
        var offset: u64 = 0;
        while (offset < x.len) {
            const count: usize = @intCast(@min(left.len, x.len - offset));
            if (try lhs.file.readPositionalAll(io, left[0..count], lhs.data_offset + x.offset + offset) != count or try rhs.file.readPositionalAll(io, right[0..count], rhs.data_offset + y.offset + offset) != count) return error.TruncatedSafetensors;
            if (!std.mem.eql(u8, left[0..count], right[0..count])) {
                std.debug.print("Conversion mismatch in {s}, byte {d}\n", .{ e.key_ptr.*, offset });
                return error.ConvertedTensorBytesMismatch;
            }
            offset += count;
        }
    }
    return lhs.header.tensors.count();
}

pub fn check(io: std.Io, directory: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const temp = arena.allocator();
    const raw = try std.fs.path.join(temp, &.{ directory, "raw" });
    const out = try std.fs.path.join(temp, &.{ directory, "native" });
    try std.Io.Dir.cwd().createDirPath(io, out);
    const mtp = try std.fs.path.join(temp, &.{ raw, "mtp.safetensors" });
    const mtp_out = try std.fs.path.join(temp, &.{ out, "mtp.safetensors" });
    try run(io, &.{ "mtp", mtp, mtp_out });
    try run(io, &.{ "mtp", try std.fs.path.join(temp, &.{ raw, "layer3.safetensors" }), try std.fs.path.join(temp, &.{ out, "layer3.safetensors" }), "--layer", "3" });
    for ([_][]const u8{ "dspark", "split" }) |kind| {
        const first = try std.fs.path.join(temp, &.{ raw, if (std.mem.eql(u8, kind, "dspark")) "model-00000.safetensors" else "split-0.safetensors" });
        const second = try std.fs.path.join(temp, &.{ raw, if (std.mem.eql(u8, kind, "dspark")) "model-00001.safetensors" else "split-1.safetensors" });
        try run(io, &.{ "dspark", first, second, try std.fs.path.join(temp, &.{ out, kind, "dspark.safetensors" }) });
    }
    {
        var source = try Source.open(io, &.{try std.fs.path.join(temp, &.{ raw, "codes.safetensors" })}, "mtp.");
        defer source.deinit();
        try mx.init();
        defer mx.shutdown();
        var s = mx.Scope{};
        defer s.deinit();
        const decoded = try fp8Block(&s, try source.read(&s, "mtp.codes.weight"), try source.read(&s, "mtp.codes.scale"));
        const map = mx.c.mlx_map_string_to_array_new();
        defer _ = mx.c.mlx_map_string_to_array_free(map);
        try insert(map, "decoded", decoded);
        try save(io, &source, map, try std.fs.path.join(temp, &.{ out, "codes.safetensors" }));
    }
    var count: usize = 0;
    for ([_][]const u8{ "mtp.safetensors", "layer3.safetensors", "dspark/dspark.safetensors", "split/dspark.safetensors", "codes.safetensors" }) |name| {
        const oracle = if (std.mem.startsWith(u8, name, "split")) "dspark/dspark.safetensors" else name;
        count += try compare(io, try std.fs.path.join(temp, &.{ directory, "expected", oracle }), try std.fs.path.join(temp, &.{ out, name }));
    }
    const expected_config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(temp, &.{ directory, "expected/dspark/config.json" }), temp, .limited(4096));
    const want = try std.json.parseFromSlice(std.json.Value, temp, expected_config, .{});
    for ([_][]const u8{ "dspark", "split" }) |kind| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(temp, &.{ out, kind, "config.json" }), temp, .limited(4096));
        const got = try std.json.parseFromSlice(std.json.Value, temp, bytes, .{});
        try std.testing.expectEqualStrings(try std.json.Stringify.valueAlloc(temp, want.value, .{}), try std.json.Stringify.valueAlloc(temp, got.value, .{}));
    }
    try std.testing.expectError(error.OutputAliasesInput, run(io, &.{ "mtp", mtp, mtp }));
    try std.testing.expectError(error.DuplicateWeight, run(io, &.{ "mtp", mtp, mtp, mtp_out }));
    try std.testing.expectError(error.InvalidFp8Block, run(io, &.{ "mtp", try std.fs.path.join(temp, &.{ raw, "bad-scale.safetensors" }), mtp_out }));
    try std.testing.expectError(error.MissingWeight, run(io, &.{ "mtp", try std.fs.path.join(temp, &.{ raw, "missing-expert.safetensors" }), mtp_out }));
    _ = try compare(io, try std.fs.path.join(temp, &.{ directory, "expected/mtp.safetensors" }), mtp_out);
    std.debug.print("PASS: {d} converted tensors match upstream byte-for-byte; all FP8/E8M0 codes, split shards, config and failure preservation checked\n", .{count});
    try mx.init();
    defer mx.shutdown();
    for ([_][]const u8{ "", "dspark" }) |draft_path| {
        var model = try @import("deepseek.zig").Model.init(io, try std.fs.path.join(temp, &.{ directory, "target" }));
        defer model.deinit();
        try model.loadDraft(io, try std.fs.path.join(temp, &.{ out, draft_path }));
        for ([_]f64{ 0, 0.8 }) |temperature| {
            var reference: std.ArrayList(u32) = .empty;
            defer reference.deinit(a);
            for ([_]usize{ 0, 1, 3, 7 }) |depth| {
                model.reset();
                var generated = try @import("serial_generation.zig").generate(&model, &.{ 1, 2, 3, 4 }, 12, .{ .seed = 1234, .temperature = temperature, .top_k = 20, .top_p = 0.95, .metal = true }, depth, null);
                defer generated.deinit();
                if (depth == 0) try reference.appendSlice(a, generated.tokens.items) else try std.testing.expectEqualSlices(u32, reference.items, generated.tokens.items);
            }
        }
    }
    std.debug.print("PASS: natively converted MTP/DSpark load and preserve greedy/seeded serial generation at draft depths 1, 3 and 7\n", .{});
}
