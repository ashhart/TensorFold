//! A runtime LoRA on the Qwen 27B's decision prompt rows (Cloudflare Clef): y = bf16(y + (bf16(x A^T)) B^T) in fp32.
const std = @import("std");
const cuda = @import("cuda");
const geometry = @import("cuda_geometry.zig");
const E = @import("cuda_engine.zig");
const layers_mod = @import("cuda_layers.zig");

const Slot = geometry.Slot;
const Allocator = std.mem.Allocator;

/// Rows a factor product takes at once; DenseRows keeps each row's bits at any row count.
const block = 256;

/// One projection's factors on the device: A [rank][k] and B [n][rank], bf16, the scale folded into B.
pub const Factor = struct { a: u64, b: u64, rank: usize, n: usize, k: usize };

pub const Lora = struct {
    gpa: Allocator,
    dir: []u8,
    buf: cuda.DeviceBuffer,
    mod: cuda.Module,
    add: cuda.Function,
    dense: cuda.dense_rows.DenseRows,
    t32: u64,
    t16: u64,
    d32: u64,
    at: []std.EnumArray(Slot, ?Factor),

    pub fn factor(l: *const Lora, layer: usize, s: Slot) ?Factor {
        return if (layer < l.at.len) l.at[layer].get(s) else null;
    }

    pub fn deinit(l: *Lora) void {
        l.buf.free();
        l.mod.unload();
        l.dense.unload();
        l.gpa.free(l.at);
        l.gpa.free(l.dir);
    }

    /// `dir`/adapter.safetensors's `layers.{i}.{module}.lora_A` / `.lora_B` pairs, checked against the engine's widths.
    pub fn open(gpa: Allocator, io: std.Io, e: *E.Engine, dir: []const u8) !*Lora {
        const path = try std.fs.path.join(gpa, &.{ dir, "adapter.safetensors" });
        defer gpa.free(path);
        var f = try Tensors.open(gpa, io, path);
        defer f.close(io);
        const at = try gpa.alloc(std.EnumArray(Slot, ?Factor), e.layers.len);
        errdefer gpa.free(at);
        for (at) |*x| x.* = .initFill(null);
        // the factors' bytes, each 256-aligned, then the scratch rows
        var total: usize = 0;
        var most_n: usize = 0;
        var most_rank: usize = 0;
        var name: [160]u8 = undefined;
        for (e.layers, 0..) |l, i| for (std.enums.values(Slot)) |s| {
            const p = projOf(l, s) orelse continue;
            const a = f.get(try std.fmt.bufPrint(&name, "layers.{d}.{s}.lora_A", .{ i, geometry.slot_modules.get(s) })) orelse continue;
            const b = f.get(try std.fmt.bufPrint(&name, "layers.{d}.{s}.lora_B", .{ i, geometry.slot_modules.get(s) })) orelse return error.LoraMissingB;
            if (!a.ok2() or !b.ok2()) return error.LoraFormat;
            const rank = a.dims[0];
            if (a.dims[1] != p.inputs() or b.dims[0] != p.outputs() or b.dims[1] != rank or rank % 8 != 0) return error.LoraShape;
            if (a.bytes.len != rank * p.inputs() * 2 or b.bytes.len != p.outputs() * rank * 2) return error.LoraShape;
            at[i].set(s, .{ .a = total, .b = 0, .rank = rank, .n = p.outputs(), .k = p.inputs() });
            total += aligned(a.bytes.len) + aligned(b.bytes.len);
            most_n = @max(most_n, p.outputs());
            most_rank = @max(most_rank, rank);
        };
        if (total == 0) return error.LoraEmpty;
        const scratch = aligned(block * most_rank * 4) + aligned(block * most_rank * 2) + aligned(block * most_n * 4);
        var buf = try cuda.DeviceBuffer.alloc(e.d, total + scratch);
        errdefer buf.free();
        for (at, 0..) |*row, i| for (std.enums.values(Slot)) |s| {
            const x = row.get(s) orelse continue;
            const a = f.get(try std.fmt.bufPrint(&name, "layers.{d}.{s}.lora_A", .{ i, geometry.slot_modules.get(s) })).?;
            const b = f.get(try std.fmt.bufPrint(&name, "layers.{d}.{s}.lora_B", .{ i, geometry.slot_modules.get(s) })).?;
            try buf.upload(x.a, a.bytes);
            try buf.upload(x.a + aligned(a.bytes.len), b.bytes);
            row.set(s, .{ .a = buf.ptr + x.a, .b = buf.ptr + x.a + aligned(a.bytes.len), .rank = x.rank, .n = x.n, .k = x.k });
        };
        var mod = try cuda.Module.load(e.d, cuda.kernels.qwen27_train);
        errdefer mod.unload();
        var dense = try cuda.dense_rows.DenseRows.load(e.d);
        errdefer dense.unload();
        const l = try gpa.create(Lora);
        const t32 = buf.ptr + total;
        const t16 = t32 + aligned(block * most_rank * 4);
        l.* = .{ .gpa = gpa, .dir = try gpa.dupe(u8, dir), .buf = buf, .mod = mod, .add = try mod.function("tf_q27_add_bf16"), .dense = dense, .t32 = t32, .t16 = t16, .d32 = t16 + aligned(block * most_rank * 2), .at = at };
        return l;
    }

    /// y [rows][n] bf16 += (bf16(x A^T)) B^T for one projection, `block` rows at a time.
    fn apply(l: *const Lora, e: *E.Engine, f: Factor, x: u64, y: u64, rows: usize) !void {
        var at: usize = 0;
        while (at < rows) : (at += block) {
            const m = @min(block, rows - at);
            try l.dense.rows(e.stream, x + at * f.k * 2, f.k, f.a, l.t32, f.rank, m, f.rank, f.k);
            try e.d.check(e.d.api.cuMemsetD8Async(l.t16, 0, m * f.rank * 2, e.stream.handle), "cuMemsetD8Async");
            try e.run(l.add, .{ (m * f.rank + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ l.t16, l.t32, @as(c_int, @intCast(m * f.rank)) });
            try l.dense.rows(e.stream, l.t16, f.rank, f.b, l.d32, f.n, m, f.n, f.rank);
            try e.run(l.add, .{ (m * f.n + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ y + at * f.n * 2, l.d32, @as(c_int, @intCast(m * f.n)) });
        }
    }
};

/// Each adapted projection of `slots` adds its term to its output in `ys`, read from `x`; nothing without a LoRA on.
pub fn after(e: *E.Engine, layer: usize, slots: []const Slot, x: u64, ys: []const u64, rows: usize) !void {
    const l = e.lora orelse return;
    // a captured graph would replay the term on chat rounds: the adapter never runs inside a capture
    if (try cuda.graph.captureStatus(e.stream) != .none) return error.QwenLoraDuringCapture;
    for (slots, ys) |s, y| if (l.factor(layer, s)) |f| try l.apply(e, f, x, y, rows);
}

/// The layer's projection in `s`, if it has one (routed experts and the other mixer's slots have none).
fn projOf(l: layers_mod.Layer, s: Slot) ?layers_mod.Proj {
    switch (l.mixer) {
        .delta => |m| switch (s) {
            .qkv => return m.qkv,
            .z => return m.z,
            .a => return m.a,
            .b => return m.b,
            .out => return m.out,
            else => {},
        },
        .attention => |m| switch (s) {
            .q => return m.q,
            .k => return m.k,
            .v => return m.v,
            .o => return m.o,
            else => {},
        },
    }
    return switch (l.mlp) {
        .dense => |m| switch (s) {
            .gate => m.gate,
            .up => m.up,
            .down => m.down,
            else => null,
        },
        .moe => null,
    };
}

fn aligned(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}

/// A safetensors file mapped read-only, its tensors by name.
const Tensors = struct {
    map: std.Io.File.MemoryMap,
    file: std.Io.File,
    data: usize,
    header: std.json.Parsed(std.json.Value),

    const T = struct {
        bf16: bool,
        bytes: []const u8,
        dims: [4]usize,
        nd: usize,
        fn ok2(t: T) bool {
            return t.nd == 2 and t.bf16;
        }
    };

    fn open(a: Allocator, io: std.Io, path: []const u8) !Tensors {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const len: usize = @intCast(try file.length(io));
        if (len < 8) return error.LoraFormat;
        const map = try std.Io.File.MemoryMap.create(io, file, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
        const hl: usize = @intCast(std.mem.readInt(u64, map.memory[0..8], .little));
        if (8 + hl > len) return error.LoraFormat;
        const header = try std.json.parseFromSlice(std.json.Value, a, map.memory[8..][0..hl], .{});
        return .{ .map = map, .file = file, .data = 8 + hl, .header = header };
    }

    fn close(t: *Tensors, io: std.Io) void {
        t.header.deinit();
        t.map.destroy(io);
        t.file.close(io);
    }

    fn get(t: *const Tensors, name: []const u8) ?T {
        const e = t.header.value.object.get(name) orelse return null;
        const off = e.object.get("data_offsets").?.array.items;
        const lo: usize = @intCast(off[0].integer);
        const hi: usize = @intCast(off[1].integer);
        const dims = e.object.get("shape").?.array.items;
        if (dims.len > 4 or lo > hi or t.data + hi > t.map.memory.len) return null;
        var out: T = .{ .bf16 = std.mem.eql(u8, e.object.get("dtype").?.string, "BF16"), .bytes = t.map.memory[t.data + lo .. t.data + hi], .dims = .{ 0, 0, 0, 0 }, .nd = dims.len };
        for (dims, 0..) |d, i| out.dims[i] = @intCast(d.integer);
        return out;
    }
};

test "Qwen adapter tensors: each lookup keeps its own shape, sizes and dtype" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header = "{\"layers.40.mlp.up_proj.lora_A\":{\"dtype\":\"BF16\",\"shape\":[8,16],\"data_offsets\":[0,256]}," ++
        "\"layers.40.mlp.up_proj.lora_B\":{\"dtype\":\"BF16\",\"shape\":[24,8],\"data_offsets\":[256,640]}}";
    var bytes: [8 + header.len + 640]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    @memset(bytes[8 + header.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "adapter.safetensors", .data = &bytes });
    const path = try tmp.dir.realPathFileAlloc(io, "adapter.safetensors", gpa);
    defer gpa.free(path);
    var t = try Tensors.open(gpa, io, path);
    defer t.close(io);
    const a = t.get("layers.40.mlp.up_proj.lora_A").?;
    const b = t.get("layers.40.mlp.up_proj.lora_B").?;
    try std.testing.expect(a.ok2() and b.ok2());
    try std.testing.expectEqual([2]usize{ 8, 16 }, a.dims[0..2].*);
    try std.testing.expectEqual([2]usize{ 24, 8 }, b.dims[0..2].*);
    try std.testing.expectEqual(@as(usize, 256), a.bytes.len);
    try std.testing.expectEqual(@as(usize, 384), b.bytes.len);
    try std.testing.expect(t.get("layers.40.mlp.down_proj.lora_A") == null);
}
