//! Checkpoint bytes for one drafts-off prompt: the safetensors reader and the quantized linears.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hc = @import("cuda_hc.zig");
const qmm = @import("cuda_qmm.zig");
const gdn = @import("cuda_gdn.zig");

pub const layers_n: usize = 48;
pub const linear_n: usize = 36;
pub const attn_n: usize = 12;
pub const dims: usize = 2560;
pub const streams: usize = 4;
pub const wide: usize = streams * dims;
pub const low: usize = 320;
pub const eps: f32 = 1e-6;
pub const vocab: usize = 248320;
pub const experts_n: usize = 512;
pub const slots: usize = 11;
pub const moe_w: usize = 640;
pub const block_bytes: usize = 160 * 4;
pub const tile: usize = 16;
pub const q_heads: usize = 24;
pub const kv_heads: usize = 2;
pub const head_dim: usize = 256;
pub const index_heads: usize = 4;
pub const index_dim: usize = 128;
pub const half: usize = 32;
pub const proj_n: usize = 13952;
pub const out_k: usize = 6144;
pub const nch: usize = 3;
pub const ple_layer: usize = 1;
pub const inj_stride: usize = 16;
pub const wts_stride: usize = 48;
pub const ple_tail_n: usize = 9;
pub const state_n: usize = gdn.nv * gdn.dv * gdn.dk;
pub const hc_sk: usize = 32;
pub const out_sk: usize = 8;

pub const pack_symbol: [:0]const u8 = "_ZN15tf_experts_pack11pack_kernelILi1EEEvPKjPKtS4_Pjiiii";
pub const plan_symbol: [:0]const u8 = "_ZN10tf_experts11plan_kernelEPKiiiiPiS2_S2_";
pub const up_symbol: [:0]const u8 = "_ZN10tf_experts13expert_kernelILi32ELi2ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
pub const down_symbol: [:0]const u8 = "_ZN10tf_experts13expert_kernelILi32ELi1ELi0ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";

const Tensor = core.safetensors.Tensor;

const Held = struct { rel: []u8, file: Shard };

const Shard = struct {
    io: std.Io,
    file: std.Io.File,
    map: std.Io.File.MemoryMap,
    at: usize,
    arena: std.heap.ArenaAllocator,
    names: std.StringHashMapUnmanaged(core.safetensors.Entry),

    fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Shard {
        var file = try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const len: usize = @intCast(try file.length(io));
        if (len < 8) return error.BadSafetensors;
        var map = try std.Io.File.MemoryMap.create(io, file, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
        errdefer map.destroy(io);
        const header_len: usize = @intCast(std.mem.readInt(u64, map.memory[0..8], .little));
        if (header_len > len - 8) return error.BadSafetensors;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        var names: std.StringHashMapUnmanaged(core.safetensors.Entry) = .empty;
        const parsed = try std.json.parseFromSlice(std.json.Value, arena.allocator(), map.memory[8..][0..header_len], .{});
        defer parsed.deinit();
        var it = parsed.value.object.iterator();
        while (it.next()) |kv| {
            if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
            const o = kv.value_ptr.object;
            const dtype = core.safetensors.DType.parse(o.get("dtype").?.string) orelse return error.UnsupportedDType;
            const shape = o.get("shape").?.array.items;
            if (shape.len > core.safetensors.max_rank) continue;
            var e: core.safetensors.Entry = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .begin = 0, .end = 0 };
            var n: usize = dtype.size();
            for (shape, 0..) |d, i| {
                e.shape[i] = @intCast(d.integer);
                n *= e.shape[i];
            }
            const offs = o.get("data_offsets").?.array.items;
            e.begin = @intCast(offs[0].integer);
            e.end = @intCast(offs[1].integer);
            if (e.end < e.begin or e.end - e.begin != n or e.end > len - 8 - header_len) return error.BadSafetensors;
            try names.put(arena.allocator(), try arena.allocator().dupe(u8, kv.key_ptr.*), e);
        }
        return .{ .io = io, .file = file, .map = map, .at = 8 + header_len, .arena = arena, .names = names };
    }

    fn close(self: *Shard) void {
        self.map.destroy(self.io);
        self.file.close(self.io);
        self.arena.deinit();
        self.* = undefined;
    }

    fn get(self: *const Shard, name: []const u8) ?Tensor {
        const e = self.names.get(name) orelse return null;
        return .{ .dtype = e.dtype, .rank = e.rank, .shape = e.shape, .bytes = self.map.memory[self.at + e.begin .. self.at + e.end] };
    }
};

pub const Store = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    parsed: std.json.Parsed(std.json.Value),
    path: []u8 = &.{},
    rel: []u8 = &.{},
    file: ?Shard = null,
    held: [12]Held = undefined,
    nheld: usize = 0,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Store {
        const index = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
        defer gpa.free(index);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, index, gpa, .limited(1 << 26));
        defer gpa.free(text);
        return .{ .gpa = gpa, .io = io, .dir = dir, .parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{}) };
    }

    pub fn deinit(self: *Store) void {
        self.release();
        self.closeFile();
        self.parsed.deinit();
    }

    pub fn release(self: *Store) void {
        for (self.held[0..self.nheld]) |*h| {
            h.file.close();
            self.gpa.free(h.rel);
        }
        self.nheld = 0;
    }

    pub fn hold(self: *Store, key: []const u8) !Tensor {
        const file = try self.shard(key);
        for (self.held[0..self.nheld]) |*h| {
            if (std.mem.eql(u8, h.rel, file)) return h.file.get(key) orelse return error.MissingTensor;
        }
        if (self.nheld == self.held.len) return error.UnexpectedTensor;
        const rel = try self.gpa.dupe(u8, file);
        errdefer self.gpa.free(rel);
        const path = try std.fs.path.join(self.gpa, &.{ self.dir, file });
        defer self.gpa.free(path);
        self.held[self.nheld] = .{ .rel = rel, .file = try Shard.load(self.gpa, self.io, path) };
        self.nheld += 1;
        return self.held[self.nheld - 1].file.get(key) orelse {
            std.debug.print("absent {s}\n", .{key});
            return error.MissingTensor;
        };
    }

    pub fn closeFile(self: *Store) void {
        if (self.file) |*f| f.close();
        self.file = null;
        if (self.path.len != 0) self.gpa.free(self.path);
        self.path = &.{};
        if (self.rel.len != 0) self.gpa.free(self.rel);
        self.rel = &.{};
    }

    fn shard(self: *Store, key: []const u8) ![]const u8 {
        const wm = self.parsed.value.object.get("weight_map") orelse return error.MissingTensor;
        const v = wm.object.get(key) orelse {
            std.debug.print("missing {s}\n", .{key});
            return error.MissingTensor;
        };
        return v.string;
    }

    fn use(self: *Store, key: []const u8) !void {
        const file = try self.shard(key);
        if (self.file != null and std.mem.eql(u8, file, self.rel)) return;
        self.closeFile();
        self.rel = try self.gpa.dupe(u8, file);
        self.path = try std.fs.path.join(self.gpa, &.{ self.dir, file });
        self.file = try Shard.load(self.gpa, self.io, self.path);
    }

    pub fn tensor(self: *Store, key: []const u8) !Tensor {
        try self.use(key);
        return self.file.?.get(key) orelse {
            std.debug.print("absent {s}\n", .{key});
            return error.MissingTensor;
        };
    }

    pub fn copyOf(self: *Store, key: []const u8) ![]u8 {
        const t = try self.tensor(key);
        const out = try self.gpa.alloc(u8, t.bytes.len);
        @memcpy(out, t.bytes);
        return out;
    }
};

pub const Face = struct {
    w: []const u8,
    s: []const u8,
    b: []const u8,
    n: usize,
    k: usize,
    routed: bool,
};

pub const Q4 = struct {
    w: cuda.DeviceBuffer,
    s: cuda.DeviceBuffer,
    b: cuda.DeviceBuffer,
    n: usize,
    k: usize,

    pub fn free(self: *Q4) void {
        self.w.free();
        self.s.free();
        self.b.free();
    }

    pub fn empty(d: *cuda.Driver) Q4 {
        const z: cuda.DeviceBuffer = .{ .d = d, .ptr = 0, .len = 0 };
        return .{ .w = z, .s = z, .b = z, .n = 0, .k = 0 };
    }
};

pub const HcW = struct {
    scale: cuda.DeviceBuffer,
    down: Q4,
    up: Q4,
    n_down: usize,
    has_inj: usize,

    pub fn free(self: *HcW) void {
        self.scale.free();
        self.down.free();
        self.up.free();
    }
};

pub fn linearLayer(layer: usize) bool {
    return layer % 4 != 3;
}

pub fn linearIndex(layer: usize) usize {
    return layer - layer / 4;
}

pub fn attnIndex(layer: usize) usize {
    return layer / 4;
}

pub fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

pub fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub fn cint(v: usize) c_int {
    return @intCast(v);
}

pub fn cat(gpa: std.mem.Allocator, parts: []const []const u8) ![]u8 {
    var n: usize = 0;
    for (parts) |p| n += p.len;
    const out = try gpa.alloc(u8, n);
    var at: usize = 0;
    for (parts) |p| {
        @memcpy(out[at..][0..p.len], p);
        at += p.len;
    }
    return out;
}

pub fn asF32(gpa: std.mem.Allocator, t: Tensor) ![]f32 {
    if (t.rank != 1) return error.UnexpectedTensor;
    const out = try gpa.alloc(f32, t.dim(0));
    if (t.dtype == .f32 and t.bytes.len == out.len * 4) {
        for (out, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little));
    } else if (t.dtype == .bf16 and t.bytes.len == out.len * 2) {
        for (out, 0..) |*o, i| o.* = promote(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little));
    } else return error.UnexpectedTensor;
    return out;
}

pub fn uploadQ4(gpa: std.mem.Allocator, driver: *cuda.Driver, words: []const u8, scales: []const u8, biases: []const u8, n: usize, k8: usize) !Q4 {
    var lane = try qmm.pack(gpa, words, scales, biases, n, k8);
    defer lane.deinit(gpa);
    var w = try cuda.DeviceBuffer.fromHost(driver, lane.weight);
    errdefer w.free();
    var s = try cuda.DeviceBuffer.fromHost(driver, lane.scales);
    errdefer s.free();
    const b = try cuda.DeviceBuffer.fromHost(driver, lane.biases);
    return .{ .w = w, .s = s, .b = b, .n = n, .k = k8 * 8 };
}

pub fn uploadTiled(gpa: std.mem.Allocator, driver: *cuda.Driver, words: []const u8, scales: []const u8, biases: []const u8, n: usize, k8: usize) !Q4 {
    const kg = k8 / 4;
    const tiled = try hc.tileWords(gpa, words, n, k8);
    defer gpa.free(tiled);
    const ts = try hc.transposeBf16(gpa, scales, n, kg);
    defer gpa.free(ts);
    const tb = try hc.transposeBf16(gpa, biases, n, kg);
    defer gpa.free(tb);
    var w = try cuda.DeviceBuffer.fromHost(driver, tiled);
    errdefer w.free();
    var s = try cuda.DeviceBuffer.fromHost(driver, ts);
    errdefer s.free();
    const b = try cuda.DeviceBuffer.fromHost(driver, tb);
    return .{ .w = w, .s = s, .b = b, .n = n, .k = k8 * 8 };
}

pub fn join3(gpa: std.mem.Allocator, store: *Store, buf: []u8, prefix: []const u8, name: []const u8) !struct { w: []u8, s: []u8, b: []u8 } {
    const w = try store.copyOf(try std.fmt.bufPrint(buf, "{s}{s}.weight", .{ prefix, name }));
    errdefer gpa.free(w);
    const s = try store.copyOf(try std.fmt.bufPrint(buf, "{s}{s}.scales", .{ prefix, name }));
    errdefer gpa.free(s);
    const b = try store.copyOf(try std.fmt.bufPrint(buf, "{s}{s}.biases", .{ prefix, name }));
    return .{ .w = w, .s = s, .b = b };
}

pub fn loadHc(gpa: std.mem.Allocator, driver: *cuda.Driver, store: *Store, prefix: []const u8, inject: bool) !HcW {
    var buf: [180]u8 = undefined;
    const down = try join3(gpa, store, &buf, prefix, ".input_mix_weight_down");
    defer gpa.free(down.w);
    defer gpa.free(down.s);
    defer gpa.free(down.b);
    const up = try join3(gpa, store, &buf, prefix, ".input_mix_weight_up");
    defer gpa.free(up.w);
    defer gpa.free(up.s);
    defer gpa.free(up.b);
    const gamma = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.hc_norm.weight", .{prefix}));
    const scale = try asF32(gpa, gamma);
    defer gpa.free(scale);
    if (scale.len != wide) return error.UnexpectedTensor;
    var n: usize = low;
    var words: []u8 = undefined;
    var scales: []u8 = undefined;
    var biases: []u8 = undefined;
    var inj_w: []u8 = &.{};
    var inj_s: []u8 = &.{};
    var inj_b: []u8 = &.{};
    if (inject) {
        const inj = try join3(gpa, store, &buf, prefix, ".block_inject_weight");
        inj_w = inj.w;
        inj_s = inj.s;
        inj_b = inj.b;
        n = low + 4;
        words = try cat(gpa, &.{ down.w, inj.w });
        scales = try cat(gpa, &.{ down.s, inj.s });
        biases = try cat(gpa, &.{ down.b, inj.b });
    } else {
        words = try gpa.dupe(u8, down.w);
        scales = try gpa.dupe(u8, down.s);
        biases = try gpa.dupe(u8, down.b);
    }
    defer gpa.free(words);
    defer gpa.free(scales);
    defer gpa.free(biases);
    defer if (inject) gpa.free(inj_w);
    defer if (inject) gpa.free(inj_s);
    defer if (inject) gpa.free(inj_b);
    const k8 = wide / 8;
    if (words.len != n * k8 * 4) return error.UnexpectedTensor;
    var down_q = try uploadTiled(gpa, driver, words, scales, biases, n, k8);
    errdefer down_q.free();
    var up_q = try uploadTiled(gpa, driver, up.w, up.s, up.b, wide, low / 8);
    errdefer up_q.free();
    const scale_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(scale));
    return .{ .scale = scale_b, .down = down_q, .up = up_q, .n_down = n, .has_inj = @intFromBool(inject) };
}

pub fn stitch(gpa: std.mem.Allocator, store: *Store, prefix: []const u8, names: []const []const u8) !struct { w: []u8, s: []u8, b: []u8, n: usize } {
    var buf: [180]u8 = undefined;
    var ws: [6][]u8 = undefined;
    var ss: [6][]u8 = undefined;
    var bs: [6][]u8 = undefined;
    var n: usize = 0;
    for (names, 0..) |part, i| {
        const got = try join3(gpa, store, &buf, prefix, part);
        ws[i] = got.w;
        ss[i] = got.s;
        bs[i] = got.b;
        n += 1;
    }
    defer for (0..n) |i| {
        gpa.free(ws[i]);
        gpa.free(ss[i]);
        gpa.free(bs[i]);
    };
    return .{ .w = try cat(gpa, ws[0..n]), .s = try cat(gpa, ss[0..n]), .b = try cat(gpa, bs[0..n]), .n = n };
}
