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
/// Chunks of 512 keys a row's attention reads at most: 2,048 selected keys plus a tail, or 2,051 dense keys.
pub const nch: usize = 5;
pub const dense_nch: usize = 3;
/// The block indexer (QSA): keys pooled in blocks of 4, a row past 512 complete blocks reads its 512 best.
pub const index_ratio: usize = 4;
pub const index_top: usize = 512;
/// A row's key list: the budget's 2,048 keys plus a tail of up to ratio - 1 (attention.AttnScratch.idw).
pub const index_idw: usize = index_top * index_ratio + index_ratio;
/// Keys a dense row reads at most before selection starts: (top + 1) x ratio - 1.
pub const dense_keys: usize = (index_top + 1) * index_ratio - 1;
/// The model's window (config max_position_embeddings).
pub const window: usize = 262144;
pub const ple_layer: usize = 1;
pub const inj_stride: usize = 16;
pub const wts_stride: usize = 48;
pub const ple_tail_n: usize = 9;
pub const state_n: usize = gdn.nv * gdn.dv * gdn.dk;
pub const hc_sk: usize = 32;
pub const out_sk: usize = 8;

pub const pack_symbol: [:0]const u8 = "_ZN15tf_experts_pack11pack_kernelILi1EEEvPKjPKtS4_Pjiiiiii";
pub const plan_symbol: [:0]const u8 = "_ZN10tf_experts11plan_kernelEPKiiiiPiS2_S2_";
pub const up_symbol: [:0]const u8 = "_ZN10tf_experts13expert_kernelILi32ELi2ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
pub const down_symbol: [:0]const u8 = "_ZN10tf_experts13expert_kernelILi32ELi1ELi0ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";

const Tensor = core.safetensors.Tensor;

const Held = struct { rel: []u8, file: Shard };

/// One mapped shard: core.safetensors.File.
const Shard = struct {
    f: core.safetensors.File,
    io: std.Io,

    fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Shard {
        return .{ .f = try core.safetensors.File.open(gpa, io, path), .io = io };
    }

    fn close(self: *Shard) void {
        self.f.close(self.io);
    }

    fn get(self: *const Shard, name: []const u8) ?Tensor {
        return self.f.get(name);
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
    released: [2]Held = undefined,
    nreleased: usize = 0,
    norm_offset: f32 = 0,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Store {
        const index = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
        defer gpa.free(index);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, index, gpa, .limited(1 << 26));
        defer gpa.free(text);
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        errdefer parsed.deinit();
        _ = try core.safetensors.index.weightMap(parsed.value);
        return .{ .gpa = gpa, .io = io, .dir = dir, .parsed = parsed };
    }

    /// Infer one checkpoint-wide convention from every decoder HC norm, never from the final mixer's outlier.
    pub fn detectNorms(self: *Store) !void {
        var means: [layers_n]f64 = undefined;
        var buf: [160]u8 = undefined;
        for (0..layers_n) |i| {
            const name = try std.fmt.bufPrint(&buf, "language_model.model.layers.{d}.attn_hyper_connection.hc_norm.weight", .{i});
            const values = try asF32(self.gpa, try self.tensor(name));
            defer self.gpa.free(values);
            if (values.len != wide) return error.UnexpectedTensor;
            var sum: f64 = 0;
            for (values) |v| {
                if (!std.math.isFinite(v)) return error.InvalidNormConvention;
                sum += v;
            }
            means[i] = sum / @as(f64, @floatFromInt(values.len));
        }
        self.norm_offset = try @import("cuda_norms.zig").offset(layers_n, means);
    }

    pub fn normScale(self: *const Store, gpa: std.mem.Allocator, t: Tensor) ![]f32 {
        const values = try asF32(gpa, t);
        errdefer gpa.free(values);
        for (values) |*v| {
            if (!std.math.isFinite(v.*)) return error.InvalidNormConvention;
            v.* += self.norm_offset;
        }
        return values;
    }

    pub fn deinit(self: *Store) void {
        self.release();
        self.closeReleased();
        self.closeFile();
        self.parsed.deinit();
    }

    /// Borrowers expire here; two indexed shards cover separate weight and calibration planes.
    pub fn release(self: *Store) void {
        for (self.held[0..self.nheld]) |h| {
            if (self.nreleased == self.released.len) {
                self.released[0].file.close();
                self.gpa.free(self.released[0].rel);
                self.released[0] = self.released[1];
                self.nreleased -= 1;
            }
            self.released[self.nreleased] = h;
            self.nreleased += 1;
        }
        self.nheld = 0;
    }

    fn closeReleased(self: *Store) void {
        for (self.released[0..self.nreleased]) |*h| {
            h.file.close();
            self.gpa.free(h.rel);
        }
        self.nreleased = 0;
    }

    pub fn hold(self: *Store, requested: []const u8) !Tensor {
        var mapped: [512]u8 = undefined;
        const key = try self.resolveKey(requested, &mapped);
        const file = try self.shard(key);
        for (self.held[0..self.nheld]) |*h| {
            if (std.mem.eql(u8, h.rel, file)) return h.file.get(key) orelse return error.MissingTensor;
        }
        if (self.nheld == self.held.len) return error.UnexpectedTensor;
        for (self.released[0..self.nreleased], 0..) |cached, i| {
            if (std.mem.eql(u8, cached.rel, file)) {
                self.held[self.nheld] = cached;
                self.nreleased -= 1;
                if (i < self.nreleased) self.released[i] = self.released[self.nreleased];
                self.nheld += 1;
                return self.held[self.nheld - 1].file.get(key) orelse return error.MissingTensor;
            }
        }
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

    pub fn resolveKey(self: *const Store, requested: []const u8, buf: []u8) ![]const u8 {
        const wm = self.parsed.value.object.get("weight_map") orelse return error.MissingTensor;
        if (wm.object.contains(requested)) return requested;
        const aliases = .{
            .{ "language_model.model.", "model.language_model." },
            .{ "language_model.lm_head.", "lm_head." },
            .{ "language_model.mtp.", "mtp." },
        };
        inline for (aliases) |pair| {
            if (std.mem.startsWith(u8, requested, pair[0])) {
                const name = try std.fmt.bufPrint(buf, "{s}{s}", .{ pair[1], requested[pair[0].len..] });
                if (wm.object.contains(name)) return name;
            }
        }
        return error.MissingTensor;
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

    pub fn tensor(self: *Store, requested: []const u8) !Tensor {
        var mapped: [512]u8 = undefined;
        const key = try self.resolveKey(requested, &mapped);
        try self.use(key);
        return self.file.?.get(key) orelse {
            std.debug.print("absent {s}\n", .{key});
            return error.MissingTensor;
        };
    }

    /// Check the exact or compatibility name without mapping its shard.
    pub fn has(self: *const Store, requested: []const u8) bool {
        var mapped: [512]u8 = undefined;
        _ = self.resolveKey(requested, &mapped) catch return false;
        return true;
    }

    /// Checkpoint-reader adapter: the borrowed tensor stays mapped until release().
    pub fn get(self: *Store, key: []const u8) !Tensor {
        return self.hold(key);
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

pub const Q4 = @import("cuda_projection.zig").Projection;

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

/// A tensor the engine leaves off the device: the n-gram tables, the draft head and the vision tower.
pub fn hostOnly(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".ple_embedding.ngram_embedding.") != null or
        std.mem.startsWith(u8, name, "language_model.mtp.") or
        std.mem.startsWith(u8, name, "mtp.") or
        std.mem.startsWith(u8, name, "model.visual.") or
        std.mem.startsWith(u8, name, "vision_tower.");
}

/// The checkpoint's tensor bytes the engine uploads, every shard the index names read for its header alone.
pub fn deviceBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    var store = try Store.open(gpa, io, dir);
    defer store.deinit();
    const wm = store.parsed.value.object.get("weight_map") orelse return error.MissingTensor;
    var files: std.StringHashMapUnmanaged(void) = .empty;
    defer files.deinit(gpa);
    var it = wm.object.iterator();
    while (it.next()) |kv| try files.put(gpa, kv.value_ptr.string, {});
    var total: u64 = 0;
    var fi = files.keyIterator();
    while (fi.next()) |rel| {
        const path = try std.fs.path.join(gpa, &.{ dir, rel.* });
        defer gpa.free(path);
        var sh = try Shard.load(gpa, io, path);
        defer sh.close();
        var ni = sh.f.names.iterator();
        while (ni.next()) |e| {
            if (!hostOnly(e.key_ptr.*)) total += e.value_ptr.end - e.value_ptr.begin;
        }
    }
    return total;
}

pub fn asF32(gpa: std.mem.Allocator, t: Tensor) ![]f32 {
    if (t.rank != 1) return error.UnexpectedTensor;
    const out = try gpa.alloc(f32, t.dim(0));
    errdefer gpa.free(out);
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
    const down_name = try std.fmt.bufPrint(&buf, "{s}.input_mix_weight_down.weight", .{prefix});
    const down_tensor = try store.tensor(down_name);
    if (down_tensor.dtype == .bf16) return loadDenseHc(gpa, driver, store, prefix, inject);
    const down = try join3(gpa, store, &buf, prefix, ".input_mix_weight_down");
    defer gpa.free(down.w);
    defer gpa.free(down.s);
    defer gpa.free(down.b);
    const up = try join3(gpa, store, &buf, prefix, ".input_mix_weight_up");
    defer gpa.free(up.w);
    defer gpa.free(up.s);
    defer gpa.free(up.b);
    const gamma = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.hc_norm.weight", .{prefix}));
    const scale = try store.normScale(gpa, gamma);
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

fn loadDenseHc(gpa: std.mem.Allocator, driver: *cuda.Driver, store: *Store, prefix: []const u8, inject: bool) !HcW {
    const projection = @import("cuda_projection.zig");
    var buf: [200]u8 = undefined;
    const n = low + if (inject) streams else @as(usize, 0);
    const data = try gpa.alloc(u8, n * wide * 2);
    defer gpa.free(data);
    const down = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.input_mix_weight_down.weight", .{prefix}));
    if (!down.is(.bf16, &.{ low, wide }) or down.bytes.len != low * wide * 2) return error.UnexpectedTensor;
    @memcpy(data[0..down.bytes.len], down.bytes);
    if (inject) {
        const inj = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.block_inject_weight.weight", .{prefix}));
        if (!inj.is(.bf16, &.{ streams, wide }) or inj.bytes.len != streams * wide * 2) return error.UnexpectedTensor;
        @memcpy(data[low * wide * 2 ..], inj.bytes);
    }
    var dq = try projection.dense(gpa, driver, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, wide, 1, 1, 1 }, .bytes = data });
    errdefer dq.free();
    const up = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.input_mix_weight_up.weight", .{prefix}));
    if (!up.is(.bf16, &.{ wide, low })) return error.UnexpectedTensor;
    var uq = try projection.dense(gpa, driver, up);
    errdefer uq.free();
    const gamma = try store.normScale(gpa, try store.tensor(try std.fmt.bufPrint(&buf, "{s}.hc_norm.weight", .{prefix})));
    defer gpa.free(gamma);
    if (gamma.len != wide) return error.UnexpectedTensor;
    const scale = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(gamma));
    return .{ .scale = scale, .down = dq, .up = uq, .n_down = n, .has_inj = @intFromBool(inject) };
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

test {
    _ = @import("cuda_prompt_wt_test.zig");
}
