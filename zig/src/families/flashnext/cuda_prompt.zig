//! One prompt, drafts off: each token is one fused decode row, layers in order, then the argmax.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const embed = @import("cuda_embed.zig");
const hc = @import("cuda_hc.zig");
const qmm = @import("cuda_qmm.zig");
const gdn = @import("cuda_gdn.zig");
const ple = @import("cuda_ple.zig");

const layers_n: usize = 48;
const linear_n: usize = 36;
const attn_n: usize = 12;
const dims: usize = 2560;
const streams: usize = 4;
const wide: usize = streams * dims;
const low: usize = 320;
const eps: f32 = 1e-6;
const vocab: usize = 248320;
const experts_n: usize = 512;
const slots: usize = 11;
const moe_w: usize = 640;
const block_bytes: usize = 160 * 4;
const tile: usize = 16;
const q_heads: usize = 24;
const kv_heads: usize = 2;
const head_dim: usize = 256;
const index_heads: usize = 4;
const index_dim: usize = 128;
const half: usize = 32;
const proj_n: usize = 13952;
const out_k: usize = 6144;
const nch: usize = 3;
const ple_layer: usize = 1;
const inj_stride: usize = 16;
const wts_stride: usize = 48;
const ple_tail_n: usize = 9;
const state_n: usize = gdn.nv * gdn.dv * gdn.dk;
const hc_sk: usize = 32;
const out_sk: usize = 8;

const pack_symbol: [:0]const u8 = "_ZN15tf_experts_pack11pack_kernelILi1EEEvPKjPKtS4_Pjiiii";
const plan_symbol: [:0]const u8 = "_ZN10tf_experts11plan_kernelEPKiiiiPiS2_S2_";
const up_symbol: [:0]const u8 = "_ZN10tf_experts13expert_kernelILi32ELi2ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
const down_symbol: [:0]const u8 = "_ZN10tf_experts13expert_kernelILi32ELi1ELi0ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";

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

const Store = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    parsed: std.json.Parsed(std.json.Value),
    path: []u8 = &.{},
    rel: []u8 = &.{},
    file: ?Shard = null,
    held: [12]Held = undefined,
    nheld: usize = 0,

    fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Store {
        const index = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
        defer gpa.free(index);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, index, gpa, .limited(1 << 26));
        defer gpa.free(text);
        return .{ .gpa = gpa, .io = io, .dir = dir, .parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{}) };
    }

    fn deinit(self: *Store) void {
        self.release();
        self.closeFile();
        self.parsed.deinit();
    }

    fn release(self: *Store) void {
        for (self.held[0..self.nheld]) |*h| {
            h.file.close();
            self.gpa.free(h.rel);
        }
        self.nheld = 0;
    }

    fn hold(self: *Store, key: []const u8) !Tensor {
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

    fn closeFile(self: *Store) void {
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

    fn tensor(self: *Store, key: []const u8) !Tensor {
        try self.use(key);
        return self.file.?.get(key) orelse {
            std.debug.print("absent {s}\n", .{key});
            return error.MissingTensor;
        };
    }

    fn copyOf(self: *Store, key: []const u8) ![]u8 {
        const t = try self.tensor(key);
        const out = try self.gpa.alloc(u8, t.bytes.len);
        @memcpy(out, t.bytes);
        return out;
    }
};

const Face = struct {
    w: []const u8,
    s: []const u8,
    b: []const u8,
    n: usize,
    k: usize,
    routed: bool,
};

const Q4 = struct {
    w: cuda.DeviceBuffer,
    s: cuda.DeviceBuffer,
    b: cuda.DeviceBuffer,
    n: usize,
    k: usize,

    fn free(self: *Q4) void {
        self.w.free();
        self.s.free();
        self.b.free();
    }

    fn empty(d: *cuda.Driver) Q4 {
        const z: cuda.DeviceBuffer = .{ .d = d, .ptr = 0, .len = 0 };
        return .{ .w = z, .s = z, .b = z, .n = 0, .k = 0 };
    }
};

const HcW = struct {
    scale: cuda.DeviceBuffer,
    down: Q4,
    up: Q4,
    n_down: usize,
    has_inj: usize,

    fn free(self: *HcW) void {
        self.scale.free();
        self.down.free();
        self.up.free();
    }
};

fn linearLayer(layer: usize) bool {
    return layer % 4 != 3;
}

fn linearIndex(layer: usize) usize {
    return layer - layer / 4;
}

fn attnIndex(layer: usize) usize {
    return layer / 4;
}

fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn cint(v: usize) c_int {
    return @intCast(v);
}

fn cat(gpa: std.mem.Allocator, parts: []const []const u8) ![]u8 {
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

fn asF32(gpa: std.mem.Allocator, t: Tensor) ![]f32 {
    if (t.rank != 1) return error.UnexpectedTensor;
    const out = try gpa.alloc(f32, t.dim(0));
    if (t.dtype == .f32 and t.bytes.len == out.len * 4) {
        for (out, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little));
    } else if (t.dtype == .bf16 and t.bytes.len == out.len * 2) {
        for (out, 0..) |*o, i| o.* = promote(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little));
    } else return error.UnexpectedTensor;
    return out;
}

fn uploadQ4(gpa: std.mem.Allocator, driver: *cuda.Driver, words: []const u8, scales: []const u8, biases: []const u8, n: usize, k8: usize) !Q4 {
    var lane = try qmm.pack(gpa, words, scales, biases, n, k8);
    defer lane.deinit(gpa);
    var w = try cuda.DeviceBuffer.fromHost(driver, lane.weight);
    errdefer w.free();
    var s = try cuda.DeviceBuffer.fromHost(driver, lane.scales);
    errdefer s.free();
    const b = try cuda.DeviceBuffer.fromHost(driver, lane.biases);
    return .{ .w = w, .s = s, .b = b, .n = n, .k = k8 * 8 };
}

fn uploadTiled(gpa: std.mem.Allocator, driver: *cuda.Driver, words: []const u8, scales: []const u8, biases: []const u8, n: usize, k8: usize) !Q4 {
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

fn join3(gpa: std.mem.Allocator, store: *Store, buf: []u8, prefix: []const u8, name: []const u8) !struct { w: []u8, s: []u8, b: []u8 } {
    const w = try store.copyOf(try std.fmt.bufPrint(buf, "{s}{s}.weight", .{ prefix, name }));
    errdefer gpa.free(w);
    const s = try store.copyOf(try std.fmt.bufPrint(buf, "{s}{s}.scales", .{ prefix, name }));
    errdefer gpa.free(s);
    const b = try store.copyOf(try std.fmt.bufPrint(buf, "{s}{s}.biases", .{ prefix, name }));
    return .{ .w = w, .s = s, .b = b };
}

fn loadHc(gpa: std.mem.Allocator, driver: *cuda.Driver, store: *Store, prefix: []const u8, inject: bool) !HcW {
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

fn stitch(gpa: std.mem.Allocator, store: *Store, prefix: []const u8, names: []const []const u8) !struct { w: []u8, s: []u8, b: []u8, n: usize } {
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

const Scratch = struct {
    h: cuda.DeviceBuffer,
    y: cuda.DeviceBuffer,
    wts: cuda.DeviceBuffer,
    inj: cuda.DeviceBuffer,
    pss: cuda.DeviceBuffer,
    norm: cuda.DeviceBuffer,
    dn: cuda.DeviceBuffer,
    hc_part: cuda.DeviceBuffer,
    act: cuda.DeviceBuffer,
    xs_act: cuda.DeviceBuffer,
    mixed: cuda.DeviceBuffer,
    xsm: cuda.DeviceBuffer,
    proj: cuda.DeviceBuffer,
    part: cuda.DeviceBuffer,
    gout: cuda.DeviceBuffer,
    gxs: cuda.DeviceBuffer,
    st: cuda.DeviceBuffer,
    ple_emb: cuda.DeviceBuffer,
    ple_xs: cuda.DeviceBuffer,
    ple_keys: cuda.DeviceBuffer,
    ple_vals: cuda.DeviceBuffer,
    ple_gated: cuda.DeviceBuffer,
    ple_pss: cuda.DeviceBuffer,
    ple_nrow: cuda.DeviceBuffer,
    ple_part: cuda.DeviceBuffer,
    pa: cuda.DeviceBuffer,
    q: cuda.DeviceBuffer,
    iq: cuda.DeviceBuffer,
    po: cuda.DeviceBuffer,
    pm: cuda.DeviceBuffer,
    pl: cuda.DeviceBuffer,
    ao: cuda.DeviceBuffer,
    gated: cuda.DeviceBuffer,
    gxs_attn: cuda.DeviceBuffer,
    dummy: cuda.DeviceBuffer,
    inv: cuda.DeviceBuffer,
    pos: cuda.DeviceBuffer,
    ids: cuda.DeviceBuffer,
    logits: cuda.DeviceBuffer,
    router_l: cuda.DeviceBuffer,
    picks: cuda.DeviceBuffer,

    fn alloc(driver: *cuda.Driver, rows: usize) !Scratch {
        var s: Scratch = undefined;
        s.h = try cuda.DeviceBuffer.alloc(driver, rows * wide * 2);
        s.y = try cuda.DeviceBuffer.alloc(driver, rows * slots * dims * 4);
        s.wts = try cuda.DeviceBuffer.alloc(driver, rows * wts_stride);
        s.inj = try cuda.DeviceBuffer.alloc(driver, rows * inj_stride);
        s.pss = try cuda.DeviceBuffer.alloc(driver, (dims / 256) * streams * 4);
        s.norm = try cuda.DeviceBuffer.alloc(driver, wide * 2);
        s.dn = try cuda.DeviceBuffer.alloc(driver, (low + 4) * 2);
        s.hc_part = try cuda.DeviceBuffer.alloc(driver, hc_sk * (low + 4) * 4);
        s.act = try cuda.DeviceBuffer.alloc(driver, low * 2);
        s.xs_act = try cuda.DeviceBuffer.alloc(driver, (low / 32) * 4);
        s.mixed = try cuda.DeviceBuffer.alloc(driver, dims * 2);
        s.xsm = try cuda.DeviceBuffer.alloc(driver, (dims / 32) * 4);
        s.proj = try cuda.DeviceBuffer.alloc(driver, gdn.proj_width * 2);
        s.part = try cuda.DeviceBuffer.alloc(driver, out_sk * dims * 4);
        s.gout = try cuda.DeviceBuffer.alloc(driver, gdn.value_dim * 2);
        s.gxs = try cuda.DeviceBuffer.alloc(driver, (gdn.value_dim / 32) * 4);
        s.st = try cuda.DeviceBuffer.alloc(driver, state_n * 4);
        s.ple_emb = try cuda.DeviceBuffer.alloc(driver, dims * 2);
        s.ple_xs = try cuda.DeviceBuffer.alloc(driver, (dims / 32) * 4);
        s.ple_keys = try cuda.DeviceBuffer.alloc(driver, wide * 2);
        s.ple_vals = try cuda.DeviceBuffer.alloc(driver, dims * 2);
        s.ple_gated = try cuda.DeviceBuffer.alloc(driver, wide * 2);
        s.ple_pss = try cuda.DeviceBuffer.alloc(driver, streams * 4);
        s.ple_nrow = try cuda.DeviceBuffer.alloc(driver, wide * 2);
        s.ple_part = try cuda.DeviceBuffer.alloc(driver, 4 * dims * 4);
        s.pa = try cuda.DeviceBuffer.alloc(driver, proj_n * 2);
        s.q = try cuda.DeviceBuffer.alloc(driver, q_heads * head_dim * 2);
        s.iq = try cuda.DeviceBuffer.alloc(driver, index_heads * index_dim * 2);
        s.po = try cuda.DeviceBuffer.alloc(driver, nch * q_heads * head_dim * 4);
        s.pm = try cuda.DeviceBuffer.alloc(driver, nch * q_heads * 4);
        s.pl = try cuda.DeviceBuffer.alloc(driver, nch * q_heads * 4);
        s.ao = try cuda.DeviceBuffer.alloc(driver, out_k * 2);
        s.gated = try cuda.DeviceBuffer.alloc(driver, out_k * 2);
        s.gxs_attn = try cuda.DeviceBuffer.alloc(driver, (out_k / 32) * 4);
        s.dummy = try cuda.DeviceBuffer.alloc(driver, 4096);
        var inv: [half]f32 = undefined;
        for (&inv, 0..) |*o, i| o.* = @floatCast(std.math.pow(f64, 10_000_000.0, -@as(f64, @floatFromInt(i)) / @as(f64, half)));
        s.inv = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(&inv));
        s.pos = try cuda.DeviceBuffer.alloc(driver, 4);
        s.ids = try cuda.DeviceBuffer.alloc(driver, 4);
        s.logits = try cuda.DeviceBuffer.alloc(driver, vocab * 2);
        s.router_l = try cuda.DeviceBuffer.alloc(driver, (experts_n + 1) * 4);
        s.picks = try cuda.DeviceBuffer.alloc(driver, slots * 4);
        return s;
    }

    fn free(self: *Scratch) void {
        self.h.free();
        self.y.free();
        self.wts.free();
        self.inj.free();
        self.pss.free();
        self.norm.free();
        self.dn.free();
        self.hc_part.free();
        self.act.free();
        self.xs_act.free();
        self.mixed.free();
        self.xsm.free();
        self.proj.free();
        self.part.free();
        self.gout.free();
        self.gxs.free();
        self.st.free();
        self.ple_emb.free();
        self.ple_xs.free();
        self.ple_keys.free();
        self.ple_vals.free();
        self.ple_gated.free();
        self.ple_pss.free();
        self.ple_nrow.free();
        self.ple_part.free();
        self.pa.free();
        self.q.free();
        self.iq.free();
        self.po.free();
        self.pm.free();
        self.pl.free();
        self.ao.free();
        self.gated.free();
        self.gxs_attn.free();
        self.dummy.free();
        self.inv.free();
        self.pos.free();
        self.ids.free();
        self.logits.free();
        self.router_l.free();
        self.picks.free();
    }
};

fn writeback(tri: anytype, s: *Scratch, h: u64, mode: usize, inj: u64, y: u64, wts: u64) !void {
    const dummy = s.dummy.ptr;
    if (mode == 0) {
        try tri.hcWriteback(h, h, s.pss.ptr, dummy, "*bf16", dummy, dummy, "*bf16", dummy, "*bf16", dims, 1, dims, streams, 0, 1, 1, 1);
    } else if (mode == 2) {
        try tri.hcWriteback(h, h, s.pss.ptr, dummy, "*bf16", inj, y, "*fp32", wts, "*fp32", dims, 1, dims, streams, 2, 10, 11, 1);
    } else if (mode == 4) {
        try tri.hcWriteback(h, h, s.pss.ptr, s.part.ptr, "*fp32", inj, dummy, "*bf16", dummy, "*bf16", dims, 1, dims, streams, 4, 1, 1, out_sk);
    } else return error.UnexpectedTensor;
}

fn readout(tri: anytype, s: *Scratch, hc_w: HcW, h: u64, inj: u64) !void {
    try tri.hcDown(h, s.pss.ptr, hc_w.scale.ptr, s.norm.ptr, hc_w.down.w.ptr, hc_w.down.s.ptr, hc_w.down.b.ptr, s.dn.ptr, s.hc_part.ptr, eps, 1, hc_w.n_down, wide, dims, dims / 256, streams, hc_sk);
    const inj_ptr = if (hc_w.has_inj == 1) inj else s.act.ptr;
    try tri.hcReduceAct(s.hc_part.ptr, s.act.ptr, s.xs_act.ptr, inj_ptr, hc_sk, 1, streams, low, hc_w.n_down, hc_w.has_inj);
    try tri.hcUpmix(s.act.ptr, s.xs_act.ptr, hc_w.up.w.ptr, hc_w.up.s.ptr, hc_w.up.b.ptr, s.norm.ptr, s.mixed.ptr, s.xsm.ptr, 1, wide, low, dims, streams);
}

fn shiftTail(gpa: std.mem.Allocator, stream: *cuda.Stream, window: cuda.DeviceBuffer, taps_n: usize, row_bytes: usize, fresh: []const u8) !void {
    const bytes = taps_n * row_bytes;
    const raw = try gpa.alloc(u8, bytes);
    defer gpa.free(raw);
    try stream.synchronize();
    try window.download(0, raw);
    const next = try gpa.alloc(u8, bytes);
    defer gpa.free(next);
    const keep = taps_n - 1;
    @memcpy(next[0 .. keep * row_bytes], raw[row_bytes..]);
    @memcpy(next[keep * row_bytes ..], fresh);
    try window.upload(0, next);
}

fn gdnStep(gpa: std.mem.Allocator, driver: *cuda.Driver, stream: *cuda.Stream, s: *Scratch, proj_q: Q4, out_q: Q4, conv: cuda.DeviceBuffer, cw: u64, a_log: u64, dt: u64, norm: u64, state: *cuda.DeviceBuffer) !void {
    try qmm.matmul(driver, stream.*, s.mixed.ptr, s.xsm.ptr, proj_q.w.ptr, proj_q.s.ptr, proj_q.b.ptr, s.proj.ptr, 1, proj_q.n, proj_q.k);
    try gdn.launch(driver, stream.*, s.proj.ptr, conv.ptr, cw, state.ptr, a_log, dt, norm, eps, s.gout.ptr, s.gxs.ptr, s.st.ptr);
    try state.copyFrom(0, s.st.ptr, state_n * 4, stream.handle);
    const row = try gpa.alloc(u8, gdn.conv_dim * 2);
    defer gpa.free(row);
    try stream.synchronize();
    try s.proj.download(0, row);
    try shiftTail(gpa, stream, conv, gdn.taps - 1, gdn.conv_dim * 2, row);
    try qmm.partials(driver, stream.*, s.gout.ptr, s.gxs.ptr, out_q.w.ptr, out_q.s.ptr, out_q.b.ptr, s.dummy.ptr, s.part.ptr, 1, dims, gdn.value_dim, out_sk);
}

fn attnStep(comptime Tri: type, driver: *cuda.Driver, stream: *cuda.Stream, tri: Tri, s: *Scratch, proj_q: Q4, out_q: Q4, q_scale: u64, k_scale: u64, i_scale: u64, kc: u64, vc: u64, ikc: u64, pos: i32) !void {
    try qmm.matmul(driver, stream.*, s.mixed.ptr, s.xsm.ptr, proj_q.w.ptr, proj_q.s.ptr, proj_q.b.ptr, s.pa.ptr, 1, proj_n, dims);
    try s.pos.upload(0, std.mem.asBytes(&pos));
    try tri.attnPrep(s.pa.ptr, s.pos.ptr, q_scale, k_scale, i_scale, s.inv.ptr, s.q.ptr, kc, vc, s.dummy.ptr, s.dummy.ptr, s.iq.ptr, ikc, s.dummy.ptr, s.dummy.ptr, eps, 1, 0);
    try tri.attnChunks(s.q.ptr, kc, vc, s.dummy.ptr, s.dummy.ptr, s.pos.ptr, s.po.ptr, s.pm.ptr, s.pl.ptr, s.dummy.ptr, s.dummy.ptr, s.dummy.ptr, 1, kv_heads, 1);
    try tri.attnMerge(s.po.ptr, s.pm.ptr, s.pl.ptr, s.pos.ptr, s.ao.ptr, s.dummy.ptr, s.dummy.ptr, 1, kv_heads);
    try tri.attnGate(s.ao.ptr, s.pa.ptr, s.gated.ptr, s.gxs_attn.ptr, 1, proj_n, q_heads, head_dim);
    try qmm.partials(driver, stream.*, s.gated.ptr, s.gxs_attn.ptr, out_q.w.ptr, out_q.s.ptr, out_q.b.ptr, s.dummy.ptr, s.part.ptr, 1, dims, out_k, out_sk);
}

fn copyExpert(dst_w: []u8, dst_s: []u8, dst_b: []u8, routed: Face, shared: Face, id: i32, slot: usize) void {
    const p = if (id == experts_n) shared else routed;
    const k8 = p.k / 8;
    const kg = p.k / 32;
    const wb = p.n * k8 * 4;
    const sb = p.n * kg * 2;
    const at: usize = if (id == experts_n) 0 else @intCast(id);
    @memcpy(dst_w[slot * wb ..][0..wb], p.w[at * wb ..][0..wb]);
    @memcpy(dst_s[slot * sb ..][0..sb], p.s[at * sb ..][0..sb]);
    @memcpy(dst_b[slot * sb ..][0..sb], p.b[at * sb ..][0..sb]);
}

fn packDevice(driver: *cuda.Driver, stream: cuda.Stream, pack_fn: cuda.Function, words: []const u8, scales: []const u8, biases: []const u8, e: usize, n: usize, k: usize) !cuda.DeviceBuffer {
    const kg = k / 32;
    const nb = n / 32;
    var w_b = try cuda.DeviceBuffer.fromHost(driver, words);
    defer w_b.free();
    var s_b = try cuda.DeviceBuffer.fromHost(driver, scales);
    defer s_b.free();
    var b_b = try cuda.DeviceBuffer.fromHost(driver, biases);
    defer b_b.free();
    var out_b = try cuda.DeviceBuffer.alloc(driver, e * nb * kg * block_bytes);
    errdefer out_b.free();
    var a: cuda.Args = .{};
    a.add(w_b.ptr);
    a.add(s_b.ptr);
    a.add(b_b.ptr);
    a.add(out_b.ptr);
    a.add(cint(n));
    a.add(cint(k / 8));
    a.add(cint(kg));
    a.add(cint(nb));
    try cuda.launch.launch(pack_fn, .{ .grid = .{ .x = @intCast(kg), .y = @intCast(nb), .z = @intCast(e) }, .block = .{ .x = 160 } }, stream, &a);
    try stream.synchronize();
    return out_b;
}

fn stackHalves(gpa: std.mem.Allocator, gate: []const u8, up: []const u8) ![]u8 {
    if (gate.len != up.len or gate.len % block_bytes != 0) return error.UnexpectedTensor;
    const nblk = gate.len / block_bytes;
    const out = try gpa.alloc(u8, gate.len + up.len);
    for (0..nblk) |b| {
        @memcpy(out[b * 2 * block_bytes ..][0..block_bytes], gate[b * block_bytes ..][0..block_bytes]);
        @memcpy(out[b * 2 * block_bytes + block_bytes ..][0..block_bytes], up[b * block_bytes ..][0..block_bytes]);
    }
    return out;
}

fn launchExpert(f: cuda.Function, stream: cuda.Stream, x: u64, x_stride: usize, slot_n: usize, w: u64, kg: usize, nb: usize, items: u64, counts: u64, members: u64, out: u64, n: usize, units: usize) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(cint(x_stride));
    a.add(cint(slot_n));
    a.add(w);
    a.add(cint(kg));
    a.add(cint(nb));
    a.add(items);
    a.add(counts);
    a.add(members);
    a.add(out);
    a.add(cint(n));
    a.add(@as(f32, 0));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((units + 3) / 4) }, .block = .{ .x = 128 } }, stream, &a);
}

fn moeStep(comptime Tri: type, gpa: std.mem.Allocator, driver: *cuda.Driver, stream: *cuda.Stream, tri: Tri, s: *Scratch, router_w: u64, faces: [3]Face, shared: [3]Face, y: u64, wts: u64) !void {
    try tri.router(s.mixed.ptr, router_w, s.router_l.ptr, dims);
    try tri.topkRows(s.router_l.ptr, s.picks.ptr, wts);
    var pick_raw: [slots * 4]u8 = undefined;
    try stream.synchronize();
    try s.picks.download(0, &pick_raw);
    var packed_ids: [slots]i32 = undefined;
    var npack: usize = 0;
    var local: [slots]i32 = undefined;
    for (0..slots) |slot| {
        const id = std.mem.readInt(i32, pick_raw[4 * slot ..][0..4], .little);
        if (id < 0 or id > experts_n) return error.UnexpectedTensor;
        var found: ?usize = null;
        for (packed_ids[0..npack], 0..) |prev, i| if (prev == id) {
            found = i;
        };
        if (found) |i| {
            local[slot] = @intCast(i);
        } else {
            packed_ids[npack] = id;
            local[slot] = @intCast(npack);
            npack += 1;
        }
    }
    const gw = npack * moe_w * (dims / 8) * 4;
    const gs = npack * moe_w * (dims / 32) * 2;
    const dw = npack * dims * (moe_w / 8) * 4;
    const ds = npack * dims * (moe_w / 32) * 2;
    const gathered_w = try gpa.alloc(u8, gw);
    defer gpa.free(gathered_w);
    const gathered_s = try gpa.alloc(u8, gs);
    defer gpa.free(gathered_s);
    const gathered_b = try gpa.alloc(u8, gs);
    defer gpa.free(gathered_b);
    const up_w = try gpa.alloc(u8, gw);
    defer gpa.free(up_w);
    const up_s = try gpa.alloc(u8, gs);
    defer gpa.free(up_s);
    const up_b = try gpa.alloc(u8, gs);
    defer gpa.free(up_b);
    const down_w = try gpa.alloc(u8, dw);
    defer gpa.free(down_w);
    const down_s = try gpa.alloc(u8, ds);
    defer gpa.free(down_s);
    const down_b = try gpa.alloc(u8, ds);
    defer gpa.free(down_b);
    for (packed_ids[0..npack], 0..) |id, slot| {
        copyExpert(gathered_w, gathered_s, gathered_b, faces[0], shared[0], id, slot);
        copyExpert(up_w, up_s, up_b, faces[1], shared[1], id, slot);
        copyExpert(down_w, down_s, down_b, faces[2], shared[2], id, slot);
    }
    var pack_mod = try cuda.Module.load(driver, cuda.kernels.experts_pack);
    defer pack_mod.unload();
    var exp_mod = try cuda.Module.load(driver, cuda.kernels.experts);
    defer exp_mod.unload();
    const pack_fn = try pack_mod.function(pack_symbol);
    var packed_g = try packDevice(driver, stream.*, pack_fn, gathered_w, gathered_s, gathered_b, npack, moe_w, dims);
    defer packed_g.free();
    var packed_u = try packDevice(driver, stream.*, pack_fn, up_w, up_s, up_b, npack, moe_w, dims);
    defer packed_u.free();
    var packed_d = try packDevice(driver, stream.*, pack_fn, down_w, down_s, down_b, npack, dims, moe_w);
    defer packed_d.free();
    const g_raw = try gpa.alloc(u8, packed_g.len);
    defer gpa.free(g_raw);
    const u_raw = try gpa.alloc(u8, packed_u.len);
    defer gpa.free(u_raw);
    try packed_g.download(0, g_raw);
    try packed_u.download(0, u_raw);
    const stacked = try stackHalves(gpa, g_raw, u_raw);
    defer gpa.free(stacked);
    var up_wb = try cuda.DeviceBuffer.fromHost(driver, stacked);
    defer up_wb.free();
    var local_pick: [slots]i32 = undefined;
    for (0..slots) |slot| local_pick[slot] = local[slot];
    var picks_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(&local_pick));
    defer picks_b.free();
    var members_b = try cuda.DeviceBuffer.alloc(driver, slots * 4);
    defer members_b.free();
    var items_b = try cuda.DeviceBuffer.alloc(driver, slots * 3 * 4);
    defer items_b.free();
    var counts_b = try cuda.DeviceBuffer.alloc(driver, 8);
    defer counts_b.free();
    var act_b = try cuda.DeviceBuffer.alloc(driver, slots * moe_w * 2);
    defer act_b.free();
    const plan_fn = try exp_mod.function(plan_symbol);
    var plan_a: cuda.Args = .{};
    plan_a.add(picks_b.ptr);
    plan_a.add(cint(slots));
    plan_a.add(cint(npack));
    plan_a.add(cint(tile));
    plan_a.add(members_b.ptr);
    plan_a.add(items_b.ptr);
    plan_a.add(counts_b.ptr);
    try cuda.launch.launch(plan_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream.*, &plan_a);
    const up_fn = try exp_mod.function(up_symbol);
    const down_fn = try exp_mod.function(down_symbol);
    try launchExpert(up_fn, stream.*, s.mixed.ptr, dims, slots, up_wb.ptr, dims / 32, moe_w / 32, items_b.ptr, counts_b.ptr, members_b.ptr, act_b.ptr, moe_w, slots * (moe_w / 32));
    try launchExpert(down_fn, stream.*, act_b.ptr, moe_w, 0, packed_d.ptr, moe_w / 32, dims / 32, items_b.ptr, counts_b.ptr, members_b.ptr, y, dims, slots * (dims / 32));
    try stream.synchronize();
}

fn faceOf(store: *Store, buf: []u8, prefix: []const u8, name: []const u8, routed: bool, n: usize, k: usize) !Face {
    const w = try store.hold(try std.fmt.bufPrint(buf, "{s}{s}.weight", .{ prefix, name }));
    const sc = try store.hold(try std.fmt.bufPrint(buf, "{s}{s}.scales", .{ prefix, name }));
    const b = try store.hold(try std.fmt.bufPrint(buf, "{s}{s}.biases", .{ prefix, name }));
    const k8 = k / 8;
    const kg = k / 32;
    if (routed) {
        if (!w.is(.u32, &.{ experts_n, n, k8 }) or !sc.is(.bf16, &.{ experts_n, n, kg }) or !b.is(.bf16, &.{ experts_n, n, kg })) return error.UnexpectedTensor;
    } else if (!w.is(.u32, &.{ n, k8 }) or !sc.is(.bf16, &.{ n, kg }) or !b.is(.bf16, &.{ n, kg })) return error.UnexpectedTensor;
    return .{ .w = w.bytes, .s = sc.bytes, .b = b.bytes, .n = n, .k = k, .routed = routed };
}

fn reduceBf16(gpa: std.mem.Allocator, stream: *cuda.Stream, part: cuda.DeviceBuffer, sk: usize, n: usize, dst: cuda.DeviceBuffer) !void {
    const raw = try gpa.alloc(u8, sk * n * 4);
    defer gpa.free(raw);
    try stream.synchronize();
    try part.download(0, raw);
    const row = try gpa.alloc(u16, n);
    defer gpa.free(row);
    for (0..n) |i| {
        var acc: f32 = @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little));
        for (1..sk) |s| acc += @as(f32, @bitCast(std.mem.readInt(u32, raw[4 * (s * n + i) ..][0..4], .little)));
        row[i] = toBf16(acc);
    }
    try dst.upload(0, std.mem.sliceAsBytes(row));
}

fn pleStep(comptime Tri: type, gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, stream: *cuda.Stream, tri: Tri, s: *Scratch, model_dir: []const u8, history: *[2]i64, token: u32, h: u64, key_q: Q4, val_q: Q4, nk: u64, nq: u64, nc: u64, cw: u64, tail: *cuda.DeviceBuffer) !void {
    const emb = try gpa.alloc(u16, dims);
    defer gpa.free(emb);
    const xs = try gpa.alloc(f32, dims / 32);
    defer gpa.free(xs);
    try ple.embedding(gpa, io, model_dir, history, token, emb, xs);
    try s.ple_emb.upload(0, std.mem.sliceAsBytes(emb));
    try s.ple_xs.upload(0, std.mem.sliceAsBytes(xs));
    try qmm.matmul(driver, stream.*, s.ple_emb.ptr, s.ple_xs.ptr, key_q.w.ptr, key_q.s.ptr, key_q.b.ptr, s.ple_keys.ptr, 1, wide, dims);
    try qmm.partials(driver, stream.*, s.ple_emb.ptr, s.ple_xs.ptr, val_q.w.ptr, val_q.s.ptr, val_q.b.ptr, s.ple_vals.ptr, s.ple_part.ptr, 1, dims, dims, 4);
    try reduceBf16(gpa, stream, s.ple_part, 4, dims, s.ple_vals);
    try tri.pleGate(s.ple_keys.ptr, s.ple_vals.ptr, h, nk, nq, s.ple_gated.ptr, s.ple_pss.ptr, eps, 1, dims, streams);
    try tri.pleConv(s.ple_gated.ptr, s.ple_pss.ptr, nc, tail.ptr, cw, h, h, s.ple_nrow.ptr, eps, 1, dims, streams, 4, 3);
    const nrow = try gpa.alloc(u8, wide * 2);
    defer gpa.free(nrow);
    try stream.synchronize();
    try s.ple_nrow.download(0, nrow);
    try shiftTail(gpa, stream, tail.*, ple_tail_n, wide * 2, nrow);
    history[0] = history[1];
    history[1] = token;
}

fn argmax(raw: []const u8) u32 {
    var best = promote(std.mem.readInt(u16, raw[0..2], .little));
    var at: usize = 0;
    var i: usize = 1;
    while (i < vocab) : (i += 1) {
        const v = promote(std.mem.readInt(u16, raw[2 * i ..][0..2], .little));
        if (v > best) {
            best = v;
            at = i;
        }
    }
    return @intCast(at);
}

const Pass = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    driver: *cuda.Driver,
    stream: *cuda.Stream,
    store: *Store,
    model_dir: []const u8,
    s: *Scratch,
    table: embed.Table,
    rows: usize,
    mode: []u8,
    history: *[2]i64,
    conv: []cuda.DeviceBuffer,
    state: []cuda.DeviceBuffer,
    kc: []cuda.DeviceBuffer,
    vc: []cuda.DeviceBuffer,
    ikc: []cuda.DeviceBuffer,
    tail: cuda.DeviceBuffer,
    mixer: HcW,
    head_q: Q4,
};

fn embedRow(comptime Tri: type, p: *Pass, tri: Tri, token: u32, h: u64) !void {
    const id: i32 = @intCast(token);
    try p.s.ids.upload(0, std.mem.asBytes(&id));
    try tri.embed(p.s.ids.ptr, p.table.w.ptr, p.table.s.ptr, p.table.b.ptr, h, dims, streams, 1);
}

fn oneLayer(comptime Tri: type, p: *Pass, tri: Tri, layer: usize, ids: []const u32, pos0: usize) !void {
    var buf: [200]u8 = undefined;
    const base = try std.fmt.bufPrint(&buf, "language_model.model.layers.{d}", .{layer});
    var base_buf: [80]u8 = undefined;
    @memcpy(base_buf[0..base.len], base);
    const root = base_buf[0..base.len];
    var attn_hc = try loadHc(p.gpa, p.driver, p.store, try std.fmt.bufPrint(&buf, "{s}.attn_hyper_connection", .{root}), true);
    defer attn_hc.free();
    var mlp_hc = try loadHc(p.gpa, p.driver, p.store, try std.fmt.bufPrint(&buf, "{s}.mlp_hyper_connection", .{root}), true);
    defer mlp_hc.free();
    const mlp = try std.fmt.bufPrint(&buf, "{s}.mlp", .{root});
    var mlp_buf: [96]u8 = undefined;
    @memcpy(mlp_buf[0..mlp.len], mlp);
    const mlp_root = mlp_buf[0..mlp.len];
    const gate = try p.store.tensor(try std.fmt.bufPrint(&buf, "{s}.gate.weight", .{mlp_root}));
    if (!gate.is(.bf16, &.{ experts_n, dims })) return error.UnexpectedTensor;
    const gate_copy = try p.gpa.alloc(u8, gate.bytes.len);
    defer p.gpa.free(gate_copy);
    @memcpy(gate_copy, gate.bytes);
    const sg = try join3(p.gpa, p.store, &buf, mlp_root, ".shared_expert_gate");
    defer p.gpa.free(sg.w);
    defer p.gpa.free(sg.s);
    defer p.gpa.free(sg.b);
    const router = try p.gpa.alloc(u16, (experts_n + 1) * dims);
    defer p.gpa.free(router);
    @memcpy(std.mem.sliceAsBytes(router[0 .. experts_n * dims]), gate_copy);
    try embed.dequant(sg.w, sg.s, sg.b, dims, 0, router[experts_n * dims ..][0..dims]);
    var router_b = try cuda.DeviceBuffer.fromHost(p.driver, std.mem.sliceAsBytes(router));
    defer router_b.free();

    const linear = linearLayer(layer);
    var proj_q: Q4 = undefined;
    var out_q: Q4 = undefined;
    var cw_b: cuda.DeviceBuffer = .{ .d = p.driver, .ptr = 0, .len = 0 };
    var a_b: cuda.DeviceBuffer = .{ .d = p.driver, .ptr = 0, .len = 0 };
    var dt_b: cuda.DeviceBuffer = .{ .d = p.driver, .ptr = 0, .len = 0 };
    var nw_b: cuda.DeviceBuffer = .{ .d = p.driver, .ptr = 0, .len = 0 };
    var qs_b: cuda.DeviceBuffer = .{ .d = p.driver, .ptr = 0, .len = 0 };
    var ks_b: cuda.DeviceBuffer = .{ .d = p.driver, .ptr = 0, .len = 0 };
    var is_b: cuda.DeviceBuffer = .{ .d = p.driver, .ptr = 0, .len = 0 };
    var key_q = Q4.empty(p.driver);
    var val_q = Q4.empty(p.driver);
    var nk_b: cuda.DeviceBuffer = .{ .d = p.driver, .ptr = 0, .len = 0 };
    var nq_b: cuda.DeviceBuffer = .{ .d = p.driver, .ptr = 0, .len = 0 };
    var nc_b: cuda.DeviceBuffer = .{ .d = p.driver, .ptr = 0, .len = 0 };
    var pcw_b: cuda.DeviceBuffer = .{ .d = p.driver, .ptr = 0, .len = 0 };
    defer if (cw_b.ptr != 0) cw_b.free();
    defer if (a_b.ptr != 0) a_b.free();
    defer if (dt_b.ptr != 0) dt_b.free();
    defer if (nw_b.ptr != 0) nw_b.free();
    defer if (qs_b.ptr != 0) qs_b.free();
    defer if (ks_b.ptr != 0) ks_b.free();
    defer if (is_b.ptr != 0) is_b.free();
    defer if (nk_b.ptr != 0) nk_b.free();
    defer if (nq_b.ptr != 0) nq_b.free();
    defer if (nc_b.ptr != 0) nc_b.free();
    defer if (pcw_b.ptr != 0) pcw_b.free();
    if (linear) {
        const pre = try std.fmt.bufPrint(&buf, "{s}.linear_attn", .{root});
        var pre_buf: [120]u8 = undefined;
        @memcpy(pre_buf[0..pre.len], pre);
        const lp = pre_buf[0..pre.len];
        const parts = [_][]const u8{ ".in_proj_qkv", ".in_proj_z", ".in_proj_b", ".in_proj_a" };
        const stitched = try stitch(p.gpa, p.store, lp, &parts);
        defer p.gpa.free(stitched.w);
        defer p.gpa.free(stitched.s);
        defer p.gpa.free(stitched.b);
        if (stitched.w.len != gdn.proj_width * (dims / 8) * 4) return error.UnexpectedTensor;
        proj_q = try uploadQ4(p.gpa, p.driver, stitched.w, stitched.s, stitched.b, gdn.proj_width, dims / 8);
        const out = try join3(p.gpa, p.store, &buf, lp, ".out_proj");
        defer p.gpa.free(out.w);
        defer p.gpa.free(out.s);
        defer p.gpa.free(out.b);
        out_q = try uploadQ4(p.gpa, p.driver, out.w, out.s, out.b, dims, gdn.value_dim / 8);
        const conv = try p.store.tensor(try std.fmt.bufPrint(&buf, "{s}.conv1d.weight", .{lp}));
        if (conv.bytes.len != gdn.conv_dim * gdn.taps * 2) return error.UnexpectedTensor;
        cw_b = try cuda.DeviceBuffer.fromHost(p.driver, conv.bytes);
        const a_log = try asF32(p.gpa, try p.store.tensor(try std.fmt.bufPrint(&buf, "{s}.A_log", .{lp})));
        defer p.gpa.free(a_log);
        const dt = try asF32(p.gpa, try p.store.tensor(try std.fmt.bufPrint(&buf, "{s}.dt_bias", .{lp})));
        defer p.gpa.free(dt);
        if (a_log.len != gdn.nv or dt.len != gdn.nv) return error.UnexpectedTensor;
        a_b = try cuda.DeviceBuffer.fromHost(p.driver, std.mem.sliceAsBytes(a_log));
        dt_b = try cuda.DeviceBuffer.fromHost(p.driver, std.mem.sliceAsBytes(dt));
        const norm = try p.store.copyOf(try std.fmt.bufPrint(&buf, "{s}.norm.weight", .{lp}));
        defer p.gpa.free(norm);
        nw_b = try cuda.DeviceBuffer.fromHost(p.driver, norm);
    } else {
        const pre = try std.fmt.bufPrint(&buf, "{s}.self_attn", .{root});
        var pre_buf: [120]u8 = undefined;
        @memcpy(pre_buf[0..pre.len], pre);
        const ap = pre_buf[0..pre.len];
        const parts = [_][]const u8{ ".q_proj", ".k_proj", ".v_proj", ".indexer.index_qk_proj" };
        const stitched = try stitch(p.gpa, p.store, ap, &parts);
        defer p.gpa.free(stitched.w);
        defer p.gpa.free(stitched.s);
        defer p.gpa.free(stitched.b);
        if (stitched.w.len != proj_n * (dims / 8) * 4) return error.UnexpectedTensor;
        proj_q = try uploadQ4(p.gpa, p.driver, stitched.w, stitched.s, stitched.b, proj_n, dims / 8);
        const out = try join3(p.gpa, p.store, &buf, ap, ".o_proj");
        defer p.gpa.free(out.w);
        defer p.gpa.free(out.s);
        defer p.gpa.free(out.b);
        out_q = try uploadQ4(p.gpa, p.driver, out.w, out.s, out.b, dims, out_k / 8);
        const qn = try asF32(p.gpa, try p.store.tensor(try std.fmt.bufPrint(&buf, "{s}.q_norm.weight", .{ap})));
        defer p.gpa.free(qn);
        const kn = try asF32(p.gpa, try p.store.tensor(try std.fmt.bufPrint(&buf, "{s}.k_norm.weight", .{ap})));
        defer p.gpa.free(kn);
        const iqn = try asF32(p.gpa, try p.store.tensor(try std.fmt.bufPrint(&buf, "{s}.indexer.q_layernorm.weight", .{ap})));
        defer p.gpa.free(iqn);
        if (qn.len != head_dim or kn.len != head_dim or iqn.len != index_dim) return error.UnexpectedTensor;
        qs_b = try cuda.DeviceBuffer.fromHost(p.driver, std.mem.sliceAsBytes(qn));
        ks_b = try cuda.DeviceBuffer.fromHost(p.driver, std.mem.sliceAsBytes(kn));
        is_b = try cuda.DeviceBuffer.fromHost(p.driver, std.mem.sliceAsBytes(iqn));
    }
    defer proj_q.free();
    defer out_q.free();
    if (layer == ple_layer) {
        const pre = try std.fmt.bufPrint(&buf, "{s}.ple", .{root});
        var pre_buf: [96]u8 = undefined;
        @memcpy(pre_buf[0..pre.len], pre);
        const pp = pre_buf[0..pre.len];
        const key = try join3(p.gpa, p.store, &buf, pp, ".key_proj");
        defer p.gpa.free(key.w);
        defer p.gpa.free(key.s);
        defer p.gpa.free(key.b);
        const val = try join3(p.gpa, p.store, &buf, pp, ".value_proj");
        defer p.gpa.free(val.w);
        defer p.gpa.free(val.s);
        defer p.gpa.free(val.b);
        key_q = try uploadQ4(p.gpa, p.driver, key.w, key.s, key.b, wide, dims / 8);
        val_q = try uploadQ4(p.gpa, p.driver, val.w, val.s, val.b, dims, dims / 8);
        const nk = try asF32(p.gpa, try p.store.tensor(try std.fmt.bufPrint(&buf, "{s}.norm_key.weight", .{pp})));
        defer p.gpa.free(nk);
        const nq = try asF32(p.gpa, try p.store.tensor(try std.fmt.bufPrint(&buf, "{s}.norm_query.weight", .{pp})));
        defer p.gpa.free(nq);
        const nc = try asF32(p.gpa, try p.store.tensor(try std.fmt.bufPrint(&buf, "{s}.norm_conv.weight", .{pp})));
        defer p.gpa.free(nc);
        if (nk.len != wide or nq.len != wide or nc.len != wide) return error.UnexpectedTensor;
        nk_b = try cuda.DeviceBuffer.fromHost(p.driver, std.mem.sliceAsBytes(nk));
        nq_b = try cuda.DeviceBuffer.fromHost(p.driver, std.mem.sliceAsBytes(nq));
        nc_b = try cuda.DeviceBuffer.fromHost(p.driver, std.mem.sliceAsBytes(nc));
        const conv = try p.store.tensor(try std.fmt.bufPrint(&buf, "{s}.conv1d.weight", .{pp}));
        if (conv.bytes.len != wide * 4 * 2) return error.UnexpectedTensor;
        pcw_b = try cuda.DeviceBuffer.fromHost(p.driver, conv.bytes);
    }
    defer key_q.free();
    defer val_q.free();

    // Shared experts are copied. Routed rows stay mapped, so this is the last shard switch.
    var sh_g = try ownedFace(p.gpa, p.store, &buf, mlp_root, ".shared_expert.gate_proj", moe_w, dims);
    defer sh_g.free(p.gpa);
    var sh_u = try ownedFace(p.gpa, p.store, &buf, mlp_root, ".shared_expert.up_proj", moe_w, dims);
    defer sh_u.free(p.gpa);
    var sh_d = try ownedFace(p.gpa, p.store, &buf, mlp_root, ".shared_expert.down_proj", dims, moe_w);
    defer sh_d.free(p.gpa);
    p.store.release();
    var faces: [3]Face = undefined;
    faces[0] = try faceOf(p.store, &buf, mlp_root, ".switch_mlp.gate_proj", true, moe_w, dims);
    faces[1] = try faceOf(p.store, &buf, mlp_root, ".switch_mlp.up_proj", true, moe_w, dims);
    faces[2] = try faceOf(p.store, &buf, mlp_root, ".switch_mlp.down_proj", true, dims, moe_w);
    const sh = [3]Face{ sh_g.face, sh_u.face, sh_d.face };

    std.debug.print("layer {d}\n", .{layer});
    for (ids, 0..) |token, t| {
        const h = p.s.h.ptr + @as(u64, t) * wide * 2;
        const inj = p.s.inj.ptr + @as(u64, t) * inj_stride;
        const y = p.s.y.ptr + @as(u64, t) * slots * dims * 4;
        const wt = p.s.wts.ptr + @as(u64, t) * wts_stride;
        if (layer == ple_layer and p.mode[t] == 2) {
            try writeback(tri, p.s, h, 2, inj, y, wt);
            p.mode[t] = 0;
        }
        if (layer == ple_layer) try pleStep(Tri, p.gpa, p.io, p.driver, p.stream, tri, p.s, p.model_dir, p.history, token, h, key_q, val_q, nk_b.ptr, nq_b.ptr, nc_b.ptr, pcw_b.ptr, &p.tail);
        if (p.mode[t] == 2) {
            try writeback(tri, p.s, h, 2, inj, y, wt);
            p.mode[t] = 0;
        } else try writeback(tri, p.s, h, 0, inj, y, wt);
        try readout(tri, p.s, attn_hc, h, inj);
        if (linear) {
            const li = linearIndex(layer);
            try gdnStep(p.gpa, p.driver, p.stream, p.s, proj_q, out_q, p.conv[li], cw_b.ptr, a_b.ptr, dt_b.ptr, nw_b.ptr, &p.state[li]);
        } else {
            const ai = attnIndex(layer);
            try attnStep(Tri, p.driver, p.stream, tri, p.s, proj_q, out_q, qs_b.ptr, ks_b.ptr, is_b.ptr, p.kc[ai].ptr, p.vc[ai].ptr, p.ikc[ai].ptr, @intCast(pos0 + t));
        }
        try writeback(tri, p.s, h, 4, inj, y, wt);
        try readout(tri, p.s, mlp_hc, h, inj);
        try moeStep(Tri, p.gpa, p.driver, p.stream, tri, p.s, router_b.ptr, faces, sh, y, wt);
        p.mode[t] = 2;
        if (t % 16 == 0) std.debug.print("layer {d} row {d}\n", .{ layer, t });
    }
    p.store.release();
}

const OwnedFace = struct {
    face: Face,
    w: []u8,
    s: []u8,
    b: []u8,

    fn free(self: *OwnedFace, gpa: std.mem.Allocator) void {
        gpa.free(self.w);
        gpa.free(self.s);
        gpa.free(self.b);
    }
};

fn ownedFace(gpa: std.mem.Allocator, store: *Store, buf: []u8, prefix: []const u8, name: []const u8, n: usize, k: usize) !OwnedFace {
    const got = try join3(gpa, store, buf, prefix, name);
    const k8 = k / 8;
    if (got.w.len != n * k8 * 4 or got.s.len != n * (k / 32) * 2) {
        gpa.free(got.w);
        gpa.free(got.s);
        gpa.free(got.b);
        return error.UnexpectedTensor;
    }
    return .{ .face = .{ .w = got.w, .s = got.s, .b = got.b, .n = n, .k = k, .routed = false }, .w = got.w, .s = got.s, .b = got.b };
}

fn zeros(gpa: std.mem.Allocator, driver: *cuda.Driver, n: usize, len: usize) ![]cuda.DeviceBuffer {
    const out = try gpa.alloc(cuda.DeviceBuffer, n);
    var i: usize = 0;
    errdefer {
        for (out[0..i]) |*b| b.free();
        gpa.free(out);
    }
    while (i < n) : (i += 1) {
        out[i] = try cuda.DeviceBuffer.alloc(driver, len);
        try out[i].fill8(0, null);
    }
    return out;
}

fn head(comptime Tri: type, p: *Pass, tri: Tri, row: usize) !u32 {
    if (p.mode[row] != 2) return error.UnexpectedTensor;
    const h = p.s.h.ptr + @as(u64, row) * wide * 2;
    const inj = p.s.inj.ptr + @as(u64, row) * inj_stride;
    const y = p.s.y.ptr + @as(u64, row) * slots * dims * 4;
    const wt = p.s.wts.ptr + @as(u64, row) * wts_stride;
    try writeback(tri, p.s, h, 2, inj, y, wt);
    p.mode[row] = 0;
    try readout(tri, p.s, p.mixer, h, inj);
    try qmm.matmul(p.driver, p.stream.*, p.s.mixed.ptr, p.s.xsm.ptr, p.head_q.w.ptr, p.head_q.s.ptr, p.head_q.b.ptr, p.s.logits.ptr, 1, vocab, dims);
    const raw = try p.gpa.alloc(u8, vocab * 2);
    defer p.gpa.free(raw);
    try p.stream.synchronize();
    try p.s.logits.download(0, raw);
    return argmax(raw);
}

fn forward(comptime Tri: type, p: *Pass, tri: Tri, ids: []const u32, pos0: usize) !u32 {
    @memset(p.mode, 0);
    for (ids, 0..) |token, t| try embedRow(Tri, p, tri, token, p.s.h.ptr + @as(u64, t) * wide * 2);
    for (0..layers_n) |layer| try oneLayer(Tri, p, tri, layer, ids, pos0);
    return head(Tri, p, tri, ids.len - 1);
}

/// `out` receives `new_tokens` greedy ids. The first is the prompt's argmax. The rest are fed back one at a time.
pub fn run(comptime Tri: type, gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, stream: *cuda.Stream, tri: Tri, model_dir: []const u8, tokens: []const u32, new_tokens: usize, out: []u32) !void {
    if (!cuda.kernels.available) return error.BuiltWithoutKernels;
    if (tokens.len == 0 or new_tokens == 0 or out.len < new_tokens or tokens.len + new_tokens > 256) return error.UnexpectedTensor;
    var store = try Store.open(gpa, io, model_dir);
    defer store.deinit();
    const weight = try store.tensor(embed.weight_name);
    const scales = try store.tensor(embed.scales_name);
    const biases = try store.tensor(embed.biases_name);
    var table = try embed.Table.upload(driver, weight, scales, biases);
    defer table.deinit();
    const eos = try ple.eosOf(gpa, io, model_dir);
    var history = [2]i64{ eos, eos };
    var scratch = try Scratch.alloc(driver, tokens.len);
    defer scratch.free();
    try scratch.dummy.fill8(0, null);
    const conv = try zeros(gpa, driver, linear_n, (gdn.taps - 1) * gdn.conv_dim * 2);
    defer {
        for (conv) |*b| b.free();
        gpa.free(conv);
    }
    const state = try zeros(gpa, driver, linear_n, state_n * 4);
    defer {
        for (state) |*b| b.free();
        gpa.free(state);
    }
    const cache = tokens.len + new_tokens;
    const kc = try zeros(gpa, driver, attn_n, cache * kv_heads * head_dim * 2);
    defer {
        for (kc) |*b| b.free();
        gpa.free(kc);
    }
    const vc = try zeros(gpa, driver, attn_n, cache * kv_heads * head_dim * 2);
    defer {
        for (vc) |*b| b.free();
        gpa.free(vc);
    }
    const ikc = try zeros(gpa, driver, attn_n, cache * index_dim * 2);
    defer {
        for (ikc) |*b| b.free();
        gpa.free(ikc);
    }
    var tail = try cuda.DeviceBuffer.alloc(driver, ple_tail_n * wide * 2);
    defer tail.free();
    try tail.fill8(0, null);
    var mixer = try loadHc(gpa, driver, &store, "language_model.model.hyper_connection_mixer", false);
    defer mixer.free();
    var buf: [80]u8 = undefined;
    const head_w = try join3(gpa, &store, &buf, "language_model.lm_head", "");
    defer gpa.free(head_w.w);
    defer gpa.free(head_w.s);
    defer gpa.free(head_w.b);
    if (head_w.w.len != vocab * (dims / 8) * 4) return error.UnexpectedTensor;
    var head_q = try uploadQ4(gpa, driver, head_w.w, head_w.s, head_w.b, vocab, dims / 8);
    defer head_q.free();
    const mode = try gpa.alloc(u8, tokens.len);
    defer gpa.free(mode);
    var pass = Pass{
        .gpa = gpa,
        .io = io,
        .driver = driver,
        .stream = stream,
        .store = &store,
        .model_dir = model_dir,
        .s = &scratch,
        .table = table,
        .rows = tokens.len,
        .mode = mode,
        .history = &history,
        .conv = conv,
        .state = state,
        .kc = kc,
        .vc = vc,
        .ikc = ikc,
        .tail = tail,
        .mixer = mixer,
        .head_q = head_q,
    };
    out[0] = try forward(Tri, &pass, tri, tokens, 0);
    std.debug.print("token {d}\n", .{out[0]});
    var tok = out[0];
    var pos: usize = tokens.len;
    for (1..new_tokens) |i| {
        var one = [_]u32{tok};
        // The new token reuses row 0. Clear every row so a stale prompt pending is not written back.
        tok = try forward(Tri, &pass, tri, &one, pos);
        out[i] = tok;
        pos += 1;
        std.debug.print("token {d}\n", .{tok});
    }
}

test "linear layers skip every fourth index and attention starts at 3" {
    try std.testing.expect(linearLayer(0));
    try std.testing.expect(!linearLayer(3));
    try std.testing.expect(!linearLayer(47));
    try std.testing.expectEqual(@as(usize, 0), linearIndex(0));
    try std.testing.expectEqual(@as(usize, 2), linearIndex(2));
    try std.testing.expectEqual(@as(usize, 35), linearIndex(46));
    try std.testing.expectEqual(@as(usize, 0), attnIndex(3));
    try std.testing.expectEqual(@as(usize, 11), attnIndex(47));
}
