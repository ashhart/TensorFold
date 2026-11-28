//! MOSS-Image runtime LoRAs, never merged: core.runtime_lora checks each adapter, the GPU adds it per projection.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const host = @import("host.zig");
const lt = cuda.cublaslt;
const st = core.safetensors;
const rl = core.runtime_lora;
const sys = @import("sys");

const workspace_bytes = 32 << 20;

/// A request's choice: an adapter's name in the folder and the strength it runs at.
pub const Pick = struct { name: []const u8, scale: f32 };

/// A projection a LoRA may target: dotted name, weight and bias (0: none) on the device, n out, k in, its stored bytes.
pub const Target = struct { module: []const u8, w: u64, bias: u64 = 0, n: usize, k: usize, bytes: []const u8 };

/// One projection's factors on the device: A as bf16 pieces summing to the file's A exactly, B fp32, alpha / rank.
pub const Factor = struct {
    a: cuda.DeviceBuffer,
    b: cuda.DeviceBuffer,
    rank: usize,
    /// [parts, rank, k] bf16 pieces: 1 for a bf16 A, 2 for most f16, 3 for f32; one GEMM each into fp32 sums
    parts: usize,
    n: usize,
    k: usize,
    scale: f32,
    /// alpha / rank as core.runtime_lora gave it, before f32
    exact_scale: f64,
};

const Adapter = struct {
    arena: std.heap.ArenaAllocator,
    /// keyed by the module's name as the adapter file spells it, prefixes and suffixes off
    factors: std.StringArrayHashMapUnmanaged(Factor) = .empty,
    bytes: usize = 0,

    fn deinit(ad: *Adapter, gpa: std.mem.Allocator) void {
        for (ad.factors.values()) |*f| {
            f.a.free();
            f.b.free();
        }
        ad.factors.deinit(gpa);
        ad.arena.deinit();
    }
};

pub const Use = struct { f: *const Factor, scale: f32 };

/// A request's LoRA the folder lacks or that fails a check: that request fails alone, the engine keeps serving.
pub const Refusal = error{
    LoraNotFound,
    LoraUnreadable,
    LoraKeys,
    LoraMissingFactor,
    LoraModules,
    LoraScaleOverflow,
    BadAlpha,
    BadLoraName,
    BadLoraScale,
    DuplicateFactor,
    EmptyAdapter,
    InexactAdapterFactor,
    InvalidAdapterConfig,
    InvalidAlpha,
    InvalidRank,
    InvalidTargetModules,
    MalformedAdapterName,
    MismatchedTensorShape,
    MissingFactor,
    MissingTargetFactors,
    NonFiniteAdapter,
    TargetModuleNotAllowed,
    UnsupportedAdapterDType,
    UnsupportedAdapterType,
    UnsupportedLoRAVariant,
    UnsupportedTargetModules,
    UntargetedModule,
};

/// Whether `e` is one of `Refusal`'s.
pub fn isRefusal(e: anyerror) bool {
    inline for (@typeInfo(Refusal).error_set.error_names.?) |x| if (e == @field(anyerror, x)) return true;
    return false;
}

/// A host-side read or check failure as a refusal (out of memory kept as it is): an unnamed one is LoraUnreadable.
fn refusal(e: anyerror) anyerror {
    if (e == error.OutOfMemory or isRefusal(e)) return e;
    return error.LoraUnreadable;
}

const prefixes = [_][]const u8{ "model.diffusion_model.", "diffusion_model.", "base_model.model.", "transformer.", "lora_unet_", "lora_transformer_" };
const Side = enum { a, b, alpha };
const suffixes = [_]struct { []const u8, Side }{
    .{ ".lora_down.weight", .a }, .{ ".lora_up.weight", .b },        .{ ".lora_A.weight", .a },
    .{ ".lora_B.weight", .b },    .{ ".lora_A.default.weight", .a }, .{ ".lora_B.default.weight", .b },
    .{ ".alpha", .alpha },
};

fn split(name: []const u8) ?struct { module: []const u8, side: Side } {
    var rest = name;
    // prefixes stack (`base_model.model.transformer.`): take them off while one fits
    strip: while (true) {
        for (prefixes) |p| if (std.mem.startsWith(u8, rest, p)) {
            rest = rest[p.len..];
            continue :strip;
        };
        break;
    }
    for (suffixes) |s| if (std.mem.endsWith(u8, rest, s[0])) return .{ .module = rest[0 .. rest.len - s[0].len], .side = s[1] };
    return null;
}

fn scalar(t: st.Tensor) !f64 {
    if (t.numel() != 1) return error.BadAlpha;
    return switch (t.dtype) {
        .f32 => @as(f32, @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little))),
        .bf16 => host.widen(std.mem.readInt(u16, t.bytes[0..2], .little)),
        .f16 => @as(f16, @bitCast(std.mem.readInt(u16, t.bytes[0..2], .little))),
        else => error.BadAlpha,
    };
}

/// `v` as a GEMM's fp32 scale (rounded once), refused past fp32's range, where even zero factors would give NaN.
fn fp32Scale(v: f64) !f32 {
    const s: f32 = @floatCast(v);
    return if (std.math.isFinite(s)) s else error.LoraScaleOverflow;
}

const max_parts = 3;

/// `v` as bf16 pieces summing to it exactly, each cut toward zero (stays finite); null if not held or subnormal.
fn cut(v: f32) ?[max_parts]u16 {
    var rest = v;
    var out: [max_parts]u16 = @splat(0);
    for (&out) |*p| {
        p.* = @truncate(@as(u32, @bitCast(rest)) >> 16);
        if (p.* & 0x7F80 == 0 and p.* & 0x7F != 0) return null;
        // the bits below the cut: exact in f32
        rest -= host.widen(p.*);
    }
    return if (rest == 0) out else null;
}

/// A's values (from the file, finite: core.runtime_lora checked) on the device as bf16 pieces summing to them exactly.
fn toDevice(gpa: std.mem.Allocator, d: *const cuda.Driver, values: []const f32) !struct { buf: cuda.DeviceBuffer, parts: usize } {
    const all = try gpa.alloc(u16, max_parts * values.len);
    defer gpa.free(all);
    var parts: usize = 1;
    for (values, 0..) |v, i| {
        const p = cut(v) orelse return error.InexactAdapterFactor;
        for (p, 0..) |h, j| {
            all[j * values.len + i] = h;
            if (h & 0x7FFF != 0) parts = @max(parts, j + 1);
        }
    }
    return .{ .buf = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(all[0 .. parts * values.len])), .parts = parts };
}

/// The LoRA folder: $TF_LORA_DIR, else `<checkpoint>/loras`; owned by the caller.
pub fn folder(gpa: std.mem.Allocator, checkpoint: []const u8) ![]u8 {
    if (try sys.environment.path("TF_LORA_DIR", sys.environment.Process.get(.{}, "TF_LORA_DIR"))) |dir| return gpa.dupe(u8, dir);
    return std.fs.path.join(gpa, &.{ checkpoint, "loras" });
}

/// Whether `name` is a safe file name in the folder: letters, digits, '.', '_' and '-', no leading dot.
pub fn validName(name: []const u8) bool {
    if (name.len == 0 or name.len > 128 or name[0] == '.') return false;
    for (name) |c| if (!(std.ascii.isAlphanumeric(c) or c == '.' or c == '_' or c == '-')) return false;
    return true;
}

/// How many bytes from each end of a projection's weight the base identity reads.
const sample_bytes = 4096;

/// Each projection in order: name, header entry (bf16 [n, k], checked in Model.target), byte count, both ends (4 KiB).
fn baseIdentity(shapes: []const rl.Target, targets: *const std.StringHashMapUnmanaged(Target)) rl.BaseIdentity {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("tensorfold.moss_image.projections.v2\x00");
    for (shapes) |s| {
        const bytes = targets.get(s.module).?.bytes;
        var w: [24]u8 = undefined;
        std.mem.writeInt(u64, w[0..8], s.n, .little);
        std.mem.writeInt(u64, w[8..16], s.k, .little);
        std.mem.writeInt(u64, w[16..24], bytes.len, .little);
        h.update(s.module);
        h.update("\x00BF16\x00");
        h.update(&w);
        const n = @min(sample_bytes, bytes.len);
        h.update(bytes[0..n]);
        h.update(bytes[bytes.len - n ..]);
    }
    return rl.BaseIdentity.fromDigest(h.finalResult());
}

pub const Bank = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    dir: []const u8,
    lib: lt.Library,
    handle: lt.Handle,
    plans: std.AutoHashMapUnmanaged(lt.Spec, lt.Gemm) = .empty,
    ws: cuda.DeviceBuffer,
    adapters: std.StringArrayHashMapUnmanaged(*Adapter) = .empty,
    /// The projections the current picks touch, by their weight's device address.
    active: std.AutoHashMapUnmanaged(u64, std.ArrayList(Use)) = .empty,
    /// The current picks, names owned, to skip a rebuild when a request repeats them.
    picked: std.ArrayList(Pick) = .empty,
    low: ?cuda.DeviceBuffer = null,
    sums: ?cuda.DeviceBuffer = null,
    /// What core.runtime_lora checks an adapter against: every projection by its module name, sorted.
    shapes: []rl.Target = &.{},
    /// This transformer's projections (names, shapes, a sample of each weight's bytes), the base overlays are keyed to.
    base: rl.BaseIdentity = undefined,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, targets: *const std.StringHashMapUnmanaged(Target)) !*Bank {
        const b = try gpa.create(Bank);
        errdefer gpa.destroy(b);
        var lib = try lt.Library.open();
        errdefer lib.close();
        const handle = try lt.createHandle(&lib);
        errdefer _ = lib.api.cublasLtDestroy(handle);
        const ws = try cuda.DeviceBuffer.alloc(d, workspace_bytes);
        errdefer {
            var w = ws;
            w.free();
        }
        b.* = .{ .gpa = gpa, .io = io, .d = d, .dir = try gpa.dupe(u8, dir), .lib = lib, .handle = handle, .ws = ws };
        errdefer gpa.free(b.dir);
        try b.layout(targets);
        std.log.info("loras: from {s}", .{dir});
        return b;
    }

    pub fn deinit(b: *Bank) void {
        b.clearActive();
        b.active.deinit(b.gpa);
        b.clearPicked();
        b.picked.deinit(b.gpa);
        for (b.adapters.keys(), b.adapters.values()) |k, ad| {
            ad.deinit(b.gpa);
            b.gpa.destroy(ad);
            b.gpa.free(k);
        }
        b.adapters.deinit(b.gpa);
        if (b.low) |*l| l.free();
        if (b.sums) |*s| s.free();
        var it = b.plans.valueIterator();
        while (it.next()) |g| g.deinit();
        b.plans.deinit(b.gpa);
        b.ws.free();
        _ = b.lib.api.cublasLtDestroy(b.handle);
        b.lib.close();
        b.gpa.free(b.shapes);
        b.gpa.free(b.dir);
        b.gpa.destroy(b);
    }

    fn fewer(_: void, x: rl.Target, y: rl.Target) bool {
        return std.mem.order(u8, x.module, y.module) == .lt;
    }

    /// `shapes` and `base` from the model's projections (each listed once, under its dotted name).
    fn layout(b: *Bank, targets: *const std.StringHashMapUnmanaged(Target)) !void {
        var list: std.ArrayList(rl.Target) = .empty;
        errdefer list.deinit(b.gpa);
        var it = targets.iterator();
        while (it.next()) |e| if (std.mem.eql(u8, e.key_ptr.*, e.value_ptr.module))
            try list.append(b.gpa, .{ .module = e.value_ptr.module, .n = e.value_ptr.n, .k = e.value_ptr.k, .storage = .{ .dense = .bf16 } });
        b.shapes = try list.toOwnedSlice(b.gpa);
        std.mem.sort(rl.Target, b.shapes, {}, fewer);
        b.base = baseIdentity(b.shapes, targets);
    }

    fn clearActive(b: *Bank) void {
        var it = b.active.valueIterator();
        while (it.next()) |l| l.deinit(b.gpa);
        b.active.clearRetainingCapacity();
    }

    fn clearPicked(b: *Bank) void {
        for (b.picked.items) |p| b.gpa.free(p.name);
        b.picked.clearRetainingCapacity();
    }

    fn same(b: *const Bank, picks: []const Pick) bool {
        if (picks.len != b.picked.items.len) return false;
        for (picks, b.picked.items) |p, q| if (p.scale != q.scale or !std.mem.eql(u8, p.name, q.name)) return false;
        return true;
    }

    /// The LoRAs of `picks` on `targets` from the next projection on; none for an empty list.
    pub fn select(b: *Bank, picks: []const Pick, targets: *const std.StringHashMapUnmanaged(Target)) !void {
        if (b.same(picks)) return;
        b.clearActive();
        b.clearPicked();
        errdefer {
            b.clearActive();
            b.clearPicked();
        }
        for (picks) |p| {
            if (!validName(p.name)) return error.BadLoraName;
            if (!std.math.isFinite(p.scale)) return error.BadLoraScale;
            const ad = try b.adapter(p.name, targets);
            // core.runtime_lora matched every factor to a projection of this shape when it loaded; scale 0 adds nothing
            if (p.scale != 0) for (ad.factors.keys(), ad.factors.values()) |module, *f| {
                const t = targets.get(module) orelse return error.LoraModules;
                const got = try b.active.getOrPut(b.gpa, t.w);
                if (!got.found_existing) got.value_ptr.* = .empty;
                // an f64 product of two f32s is exact, so this rounds once, as an f32 multiply does
                const scale = try fp32Scale(@as(f64, f.scale) * p.scale);
                try got.value_ptr.append(b.gpa, .{ .f = f, .scale = scale });
            };
            try b.picked.append(b.gpa, .{ .name = try b.gpa.dupe(u8, p.name), .scale = p.scale });
        }
    }

    /// The loaded adapter `name`'s factors for `module`, if it has some.
    pub fn factor(b: *const Bank, name: []const u8, module: []const u8) ?*const Factor {
        const ad = b.adapters.get(name) orelse return null;
        return ad.factors.getPtr(module);
    }

    /// The overlays on the projection whose weight sits at `w`, if any.
    pub fn uses(b: *const Bank, w: u64) ?[]const Use {
        const l = b.active.getPtr(w) orelse return null;
        return l.items;
    }

    /// An fp32 [rows, n] buffer for a targeted projection's sums.
    pub fn sumsFor(b: *Bank, rows: usize, n: usize) !u64 {
        return grow(b.d, &b.sums, rows * n * 4);
    }

    fn grow(d: *const cuda.Driver, slot: *?cuda.DeviceBuffer, bytes: usize) !u64 {
        if (slot.*) |s| if (s.len >= bytes) return s.ptr;
        if (slot.*) |*s| s.free();
        slot.* = null;
        slot.* = try cuda.DeviceBuffer.alloc(d, bytes);
        return slot.*.?.ptr;
    }

    /// sums [rows, n] (x Wᵀ, fp32) += Σ scale (x Aᵀ) Bᵀ over `list`, x [rows, k] bf16; every product and sum in fp32.
    pub fn add(b: *Bank, s: cuda.Stream, list: []const Use, x: u64, sums: u64, rows: usize, n: usize, k: usize) !void {
        for (list) |u| {
            const r = u.f.rank;
            const low = try grow(b.d, &b.low, rows * r * 4);
            // column-major lowᵀ [r, rows] = Σ (piece stored k x r)ᵀ xᵀ: low [rows, r] fp32, bf16 products exact in fp32
            for (0..u.f.parts) |p| try b.run(s, .{ .ta = .t, .m = r, .n = rows, .k = k, .a = .bf16, .b = .bf16, .c = .f32, .lda = k, .ldb = k, .ldc = r }, 1, u.f.a.ptr + p * r * k * 2, x, if (p == 0) 0 else 1, low);
            // column-major sumsᵀ [n, rows] += scale (b stored r x n)ᵀ lowᵀ, fp32 products (no TF32)
            try b.run(s, .{ .ta = .t, .m = n, .n = rows, .k = r, .a = .f32, .b = .f32, .c = .f32, .lda = r, .ldb = r, .ldc = n }, u.scale, u.f.b.ptr, low, 1, sums);
        }
    }

    fn run(b: *Bank, s: cuda.Stream, spec: lt.Spec, alpha: f32, a: u64, x: u64, beta: f32, c: u64) !void {
        const got = try b.plans.getOrPut(b.gpa, spec);
        if (!got.found_existing) got.value_ptr.* = lt.Gemm.init(&b.lib, b.handle, spec, workspace_bytes) catch |e| {
            _ = b.plans.remove(spec);
            return e;
        };
        try got.value_ptr.run(alpha, a, x, beta, c, b.ws.ptr, b.ws.len, s.handle);
    }

    /// The adapter `name` from the folder, read and checked by core.runtime_lora, on the device from its first use.
    fn adapter(b: *Bank, name: []const u8, targets: *const std.StringHashMapUnmanaged(Target)) !*Adapter {
        if (b.adapters.get(name)) |ad| return ad;
        const started = @import("cuda_model.zig").seconds();
        var prepared = b.read(name, targets) catch |e| {
            const r = refusal(e);
            if (r != e) std.log.err("lora {s}: cannot be read ({s})", .{ name, @errorName(e) });
            return r;
        };
        defer prepared.deinit();
        const ad = try b.gpa.create(Adapter);
        errdefer b.gpa.destroy(ad);
        ad.* = .{ .arena = .init(b.gpa) };
        errdefer ad.deinit(b.gpa);
        for (prepared.overlays) |o| {
            const scale = fp32Scale(o.scale) catch |e| {
                std.log.err("lora {s}: alpha / rank = {e} is past fp32's range", .{ name, o.scale });
                return e;
            };
            const up = toDevice(b.gpa, b.d, o.a) catch |e| {
                if (e == error.InexactAdapterFactor) std.log.err("lora {s}: {s} has an A value whose bf16 pieces would be subnormal (below about 2^-103)", .{ name, o.module });
                return e;
            };
            var da = up.buf;
            errdefer da.free();
            var db = try cuda.DeviceBuffer.fromHost(b.d, std.mem.sliceAsBytes(o.b));
            errdefer db.free();
            try ad.factors.put(b.gpa, try ad.arena.allocator().dupe(u8, o.module), .{ .a = da, .b = db, .rank = o.rank, .parts = up.parts, .n = o.n, .k = o.k, .scale = scale, .exact_scale = o.scale });
            ad.bytes += da.len + db.len;
        }
        std.log.info("lora {s}: {d} projections, rank {d}, scale {d}, {d} MiB on the device, {d:.1} s (overlay key {x})", .{ name, ad.factors.count(), prepared.rank, prepared.scale, ad.bytes >> 20, @import("cuda_model.zig").seconds() - started, prepared.cache_key.digest[0..6] });
        const key = try b.gpa.dupe(u8, name);
        errdefer b.gpa.free(key);
        try b.adapters.put(b.gpa, key, ad);
        return ad;
    }

    /// The adapter `name`'s overlays on the host, the file's exact values (the gate's reference); fails its request.
    pub fn read(b: *Bank, name: []const u8, targets: *const std.StringHashMapUnmanaged(Target)) !rl.Prepared {
        var scratch: std.heap.ArenaAllocator = .init(b.gpa);
        defer scratch.deinit();
        const a = scratch.allocator();
        // a PEFT folder (adapter_config.json + adapter_model.safetensors), else <name>.safetensors
        var path = try std.fs.path.join(a, &.{ b.dir, name, "adapter_model.safetensors" });
        const peft_config: ?[]const u8 = std.Io.Dir.cwd().readFileAlloc(b.io, try std.fs.path.join(a, &.{ b.dir, name, "adapter_config.json" }), a, .limited(1 << 26)) catch |e| switch (e) {
            error.FileNotFound, error.NotDir => null,
            else => return e,
        };
        if (peft_config == null) path = try std.fmt.allocPrint(a, "{s}/{s}.safetensors", .{ b.dir, name });
        var file = st.File.open(b.gpa, b.io, path) catch |e| {
            std.log.err("lora {s}: cannot open {s} ({s})", .{ name, path, @errorName(e) });
            return error.LoraNotFound;
        };
        defer file.close(b.io);

        // Every factor under its PEFT name for this transformer's module (kohya's spellings and prefixes mapped).
        const named = try a.alloc(rl.NamedTensor, file.names.count());
        var count: usize = 0;
        var rank: ?usize = null;
        var alpha: ?f64 = null;
        var mixed = false;
        var downs: usize = 0;
        var alphas: usize = 0;
        var missed: usize = 0;
        for (file.names.keys()) |key| {
            const sp = split(key) orelse {
                std.log.err("lora {s}: {s} is not a LoRA factor (lora_down/lora_up/alpha or lora_A/lora_B)", .{ name, key });
                return error.LoraKeys;
            };
            const t = targets.get(sp.module) orelse {
                if (missed < 4) std.log.err("lora {s}: {s} names no projection of this transformer", .{ name, sp.module });
                missed += 1;
                continue;
            };
            const tensor = file.get(key).?;
            switch (sp.side) {
                .alpha => {
                    const v = try scalar(tensor);
                    if (alpha) |was| mixed = mixed or was != v else alpha = v;
                    alphas += 1;
                    continue;
                },
                .a => {
                    downs += 1;
                    if (tensor.rank == 2) {
                        if (rank) |was| mixed = mixed or was != tensor.dim(0) else rank = tensor.dim(0);
                    }
                },
                .b => {},
            }
            const side = if (sp.side == .a) "lora_A" else "lora_B";
            named[count] = .{ .name = try std.fmt.allocPrint(a, "base_model.model.{s}.{s}.weight", .{ t.module, side }), .tensor = tensor };
            count += 1;
        }
        if (missed > 0) {
            std.log.err("lora {s}: {d} of its tensors match no projection here (a LoRA for another model?)", .{ name, missed });
            return error.LoraModules;
        }
        // PEFT: its own config; kohya: one rank and one alpha for all modules (alpha = rank when it has none)
        var synth: []const u8 = undefined;
        const cfg: rl.Config = if (peft_config) |text| rl.parseConfig(a, text) catch |e| {
            std.log.err("lora {s}: adapter_config.json refused ({s})", .{ name, @errorName(e) });
            return e;
        } else blk: {
            // an alpha on some modules and not others is two scales, as is a second rank or alpha value
            if (mixed or (alphas != 0 and alphas != downs)) {
                std.log.err("lora {s}: its modules differ in rank or alpha; one of each is what this engine reads", .{name});
                return error.UnsupportedLoRAVariant;
            }
            const r = rank orelse return error.LoraMissingFactor;
            if (alpha == null) std.log.info("lora {s}: no alpha and no adapter_config.json, so alpha = rank (scale 1)", .{name});
            const al = alpha orelse @as(f64, @floatFromInt(r));
            synth = try std.fmt.allocPrint(a, "{{\"peft_type\":\"LORA\",\"r\":{d},\"lora_alpha\":{d}}}", .{ r, al });
            break :blk .{ .rank = r, .alpha = al, .scale = al / @as(f64, @floatFromInt(r)) };
        };
        const id = rl.AdapterIdentity.fromBytes(peft_config orelse synth, file.map.memory);
        return rl.prepare(b.gpa, b.base, id, cfg, named[0..count], b.shapes) catch |e| {
            std.log.err("lora {s}: refused ({s})", .{ name, @errorName(e) });
            return e;
        };
    }
};

test "adapter keys: kohya and PEFT spellings, with and without the usual prefixes" {
    const k = split("diffusion_model.layers.3.proj.q.lora_down.weight").?;
    try std.testing.expectEqualStrings("layers.3.proj.q", k.module);
    try std.testing.expectEqual(Side.a, k.side);
    try std.testing.expectEqual(Side.alpha, split("diffusion_model.layers.3.proj.q.alpha").?.side);
    try std.testing.expectEqualStrings("layers.0.proj.ffn_down", split("base_model.model.layers.0.proj.ffn_down.lora_B.weight").?.module);
    try std.testing.expectEqualStrings("layers_0_proj_q", split("lora_unet_layers_0_proj_q.lora_up.weight").?.module);
    try std.testing.expectEqualStrings("layers.1.proj.o", split("base_model.model.transformer.layers.1.proj.o.lora_A.default.weight").?.module);
    try std.testing.expect(split("layers.0.proj.q.weight") == null);
    try std.testing.expect(validName("moss-8step_dmd.v2"));
    try std.testing.expect(!validName("../etc"));
    try std.testing.expect(!validName(".hidden"));
}

test "env defaults: TF_LORA_DIR unset reads <checkpoint>/loras, a set folder as given, control characters refused" {
    const a = std.testing.allocator;
    const env = sys.env;
    const was = sys.getenv("TF_LORA_DIR");
    const saved = if (was) |w| try a.dupeSentinel(u8, w, 0) else null;
    defer if (saved) |s| {
        _ = env.setenv("TF_LORA_DIR", s, 1);
        a.free(s);
    } else {
        _ = env.unsetenv("TF_LORA_DIR");
    };
    _ = env.unsetenv("TF_LORA_DIR");
    const default = try folder(a, "/fixture/moss");
    defer a.free(default);
    const want = try std.fs.path.join(a, &.{ "/fixture/moss", "loras" });
    defer a.free(want);
    try std.testing.expectEqualStrings(want, default);
    _ = env.setenv("TF_LORA_DIR", "/fixture/my loras", 1);
    const given = try folder(a, "/fixture/moss");
    defer a.free(given);
    try std.testing.expectEqualStrings("/fixture/my loras", given);
    _ = env.setenv("TF_LORA_DIR", "/fixture/two\nlines", 1);
    try std.testing.expectError(error.InvalidEnvironment, folder(a, "/fixture/moss"));
}

test "base identity: names, shapes and both ends of each weight's bytes; the middle is not read" {
    const a = std.testing.allocator;
    const weights = try a.alloc(u8, 3 * sample_bytes);
    defer a.free(weights);
    @memset(weights, 1);
    var targets: std.StringHashMapUnmanaged(Target) = .empty;
    defer targets.deinit(a);
    try targets.put(a, "layers.0.proj.q", .{ .module = "layers.0.proj.q", .w = 1, .n = 48, .k = 128, .bytes = weights });
    const shapes = [_]rl.Target{.{ .module = "layers.0.proj.q", .n = 48, .k = 128, .storage = .{ .dense = .bf16 } }};
    const first = baseIdentity(&shapes, &targets).digest;
    weights[sample_bytes + 7] = 2;
    try std.testing.expectEqualSlices(u8, &first, &baseIdentity(&shapes, &targets).digest);
    weights[3] = 2;
    const head = baseIdentity(&shapes, &targets).digest;
    try std.testing.expect(!std.mem.eql(u8, &first, &head));
    weights[weights.len - 1] = 2;
    try std.testing.expect(!std.mem.eql(u8, &head, &baseIdentity(&shapes, &targets).digest));
    const wider = [_]rl.Target{.{ .module = "layers.0.proj.q", .n = 96, .k = 64, .storage = .{ .dense = .bf16 } }};
    try std.testing.expect(!std.mem.eql(u8, &baseIdentity(&shapes, &targets).digest, &baseIdentity(&wider, &targets).digest));
}

test "A pieces: bf16 pieces sum to the f32 or f16 value exactly, finite past bf16's largest, subnormal tails refused" {
    const sum = struct {
        fn of(p: [max_parts]u16) f32 {
            // largest first, as the GEMMs add them; each partial sum is exact
            return host.widen(p[0]) + host.widen(p[1]) + host.widen(p[2]);
        }
    };
    // bf16 values take one piece
    try std.testing.expectEqual([_]u16{ 0x3F80, 0, 0 }, cut(1).?);
    // f16's 1 + 2^-10 needs two; bf16 rounding would give 1
    const f16v: f32 = @as(f16, 1.0 + 1.0 / 1024.0);
    const two = cut(f16v).?;
    try std.testing.expect(two[1] != 0 and two[2] == 0);
    try std.testing.expectEqual(f16v, sum.of(two));
    // f32's largest and a neighbour: bf16 rounding makes both infinite, the pieces stay finite and exact
    for ([_]u32{ 0x7F7FFFFF, 0x7F7FFF00, 0xFF7FFFFF }) |bits| {
        const v: f32 = @bitCast(bits);
        try std.testing.expect(std.math.isInf(host.widen(host.bf16(v))));
        const p = cut(v).?;
        for (p) |h| try std.testing.expect(std.math.isFinite(host.widen(h)));
        try std.testing.expectEqual(v, sum.of(p));
    }
    // seeded f32 values over a wide range, and the cancellation a + (-a') that bf16 rounding loses
    var prng = std.Random.DefaultPrng.init(0x50696563);
    for (0..100_000) |_| {
        const v: f32 = @floatCast(prng.random().floatNorm(f64) * std.math.pow(f64, 2, @floatFromInt(prng.random().intRangeAtMost(i32, -90, 100))));
        try std.testing.expectEqual(v, sum.of(cut(v).?));
    }
    const a: f32 = 1 + 1.0 / 4096.0;
    try std.testing.expectEqual(@as(f32, 1.0 / 4096.0), sum.of(cut(a).?) - sum.of(cut(1).?));
    try std.testing.expectEqual(@as(f32, 0), host.round(a) - 1);
    // a subnormal piece (below bf16's smallest step, or one the GPU might flush) is refused, not rounded
    try std.testing.expect(cut(@bitCast(@as(u32, 0x00000001))) == null);
    try std.testing.expect(cut(@bitCast(@as(u32, 0x00010000))) == null);
    try std.testing.expectEqual([_]u16{ 0x8000, 0, 0 }, cut(-0.0).?);
}

test "refusal: a host-side failure fails the request alone, out of memory stays itself" {
    try std.testing.expectEqual(error.LoraUnreadable, refusal(error.SyntaxError));
    try std.testing.expectEqual(error.LoraUnreadable, refusal(error.AccessDenied));
    try std.testing.expectEqual(error.MismatchedTensorShape, refusal(error.MismatchedTensorShape));
    try std.testing.expectEqual(error.OutOfMemory, refusal(error.OutOfMemory));
    try std.testing.expect(isRefusal(error.LoraUnreadable) and !isRefusal(error.OutOfMemory));
}

test "scales: alpha / rank and its product with a strength are refused past fp32's range, exact up to it" {
    const max: f32 = std.math.floatMax(f32);
    try std.testing.expectEqual(max, try fp32Scale(max));
    for ([_]f64{ 1e39, @as(f64, max) * 2, -@as(f64, max) * 2 }) |v| try std.testing.expectError(error.LoraScaleOverflow, fp32Scale(v));
}
