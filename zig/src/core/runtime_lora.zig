//! Runtime PEFT loading into immutable owned overlays; base weights are never read or modified here.
const std = @import("std");
const st = @import("safetensors.zig");
const config = @import("runtime_lora_config.zig");
const identity = @import("runtime_lora_identity.zig");
const ov = @import("runtime_lora_overlay.zig");
const rt = @import("runtime_lora_target.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const Config = config.Config;
pub const parseConfig = config.parse;
pub const BaseIdentity = identity.BaseIdentity;
pub const AdapterIdentity = identity.AdapterIdentity;
pub const CacheKey = identity.CacheKey;
pub const baseProjectionKey = identity.baseProjectionKey;
pub const cache_namespace = identity.cache_namespace;
pub const base_projection_namespace = identity.base_projection_namespace;
pub const Target = rt.Target;
pub const Storage = rt.Storage;
pub const Affine = rt.Affine;
pub const Overlay = ov.Overlay;

pub const NamedTensor = struct {
    name: []const u8,
    tensor: st.Tensor,
};

pub const Prepared = struct {
    arena: std.heap.ArenaAllocator,
    base: BaseIdentity,
    adapter: AdapterIdentity,
    cache_key: CacheKey,
    rank: usize,
    alpha: f64,
    scale: f64,
    overlays: []const Overlay,

    pub fn deinit(p: *Prepared) void {
        p.arena.deinit();
        p.* = undefined;
    }

    pub fn find(p: *const Prepared, module: []const u8) ?*const Overlay {
        for (p.overlays) |*overlay| if (std.mem.eql(u8, overlay.module, module)) return overlay;
        return null;
    }

    pub fn apply(p: *const Prepared, module: []const u8, x: []const f64, x_stride: usize, y: []f64, y_stride: usize, rows: usize, work: []f64) !void {
        const overlay = p.find(module) orelse return error.MissingOverlay;
        return overlay.apply(x, x_stride, y, y_stride, rows, work);
    }
};

const Pair = struct {
    seen: bool = false,
    a: ?st.Tensor = null,
    b: ?st.Tensor = null,
};

fn fewer(_: void, left: Overlay, right: Overlay) bool {
    return std.mem.order(u8, left.module, right.module) == .lt;
}

pub fn prepare(gpa: Allocator, base: BaseIdentity, adapter: AdapterIdentity, cfg: Config, tensors: []const NamedTensor, targets: []const Target) !Prepared {
    try cfg.validate();
    for (targets, 0..) |target, i| {
        try target.validate();
        for (targets[0..i]) |other| if (std.mem.eql(u8, target.module, other.module)) return error.DuplicateTarget;
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const pairs = try a.alloc(Pair, targets.len);
    for (pairs) |*pair| pair.* = .{};
    var adapters: usize = 0;
    for (tensors) |named| {
        const key = ov.parseName(named.name) orelse return error.MalformedAdapterName;
        if (!cfg.targets(key.module)) return error.TargetModuleNotAllowed;
        const index = for (targets, 0..) |target, i| {
            if (std.mem.eql(u8, key.module, target.module)) break i;
        } else return error.UntargetedModule;
        const pair = &pairs[index];
        if (!pair.seen) {
            pair.seen = true;
            adapters += 1;
        }
        switch (key.side) {
            .a => {
                if (pair.a != null) return error.DuplicateFactor;
                pair.a = named.tensor;
            },
            .b => {
                if (pair.b != null) return error.DuplicateFactor;
                pair.b = named.tensor;
            },
        }
    }
    if (adapters == 0) return error.EmptyAdapter;
    for (targets, pairs) |target, pair| {
        if (pair.seen and (pair.a == null or pair.b == null)) return error.MissingFactor;
        if (!pair.seen and cfg.target_modules != null and cfg.targets(target.module)) return error.MissingTargetFactors;
    }
    const overlays = try a.alloc(Overlay, adapters);
    var out: usize = 0;
    for (targets, pairs) |target, pair| {
        if (!pair.seen) continue;
        try ov.validateFactor(pair.a.?, cfg.rank, target.k);
        try ov.validateFactor(pair.b.?, target.n, cfg.rank);
        const factors_a = try a.alloc(f32, try std.math.mul(usize, cfg.rank, target.k));
        const factors_b = try a.alloc(f32, try std.math.mul(usize, target.n, cfg.rank));
        try ov.convertFactor(pair.a.?, cfg.rank, target.k, factors_a);
        try ov.convertFactor(pair.b.?, target.n, cfg.rank, factors_b);
        overlays[out] = .{
            .module = try a.dupe(u8, target.module),
            .n = target.n,
            .k = target.k,
            .rank = cfg.rank,
            .scale = cfg.scale,
            .storage = target.storage,
            .a = factors_a,
            .b = factors_b,
        };
        out += 1;
    }
    std.mem.sort(Overlay, overlays, {}, fewer);
    var key = identity.KeyBuilder.init(base, adapter);
    for (overlays) |overlay| key.add(overlay.module, overlay.n, overlay.k, overlay.rank, overlay.scale, overlay.storage);
    return .{
        .arena = arena,
        .base = base,
        .adapter = adapter,
        .cache_key = key.finish(base, adapter),
        .rank = cfg.rank,
        .alpha = cfg.alpha,
        .scale = cfg.scale,
        .overlays = overlays,
    };
}

pub fn load(gpa: Allocator, io: Io, base: BaseIdentity, adapter_dir: []const u8, targets: []const Target) !Prepared {
    const config_path = try std.fs.path.join(gpa, &.{ adapter_dir, "adapter_config.json" });
    defer gpa.free(config_path);
    const weights_path = try std.fs.path.join(gpa, &.{ adapter_dir, "adapter_model.safetensors" });
    defer gpa.free(weights_path);
    const config_bytes = try Io.Dir.cwd().readFileAlloc(io, config_path, gpa, .limited(1 << 26));
    defer gpa.free(config_bytes);
    var scratch: std.heap.ArenaAllocator = .init(gpa);
    defer scratch.deinit();
    const cfg = try config.parse(scratch.allocator(), config_bytes);
    var file = try st.File.open(gpa, io, weights_path);
    defer file.close(io);
    const adapter = AdapterIdentity.fromBytes(config_bytes, file.map.memory);
    const tensors = try scratch.allocator().alloc(NamedTensor, file.names.count());
    for (file.names.keys(), tensors) |name, *named| named.* = .{ .name = name, .tensor = file.get(name).? };
    return prepare(gpa, base, adapter, cfg, tensors, targets);
}

test {
    _ = @import("runtime_lora_test.zig");
}
