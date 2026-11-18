//! BF16 dense MTP weights for SenseNova Loop, packed in the target's lane format.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const engine = @import("cuda_engine.zig");
const geometry = @import("cuda_geometry.zig");
const mtp = @import("cuda_mtp_weights.zig");
const files = @import("cuda_mtp_dense_file.zig");
const weights = @import("cuda_weights.zig");

fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !core.Checkpoint {
    const loc = try files.discover(io, dir);
    var ck: core.Checkpoint = .{ .gpa = gpa, .io = io };
    errdefer ck.close();
    if (loc == .root) {
        try ck.add(dir, "mtp.safetensors");
    } else {
        const path = if (loc == .child) try std.fs.path.join(gpa, &.{ dir, "mtp" }) else try gpa.dupe(u8, dir);
        defer gpa.free(path);
        var d = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
        defer d.close(io);
        var it = d.iterate();
        while (try it.next(io)) |entry| {
            if ((entry.kind == .file or entry.kind == .sym_link) and std.mem.endsWith(u8, entry.name, ".safetensors")) try ck.add(path, entry.name);
        }
    }
    return ck;
}

fn packedBytes(rows: usize, cols: usize) !usize {
    const padded = try mtp.checkedMul(try mtp.checkedAdd(rows, 127) / 128, 128);
    return mtp.checkedAligned(try mtp.checkedMul(try mtp.checkedMul(padded, cols), 2));
}

// Shards are discovered by directory iteration. Never choose between duplicated weights by iteration order.
fn headTensor(ck: *const core.Checkpoint, name: []const u8) !core.checkpoint.Tensor {
    var found: ?core.checkpoint.Tensor = null;
    for (ck.files.items) |*shard| {
        const tensor = shard.get(name) orelse continue;
        if (found != null) return error.DuplicateTensor;
        found = tensor;
    }
    return found orelse error.MissingTensor;
}

fn normBytes(ck: *const core.Checkpoint, name: []const u8, n: usize) !usize {
    _ = try files.rawNorm(try headTensor(ck, name), n, true);
    return mtp.checkedAligned(try mtp.checkedMul(n, 2));
}

// Use the same validated layout for admission and both device allocations.
fn plan(ck: *core.Checkpoint, g: geometry.Geometry) !mtp.AllocationPlan {
    try mtp.validateGeometry(g);
    var bytes: mtp.AllocationPlan = .{ .resident = 0, .staging = 0 };
    for (mtp.denseTensors(g)) |spec| {
        const n, const k = spec.shape;
        if (spec.rank == 1) {
            bytes.resident = try mtp.checkedAdd(bytes.resident, try normBytes(ck, spec.name, n));
            continue;
        }
        try files.projection(try headTensor(ck, spec.name), n, k);
        const parts: usize = if (spec.split_fc) 2 else 1;
        const cols = k / parts;
        const packed_bytes = try packedBytes(n, cols);
        bytes.resident = try mtp.checkedAdd(bytes.resident, try mtp.checkedMul(parts, packed_bytes));
        bytes.staging = @max(bytes.staging, try mtp.checkedMul(try mtp.checkedMul(n, cols), 2));
    }
    if (ck.has("mtp.main_norm.weight"))
        bytes.resident = try mtp.checkedAdd(bytes.resident, try normBytes(ck, "mtp.main_norm.weight", g.hidden));
    return bytes;
}

pub fn allocationPlan(gpa: std.mem.Allocator, io: std.Io, g: geometry.Geometry, dir: []const u8) !mtp.AllocationPlan {
    var ck = try open(gpa, io, dir);
    defer ck.close();
    return plan(&ck, g);
}

const Loader = struct {
    gpa: std.mem.Allocator,
    e: *engine.Engine,
    w: *mtp.Weights,
    ck: *core.Checkpoint,
    stage: *cuda.DeviceBuffer,

    fn pack(l: *Loader, bytes: []const u8, n: usize, k: usize) !engine.Proj {
        try l.stage.upload(0, bytes);
        const dst = try l.w.alloc(try packedBytes(n, k));
        try (l.e.mo orelse return error.FormatNotLoaded).pack.bf16Lane(l.e.stream, l.stage.ptr, dst, n, k);
        try l.e.stream.synchronize();
        return .{ .ql = .{ .bf16 = cuda.modelopt.bf16Weight(dst, n, k) } };
    }

    fn linear(l: *Loader, name: []const u8, n: usize, k: usize) !engine.Proj {
        const t = try l.ck.get(name);
        try files.projection(t, n, k);
        return l.pack(t.bytes, n, k);
    }

    fn norm(l: *Loader, name: []const u8, n: usize, raw_flag: ?*bool) !u64 {
        const t = try l.ck.get(name);
        if (try files.rawNorm(t, n, raw_flag != null)) {
            const ptr = try weights.upload(l.w, t.bytes);
            raw_flag.?.* = true;
            return ptr;
        }
        const out = try l.gpa.alloc(u16, n);
        defer l.gpa.free(out);
        try files.shift(t, out);
        return weights.upload(l.w, std.mem.sliceAsBytes(out));
    }
};

pub fn load(gpa: std.mem.Allocator, io: std.Io, e: *engine.Engine, dir: []const u8) !mtp.Weights {
    const g = e.g;
    const h = g.hidden;
    var ck = try open(gpa, io, dir);
    defer ck.close();
    const bytes = try plan(&ck, g);
    const aw = g.attnWidth();
    const kw = g.kv_heads * geometry.head_dim;
    const ni = g.intermediate;
    var w: mtp.Weights = .{ .arena = try cuda.DeviceBuffer.alloc(e.d, bytes.resident), .fc_e = undefined, .fc_h = undefined, .norm_e = 0, .norm_h = 0, .input_norm = 0, .post_norm = 0, .q_norm = 0, .k_norm = 0, .norm = 0, .q = undefined, .k = undefined, .v = undefined, .o = undefined, .mlp = undefined };
    errdefer w.deinit();
    var stage = try cuda.DeviceBuffer.alloc(e.d, bytes.staging);
    defer stage.free();
    var l: Loader = .{ .gpa = gpa, .e = e, .w = &w, .ck = &ck, .stage = &stage };
    const fc = try ck.expect("mtp.fc.weight", .bf16, &.{ h, 2 * h });
    const half = try gpa.alloc(u8, h * h * 2);
    defer gpa.free(half);
    for (0..2) |part| {
        for (0..h) |row| @memcpy(half[row * h * 2 ..][0 .. h * 2], fc.bytes[(row * 2 * h + part * h) * 2 ..][0 .. h * 2]);
        if (part == 0) w.fc_e = try l.pack(half, h, h) else w.fc_h = try l.pack(half, h, h);
    }
    w.norm_e = try l.norm("mtp.pre_fc_norm_embedding.weight", h, &w.raw.norm_e);
    w.norm_h = try l.norm("mtp.pre_fc_norm_hidden.weight", h, &w.raw.norm_h);
    w.norm = try l.norm("mtp.norm.weight", h, &w.raw.norm);
    if (ck.has("mtp.main_norm.weight")) w.main_norm = try l.norm("mtp.main_norm.weight", h, &w.raw.main);
    w.input_norm = try l.norm("mtp.layers.0.input_layernorm.weight", h, &w.raw.input);
    w.post_norm = try l.norm("mtp.layers.0.post_attention_layernorm.weight", h, &w.raw.post);
    w.q_norm = try l.norm("mtp.layers.0.self_attn.q_norm.weight", geometry.head_dim, null);
    w.k_norm = try l.norm("mtp.layers.0.self_attn.k_norm.weight", geometry.head_dim, null);
    w.q = try l.linear("mtp.layers.0.self_attn.q_proj.weight", 2 * aw, h);
    w.k = try l.linear("mtp.layers.0.self_attn.k_proj.weight", kw, h);
    w.v = try l.linear("mtp.layers.0.self_attn.v_proj.weight", kw, h);
    w.o = try l.linear("mtp.layers.0.self_attn.o_proj.weight", h, aw);
    w.mlp = .{ .dense = .{
        .gate = try l.linear("mtp.layers.0.mlp.gate_proj.weight", ni, h),
        .up = try l.linear("mtp.layers.0.mlp.up_proj.weight", ni, h),
        .down = try l.linear("mtp.layers.0.mlp.down_proj.weight", h, ni),
    } };
    return w;
}
