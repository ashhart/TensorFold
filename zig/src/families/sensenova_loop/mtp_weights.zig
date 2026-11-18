//! Official SenseNova-Loop MTP safetensors weight loader (`mtp.safetensors` beside checkpoint).
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const cp = core.checkpoint_metal;
const linear_gpu = @import("linear_gpu.zig");

pub const file = "mtp.safetensors";

pub const MtpMat = union(enum) {
    bf16: linear_gpu.Bf16Matrix,
    awq: linear_gpu.AwqMatrix,

    pub fn deinit(m: *MtpMat) void {
        switch (m.*) {
            .awq => |*a| a.deinit(),
            .bf16 => {},
        }
    }
};

pub const Weights = struct {
    allocator: std.mem.Allocator,
    ck: ?cp.Checkpoint = null,
    owned: std.ArrayList(mtl.Buffer) = .empty,
    bits: u32 = 16,
    norm_e: cp.Tensor,
    norm_h: cp.Tensor,
    input_norm: cp.Tensor,
    post_norm: cp.Tensor,
    q_norm: cp.Tensor,
    k_norm: cp.Tensor,
    norm: cp.Tensor,
    fc: MtpMat,
    qkv: MtpMat,
    o: MtpMat,
    gate_up: MtpMat,
    down: MtpMat,

    /// True when `{dir}/mtp.safetensors` exists and can be opened.
    pub fn hasFile(io: std.Io, dir: []const u8) bool {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, file }) catch return false;
        std.Io.Dir.cwd().access(io, path, .{}) catch return false;
        return true;
    }

    pub fn load(
        gpa: std.mem.Allocator,
        io: std.Io,
        device: mtl.Device,
        queue: mtl.Queue,
        dir: []const u8,
        hidden: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        intermediate: usize,
    ) !Weights {
        _ = io;
        const bits = try @import("sys").environment.read(u32, "TF_MTP_BITS", 4);
        if (bits != 4 and bits != 16) return error.BadMtpBits;
        const path = try std.fs.path.joinZ(gpa, &.{ dir, file });
        defer gpa.free(path);
        var ck = cp.Checkpoint.init(gpa);
        var keep_ck = true;
        errdefer if (keep_ck) ck.deinit();
        try ck.addFile(device, path, "");
        var temp_owned: std.ArrayList(mtl.Buffer) = .empty;
        var keep_temp = true;
        errdefer if (keep_temp) {
            for (temp_owned.items) |b| b.deinit();
            temp_owned.deinit(gpa);
        };
        const m = try resolvePrefix(&ck);
        const p = if (std.mem.eql(u8, m, "mtp.")) "mtp.layers.0." else "language_model.mtp.layers.0.";
        const q_rows = 2 * heads * head_dim;
        const kv_rows = kv_heads * head_dim;
        const o_cols = heads * head_dim;
        var norm_e = try normTensor(gpa, &ck, m, "pre_fc_norm_embedding.weight", hidden);
        var norm_h = try normTensor(gpa, &ck, m, "pre_fc_norm_hidden.weight", hidden);
        var norm = try normTensor(gpa, &ck, m, "norm.weight", hidden);
        var input_norm = try normTensor(gpa, &ck, p, "input_layernorm.weight", hidden);
        var post_norm = try normTensor(gpa, &ck, p, "post_attention_layernorm.weight", hidden);
        var q_norm = try normTensor(gpa, &ck, p, "self_attn.q_norm.weight", head_dim);
        var k_norm = try normTensor(gpa, &ck, p, "self_attn.k_norm.weight", head_dim);
        const bf_fc = try singleMatrix(gpa, device, &ck, &temp_owned, m, "fc.weight", hidden, 2 * hidden);
        const bf_qkv = try stackMatrices(gpa, device, &ck, &temp_owned, p, &.{
            .{ .suffix = "self_attn.q_proj.weight", .n = q_rows },
            .{ .suffix = "self_attn.k_proj.weight", .n = kv_rows },
            .{ .suffix = "self_attn.v_proj.weight", .n = kv_rows },
        }, hidden);
        const bf_o = try singleMatrix(gpa, device, &ck, &temp_owned, p, "self_attn.o_proj.weight", hidden, o_cols);
        const bf_gu = try stackMatrices(gpa, device, &ck, &temp_owned, p, &.{
            .{ .suffix = "mlp.gate_proj.weight", .n = intermediate },
            .{ .suffix = "mlp.up_proj.weight", .n = intermediate },
        }, hidden);
        const bf_down = try singleMatrix(gpa, device, &ck, &temp_owned, p, "mlp.down_proj.weight", hidden, intermediate);

        if (comptime @hasDecl(linear_gpu.LinearGpu, "quantizeBf16ToAwq")) {
            if (bits == 4) {
                var owned: std.ArrayList(mtl.Buffer) = .empty;
                errdefer {
                    for (owned.items) |b| b.deinit();
                    owned.deinit(gpa);
                }
                try packNorms(gpa, device, &owned, .{ &norm_e, &norm_h, &norm, &input_norm, &post_norm, &q_norm, &k_norm });
                var sn_gpu = linear_gpu.LinearGpu.init(gpa, device);
                defer sn_gpu.deinit();
                const src = [5]linear_gpu.Bf16Matrix{ bf_fc, bf_qkv, bf_o, bf_gu, bf_down };
                var q_mats: [5]?linear_gpu.AwqMatrix = @splat(null);
                errdefer for (&q_mats) |*qm| if (qm.*) |*am| am.deinit();
                for (src, &q_mats) |mat, *dst| dst.* = try quantizeMat(&sn_gpu, queue, mat);
                for (temp_owned.items) |b| b.deinit();
                temp_owned.deinit(gpa);
                keep_temp = false;
                ck.deinit();
                keep_ck = false;
                return .{
                    .allocator = gpa,
                    .ck = null,
                    .owned = owned,
                    .bits = 4,
                    .norm_e = norm_e,
                    .norm_h = norm_h,
                    .norm = norm,
                    .input_norm = input_norm,
                    .post_norm = post_norm,
                    .q_norm = q_norm,
                    .k_norm = k_norm,
                    .fc = .{ .awq = q_mats[0].? },
                    .qkv = .{ .awq = q_mats[1].? },
                    .o = .{ .awq = q_mats[2].? },
                    .gate_up = .{ .awq = q_mats[3].? },
                    .down = .{ .awq = q_mats[4].? },
                };
            }
        }
        keep_ck = false;
        keep_temp = false;
        return .{
            .allocator = gpa,
            .ck = ck,
            .owned = temp_owned,
            .bits = 16,
            .norm_e = norm_e,
            .norm_h = norm_h,
            .norm = norm,
            .input_norm = input_norm,
            .post_norm = post_norm,
            .q_norm = q_norm,
            .k_norm = k_norm,
            .fc = .{ .bf16 = bf_fc },
            .qkv = .{ .bf16 = bf_qkv },
            .o = .{ .bf16 = bf_o },
            .gate_up = .{ .bf16 = bf_gu },
            .down = .{ .bf16 = bf_down },
        };
    }

    pub fn appendBuffers(w: *const Weights, a: std.mem.Allocator, list: *std.ArrayList(mtl.Buffer)) !void {
        if (w.ck) |*ck| for (ck.shards.items) |s| try list.append(a, s.buffer);
        try list.appendSlice(a, w.owned.items);
        for ([_]MtpMat{ w.fc, w.qkv, w.o, w.gate_up, w.down }) |mat| switch (mat) {
            .awq => |awq_m| {
                try list.appendSlice(a, &.{ awq_m.w, awq_m.sb });
                if (awq_m.input_scale) |ib| try list.append(a, ib);
            },
            .bf16 => {},
        };
    }

    pub fn applyResident(w: *const Weights, f: mtl.ResidencySet.Follow, joining: bool) void {
        if (w.ck) |*ck| for (ck.shards.items) |s| mtl.resident.fields.one(s.buffer, f, joining);
        for (w.owned.items) |b| mtl.resident.fields.one(b, f, joining);
        for ([_]MtpMat{ w.fc, w.qkv, w.o, w.gate_up, w.down }) |mat| switch (mat) {
            .awq => |awq_m| {
                mtl.resident.fields.one(awq_m.w, f, joining);
                mtl.resident.fields.one(awq_m.sb, f, joining);
                if (awq_m.input_scale) |ib| mtl.resident.fields.one(ib, f, joining);
            },
            .bf16 => {},
        };
    }

    pub fn deinit(w: *Weights) void {
        for ([_]*MtpMat{ &w.fc, &w.qkv, &w.o, &w.gate_up, &w.down }) |mat| mat.deinit();
        for (w.owned.items) |b| b.deinit();
        w.owned.deinit(w.allocator);
        if (w.ck) |*ck| ck.deinit();
    }
};

pub fn quantizeMat(sn: *linear_gpu.LinearGpu, queue: mtl.Queue, mat: linear_gpu.Bf16Matrix) !linear_gpu.AwqMatrix {
    if (comptime @hasDecl(linear_gpu.LinearGpu, "quantizeBf16ToAwq")) {
        return sn.quantizeBf16ToAwq(queue, mat);
    } else return error.UnsupportedQuantization;
}

fn packNorms(a: std.mem.Allocator, device: mtl.Device, owned: *std.ArrayList(mtl.Buffer), norms: [7]*cp.Tensor) !void {
    var total: usize = 0;
    for (norms) |t| total += std.mem.alignForward(usize, t.bytes, 16);
    const buf = try ownBuffer(a, device, owned, total);
    var off: usize = 0;
    for (norms) |t| {
        @memcpy(buf.contents()[off..][0..t.bytes], t.buffer.contents()[t.offset..][0..t.bytes]);
        t.buffer = buf;
        t.offset = off;
        off += std.mem.alignForward(usize, t.bytes, 16);
    }
}

fn resolvePrefix(ck: *const cp.Checkpoint) ![]const u8 {
    if (ck.has("mtp.fc.weight")) return "mtp.";
    if (ck.has("language_model.mtp.fc.weight")) return "language_model.mtp.";
    return error.MissingMtpWeights;
}

fn checked(t: cp.Tensor, shape: []const usize) !cp.Tensor {
    if (t.dtype != .bf16 or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) return error.BadMtpTensor;
    return t;
}

fn normTensor(a: std.mem.Allocator, ck: *const cp.Checkpoint, prefix: []const u8, suffix: []const u8, n: usize) !cp.Tensor {
    const name = try std.mem.concat(a, u8, &.{ prefix, suffix });
    defer a.free(name);
    return checked(try ck.get(name), &.{n});
}

fn ownBuffer(a: std.mem.Allocator, device: mtl.Device, owned: *std.ArrayList(mtl.Buffer), bytes: usize) !mtl.Buffer {
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const b = try device.buffer(bytes, opts);
    owned.append(a, b) catch |err| {
        b.deinit();
        return err;
    };
    return b;
}

fn singleMatrix(
    a: std.mem.Allocator,
    device: mtl.Device,
    ck: *const cp.Checkpoint,
    owned: *std.ArrayList(mtl.Buffer),
    prefix: []const u8,
    suffix: []const u8,
    n: usize,
    k: usize,
) !linear_gpu.Bf16Matrix {
    const name = try std.mem.concat(a, u8, &.{ prefix, suffix });
    defer a.free(name);
    const t = try checked(try ck.get(name), &.{ n, k });
    if (t.offset % 16 == 0) return linear_gpu.LinearGpu.prepareBf16(t.buffer, t.offset, @intCast(n), @intCast(k));
    const buf = try ownBuffer(a, device, owned, n * k * 2);
    @memcpy(buf.contents()[0 .. n * k * 2], t.buffer.contents()[t.offset..][0 .. n * k * 2]);
    return linear_gpu.LinearGpu.prepareBf16(buf, 0, @intCast(n), @intCast(k));
}

fn stackMatrices(
    a: std.mem.Allocator,
    device: mtl.Device,
    ck: *const cp.Checkpoint,
    owned: *std.ArrayList(mtl.Buffer),
    prefix: []const u8,
    parts: []const struct { suffix: []const u8, n: usize },
    k: usize,
) !linear_gpu.Bf16Matrix {
    var total_n: usize = 0;
    for (parts) |part| total_n += part.n;
    const buf = try ownBuffer(a, device, owned, total_n * k * 2);
    var dst_bytes: usize = 0;
    for (parts) |part| {
        const name = try std.mem.concat(a, u8, &.{ prefix, part.suffix });
        defer a.free(name);
        const t = try checked(try ck.get(name), &.{ part.n, k });
        const row_bytes = part.n * k * 2;
        @memcpy(buf.contents()[dst_bytes..][0..row_bytes], t.buffer.contents()[t.offset..][0..row_bytes]);
        dst_bytes += row_bytes;
    }
    return linear_gpu.LinearGpu.prepareBf16(buf, 0, @intCast(total_n), @intCast(k));
}
