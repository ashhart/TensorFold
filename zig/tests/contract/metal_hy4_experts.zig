//! Hy4-flash's experts group: the portable routed rows, the units' affine_mm gathers, tensor_dispatch's tables.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const h = @import("metal_hy4.zig");
const mmx = @import("metal_hy4_experts_mm.zig");
const ix = @import("metal_hy4_experts_index.zig");
const g = contract.gates;

// --- the planted faults; each is also a gate-5 launch named "hy4 [units ]<stem> [qN ]<label> <function>" ----------

const silu_point: mmx.Fault = .{ .label = "silu-unrounded", .what = "the SiLU's bf16 point dropped", .gate = .shapes, .edit = .{ .kernel = "affine_expert_rows", .from = "const float a = float(bfloat(precise::divide(gate, 1.0f + precise::exp(-gate))));", .to = "const float a = precise::divide(gate, 1.0f + precise::exp(-gate));" } };
const weight_point: mmx.Fault = .{ .label = "weight-unrounded", .what = "each dequantized weight's bf16 point dropped", .gate = .shapes, .edit = .{ .kernel = "affine_expert_dot", .from = "const float w = float(bfloat(fma(code, s, b)));", .to = "const float w = fma(code, s, b);" } };
const half_sums: mmx.Fault = .{ .label = "half-sums", .what = "each lane's sum rounded to fp16 a step", .gate = .sums, .edit = .{ .kernel = "affine_expert_dot", .from = "acc = fma(float(input[index]), w, acc);", .to = "acc = float(half(fma(float(input[index]), w, acc)));" } };
const glu_map: mmx.Fault = .{ .label = "map-ignored-glu", .what = "a job's gate and up read at the job's index, not its expert's", .gate = .shapes, .edit = .{ .kernel = "tf_affine_gather_glu_##BM", .from = "long(TF_AM_EXPERT(expert)) * a.n + c0", .to = "long(expert) * a.n + c0" } };
const mlx_silu: mmx.Fault = .{ .label = "mlx-sigmoid", .what = "the SiLU in MLX's bf16 sigmoid steps (TF_GLU_ROUND_SILU off)", .gate = .shapes, .mlx_silu = true };
const scatter_map: mmx.Fault = .{ .label = "map-ignored-scatter", .what = "a job's down weights read at the job's index, not its expert's", .gate = .shapes, .edit = .{ .kernel = "tf_affine_gather_scatter_##BM", .from = "long(TF_AM_EXPERT(expert)) * a.n + col", .to = "long(expert) * a.n + col" } };
const split_short: ix.Fault = .{ .label = "kv-short", .what = "each row's last kv column written as its gate", .gate = .shapes, .edit = .{ .kernel = "tf_hy4_qkv_split", .from = "else if (col < qn + kn)", .to = "else if (col + 1 < qn + kn)" } };
const pair0: ix.Fault = .{ .label = "pair0-dropped", .what = "a member at pair 0 ending its expert's list", .gate = .shapes, .edit = .{ .kernel = "tf_hy4_active_experts", .from = "members[t * shape.y + i] >= 0", .to = "members[t * shape.y + i] > 0" } };
const own_row: ix.Fault = .{ .label = "own-row", .what = "each pair's rank counting its own row", .gate = .shapes, .edit = .{ .kernel = "tf_hy4_group_pairs", .from = "for (uint q = 0; q < r; ++q)", .to = "for (uint q = 0; q <= r; ++q)" } };

const mm_faults = [_]mmx.Fault{ silu_point, weight_point, half_sums, glu_map, mlx_silu, scatter_map };
const index_faults = [_]ix.Fault{ split_short, pair0, own_row };

const Reg = union(enum) {
    mm: mmx.Variant,
    index: struct { which: ix.Which, fault: ?ix.Fault = null },

    fn tensorOnly(r: Reg) bool {
        return switch (r) {
            .mm => |v| v.kind == .glu or v.kind == .scatter,
            .index => true,
        };
    }
};

/// Every kernel at both checkpoint widths (4 and 8 bits, groups of 64), then the planted controls.
const regs = [_]Reg{
    .{ .mm = .{ .kind = .up, .bits = 4 } },
    .{ .mm = .{ .kind = .up, .bits = 8 } },
    .{ .mm = .{ .kind = .down, .bits = 4 } },
    .{ .mm = .{ .kind = .down, .bits = 8 } },
    .{ .mm = .{ .kind = .glu, .bits = 4 } },
    .{ .mm = .{ .kind = .glu, .bits = 8 } },
    .{ .mm = .{ .kind = .scatter, .bits = 4 } },
    .{ .mm = .{ .kind = .scatter, .bits = 8 } },
    .{ .index = .{ .which = .split } },
    .{ .index = .{ .which = .active } },
    .{ .index = .{ .which = .group } },
    .{ .mm = .{ .kind = .up, .bits = 4, .fault = silu_point } },
    .{ .mm = .{ .kind = .down, .bits = 8, .fault = weight_point } },
    .{ .mm = .{ .kind = .down, .bits = 4, .fault = half_sums } },
    .{ .mm = .{ .kind = .glu, .bits = 4, .fault = glu_map } },
    .{ .mm = .{ .kind = .glu, .bits = 8, .fault = mlx_silu } },
    .{ .mm = .{ .kind = .scatter, .bits = 8, .fault = scatter_map } },
    .{ .index = .{ .which = .split, .fault = split_short } },
    .{ .index = .{ .which = .active, .fault = pair0 } },
    .{ .index = .{ .which = .group, .fault = own_row } },
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    projs: h.fl.List(mmx.Proj) = .{},
    indexes: h.fl.List(ix.Index) = .{},

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        const parsed = try h.text(s.a);
        defer parsed.deinit();
        const t = parsed.value.text_config;
        const w = mmx.Widths.of(t);
        const mla = h.hy.mla_weights;
        const cols: ix.Split = .{ .q = t.q_lora_rank, .kv = mla.absorbed_stride, .value = mla.value_width, .heads = .{ t.num_attention_heads, t.swa_num_attention_heads } };
        var buf: [200]u8 = undefined;
        for (regs) |r| {
            const base = switch (r) {
                .mm => |v| mmx.baseName(v, &buf),
                .index => |x| ix.Index.baseName(x.which, &buf),
            };
            if (!h.fl.wanted(only, base)) continue;
            // the loaders build these only on tensor units (metal_routed.loadForModel, metal_dense's tensors)
            if (r.tensorOnly() and !@import("metal_support.zig").tensor(s.device.tensorUnits(), base)) continue;
            switch (r) {
                .mm => |v| try s.projs.add(s.a, s.kernels, try mmx.Proj.init(s.a, s.device, s.queue, v, w)),
                .index => |x| try s.indexes.add(s.a, s.kernels, try ix.Index.init(s.a, s.device, s.queue, x.which, x.fault, cols, w.experts, w.topk)),
            }
        }
    }

    pub fn deinit(s: *Suite) void {
        s.projs.deinit(s.a);
        s.indexes.deinit(s.a);
    }
};

fn label(name: []const u8, head: []const u8, fun: []const u8) []const u8 {
    return std.mem.trim(u8, name[head.len .. name.len - fun.len], " ");
}

/// Gate 5's texts for this group's launches (each loader's library, or a control's edit of it); null for others'.
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, name: []const u8) !?h.fl.Texts {
    const fun = name[std.mem.lastIndexOfScalar(u8, name, ' ').? + 1 ..];
    const portable = "hy4 affine_experts ";
    const units_mm = "hy4 units affine_mm ";
    const units_index = "hy4 units tensor_dispatch ";
    if (std.mem.startsWith(u8, name, portable)) {
        const fault = try find(mmx.Fault, &mm_faults, label(name, portable, fun));
        const body = try mmx.text(a, device, .{ .kind = .up, .bits = 4, .fault = fault });
        return .{ .here = body, .pre_m5 = body };
    }
    if (std.mem.startsWith(u8, name, units_mm)) {
        // "q4 " or "q8 ", then a control's label; the loader has no M1/M2 library
        const rest = name[units_mm.len..];
        const bits: u8 = if (std.mem.startsWith(u8, rest, "q4 ")) 4 else if (std.mem.startsWith(u8, rest, "q8 ")) 8 else return error.UnknownLaunch;
        const fault = try find(mmx.Fault, &mm_faults, label(name, name[0 .. units_mm.len + 3], fun));
        return .{ .here = try mmx.text(a, device, .{ .kind = .glu, .bits = bits, .fault = fault }), .pre_m5 = "" };
    }
    if (std.mem.startsWith(u8, name, units_index)) {
        const fault = try find(ix.Fault, &index_faults, label(name, units_index, fun));
        return .{ .here = try ix.text(a, fault), .pre_m5 = "" };
    }
    return null;
}

fn find(comptime F: type, list: []const F, want: []const u8) !?F {
    if (want.len == 0) return null;
    for (list) |f| if (std.mem.eql(u8, f.label, want)) return f;
    return error.UnknownLaunch;
}
