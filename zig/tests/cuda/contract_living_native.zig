//! The production native learner pipelines run on guarded device buffers through the common primitive contracts.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const contract = @import("core").contract;
const F = @import("qwen4").family;
const tf = @import("core");
const rig_ = @import("contract_rig.zig");
fn Group(comptime spec: type) type {
    const K = F.train_rows_cuda.Adapter(cuda, spec);
    const Backend = struct {
        pub const Rig = rig_.Rig;
        pub const probability_cut = true;
        kernels: *K,
        views: [11]K.View = undefined,
        pub fn bind(b: *@This(), r: *Rig, f: anytype) !void {
            const needs = try spec.needs(f.raw.op, f.raw.p);
            for (&b.views, needs.bytes, 0..) |*view, size, i| {
                const ptr = if (i < 8) try r.input(f.raw.data[i][0..size]) else try r.output(size);
                view.* = .{ .buffer = .{ .d = r.gpu.d, .ptr = ptr, .len = size }, .len = size };
            }
        }
        pub fn launch(b: *@This(), r: *Rig, f: anytype) !void {
            for (b.views[8..], 8..) |view, i| {
                if (spec == tf.living_gemma) {
                    @memset(f.raw.data[i][0..view.len], 0x7b);
                    try view.buffer.upload(0, f.raw.data[i][0..view.len]);
                } else try r.zero(view.buffer.ptr, view.len);
            }
            try b.kernels.encode(r.s, f.raw.op, f.raw.p, b.views);
            try r.s.synchronize();
            for (b.views[8..], 8..) |view, i| @memcpy(f.raw.data[i][0..view.len], try r.bytes(view.buffer.ptr, view.len));
        }
    };
    const Case = F.learning_contract.Case(Backend, spec);
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        kernels: [5]?K = @splat(null),
        cases: [spec.names.len * 10]Case = undefined,
        made: usize = 0,
        pub fn create(a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, image: []const u8, out: *std.ArrayList(contract.gates.Kernel), only: ?[]const u8) !*Self {
            const group = try a.create(Self);
            errdefer a.destroy(group);
            group.* = .{ .a = a };
            errdefer group.cleanup();
            for (std.enums.values(tf.living_rows.Control)) |control| for (std.enums.values(spec.Op), spec.names) |op, name| for (0..2) |variant| {
                if (variant == 1 and !(spec == tf.living_gdn and op == .readout) and !(spec == tf.living_rows and std.mem.startsWith(u8, name, "tf_lw_pocket_")) and spec != tf.living_qsa) continue;
                if (spec == tf.living_gemma and op == .zero and control == .nan) continue;
                if (control == .sums and !F.learning_contract_data.reduces(spec, op)) continue;
                const label = try std.fmt.allocPrint(a, "cuda living-native {s}{s}{s}", .{ name, tf.living_rows.suffixes[@backingInt(control)], if (variant == 1) (if (spec == tf.living_gdn) " sigmoid" else if (spec == tf.living_rows) " geglu" else " gemma") else "" });
                if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) {
                    a.free(label);
                    continue;
                };
                errdefer a.free(label);
                const index = @backingInt(control);
                if (group.kernels[index] == null) group.kernels[index] = try K.loadTest(gpu.d, image, control);
                const c = &group.cases[group.made];
                c.* = .{ .a = a, .backend = .{ .kernels = &group.kernels[index].? }, .rig = .{ .base = .{ .gpu = gpu, .s = stream } }, .op = op, .control = control, .sigmoid = variant == 1 and spec == tf.living_gdn, .geglu = variant == 1 and spec == tf.living_rows, .gemma_attention = variant == 1 and spec == tf.living_qsa, .label = label };
                if (spec == tf.living_qsa and variant == 1) c.useGemmaAttention();
                try out.append(a, c.kernel());
                group.made += 1;
            };
            return group;
        }
        fn cleanup(g: *Self) void {
            for (g.cases[0..g.made]) |*c| {
                g.a.free(c.label);
                c.deinit();
            }
            for (&g.kernels) |*k| if (k.*) |*value| value.deinitDrained();
        }
        pub fn destroy(g: *Self) void {
            g.cleanup();
            g.a.destroy(g);
        }
    };
}
pub const Rows = Group(tf.living_rows);
pub const Gdn = Group(tf.living_gdn);
pub const Qsa = Group(tf.living_qsa);
pub const Gemma = Group(tf.living_gemma);

pub const Sets = struct {
    rows: *Rows,
    gdn: *Gdn,
    qsa: *Qsa,
    gemma: *Gemma,
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, out: *std.ArrayList(contract.gates.Kernel)) !@This() {
        const rows = try Rows.create(a, gpu, s, cuda.kernels.living_rows, out, null);
        errdefer rows.destroy();
        const gdn = try Gdn.create(a, gpu, s, cuda.kernels.living_gdn, out, null);
        errdefer gdn.destroy();
        const qsa = try Qsa.create(a, gpu, s, cuda.kernels.living_qsa, out, null);
        errdefer qsa.destroy();
        return .{ .rows = rows, .gdn = gdn, .qsa = qsa, .gemma = try Gemma.create(a, gpu, s, cuda.kernels.living_gemma, out, null) };
    }
    pub fn destroy(s: @This()) void {
        s.gemma.destroy();
        s.qsa.destroy();
        s.gdn.destroy();
        s.rows.destroy();
    }
};
