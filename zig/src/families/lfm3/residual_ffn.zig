//! Development-only residual-weight FFN tiles; broader numerical qualification remains open.
const std = @import("std");
const mtl = @import("metal");
const row = @import("../../core/row_projection.zig");
const pairs = @import("row_projection.zig");
const Args = extern struct { k: i32, n: i32, rows: i32, one: f32 = 1 };
const sources = @import("kernel_sources");
const source = sources.core_row_projection ++ "\n" ++ sources.nax ++ "\n" ++ sources.lfm3_residual_ffn;

pub const Candidate = struct {
    pipes: ?[2]mtl.Pipeline = null,
    pub fn init(device: mtl.Device) !Candidate {
        if (!device.tensorUnits()) return .{};
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var out: [2]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (out[0..made]) |p| p.deinit();
        for ([_][:0]const u8{ "lfm3_nax_prompt_q4", "lfm3_nax_nf_repair_q4" }, 0..) |name, i| {
            out[i] = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
            if (out[i].simdWidth() != 32 or out[i].maxThreads() < (if (i == 0) @as(usize, 128) else 64)) return error.Reflection;
            if (i == 0 and mtl.objc.msg(usize, out[i].id, "staticThreadgroupMemoryLength", .{}) != 18448) return error.Reflection;
        }
        return .{ .pipes = out };
    }
    pub fn deinit(self: *Candidate) void {
        if (self.pipes) |ps| for (ps) |p| p.deinit();
        self.* = .{};
    }
    pub fn encode(self: *Candidate, e: mtl.ComputeEncoder, fallback: *const pairs.Pipelines, w: row.Weights, x: mtl.Buffer, x_off: usize, scratch: mtl.Buffer, y: mtl.Buffer, y_off: usize, rows: usize) !void {
        try w.validate();
        const eligible = supports(w, rows);
        if (eligible) if (self.pipes) |ps| {
            const flags = ((rows + 63) / 64) * (w.n / 64) * 4;
            if (scratch.length() < flags) return error.MissingTileFlags;
            const args = Args{ .k = @intCast(w.k), .n = @intCast(w.n), .rows = @intCast(rows) };
            for (ps, 0..) |p, i| {
                e.setPipeline(p);
                e.setBuffer(x, x_off, 0);
                e.setBuffer(w.w, w.w_off, 1);
                e.setBuffer(w.scales, w.s_off, 2);
                e.setBuffer(w.biases, w.b_off, 3);
                e.setValue(args, 4);
                e.setBuffer(y, y_off, 5);
                e.setBuffer(scratch, 0, 6);
                if (i == 0) {
                    e.setBuffer(scratch, 0, 7);
                    e.dispatchGroups(.of(w.n / 64, (rows + 63) / 64, 1), .of(128, 1, 1));
                } else e.dispatchGroups(.of((rows + 63) / 64, w.n / 64, 1), .of(64, 1, 1));
                e.barrier();
            }
            return;
        };
        try pairs.encode(e, fallback, w, x, x_off, y, y_off, rows);
    }
};

pub fn supports(w: row.Weights, rows: usize) bool {
    return rows > 32 and rows <= @import("lfm3_options").prompt_rows and w.bits == 4 and w.group == 64 and w.sum == .f32 and w.n % 64 == 0 and w.k % 64 == 0 and w.k <= 12288;
}
