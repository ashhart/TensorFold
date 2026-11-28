//! ModelOpt's expert-first router, with the standard router owning every fallback launch.
const cuda = @import("root.zig");
pub const Fns = struct {
    base: cuda.moe.Fns,
    router_expert: ?cuda.Function = null,
    pub fn resolve(mod: cuda.Module) !Fns {
        return resolveMode(mod, try @import("sys").environment.read(bool, "TF_CUDA_MOE_EXPERT_FIRST", true));
    }
    pub fn expertFirst(value: ?[]const u8) !bool {
        return (try @import("sys").environment.parse("TF_CUDA_MOE_EXPERT_FIRST", value) orelse 1) != 0;
    }
    pub fn resolveMode(mod: cuda.Module, on: bool) !Fns {
        return .{ .base = try cuda.moe.Fns.resolve(mod), .router_expert = if (on) try mod.function("tf_moe_router_expert_first") else null };
    }
    pub fn route(f: Fns, s: cuda.Stream, x: u64, stride: usize, rows: usize, w: u64, logits: u64, dims: usize, count: usize, params: u64) !void {
        const kernel = f.router_expert orelse return f.base.route(s, x, stride, rows, w, logits, dims, count, params);
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_int, @intCast(stride)));
        a.add(w);
        a.add(logits);
        for ([_]usize{ rows, dims, count }) |n| a.add(@as(c_int, @intCast(n)));
        a.add(params);
        try cuda.launch.launch(kernel, .{ .grid = .{ .x = @intCast((count + 15) / 16), .y = @intCast((rows + 15) / 16) }, .block = .{ .x = 256 } }, s, &a);
    }
};
