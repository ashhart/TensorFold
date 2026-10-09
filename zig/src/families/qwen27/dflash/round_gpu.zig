//! Target picks and tree acceptance run on the GPU with one bounded result readback.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const contract = core.tree_round;
const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
pub const Matcher = struct {
    ops: core.tree_round_gpu.Ops,
    buffers: [5]mtl.Buffer,
    pub fn init(device: mtl.Device) !Matcher {
        const ops = try core.tree_round_gpu.Ops.init(device);
        errdefer ops.deinit();
        var buffers: [5]mtl.Buffer = undefined;
        var made: usize = 0;
        errdefer for (buffers[0..made]) |buffer| buffer.deinit();
        for (&buffers, [_]usize{ 16 * 4, 16 * 4, 16 * @sizeOf(contract.Pick), 16 * 4, @sizeOf(contract.Result) }) |*buffer, bytes| {
            buffer.* = try device.buffer(bytes, options);
            made += 1;
        }
        return .{ .ops = ops, .buffers = buffers };
    }
    pub fn deinit(m: Matcher) void {
        m.ops.deinit();
        for (m.buffers) |buffer| buffer.deinit();
    }
    pub fn encode(m: *Matcher, e: mtl.ComputeEncoder, logits: core.tree_round_gpu.Ref, ids: []const u32, parents: []const i32, vocab: u32, budget: u32, eos: []const u32) !contract.Match {
        if (ids.len == 0 or ids.len > 16 or parents.len != ids.len or eos.len > 16) return error.BadRoundShape;
        @memcpy(m.buffers[0].slice(u32, ids.len), ids);
        @memcpy(m.buffers[1].slice(i32, parents.len), parents);
        @memcpy(m.buffers[3].slice(u32, eos.len), eos);
        const shape = contract.Match{ .rows = @intCast(ids.len), .vocab = vocab, .budget = budget, .eos_count = @intCast(eos.len) };
        try m.ops.argmax(e, logits, .{ .buf = m.buffers[2] }, .{ .rows = shape.rows, .vocab = vocab, .stride = vocab });
        e.barrier();
        try m.ops.match(e, .{ .buf = m.buffers[0] }, .{ .buf = m.buffers[1] }, .{ .buf = m.buffers[2] }, if (eos.len == 0) null else .{ .buf = m.buffers[3] }, .{ .buf = m.buffers[4] }, shape);
        e.barrier();
        return shape;
    }
    pub fn read(m: *Matcher, shape: contract.Match) !contract.Decoded {
        const result: *const contract.Result = @ptrCast(@alignCast(m.buffers[4].contents()));
        const decoded = try contract.decode(result, shape);
        if (decoded.status != .ok) return error.InvalidTargetDraftRound;
        return decoded;
    }
};
