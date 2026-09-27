//! Queue the next one-token forward before reading the current sampled token.
//! Only an input already emitted to the caller is committed. A queued EOS suffix
//! is drained and discarded, preserving the synchronous driver's cache position.
const std = @import("std");
const mx = @import("mlx.zig");
const Sampling = @import("sampling.zig").Sampling;
const Hash = std.crypto.hash.sha2.Sha256;

pub const Result = struct { rounds: usize = 0, queued_ahead: usize = 0 };

pub fn generate(comptime M: type, m: *M, a: std.mem.Allocator, generated: *std.ArrayList(u32), limit: usize, settings: Sampling, comptime eos: fn (i32) bool, hash: ?*Hash) !Result {
    if (!settings.metal or generated.items.len != @min(limit, 1)) return error.InvalidSerialPipeline;
    if (generated.items.len == limit or eos(@intCast(generated.items[0]))) return .{};
    const Step = struct {
        pass: M.SerialPass,
        draw: mx.Array,
        fn prepare(model: *M, token: mx.Array, sample: Sampling) !@This() {
            var pass = try model.forwardSerialArray(token);
            errdefer pass.deinit();
            const draw = try @import("gpu_sampling.zig").sample(&model.kernels, &pass.scope, pass.logits, &.{model.position + 1}, sample, null);
            try mx.evalMany(&.{draw}, true);
            return .{ .pass = pass, .draw = draw };
        }
        fn deinit(step: *@This()) void {
            step.pass.deinit();
        }
    };
    var initial_scope = mx.Scope{};
    defer initial_scope.deinit();
    var pending: i32 = @intCast(generated.items[0]);
    var current: ?Step = try Step.prepare(m, try initial_scope.ints(&.{pending}), settings);
    defer if (current) |*step| step.deinit();
    var result = Result{};
    while (current) |step| {
        var landed = step;
        current = null;
        defer landed.deinit();
        try m.commitSerialQueued(&landed.pass);
        if (generated.items.len + 1 < limit) {
            current = try Step.prepare(m, landed.draw, settings);
            result.queued_ahead += 1;
        }
        // Everything above accepts a GPU array; this is the first token read.
        try mx.eval(landed.draw);
        const token = mx.c.mlx_array_data_uint32(landed.draw)[0];
        try generated.append(a, token);
        if (hash) |h| {
            h.update(&.{1});
            h.update(std.mem.asBytes(&pending));
        }
        result.rounds += 1;
        pending = @intCast(token);
        if (eos(pending)) {
            // Finish already-submitted work before returning timing/memory stats.
            // Its pass remains uncommitted and is freed by the owning defer.
            if (current) |queued| try mx.eval(queued.draw);
            break;
        }
    }
    // A final recurrent replay may have no subsequent forward depending on it.
    // Include its execution in decode time and leave a fully evaluated cache.
    var arrays: [128]mx.Array = undefined;
    var count: usize = 0;
    for (m.cache) |cache| inline for (.{ "a", "b" }) |field| {
        const value = @field(cache, field);
        if (value.ctx != null) {
            arrays[count] = value;
            count += 1;
        }
    };
    if (count > 0) try mx.evalMany(arrays[0..count], false);
    return result;
}
