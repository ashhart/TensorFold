//! The MTP adapter that exposes its model state through the shared drafter contract.
const lanes = @import("lanes");
const dr = lanes.drafter;

const prior = [_]f64{ 0.7, 0.55, 0.45, 0.4, 0.35, 0.3, 0.3 };

pub fn Adapter(comptime Mtp: type) type {
    return struct {
        fn cast(p: *anyopaque) *Mtp {
            return @ptrCast(@alignCast(p));
        }

        pub fn taps(p: *anyopaque) []const u32 {
            return cast(p).taps();
        }

        pub fn facts(p: *anyopaque) dr.Facts {
            const d = cast(p);
            return .{ .limit_shared_drafts = true, .depth = d.depth, .step_ms = d.step_ms, .prior = &prior, .plain_guard = true, .batched = !d.trees, .draft_kv = .{ .bytes = d.ringBytes(), .dtype = .bf16 }, .stream_bytes = d.stream_bytes, .trees = d.trees, .window_costs = d.costs[0..d.n_costs], .lane_costs = d.costs[0..d.n_costs] };
        }
    };
}
