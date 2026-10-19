//! A persistence failure keeps accepted live lessons; only an uncommitted lesson follows the rejection path.
const learner = @import("learner.zig");
const whole = @import("whole.zig");
pub fn failed(l: anytype, operation: anytype, e: anyerror, comptime sites: bool, comptime pocket: bool) learner.Step {
    if (sites and operation == .steps) l.trainer.?.sites.restore();
    var stale = false;
    if (pocket and l.whole and l.pocketed and operation != .retire and operation != .save) {
        l.pockets.undo(l.trainer.?, whole.Hooks(@TypeOf(l.*).Host){ .b = l.b, .gpa = l.gpa, .io = l.io }) catch {
            stale = true;
        };
    }
    if (pocket and operation == .save and l.method == .pocket) l.pockets.save_pending = true;
    l.phase = .idle;
    return .{ .done = true, .changed = true, .stale = stale, .report = .{ .failed = @errorName(e) } };
}
