//! The accuracy checks of `check`: every row's logits, from a cold prefill and from decode rounds, against fp64 truth.

const std = @import("std");
const check_ctx = @import("check_ctx.zig");
const logits_run = @import("logits_run.zig");
const truth_score = @import("truth_score.zig");

const Ctx = check_ctx.Ctx;

/// What a score must stay within.
pub const Bar = struct { kl_mean: f64 = 0.01, top1: f64 = 90.0, ppl_ratio: f64 = 1.02 };

pub fn run(c: *Ctx, truth: []const u8, bar: Bar) void {
    if (c.tp()) return c.report.skip("accuracy", "the logits run on one rank");
    const mode0: logits_run.Mode = .{ .wide = true };
    var scorer = truth_score.Scorer.init(c.gpa, c.io, truth, c.ids, logits_run.kindOf(c.e, mode0), logits_run.vocab(c.e)) catch |err| return c.report.broke("accuracy", err);
    defer scorer.deinit();
    for ([_]bool{ false, true }) |decode| {
        const name = if (decode) "accuracy decode" else "accuracy prefill";
        scorer.reset();
        logits_run.stream(c.gpa, c.e, c.ids, .{ .wide = true, .decode = decode }, .{ .ctx = &scorer, .put = truth_score.Scorer.put }) catch |err| {
            c.report.broke(name, err);
            continue;
        };
        const s = scorer.result() catch |err| {
            c.report.broke(name, err);
            continue;
        };
        const ok = s.kl_mean <= bar.kl_mean and s.top1 >= bar.top1 and s.ppl <= s.ppl_truth * bar.ppl_ratio;
        const line = "rows={d} KL mean={e:.3} max={e:.3} top1={d:.2}% max|dlogit|={d:.4} ppl truth={d:.4} cand={d:.4}";
        const args = .{ s.rows, s.kl_mean, s.kl_max, s.top1, s.max_abs, s.ppl_truth, s.ppl };
        if (ok) c.report.pass(name, line, args) else c.report.fail(name, line ++ " (bar: KL mean <= {e:.1}, top1 >= {d:.0}%, ppl within {d:.0}%)", args ++ .{ bar.kl_mean, bar.top1, (bar.ppl_ratio - 1) * 100 });
    }
}
