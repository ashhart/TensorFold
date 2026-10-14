//! The speed lines of `check`: cold prefill and decode with drafts off and on; they fail only when a run does.

const std = @import("std");
const check_ctx = @import("check_ctx.zig");
const ids_file = @import("ids_file.zig");
const lanes_session = @import("lanes_session.zig");
const prefill_run = @import("prefill_run.zig");

const Ctx = check_ctx.Ctx;
const Session = lanes_session.Session;

/// The longest prompt `run` times: the engine's capacity must hold it.
pub const longest_prefill = 8192;

const prefill_lengths = [_]usize{ 13, 2048, longest_prefill };

pub fn run(c: *Ctx) void {
    prefill(c);
    for ([_]bool{ false, true }) |drafts| {
        decode(c, drafts, c.short[0..1], 256);
        decode(c, drafts, c.short, 128);
    }
}

fn prefill(c: *Ctx) void {
    if (c.tp()) return c.report.skip("speed prefill", "a prompt pass runs on one rank");
    const prompt = ids_file.synthetic(c.arena, longest_prefill) catch |err| return c.report.broke("speed prefill", err);
    for (prefill_lengths) |len| {
        const name = std.fmt.allocPrint(c.arena, "speed prefill {d} tokens", .{len}) catch return;
        const t = prefill_run.seconds(c.gpa, c.io, c.e, prompt[0..len]) catch |err| {
            c.report.broke(name, err);
            continue;
        };
        c.report.pass(name, "{d:.3} s, {d:.0} tok/s", .{ t, @as(f64, @floatFromInt(len)) / t });
    }
}

/// `prompts` generating `max_new` tokens each, the last of three runs (the first two capture the graphs).
fn decode(c: *Ctx, drafts: bool, prompts: []const ids_file.Prompt, max_new: u32) void {
    const name = std.fmt.allocPrint(c.arena, "speed decode {d} stream{s}, drafts {s}", .{ prompts.len, if (prompts.len == 1) "" else "s", if (drafts) "on" else "off" }) catch return;
    const job: Session.Job = .{ .prompts = prompts, .max_new = max_new, .drafts = drafts };
    for (0..2) |_| _ = c.session.run(c.arena, job) catch |err| return c.report.broke(name, err);
    const done = c.session.run(c.arena, job) catch |err| return c.report.broke(name, err);
    var rounds: u64 = 0;
    for (done.replies) |r| rounds += r.rounds;
    const n = done.tokens();
    const shared: f64 = @floatFromInt(@max(done.rounds, 1));
    c.report.pass(name, "{d:.1} tok/s, {d} tokens in {d:.3} s, {d:.2} tokens a stream round, {d:.1}% of {d} rounds replayed, {d:.0} us a round submitting, {d:.0} us a round in all (backend, a call: verify {d:.0} us, keep {d:.0} us, draft {d:.0} us)", .{
        @as(f64, @floatFromInt(n)) / done.seconds,
        n,
        done.seconds,
        @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(@max(rounds, 1))),
        100.0 * @as(f64, @floatFromInt(done.replayed)) / shared,
        done.rounds,
        @as(f64, @floatFromInt(done.submit_ns)) / shared / 1e3,
        done.seconds * 1e6 / shared,
        each(done, 0),
        each(done, 1),
        each(done, 2),
    });
}

/// Microseconds a backend call of kind `which` (verify, keep, draft) took on average.
fn each(d: Session.Done, which: usize) f64 {
    return @as(f64, @floatFromInt(d.spent[which])) / @as(f64, @floatFromInt(@max(d.calls[which], 1))) / 1e3;
}
