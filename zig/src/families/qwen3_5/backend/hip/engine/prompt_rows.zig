//! The rows a prompt pass takes: what the policy asks, the fallback that keeps a stream or a cache, a piece, a step.

const std = @import("std");
const hip = @import("hip");
const prefix = @import("prefix.zig");
const fwd = @import("../forward/forward.zig");

/// The rows a policy asks of a prompt pass: its step, no less than the rows a chunk takes while other streams decode.
pub fn asked(policy: hip.Policy) usize {
    return @max(policy.prefill_step, policy.fill_rows);
}

/// The rows an engine opens with: the policy's, or the chunk for a bf16 checkpoint (rocBLAS bits follow the rows).
pub fn opening(policy: hip.Policy, plain: bool) usize {
    return if (plain) prefix.chunk else asked(policy);
}

/// The smaller rows admission starts from: the plain step behind a GPU's larger default, if below `want` (null: none).
pub fn fallback(policy: hip.Policy, plain_step: ?u32, want: usize) ?usize {
    const rows: usize = @max(plain_step orelse return null, policy.fill_rows);
    return if (rows < want) rows else null;
}

/// The rows a server took; the rows its policy asked where the fallback's replaced them, and whether for the cache.
pub const Opened = struct {
    rows: usize,
    narrowed: ?usize = null,
    cache: bool = false,

    /// What the startup line says after the rows taken: the rows asked for and what they would cost, if they gave way.
    pub fn note(o: Opened, buf: []u8) []const u8 {
        const n = o.narrowed orelse return "";
        const cost = if (o.cache) "leave less than the prompt cache" else "admit fewer of the streams";
        return std.fmt.bufPrint(buf, " ({d}-row chunks would {s} asked for)", .{ n, cost }) catch "";
    }
};

/// Bytes a prompt cache claims beside the streams: what `--prompt-cache-gib` gave, or 0 when it may pass the cap.
pub fn claim(flag: ?u64, over_cap: bool) u64 {
    return if (over_cap) 0 else flag orelse 0;
}

/// `want` rows, or `small` when the larger scratch costs a stream `small`'s admits (`beside` bytes) or the `cache`.
pub fn settle(want: usize, small: usize, room: u64, cache: u64, larger: u64, smaller: u64, beside: u64) Opened {
    // the admitted streams count as one: the larger scratch holds them all, or costs one the server asked for
    if (hip.policy.narrows(room, 0, larger, smaller, beside, 1)) return .{ .rows = small, .narrowed = want };
    // a cache a flag asked for is sized after the rows from what they leave: it rides as `narrows`'s slots
    const lost = hip.policy.narrows(room, cache, larger, smaller, beside, 1);
    return if (lost) .{ .rows = small, .narrowed = want, .cache = true } else .{ .rows = want };
}

/// The admitted streams' device bytes at some prompt rows: the whole plan, and the share the prompt scratch takes.
pub const Plan = struct { whole: u64, prompt: u64 };

/// `settle` over the streams' plans at `small` and `want` rows (`plans.at(rows)`), with all that grows with the rows.
pub fn settled(plans: anytype, want: usize, small: usize, room: u64, cache: u64) !Opened {
    const at_small: Plan = try plans.at(small);
    const beside = at_small.whole - at_small.prompt;
    // what grows with the rows beside the prompt scratch (a drafter's tap rings) rides with the larger scratch
    const larger = (try plans.at(want)).whole - beside;
    return settle(want, small, room, cache, larger, at_small.prompt, beside);
}

/// Rows a plain prompt's piece takes when nothing decodes: the policy's, at most the `held` rows a pass takes.
pub fn idle(held: usize, policy: hip.Policy) usize {
    return @min(held, asked(policy));
}

/// Rows a pass takes through a layer at once: the policy's once they pass SPAN, at most the `held` rows of a pass.
pub fn step(held: usize, policy: hip.Policy) usize {
    return @min(held, @max(fwd.SPAN, asked(policy)));
}

const T = std.testing;

fn igpu() hip.Policy {
    return hip.Policy.forCaps(hip.Caps.of("gfx1151").?, 1);
}

test "a lone gfx1151 opens with its policy's 4,096 rows; gfx1100, a group and prefill_step=1024 keep 1,024" {
    try T.expectEqual(@as(usize, 4096), opening(igpu(), false));
    try T.expectEqual(@as(usize, 1024), opening(hip.Policy.forCaps(hip.Caps.of("gfx1100").?, 1), false));
    try T.expectEqual(@as(usize, 1024), opening(hip.Policy.forCaps(hip.Caps.of("gfx1151").?, 2), false));
    var notes: hip.Policy.Notes = .{};
    const flagged = try hip.Policy.resolveFrom(igpu(), "prefill_step=1024", .none, &notes);
    try T.expectEqual(@as(usize, 1024), opening(flagged, false));
    try T.expectEqual(@as(usize, prefix.chunk), opening(igpu(), true)); // a bf16 checkpoint keeps the chunk
}

test "admission has smaller rows to start from only behind a GPU's larger default step" {
    try T.expectEqual(@as(?usize, 1024), fallback(igpu(), 1024, 4096));
    try T.expectEqual(@as(?usize, null), fallback(igpu(), null, 4096)); // a group, or a step a flag asked for
    try T.expectEqual(@as(?usize, null), fallback(igpu(), 1024, prefix.chunk)); // a bf16 checkpoint's chunk
}

test "the policy's rows give way only when their scratch costs a stream the server asked for" {
    const gib: u64 = 1 << 30;
    // the streams fit beside either scratch, the second time to the byte: the policy's rows stay
    try T.expectEqual(Opened{ .rows = 4096 }, settle(4096, 1024, 40 * gib, 0, 5 * gib, 1 * gib, 20 * gib));
    try T.expectEqual(Opened{ .rows = 4096 }, settle(4096, 1024, 25 * gib, 0, 5 * gib, 1 * gib, 20 * gib));
    // they fit only beside the smaller scratch: the fallback's rows, and the rows they replaced for the startup line
    try T.expectEqual(Opened{ .rows = 1024, .narrowed = 4096 }, settle(4096, 1024, 25 * gib - 1, 0, 5 * gib, 1 * gib, 20 * gib));
    try T.expectEqual(Opened{ .rows = 1024, .narrowed = 4096 }, settle(4096, 1024, 21 * gib, 0, 30 * gib, 1 * gib, 20 * gib));
    // a cache that fits beside neither scratch changes none of it: the plan it is refused from must fit the room
    try T.expectEqual(Opened{ .rows = 1024, .narrowed = 4096 }, settle(4096, 1024, 21 * gib, 9 * gib, 30 * gib, 1 * gib, 20 * gib));
}

test "a prompt cache a flag asked for that fits beside the fallback's scratch alone takes the fallback's rows" {
    const gib: u64 = 1 << 30;
    const for_cache: Opened = .{ .rows = 1024, .narrowed = 4096, .cache = true };
    // 40 GiB of room and 20 of streams: the policy's scratch leaves 15 GiB, the fallback's 19; both ends of the band
    try T.expectEqual(for_cache, settle(4096, 1024, 40 * gib, claim(15 * gib + 1, false), 5 * gib, 1 * gib, 20 * gib));
    try T.expectEqual(for_cache, settle(4096, 1024, 40 * gib, claim(19 * gib, false), 5 * gib, 1 * gib, 20 * gib));
}

test "with no prompt cache asked for, the policy's rows stay wherever their scratch holds the streams" {
    const gib: u64 = 1 << 30;
    try T.expectEqual(Opened{ .rows = 4096 }, settle(4096, 1024, 40 * gib, claim(null, false), 5 * gib, 1 * gib, 20 * gib));
}

test "a prompt cache that fits beside the policy's scratch keeps the policy's rows" {
    const gib: u64 = 1 << 30;
    try T.expectEqual(Opened{ .rows = 4096 }, settle(4096, 1024, 40 * gib, claim(15 * gib, false), 5 * gib, 1 * gib, 20 * gib));
}

test "a prompt cache that may pass the cap weighs nothing: the policy's rows stay" {
    const gib: u64 = 1 << 30;
    try T.expectEqual(Opened{ .rows = 4096 }, settle(4096, 1024, 40 * gib, claim(16 * gib, true), 5 * gib, 1 * gib, 20 * gib));
}

/// The engine's plan for four streams in small: a MiB a prompt row, 4 GiB a stream, and a drafter's bytes beside each.
const Drafted = struct {
    drafter: bool,

    pub fn at(p: Drafted, rows: usize) !Plan {
        const mib: u64 = 1 << 20;
        const r: u64 = rows;
        // a 256 MiB context, and a tap ring of 64 KiB a row: the drafter's 2,048-row window past the rows a pass writes
        const drafts: u64 = if (p.drafter) 256 * mib + (2048 + r) * 64 * 1024 else 0;
        return .{ .whole = r * mib + 4 * (4096 * mib + drafts), .prompt = r * mib };
    }
};

test "a drafter's contexts and tap rings count with a prompt cache's claim, the larger rows' longer rings too" {
    const gib: u64 = 1 << 30;
    const drafted: Drafted = .{ .drafter = true };
    const alone: Drafted = .{ .drafter = false };
    // 30 GiB and an 8 GiB cache: the drafter's 2.5 GiB at 4,096 rows leave too little for it, its 1.75 at 1,024 do not
    const for_cache: Opened = .{ .rows = 1024, .narrowed = 4096, .cache = true };
    try T.expectEqual(for_cache, try settled(drafted, 4096, 1024, 30 * gib, claim(8 * gib, false)));
    try T.expectEqual(Opened{ .rows = 4096 }, try settled(alone, 4096, 1024, 30 * gib, claim(8 * gib, false)));
    // 22 GiB and no cache: the 22.5 GiB plan at 4,096 rows costs a stream, the 18.75 GiB at 1,024 holds all four
    const for_streams: Opened = .{ .rows = 1024, .narrowed = 4096 };
    try T.expectEqual(for_streams, try settled(drafted, 4096, 1024, 22 * gib, claim(null, false)));
    try T.expectEqual(Opened{ .rows = 4096 }, try settled(alone, 4096, 1024, 22 * gib, claim(null, false)));
    // the startup line names the rows that gave way and why
    var why: [96]u8 = undefined;
    try T.expectEqualStrings(" (4096-row chunks would leave less than the prompt cache asked for)", for_cache.note(&why));
    try T.expectEqualStrings(" (4096-row chunks would admit fewer of the streams asked for)", for_streams.note(&why));
}

test "a pass crosses a layer in the policy's rows once they pass the span, never in more rows than it holds" {
    try T.expectEqual(@as(usize, 4096), step(4096, igpu())); // a lone gfx1151's chunk is one step a layer
    try T.expectEqual(@as(usize, 1024), step(1024, igpu())); // narrowed to the fallback's rows
    try T.expectEqual(@as(usize, 1024), step(1024, .{}));
    try T.expectEqual(@as(usize, fwd.SPAN), step(9000, .{})); // a check's longest prompt keeps the span
}
