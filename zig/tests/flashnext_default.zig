const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const tf = @import("tensorfold");
const fz = tf.flashnext_replay;

const opts = fz.opts;
const D = fz.D;
const WIDE = fz.WIDE;
const LAYERS = fz.LAYERS;
const VOCAB = fz.VOCAB;
const CAP = fz.CAP;
const PLE_TAIL = fz.PLE_TAIL;
const GROUPS = fz.GROUPS;
const MAXR = fz.MAXR;
const CS_ROW = fz.CS_ROW;
const SO_ROW = fz.SO_ROW;
const Buf = fz.Buf;
const Entry = fz.Entry;
const Variant = fz.Variant;
const Site = fz.Site;
const readAll = fz.readAll;
const Run = fz.Run;
const TOP = fz.TOP;
const KW = fz.KW;
const Select = fz.Select;
const GSelect = fz.GSelect;
const Hc = fz.Hc;
const Layer = fz.Layer;
const hcOf = fz.hcOf;
const laneOf = fz.laneOf;
const Ple = fz.Ple;
const Tmp = fz.Tmp;
const f32Buf = fz.f32Buf;
const i32Buf = fz.i32Buf;
const Mtp = fz.Mtp;
const Model = fz.Model;
const DepthRule = fz.DepthRule;
const copyDrafts = fz.copyDrafts;
const PMAX = fz.PMAX;
const Prompt = fz.Prompt;
const jsonInt = fz.jsonInt;

pub fn runDefault(r: *Run, m: *Model, ref: std.json.Value, gpa: std.mem.Allocator, args: anytype, own_prompt: bool) !void {
    const prompt = ref.object.get("prompt").?.array.items;
    const want = ref.object.get("tokens").?.array.items;
    var pick: [MAXR]u32 = undefined;

    // FZ_LONG: prompts of any length through the prompt chunks (reference and GPU-side rounds alike)
    const long_mode = platform.c.getenv("FZ_LONG") != null;
    if (platform.c.getenv("FZ_PRE1") != null) try m.window(&.{1000}, &pick); // a decode window before the prompt chunks
    var pr_long: ?Prompt = if (long_mode) try Prompt.init(r, args[2], r.xnew_header) else null;
    if (pr_long) |*pr| if (platform.c.getenv("FZ_STEP")) |v| {
        pr.step = try std.fmt.parseInt(usize, std.mem.span(v), 10);
    };
    const ptoks = try gpa.alloc(u32, prompt.len);
    defer gpa.free(ptoks);
    for (prompt, 0..) |x, i| ptoks[i] = @intCast(x.integer);

    // 1. one-row greedy steps against the Python engine's tokens
    m.reset();
    if (pr_long) |*pr| {
        const p0 = mtl.clock.seconds();
        var at: usize = 0;
        while (at < ptoks.len) {
            const n = @min(pr.step, ptoks.len - at);
            pick[0] = try pr.chunk(m, gpa, ptoks[at .. at + n]);
            at += n;
        }
        const ps = mtl.clock.seconds() - p0;
        std.debug.print("prompt {d} tokens in prompt chunks: {d:.2} s ({d:.0} tok/s)\n", .{ ptoks.len, ps, @as(f64, @floatFromInt(ptoks.len)) / ps });
    } else for (prompt) |tok| {
        try m.window(&.{@intCast(tok.integer)}, &pick);
        m.keepRows(&.{@intCast(tok.integer)}, 1);
    }
    var got: std.ArrayList(u32) = .empty;
    try got.append(gpa, pick[0]);
    const t3 = mtl.clock.seconds();
    m.gpu_seconds = 0;
    while (got.items.len < want.len) {
        const last = got.items[got.items.len - 1];
        try m.window(&.{last}, &pick);
        m.keepRows(&.{last}, 1);
        try got.append(gpa, pick[0]);
    }
    const t4 = mtl.clock.seconds();
    var same: usize = 0;
    while (same < want.len and got.items[same] == @as(u32, @intCast(want[same].integer))) same += 1;
    const steps: f64 = @floatFromInt(want.len - 1);
    std.debug.print("one-row reference starts {any}\n", .{got.items[0..@min(24, got.items.len)]});
    std.debug.print("one row: {d}/{d} tokens equal to Python's; {d:.1} tok/s ({d:.2} ms a step, GPU {d:.2} ms)\n", .{ same, want.len, steps / (t4 - t3), (t4 - t3) * 1e3 / steps, m.gpu_seconds * 1e3 / steps });
    if (platform.c.getenv("FZ_ONLY1") != null) {
        if (same < want.len) std.debug.print("first difference at token {d}: got {d}, Python {d}\n", .{ same, got.items[same], want[same].integer });
        return;
    }

    // 2. the Python engine's drafted windows, every row's pick, with rollback
    m.reset();
    if (!long_mode and platform.c.getenv("FZ_ONLY17") == null) for (prompt) |tok| {
        try m.window(&.{@intCast(tok.integer)}, &pick);
        m.keepRows(&.{@intCast(tok.integer)}, 1);
    };
    var bad: usize = 0;
    var total_rows: usize = 0;
    const skip_mid = long_mode or platform.c.getenv("FZ_ONLY17") != null; // the one-row reference, then GPU-side rounds only
    const rounds = if (own_prompt or skip_mid) &[_]std.json.Value{} else ref.object.get("rounds").?.array.items;
    for (rounds, 0..) |round, ri| {
        const o = round.object;
        const win = o.get("window").?.array.items;
        var tokens: [MAXR]u32 = undefined;
        for (win, 0..) |x, i| tokens[i] = @intCast(x.integer);
        try m.window(tokens[0..win.len], &pick);
        const exp = o.get("picks").?.array.items;
        for (exp, 0..) |x, i| {
            total_rows += 1;
            if (pick[i] != @as(u32, @intCast(x.integer))) {
                bad += 1;
                if (bad <= 5) std.debug.print("round {d} ({d} rows) row {d}: got {d}, Python {d}\n", .{ ri, win.len, i, pick[i], x.integer });
            }
        }
        m.keepRows(tokens[0..win.len], @intCast(o.get("keep").?.integer));
    }
    std.debug.print("drafted windows: {d} rounds, {d}/{d} rows equal to Python's\n", .{ rounds.len, total_rows - bad, total_rows });

    const ref_tokens: []const u32 = got.items;
    // 4. the MTP head against the Python engine's drafts (absorb windows of 1-8 rows, then a chain)
    if (!own_prompt and !skip_mid) {
        const mref = ref.object.get("mtp").?.object;
        var seq: std.ArrayList(u32) = .empty;
        for (prompt) |x| try seq.append(gpa, @intCast(x.integer));
        for (want[0..16]) |x| try seq.append(gpa, @intCast(x.integer));
        const all = try r.buffer(seq.items.len * WIDE * 2);
        m.reset();
        for (seq.items, 0..) |tok, i| {
            try m.window(&.{tok}, &pick);
            m.keepRows(&.{tok}, 1);
            @memcpy(all.contents()[i * WIDE * 2 .. (i + 1) * WIDE * 2], m.last.b.contents()[0 .. WIDE * 2]);
        }
        m.mtp.pos = 0;
        m.mtp.drafted = 0;
        var ok: usize = 0;
        var n: usize = 0;
        var d: u32 = 0;
        for (mref.get("absorb").?.array.items) |a| {
            const start: usize = @intCast(a.object.get("start").?.integer);
            const nx = a.object.get("next").?.array.items;
            var nexts: [MAXR]u32 = undefined;
            for (nx, 0..) |x, i| nexts[i] = @intCast(x.integer);
            d = try m.mtpRun(nexts[0..nx.len], .{ .b = all, .off = start * WIDE * 2 });
            n += 1;
            if (d == @as(u32, @intCast(a.object.get("draft").?.integer))) ok += 1 else std.debug.print("absorb of {d} rows: draft {d}, Python {d}\n", .{ nx.len, d, a.object.get("draft").?.integer });
        }
        for (mref.get("chain").?.array.items) |x| {
            d = try m.mtpChain(d);
            n += 1;
            if (d == @as(u32, @intCast(x.integer))) ok += 1 else std.debug.print("chained draft {d}, Python {d}\n", .{ d, x.integer });
        }
        std.debug.print("MTP drafts equal to Python's: {d}/{d}\n", .{ ok, n });
    }

    // 5. one stream with MTP chains of a fixed depth: tokens against the one-row reference, lanes landed, tok/s
    const depths5: []const usize = if (own_prompt or skip_mid) &.{} else &.{ 1, 2, 3, 4, 6 };
    for (depths5) |depth| {
        m.reset();
        m.mtp.pos = 0;
        m.mtp.drafted = 0;
        const all = try r.buffer(prompt.len * WIDE * 2);
        for (prompt, 0..) |x, i| {
            const tok: u32 = @intCast(x.integer);
            try m.window(&.{tok}, &pick);
            m.keepRows(&.{tok}, 1);
            @memcpy(all.contents()[i * WIDE * 2 .. (i + 1) * WIDE * 2], m.last.b.contents()[0 .. WIDE * 2]);
        }
        var out: std.ArrayList(u32) = .empty;
        try out.append(gpa, pick[0]);
        const s0 = mtl.clock.seconds();
        var nexts: std.ArrayList(u32) = .empty;
        try nexts.appendSlice(gpa, ptoks[1..]);
        try nexts.append(gpa, pick[0]);
        var drafts: [MAXR]u32 = undefined;
        drafts[0] = try m.mtpAbsorb(nexts.items, .{ .b = all });
        for (1..depth) |j| drafts[j] = try m.mtpChain(drafts[j - 1]);
        var n_rounds: usize = 0;
        var landed: usize = 0;
        while (out.items.len < want.len) {
            var win: [MAXR]u32 = undefined;
            win[0] = out.items[out.items.len - 1];
            @memcpy(win[1 .. depth + 1], drafts[0..depth]);
            try m.window(win[0 .. depth + 1], &pick);
            var keep: usize = 1;
            while (keep <= depth and win[keep] == pick[keep - 1]) keep += 1;
            try out.appendSlice(gpa, pick[0..keep]);
            n_rounds += 1;
            landed += keep - 1;
            m.keepRows(win[0 .. depth + 1], keep);
            drafts[0] = try m.mtpAbsorb(pick[0..keep], m.last);
            for (1..depth) |j| drafts[j] = try m.mtpChain(drafts[j - 1]);
        }
        const wall = mtl.clock.seconds() - s0;
        var eq: usize = 0;
        while (eq < want.len and out.items[eq] == (if (r.xnew) ref_tokens[eq] else @as(u32, @intCast(want[eq].integer)))) eq += 1;
        std.debug.print("depth {d}: {d}/{d} tokens equal; {d} rounds, {d:.2} tokens a round, {d:.2} of {d} drafts landing; {d:.1} tok/s\n", .{ depth, eq, want.len, n_rounds, @as(f64, @floatFromInt(out.items.len - 1)) / @as(f64, @floatFromInt(n_rounds)), @as(f64, @floatFromInt(landed)) / @as(f64, @floatFromInt(n_rounds)), depth, @as(f64, @floatFromInt(out.items.len - 1)) / wall });
        if (eq < want.len) bad += 1;
    }

    // 6. one command buffer a round: head chains drafts on GPU, window hashes n-grams, host reads picks once
    const wids: Buf = .{ .b = try r.buffer(64) };
    const adapt_story = [_][3]usize{ .{ 3, 3, 0 }, .{ 0, 7, 9 } };
    const adapt_own = [_][3]usize{ .{ 3, 3, 0 }, .{ 6, 6, 0 }, .{ 0, 7, 9 } };
    const adapt: []const [3]usize = if (own_prompt) &adapt_own else &adapt_story;
    const copy_min: usize = if (platform.c.getenv("FZ_COPY_MIN")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 3;
    var hist: std.ArrayList(u32) = .empty;
    for (prompt) |x| try hist.append(gpa, @intCast(x.integer));
    const n_prompt = hist.items.len;
    for (0..if (skip_mid) 0 else if (r.copy) 2 else adapt.len) |run6| {
        const cfg_a = if (r.copy) adapt[0] else adapt[run6];
        const use_copy = r.copy and run6 == 1;
        var copy_rounds: usize = 0;
        var copy_landed: usize = 0;
        const ruled = cfg_a[2] == 9;
        var rule: DepthRule = .{};
        var depth: usize = if (ruled) rule.pick() else cfg_a[0];
        m.reset();
        m.mtp.pos = 0;
        m.mtp.drafted = 0;
        const all = try r.buffer(prompt.len * WIDE * 2);
        var nexts: std.ArrayList(u32) = .empty;
        const p0 = mtl.clock.seconds();
        var at_p: usize = 0;
        var last_n: usize = 1;
        while (at_p < prompt.len) { // the prompt in windows of up to MAXR rows (each row's bits equal a one-row step)
            const n: usize = @min(MAXR, prompt.len - at_p);
            var toks: [MAXR]u32 = undefined;
            for (0..n) |i| toks[i] = @intCast(prompt[at_p + i].integer);
            try m.window(toks[0..n], &pick);
            m.keepRows(toks[0..n], n);
            @memcpy(all.contents()[at_p * WIDE * 2 .. (at_p + n) * WIDE * 2], m.last.b.contents()[m.last.off .. m.last.off + n * WIDE * 2]);
            for (0..n) |i| if (at_p + i > 0) try nexts.append(gpa, toks[i]);
            at_p += n;
            last_n = n;
        }
        const prefill_s = mtl.clock.seconds() - p0;
        const first = pick[last_n - 1];
        try nexts.append(gpa, first);
        var out: std.ArrayList(u32) = .empty;
        try out.append(gpa, first);
        const w = wids.b.slice(u32, 16);
        var absorb_rows: []const u32 = nexts.items;
        var absorb_from: Buf = .{ .b = all };
        if (absorb_rows.len > MAXR) { // a long prompt: the head absorbs it before the rounds, a command buffer a chunk
            w[1] = try m.mtpAbsorb(absorb_rows, absorb_from);
            absorb_rows = &.{};
        }
        var n_rounds: usize = 0;
        var landed: usize = 0;
        const s0 = mtl.clock.seconds();
        m.gpu_seconds = 0;
        while (out.items.len < want.len) {
            w[0] = out.items[out.items.len - 1];
            var d = depth;
            var copied: usize = 0;
            if (use_copy) { // copy lanes: the window takes the tokens after an earlier match of its suffix
                hist.shrinkRetainingCapacity(n_prompt);
                try hist.appendSlice(gpa, out.items);
                copied = copyDrafts(hist.items, copy_min, w[1..MAXR]);
                if (copied > 0) d = copied;
            }
            const cb = r.queue.commandBuffer();
            r.enc = cb.compute(if (r.serial) .serial else .concurrent);
            // the head: absorb the kept rows (chunks of up to MAXR), its draft into slot 1, then chain into 2..depth
            m.mtp.pos -= m.mtp.drafted;
            m.mtp.drafted = 0;
            var at: usize = 0;
            var slot: usize = 0;
            while (at < absorb_rows.len) : (slot += 1) {
                const n = @min(MAXR, absorb_rows.len - at);
                const sl = &m.mtp.slots[slot];
                const ids = sl.ids8.b.slice(u32, MAXR);
                for (0..MAXR) |i| ids[i] = if (i < n) absorb_rows[at + i] else 0;
                try m.mtpEncode(slot, n, sl.ids8, .{ .b = absorb_from.b, .off = absorb_from.off + at * WIDE * 2 }, .{ .b = wids.b, .off = if (copied > 0) 4 * 12 else 4 });
                m.mtp.pos += n;
                at += n;
            }
            if (copied == 0) for (1..depth) |j| {
                try m.mtpEncode(slot, 1, .{ .b = wids.b, .off = 4 * j }, m.mtp.last, .{ .b = wids.b, .off = 4 * (j + 1) });
                m.mtp.pos += 1;
                m.mtp.drafted += 1;
                slot += 1;
            };
            // target window [pending, drafts]: FZ_SPLIT commits head first, encodes window while GPU runs (same queue)
            var wcb = cb;
            if (r.split) { // untracked buffers: the window waits on the head's signal, not on queue order alone
                r.enc.end();
                r.event_value += 1;
                cb.signalEvent(r.event, r.event_value);
                cb.commit();
                wcb = r.queue.commandBuffer();
                wcb.waitForEvent(r.event, r.event_value);
                r.enc = wcb.compute(if (r.serial) .serial else .concurrent);
            }
            m.windowMeta(d + 1);
            m.pleIdsGpu(d + 1, wids);
            try m.windowEncode(d + 1, wids);
            try m.finish(wcb);
            if (r.split) m.gpu_seconds += cb.gpuSeconds();
            const picks = m.t.picks.b.slice(u32, d + 1);
            var keep: usize = 1;
            while (keep <= d and w[keep] == picks[keep - 1]) keep += 1;
            try out.appendSlice(gpa, picks[0..keep]);
            n_rounds += 1;
            landed += keep - 1;
            if (copied > 0) {
                copy_rounds += 1;
                copy_landed += keep - 1;
            }
            var win: [MAXR]u32 = undefined;
            @memcpy(win[0 .. d + 1], w[0 .. d + 1]);
            m.keepRows(win[0 .. d + 1], keep);
            @memcpy(pick[0..keep], picks[0..keep]);
            absorb_rows = pick[0..keep];
            absorb_from = m.last;
            if (ruled) {
                rule.update(d, keep - 1);
                depth = rule.pick();
            } else if (cfg_a[0] != cfg_a[1]) depth = @max(cfg_a[0], @min(cfg_a[1], keep - 1 + cfg_a[2]));
        }
        const wall = mtl.clock.seconds() - s0;
        var eq: usize = 0;
        while (eq < want.len and out.items[eq] == (if (r.xnew) ref_tokens[eq] else @as(u32, @intCast(want[eq].integer)))) eq += 1;
        const made: f64 = @floatFromInt(out.items.len - 1);
        std.debug.print("one buffer a round, depth {d}-{d} (+{d}){s}: {d}/{d} tokens equal; {d:.2} tokens a round, {d:.2} drafts landing; {d:.1} tok/s (GPU busy {d:.0}%)\n", .{ cfg_a[0], cfg_a[1], cfg_a[2], if (use_copy) " + copy lanes" else "", eq, want.len, made / @as(f64, @floatFromInt(n_rounds)), @as(f64, @floatFromInt(landed)) / @as(f64, @floatFromInt(n_rounds)), made / wall, 100 * m.gpu_seconds / wall });
        std.debug.print("  prompt {d} tokens in {d:.2} s ({d:.0} tok/s){s}\n", .{ prompt.len, prefill_s, @as(f64, @floatFromInt(prompt.len)) / prefill_s, if (ruled) ", drafts chosen each round by the depth rule" else "" });
        if (use_copy) std.debug.print("  copy rounds {d} of {d}, {d:.2} copied lanes landing a copy round\n", .{ copy_rounds, n_rounds, @as(f64, @floatFromInt(copy_landed)) / @as(f64, @floatFromInt(@max(copy_rounds, 1))) });
        if (eq < want.len) bad += 1;
    }

    // 7. GPU-side rounds: state on GPU, host encodes N+1 while N runs reading a ring; depth 0 depth-rules drafts
    var depths7: std.ArrayList(usize) = .empty;
    if (platform.c.getenv("FZ_D7")) |v| { // e.g. FZ_D7=6,0
        var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
        while (it.next()) |d| try depths7.append(gpa, try std.fmt.parseInt(usize, d, 10));
    } else try depths7.appendSlice(gpa, &.{ 3, 6, 0 });
    if (r.xnew) for (depths7.items) |depth_cfg| {
        const ruled = depth_cfg == 0;
        var rule: DepthRule = .{};
        const depth0 = if (ruled) rule.pick() else depth_cfg;
        const n_lin: usize = 36;
        const g_cs = try r.buffer(n_lin * CS_ROW);
        const g_so = try r.buffer(n_lin * SO_ROW);
        const o_cs = try r.buffer(n_lin * MAXR * CS_ROW);
        const o_so = try r.buffer(n_lin * MAXR * SO_ROW);
        var saved: [LAYERS][4]Buf = undefined;
        var gi: usize = 0;
        for (&m.layers, 0..) |*L, i| if (L.linear) {
            saved[i] = .{ L.cs[0], L.cs[1], L.so[0], L.so[1] };
            L.cs[0] = .{ .b = g_cs, .off = gi * CS_ROW };
            L.cs[1] = .{ .b = o_cs, .off = gi * MAXR * CS_ROW };
            L.so[0] = .{ .b = g_so, .off = gi * SO_ROW };
            L.so[1] = .{ .b = o_so, .off = gi * MAXR * SO_ROW };
            gi += 1;
        };
        const cin_saved = m.ple.cin;
        const cins = [2]Buf{ m.ple.cin, .{ .b = try r.buffer((PLE_TAIL + MAXR) * WIDE * 2) } };
        r.ar = .{ .b = try r.buffer(4 * fz.AR_WORDS) };
        const ring = try r.buffer(fz.RING_WORDS * 4 * 512);
        const g_hist = try r.buffer((CAP + 64) * 4);
        const ar = r.ar.b.slice(i32, fz.AR_WORDS);
        const Copy = struct { // the kept row of every DeltaNet layer's window output into its state
            fn states(rr: *Run, gcs: mtl.Buffer, gso: mtl.Buffer, ocs: mtl.Buffer, oso: mtl.Buffer) void {
                rr.copyKept(.{ .b = oso }, .{ .b = gso }, SO_ROW / 4, SO_ROW / 4, MAXR * SO_ROW / 4, SO_ROW / 4, 36, -1);
                rr.copyKept(.{ .b = ocs }, .{ .b = gcs }, CS_ROW / 4, CS_ROW / 4, MAXR * CS_ROW / 4, CS_ROW / 4, 36, -1);
            }
        };
        m.reset();
        m.mtp.pos = 0;
        m.mtp.drafted = 0;
        var drafts: [MAXR]u32 = undefined;
        const p0 = mtl.clock.seconds();
        if (pr_long) |*pr| { // prompt chunks; the head's keys for every prompt row, its layer on the last
            var at: usize = 0;
            var last_n: usize = 1;
            while (at < ptoks.len) {
                const n = @min(pr.step, ptoks.len - at);
                pick[0] = try pr.chunk(m, gpa, ptoks[at .. at + n]);
                const k = if (at + n < ptoks.len) n else n - 1;
                try pr.mtpKeys(m, at, ptoks[at + 1 .. at + 1 + k], m.last);
                at += n;
                last_n = n;
            }
            if (m.state == 1) { // the state into the rounds' state buffers
                ar[0] = 1;
                const cb = r.queue.commandBuffer();
                r.enc = cb.compute(if (r.serial) .serial else .concurrent);
                Copy.states(r, g_cs, g_so, o_cs, o_so);
                try m.finish(cb);
                m.state = 0;
                m.state_row = 0;
            }
            m.mtp.pos = ptoks.len - 1;
            drafts[0] = try m.mtpRun(&.{pick[0]}, .{ .b = m.last.b, .off = m.last.off + (last_n - 1) * WIDE * 2 });
        } else {
            const all = try r.buffer(prompt.len * WIDE * 2);
            var nexts: std.ArrayList(u32) = .empty;
            for (prompt, 0..) |x, i| {
                const tok: u32 = @intCast(x.integer);
                try m.window(&.{tok}, &pick);
                ar[0] = 1;
                const cb = r.queue.commandBuffer();
                r.enc = cb.compute(if (r.serial) .serial else .concurrent);
                Copy.states(r, g_cs, g_so, o_cs, o_so);
                try m.finish(cb);
                const cin = m.ple.cin.b.contents()[m.ple.cin.off..];
                std.mem.copyForwards(u8, cin[0 .. PLE_TAIL * WIDE * 2], cin[WIDE * 2 .. (PLE_TAIL + 1) * WIDE * 2]);
                m.ple.hist = .{ m.ple.hist[1], tok };
                m.pos += 1;
                @memcpy(all.contents()[i * WIDE * 2 .. (i + 1) * WIDE * 2], m.last.b.contents()[0 .. WIDE * 2]);
                if (i > 0) try nexts.append(gpa, tok);
            }
            try nexts.append(gpa, pick[0]);
            drafts[0] = try m.mtpAbsorb(nexts.items, .{ .b = all });
        }
        for (1..depth0) |j| drafts[j] = try m.mtpChain(drafts[j - 1]);
        const first_s = mtl.clock.seconds() - p0;
        const w = wids.b.slice(u32, 16);
        w[0] = pick[0];
        for (0..depth0) |j| w[1 + j] = drafts[j];
        const W0 = depth0 + 1;
        const T: i32 = @intCast(m.pos);
        @memset(ar, 0);
        ar[1] = T;
        ar[fz.AR_HLEN] = @intCast(prompt.len + 1); // the accept kernel appends emitted tokens to the history
        for (0..MAXR) |i| {
            ar[fz.AR_POS + i] = if (i < W0) T + @as(i32, @intCast(i)) else 0;
            ar[fz.AR_NK + i] = if (i < W0) T + @as(i32, @intCast(i)) + 1 else 0;
        }
        ar[fz.AR_KV], ar[fz.AR_KV + 1], ar[fz.AR_KV + 2] = .{ T, CAP, @intCast(W0) };
        const pm = m.t.ple_meta.b.slice(i64, 39);
        pm[0], pm[1], pm[2], pm[3] = .{ m.ple.hist[0], m.ple.hist[1], m.ple.eos, 0 };
        for (0..3) |k| pm[4 + k] = m.ple.mult[k];
        for (0..16) |k| {
            pm[7 + k] = m.ple.sizes[k];
            pm[23 + k] = m.ple.offsets[k];
        }
        const t_saved = .{ m.t.rows, m.t.mdims, m.t.pos8, m.t.nk8, m.t.kvmeta };
        var rows_w: [MAXR + 1]Buf = undefined; // the window's row-count buffers, one set a width
        var mdims_w: [MAXR + 1]Buf = undefined;
        for (1..MAXR + 1) |n| {
            rows_w[n] = try i32Buf(r, &.{@intCast(n)});
            mdims_w[n] = try i32Buf(r, &.{ @intCast(n), 16, 0, 0, 0, 0, 0, 0 });
        }
        m.t.pos8 = .{ .b = r.ar.b, .off = fz.AR_POS * 4 };
        m.t.nk8 = .{ .b = r.ar.b, .off = fz.AR_NK * 4 };
        m.t.kvmeta = .{ .b = r.ar.b, .off = fz.AR_KV * 4 };
        const slots_saved = m.mtp.slots;
        for (1..MAXR + 1) |n| { // the head absorbing a window of n rows: slot MAXR - 1 + n
            const sl = &m.mtp.slots[MAXR - 1 + n];
            try m.mtpMeta(sl, n);
            sl.pos8 = .{ .b = r.ar.b, .off = fz.AR_ABS_POS * 4 };
            sl.nk8 = .{ .b = r.ar.b, .off = fz.AR_ABS_NK * 4 };
            sl.kvmeta = .{ .b = r.ar.b, .off = fz.AR_ABS_KV * 4 };
        }
        for (1..MAXR - 1) |j| { // chained draft j: slot j
            try m.mtpMeta(&m.mtp.slots[j], 1);
            const b = (fz.AR_CHAIN + (j - 1) * fz.AR_CHAIN_STRIDE) * 4;
            m.mtp.slots[j].pos8 = .{ .b = r.ar.b, .off = b };
            m.mtp.slots[j].nk8 = .{ .b = r.ar.b, .off = b + 16 * 4 };
            m.mtp.slots[j].kvmeta = .{ .b = r.ar.b, .off = b + 32 * 4 };
        }
        if (r.sel != null and prompt.len + want.len + 8 > 4 * TOP) { // long context: selection on the GPU
            const cb0 = r.queue.commandBuffer();
            r.enc = cb0.compute(if (r.serial) .serial else .concurrent);
            try r.sel.?.catchUp(r, m, m.pos / 4); // the target's layers and the head's absorbed rows
            try m.finish(cb0);
            r.gsel = try GSelect.init(r, m.pos / 4);
            m.mtp.gsel = try GSelect.init(r, m.pos / 4);
        }
        r.gpu_round = true;
        const base = r.event_value;
        var cbs: std.ArrayList(mtl.CommandBuffer) = .empty;
        var widths: std.ArrayList(usize) = .empty; // each round's window rows (decided a round ahead)
        try widths.append(gpa, W0);
        var out: std.ArrayList(u32) = .empty;
        try out.append(gpa, pick[0]);
        m.gpu_seconds = 0;
        const s0 = mtl.clock.seconds();
        var s1 = s0;
        var round: usize = 0;
        var done: usize = 0;
        var w_sum: usize = 0;
        var landed: usize = 0;
        var offered: usize = 0;
        var at_w: [MAXR + 1]usize = @splat(0);
        const rg = ring.slice(u32, fz.RING_WORDS * 512);
        while (true) {
            const wr = widths.items[round];
            const cb = r.queue.commandBuffer();
            if (round > 0) cb.waitForEvent(r.event, base + round);
            r.enc = cb.compute(if (r.serial) .serial else .concurrent);
            m.t.rows = rows_w[wr];
            m.t.mdims = mdims_w[wr];
            if (round > 0) {
                const wp = widths.items[round - 1];
                Copy.states(r, g_cs, g_so, o_cs, o_so);
                r.copyKept(cins[(round - 1) % 2], cins[round % 2], PLE_TAIL * WIDE / 2, WIDE / 2, 0, 0, 1, 0);
                try m.mtpEncode(MAXR - 1 + wp, wp, m.t.picks, m.last, .{ .b = wids.b, .off = 4 });
                for (1..wr - 1) |j| try m.mtpEncode(j, 1, .{ .b = wids.b, .off = 4 * j }, m.mtp.h[1], .{ .b = wids.b, .off = 4 * (j + 1) });
            }
            m.ple.cin = cins[round % 2];
            m.pleIdsGpu(wr, wids);
            w_sum += wr;
            if (r.gsel) |*g| g.nb_ub = (@as(usize, @intCast(T)) + w_sum) / 4 + 2;
            if (m.mtp.gsel) |*g| g.nb_ub = (@as(usize, @intCast(T)) + w_sum) / 4 + 2;
            try m.windowEncode(wr, wids);
            const wn = (if (ruled) rule.pick() else depth_cfg) + 1;
            try widths.append(gpa, wn);
            const cfg = [4]u32{ @intCast(wr), @intCast(wn), CAP, 0 };
            r.enc.setPipeline(r.accept_pipe);
            for ([_]Buf{ wids, m.t.picks, r.ar, .{ .b = ring }, m.t.ple_meta }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
            r.enc.setBytes(std.mem.asBytes(&cfg), 5);
            r.enc.setBuffer(g_hist, 0, 6);
            r.enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
            r.enc.end();
            cb.signalEvent(r.event, base + round + 1);
            cb.commit();
            try cbs.append(gpa, cb);
            round += 1;
            if (round >= 2) {
                const prev = cbs.items[done];
                prev.wait();
                if (prev.failure()) |msg| {
                    std.log.err("command buffer failed: {s}", .{msg});
                    return error.GpuFailed;
                }
                m.gpu_seconds += prev.gpuSeconds();
                const slot = (done % 512) * fz.RING_WORDS;
                const keep = rg[slot];
                try out.appendSlice(gpa, rg[slot + 3 .. slot + 3 + keep]);
                const wd = widths.items[done];
                if (ruled) rule.update(wd - 1, keep - 1);
                landed += keep - 1;
                offered += wd - 1;
                at_w[wd] += 1;
                done += 1;
                if (out.items.len >= want.len) {
                    s1 = mtl.clock.seconds();
                    break;
                }
            }
        }
        for (cbs.items[done..]) |cb| cb.wait();
        r.event_value = base + round + 1;
        r.gpu_round = false;
        r.gsel = null;
        m.mtp.gsel = null;
        m.t.rows, m.t.mdims, m.t.pos8, m.t.nk8, m.t.kvmeta = t_saved;
        m.mtp.slots = slots_saved;
        m.ple.cin = cin_saved;
        for (&m.layers, 0..) |*L, i| if (L.linear) {
            L.cs[0], L.cs[1], L.so[0], L.so[1] = saved[i];
        };
        var eq: usize = 0;
        while (eq < want.len and out.items[eq] == ref_tokens[eq]) eq += 1;
        const made: f64 = @floatFromInt(out.items.len - 1);
        const nd: f64 = @floatFromInt(done);
        if (ruled) {
            std.debug.print("GPU-side rounds, depth rule", .{});
            for (at_w, 0..) |c, n| if (c > 0) std.debug.print(" {d} rounds at {d} drafts", .{ c, n - 1 });
        } else std.debug.print("GPU-side rounds, depth {d}", .{depth_cfg});
        std.debug.print(": {d}/{d} tokens equal; {d:.2} tokens a round, {d} of {d} drafts landed; {d:.1} tok/s (GPU busy {d:.0}%); prompt to first drafts {d:.2} s\n", .{ eq, want.len, made / nd, landed, offered, made / (s1 - s0), 100 * m.gpu_seconds / (s1 - s0), first_s });
        if (eq < want.len) bad += 1;
    };

    // 3. each window size's cost
    for ([_]usize{ 1, 2, 3, 4, 6, 8 }) |rows| {
        var toks: [MAXR]u32 = undefined;
        for (0..rows) |i| toks[i] = got.items[i];
        m.gpu_seconds = 0;
        const s0 = mtl.clock.seconds();
        for (0..10) |_| try m.window(toks[0..rows], &pick);
        const wall = (mtl.clock.seconds() - s0) * 1e2;
        std.debug.print("window of {d} rows: {d:.2} ms (GPU {d:.2} ms)\n", .{ rows, wall, m.gpu_seconds * 1e2 });
    }
    if (same != want.len or bad != 0) std.process.exit(1);
}
