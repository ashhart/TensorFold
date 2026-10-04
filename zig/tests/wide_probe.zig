//! A probe's own Nemotron stream on the Metal backend: prefill, serial steps, and windows run from one state uncommitted.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const nm = tf.nemotron;
const st = nm.state;
const fwd = nm.forward;
const kern = nm.kernels;
const Metal = nm.backend.Metal;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;


pub const Probe = struct {
    b: *Metal,
    c: *st.Cache,
    ids: mtl.Buffer, // u32 [max_rows]: a forward's input ids
    out: mtl.Buffer, // u32 [max_rows]: its drawn tokens
    topk_checked: usize = 0, // the head's GPU top-4 checked against the host's ranking
    topk_bad: usize = 0,

    pub fn init(b: *Metal) !Probe {
        const c = try b.gpa.create(st.Cache);
        c.* = try st.Cache.init(b.m.device, b.m.config, b.o.capacity, b.head != null);
        c.rid = try b.pool.takeRid();
        return .{ .b = b, .c = c, .ids = try b.m.device.buffer(st.max_rows * 4, opts), .out = try b.m.device.buffer(st.max_rows * 4, opts) };
    }

    pub fn deinit(p: *Probe) void {
        p.c.deinit(&p.b.pool);
        p.b.gpa.destroy(p.c);
        p.ids.deinit();
        p.out.deinit();
    }

    /// Run `body.encode(p, e)` in one concurrent command buffer, leaving out `drop`; its GPU ms.
    pub fn run(p: *Probe, drop: []const mtl.objc.Id, body: anytype) !f64 {
        try p.b.drain();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = p.b.m.queue.commandBuffer();
        var e = fwd.Enc{ .e = cb.compute(.concurrent), .concurrent = true, .drop = drop };
        try body.encode(p, &e);
        e.e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |text| {
            std.log.err("command buffer failed: {s}", .{text});
            return error.GpuFailed;
        }
        return cb.gpuSeconds() * 1e3;
    }

    /// Prefill `ids` in the backend's prompt chunks, the head absorbing each row whose next token the prompt holds.
    pub fn prefill(p: *Probe, ids: []const u32) !void {
        @memcpy(p.b.prompt.slice(u32, ids.len), ids);
        const Chunks = struct {
            n: usize,
            fn encode(j: @This(), q: *Probe, e: *fwd.Enc) !void {
                const b = q.b;
                const c = q.c;
                var at: usize = 0;
                while (at < j.n) {
                    const rows = @min(b.o.chunk, j.n - at);
                    const slot = try b.slotFor(c, .full);
                    const segs = [_]fwd.Seg{.{ .rows = rows, .cache = c, .store = .full, .slot = slot }};
                    b.forward().body(e, &segs, b.prompt, at * 4);
                    Metal.hold(c, rows, .full, slot);
                    c.commit(&b.pool, rows);
                    c.start = 0;
                    c.rows = rows;
                    if (b.head) |*h| h.absorb(e, c, @min(at + rows, j.n - 1) - at, b.scratch.x, 0, b.prompt, (at + 1) * 4);
                    at += rows;
                }
            }
        };
        _ = try p.run(&.{}, Chunks{ .n = ids.len });
    }

    /// The token after the prompt's last row (greedy).
    pub fn first(p: *Probe) !u32 {
        const Draw = struct {
            fn encode(_: @This(), q: *Probe, e: *fwd.Enc) !void {
                const segs = [_]fwd.Seg{.{ .rows = q.c.rows, .cache = q.c }};
                q.b.forward().draw(e, &segs, .last, q.out, 0);
            }
        };
        _ = try p.run(&.{}, Draw{});
        return p.out.slice(u32, 1)[0];
    }

    /// One committed row for `token`; the greedy token after it. scratch.x row 0 then holds its hidden row.
    pub fn step(p: *Probe, token: u32) !u32 {
        p.ids.slice(u32, 1)[0] = token;
        const slot = try p.b.slotFor(p.c, .full);
        const Step = struct {
            slot: i32,
            fn encode(j: @This(), q: *Probe, e: *fwd.Enc) !void {
                const segs = [_]fwd.Seg{.{ .rows = 1, .cache = q.c, .store = .full, .slot = j.slot }};
                const f = q.b.forward();
                f.body(e, &segs, q.ids, 0);
                f.draw(e, &segs, .all, q.out, 0);
            }
        };
        _ = try p.run(&.{}, Step{ .slot = slot });
        Metal.hold(p.c, 1, .full, slot);
        p.c.commit(&p.b.pool, 1);
        p.c.start = 0;
        p.c.rows = 1;
        return p.out.slice(u32, 1)[0];
    }

    /// A verify-shaped forward of `ids` from the stream's state (its replayed rows first), never committed: GPU ms.
    pub fn window(p: *Probe, ids: []const u32, drop: []const mtl.objc.Id, prof: ?*fwd.Profiler) !f64 {
        return p.windowAs(ids, .lag, drop, prof);
    }

    /// A tree window of `ids` (row parents, row 0's -1) from the stream's state, never committed: its GPU ms.
    pub fn treeWindow(p: *Probe, ids: []const u32, parents: []const i32) !f64 {
        @memcpy(p.ids.slice(u32, ids.len), ids);
        const slot = try p.b.slotFor(p.c, .lag);
        defer if (slot >= 0) p.b.pool.give(@intCast(slot));
        const Win = struct {
            rows: usize,
            slot: i32,
            parents: []const i32,
            fn encode(j: @This(), q: *Probe, e: *fwd.Enc) !void {
                const segs = [_]fwd.Seg{.{ .rows = j.rows, .cache = q.c, .store = .lag, .slot = j.slot, .parents = j.parents }};
                const f = q.b.forward();
                f.body(e, &segs, q.ids, 0);
                f.draw(e, &segs, .all, q.out, 0);
            }
        };
        return p.run(&.{}, Win{ .rows = ids.len, .slot = slot, .parents = parents });
    }

    /// As window(), or with `.all`: every row's state stored and nothing replayed (what replay replaces).
    pub fn windowAs(p: *Probe, ids: []const u32, store: fwd.Store, drop: []const mtl.objc.Id, prof: ?*fwd.Profiler) !f64 {
        const rows = ids.len;
        @memcpy(p.ids.slice(u32, rows), ids);
        const slot = try p.b.slotFor(p.c, .lag);
        defer if (slot >= 0) p.b.pool.give(@intCast(slot));
        var slots: [st.max_rows]i32 = @splat(-1);
        if (store == .all) for (slots[0..rows]) |*x| {
            x.* = @intCast(try p.b.pool.take());
        };
        defer if (store == .all) for (slots[0..rows]) |x| p.b.pool.give(@intCast(x));
        const replay = p.c.replay;
        if (store == .all) p.c.replay = 0;
        defer p.c.replay = replay;
        const Win = struct {
            rows: usize,
            slot: i32,
            store: fwd.Store,
            slots: []const i32,
            fn encode(j: @This(), q: *Probe, e: *fwd.Enc) !void {
                const segs = [_]fwd.Seg{.{ .rows = j.rows, .cache = q.c, .store = j.store, .slot = j.slot, .slots = j.slots }};
                const f = q.b.forward();
                f.body(e, &segs, q.ids, 0);
                f.draw(e, &segs, .all, q.out, 0);
            }
        };
        const win = Win{ .rows = rows, .slot = slot, .store = store, .slots = slots[0..rows] };
        if (prof) |pr| {
            try p.b.drain();
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            var e = fwd.Enc{ .e = pr.begin(), .prof = pr };
            try win.encode(p, &e);
            e.e.end();
            pr.cb.?.commit();
            pr.cb.?.wait();
            return 0;
        }
        return p.run(drop, win);
    }

    /// Commit a window of `ids` keeping all its rows (they replay first in the next forward), as a verify would.
    pub fn keep(p: *Probe, ids: []const u32) !void {
        @memcpy(p.ids.slice(u32, ids.len), ids);
        const slot = try p.b.slotFor(p.c, .lag);
        const Win = struct {
            rows: usize,
            slot: i32,
            fn encode(j: @This(), q: *Probe, e: *fwd.Enc) !void {
                const segs = [_]fwd.Seg{.{ .rows = j.rows, .cache = q.c, .store = .lag, .slot = j.slot }};
                q.b.forward().body(e, &segs, q.ids, 0);
            }
        };
        _ = try p.run(&.{}, Win{ .rows = ids.len, .slot = slot });
        Metal.hold(p.c, ids.len, .lag, slot);
        p.c.commit(&p.b.pool, ids.len);
    }

    /// The head's cache takes scratch.x row `row` with `token`, the token after it.
    pub fn absorb(p: *Probe, row: usize, token: u32) !void {
        p.ids.slice(u32, 1)[0] = token;
        const Abs = struct {
            row: usize,
            fn encode(j: @This(), q: *Probe, e: *fwd.Enc) !void {
                const d = q.b.m.config.hidden * 2;
                q.b.head.?.absorb(e, q.c, 1, q.b.scratch.x, j.row * d, q.ids, 0);
            }
        };
        _ = try p.run(&.{}, Abs{ .row = row });
    }

    /// `depth` chained head drafts from scratch.x row `row` and `token` (the cache keeps the first step's row).
    pub fn drafts(p: *Probe, row: usize, token: u32, depth: usize) ![]const u32 {
        p.ids.slice(u32, 1)[0] = token;
        const Chain = struct {
            row: usize,
            depth: usize,
            fn encode(j: @This(), q: *Probe, e: *fwd.Enc) !void {
                const d = q.b.m.config.hidden * 2;
                q.b.head.?.chain(e, q.c, j.depth, q.b.scratch.x, j.row * d, q.ids, 0, .greedy, q.out, 0);
            }
        };
        _ = try p.run(&.{}, Chain{ .row = row, .depth = depth });
        return p.out.slice(u32, depth);
    }

    /// The head chained one step a command buffer: each level's top `k` tokens and softmax probabilities, as chain().
    pub fn headLevels(p: *Probe, row: usize, token: u32, depth: usize, k: usize, toks: [][8]u32, probs: [][8]f64) !void {
        p.ids.slice(u32, 1)[0] = token;
        const kept = p.c.mtp_len + 1;
        for (0..depth) |j| {
            if (j > 0) p.c.mtp_len += 1;
            const Level = struct {
                j: usize,
                row: usize,
                fn encode(l: @This(), q: *Probe, e: *fwd.Enc) !void {
                    const h = &q.b.head.?;
                    const d = q.b.m.config.hidden * 2;
                    const lv: ?usize = if (h.topk) l.j else null;
                    if (l.j == 0) h.stepAt(e, q.c, q.b.scratch.x, l.row * d, q.ids, 0, .greedy, q.out, 0, lv) else h.stepAt(e, q.c, h.hid, 0, q.out, (l.j - 1) * 4, .greedy, q.out, l.j * 4, lv);
                }
            };
            _ = try p.run(&.{}, Level{ .j = j, .row = row });
            p.topDrafts(k, &toks[j]);
            const h = &p.b.head.?;
            const logits = h.scratch.logits.slice(u16, h.w.vocab);
            var most: f64 = -std.math.inf(f64);
            for (logits) |raw| most = @max(most, @as(f64, @as(f32, @bitCast(@as(u32, raw) << 16))));
            var total: f64 = 0;
            for (logits) |raw| total += @exp(@as(f64, @as(f32, @bitCast(@as(u32, raw) << 16))) - most);
            const ids = p.b.m.draft_ids;
            for (0..k) |r| {
                const at = std.mem.indexOfScalar(u32, ids, toks[j][r]).?;
                probs[j][r] = @exp(@as(f64, @as(f32, @bitCast(@as(u32, logits[at]) << 16))) - most) / total;
            }
            if (h.topk) {
                const gi = p.c.topk.slice(u32, st.max_levels * 4)[j * 4 ..][0..4];
                const gp = p.c.topk.slice(f32, 2 * st.max_levels * 4)[st.max_levels * 4 + j * 4 ..][0..4];
                for (0..@min(k, 4)) |r| {
                    p.topk_checked += 1;
                    if (gi[r] != toks[j][r] or @abs(gp[r] - probs[j][r]) > 1e-4) p.topk_bad += 1;
                }
            }
        }
        p.c.mtp_len = kept;
    }

    /// The head's last step's `k` best draft tokens (draft logits ranked on the host, ids through the draft vocabulary).
    pub fn topDrafts(p: *Probe, k: usize, out: []u32) void {
        const h = &p.b.head.?;
        const logits = h.scratch.logits.slice(u16, h.w.vocab);
        const map = p.b.m.draft_ids;
        var best: [8]usize = undefined;
        var vals: [8]f32 = undefined;
        var n: usize = 0;
        for (logits, 0..) |raw, i| {
            const v: f32 = @bitCast(@as(u32, raw) << 16);
            if (n == k and !(v > vals[k - 1])) continue;
            if (n < k) n += 1;
            var at = n - 1;
            while (at > 0 and v > vals[at - 1]) : (at -= 1) {
                vals[at] = vals[at - 1];
                best[at] = best[at - 1];
            }
            vals[at] = v;
            best[at] = i;
        }
        for (out[0..k], best[0..k]) |*o, i| o.* = map[i];
    }
};

/// Pipeline ids of every kernel whose key starts with one of `prefixes`.
pub fn pipelines(k: *const kern.Kernels, prefixes: []const []const u8, out: []mtl.objc.Id) []mtl.objc.Id {
    var n: usize = 0;
    for (0..kern.total) |i| {
        for (prefixes) |pre| if (std.mem.startsWith(u8, kern.keyOf(i), pre)) {
            out[n] = k.pipelines[i].id;
            n += 1;
            break;
        };
    }
    return out[0..n];
}

/// Comma-separated integers.
pub fn parseList(arena: std.mem.Allocator, text: []const u8) ![]usize {
    var list: std.ArrayList(usize) = .empty;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |t| try list.append(arena, try std.fmt.parseInt(usize, t, 10));
    return list.items;
}

/// A prompts file ({"name": [ids...]}) as names and id lists, in file order.
pub fn readPrompts(arena: std.mem.Allocator, path: []const u8) !std.json.ArrayHashMap([]u32) {
    const z = try arena.dupeSentinel(u8, path, 0);
    const f = try mtl.MappedFile.open(z);
    defer f.deinit();
    return std.json.parseFromSliceLeaky(std.json.ArrayHashMap([]u32), arena, f.bytes[0..f.size], .{ .allocate = .alloc_always });
}

/// Median of a small sample (sorted in place).
pub fn median(v: []f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return if (v.len % 2 == 1) v[v.len / 2] else (v[v.len / 2 - 1] + v[v.len / 2]) / 2;
}
