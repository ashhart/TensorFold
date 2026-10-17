//! tf-train-pack's checks on either backend: packed passes against one a pass, and passes after kept prefixes.
const std = @import("std");
const platform = @import("sys");

/// What one sequence's pass left: its loss, whether it came back, its pushes and its keys at the pocket's layers.
const Seen = struct { loss: f32, recalled: bool, push: []f32, keys: []u16 };

/// The checks for module `train` (either backend's trainer: the same pass API).
pub fn For(comptime train: type) type {
    return struct {
        const Trainer = train.Trainer;
        const Seq = train.Seq;

        /// `count` of a dump's examples spread over it, each nudged at its decision row (and its subject's, if named).
        pub fn examples(gpa: std.mem.Allocator, io: std.Io, path: []const u8, count: usize, layers: []const usize, rng: std.Random) ![]Seq {
            const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
            var all: std.ArrayList(Seq) = .empty;
            var lines = std.mem.tokenizeScalar(u8, text, '\n');
            while (lines.next()) |line| {
                var f = std.mem.tokenizeScalar(u8, line, ' ');
                _ = f.next();
                _ = f.next();
                const start = try std.fmt.parseInt(usize, f.next().?, 10);
                const decide = try std.fmt.parseInt(usize, f.next().?, 10);
                const subject = try std.fmt.parseInt(usize, f.next().?, 10);
                _ = f.next();
                var ids: std.ArrayList(u32) = .empty;
                while (f.next()) |x| try ids.append(gpa, try std.fmt.parseInt(u32, x, 10));
                if (ids.items.len > train.max_rows + 1 or start >= ids.items.len) continue;
                var nudge: train.Nudge = .{ .rows = .{ decide, if (subject > 0 and subject < start) subject else 0, 0 } };
                for (layers, 0..) |l, k| {
                    nudge.layers[0] |= @as(u64, 1) << @intCast(l);
                    if (k < 2 and nudge.rows[1] > 0) nudge.layers[1] |= @as(u64, 1) << @intCast(l);
                }
                try all.append(gpa, .{ .ids = ids.items, .start = start, .decide = decide, .scale = 0.5 + 0.5 * rng.float(f32), .nudge = nudge });
            }
            const out = try gpa.alloc(Seq, @min(count, all.items.len));
            for (out, 0..) |*s, i| s.* = all.items[i * all.items.len / out.len];
            return out;
        }

        /// Pairs of `short` rows in all, which take a pass of one's kernels: keys must keep their bits when `exact`.
        pub fn pairs(gpa: std.mem.Allocator, t: *Trainer, seqs: []const Seq, layers: []const usize, short: usize, exact: bool) !void {
            var tried: usize = 0;
            var kept: [3]usize = @splat(0);
            var diffs: [3]Diff = @splat(.{});
            var i: usize = 0;
            while (i + 1 < seqs.len) : (i += 1) {
                const two = [2]Seq{ seqs[i], seqs[i + 1] };
                if (rows(&two) > short) continue;
                for ([_]train.Mode{ .keys, .loss, .target }, &kept, &diffs) |mode, *k, *w| {
                    var one: [2]Seen = undefined;
                    var both: [2]Seen = undefined;
                    for (0..2) |j| try runPass(gpa, t, two[j..][0..1], mode, layers, one[j..][0..1]);
                    try runPass(gpa, t, &two, mode, layers, &both);
                    for (one, both) |x, y| {
                        k.* += @intFromBool(same(x, y));
                        w.add(x, y);
                    }
                }
                tried += 2;
                i += 1;
            }
            std.debug.print("RESULT pairs: of {d} sequences in passes of two short ones, the same to the bit as one a pass: keys {d}, losses {d}, targets {d}\n", .{ tried, kept[0], kept[1], kept[2] });
            for (&diffs) |*w| w.print();
            if (!exact) return;
            std.debug.print("{s} train-pack: short pairs keep every key's bits\n", .{if (kept[0] == tried) "PASS" else "FAIL"});
            if (kept[0] != tried) return error.PairKeysDiffer;
        }

        /// Every sequence one a pass and side by side in each mode, timed; then each pass again, and the others turned.
        pub fn check(gpa: std.mem.Allocator, io: std.Io, t: *Trainer, seqs: []const Seq, layers: []const usize) !void {
            var leaks: usize = 0;
            var tried: usize = 0;
            var repeats: usize = 0; // sequences whose results differed when their very pass ran again
            for ([_]train.Mode{ .loss, .keys, .target }) |mode| {
                const one = try gpa.alloc(Seen, seqs.len);
                const many = try gpa.alloc(Seen, seqs.len);
                var times: [2]f64 = @splat(std.math.inf(f64));
                for (0..3) |_| {
                    const t0 = now(io);
                    for (seqs, 0..) |s, i| try runPass(gpa, t, &.{s}, mode, layers, one[i..][0..1]);
                    const t1 = now(io);
                    try packed_(gpa, t, seqs, mode, layers, many);
                    times[0] = @min(times[0], t1 - t0);
                    times[1] = @min(times[1], now(io) - t1);
                }
                var w: Diff = .{};
                for (one, many) |a, z| w.add(a, z);
                std.debug.print("RESULT {s}: {d} sequences, {d} rows: one a pass {d:.1} ms, side by side {d:.1} ms ({d:.2}x)\n", .{ @tagName(mode), seqs.len, rows(seqs), 1000 * times[0], 1000 * times[1], times[0] / times[1] });
                w.print();
                if (mode == .target) try dump(gpa, io, many);
                var at: usize = 0;
                while (at < seqs.len) {
                    const n = fits(t, seqs[at..]);
                    var twice: [train.max_seqs]Seen = undefined;
                    try runPass(gpa, t, seqs[at..][0..n], mode, layers, twice[0..n]);
                    for (twice[0..n], many[at..][0..n]) |x, y| repeats += @intFromBool(!same(x, y));
                    for (0..n) |i| {
                        const other = try gpa.alloc(Seq, n);
                        for (other, seqs[at..][0..n], 0..) |*o, s, j| o.* = if (j == i) s else try turned(gpa, s);
                        var again: [train.max_seqs]Seen = undefined;
                        try runPass(gpa, t, other, mode, layers, again[0..n]);
                        tried += 1;
                        leaks += @intFromBool(!same(many[at + i], again[i]));
                    }
                    at += n;
                }
            }
            std.debug.print("RESULT repeats: {d} of {d} sequences changed when their very pass ran again\n", .{ repeats, tried });
            std.debug.print("RESULT leaks: {d} of {d} sequences changed when the others in their pass held other tokens\n", .{ leaks, tried });
            std.debug.print("{s} train-pack: no sequence reads another's rows\n", .{if (leaks == 0) "PASS" else "FAIL"});
            if (leaks != 0) return error.Leak;
        }

        /// Each sequence's rows before its decision, answer and nudges primed into a slot, then the rest against whole.
        pub fn prefixes(gpa: std.mem.Allocator, io: std.Io, t: *Trainer, all: []const Seq, layers: []const usize) !void {
            const most = @TypeOf(t.prefixes).most;
            var whole: std.ArrayList(Seq) = .empty;
            var cut: std.ArrayList(Seq) = .empty;
            var primes: std.ArrayList(Seq) = .empty;
            var skipped: usize = 0;
            for (all) |s| {
                var skip = @min(s.decide, s.start - 1);
                for (s.nudge.layers, s.nudge.rows) |mask, row| if (mask != 0) {
                    skip = @min(skip, row);
                };
                if (skip == 0 or whole.items.len == most) continue;
                const slot: u32 = @intCast(whole.items.len);
                try whole.append(gpa, s);
                var c = s;
                c.skip = skip;
                c.slot = slot;
                try cut.append(gpa, c);
                try primes.append(gpa, .{ .ids = s.ids[0 .. skip + 1], .start = skip, .decide = skip - 1, .scale = s.scale, .slot = slot });
                skipped += skip;
            }
            const n = whole.items.len;
            try t.prefixSlots(n);
            defer t.prefixSlots(0) catch {};
            var repeats: usize = 0;
            for ([_]train.Mode{ .loss, .target }) |mode| {
                var times: [3]f64 = @splat(std.math.inf(f64));
                const a = try gpa.alloc(Seen, n);
                const b = try gpa.alloc(Seen, n);
                const again = try gpa.alloc(Seen, n);
                for (0..3) |_| {
                    const t0 = now(io);
                    try packed_(gpa, t, whole.items, mode, layers, a);
                    const t1 = now(io);
                    try packed_(gpa, t, primes.items, .prime, &.{}, again);
                    const t2 = now(io);
                    try packed_(gpa, t, cut.items, mode, layers, b);
                    const t3 = now(io);
                    times[0] = @min(times[0], t1 - t0);
                    times[1] = @min(times[1], t2 - t1);
                    times[2] = @min(times[2], t3 - t2);
                }
                try packed_(gpa, t, cut.items, mode, layers, again);
                for (b, again) |x, y| repeats += @intFromBool(!same(x, y));
                var w: Diff = .{};
                for (a, b, cut.items) |x, y, c| w.add(try tail(gpa, x, c, layers.len), y);
                std.debug.print("RESULT prefixes {s}: {d} sequences, {d} rows, {d} of them before the cut: whole {d:.1} ms; the prefixes once {d:.1} ms, then from the cut {d:.1} ms ({d:.2}x)\n", .{ @tagName(mode), n, rows(whole.items), skipped, 1000 * times[0], 1000 * times[1], 1000 * times[2], times[0] / times[2] });
                w.print();
            }
            std.debug.print("RESULT prefix repeats: {d} of {d} sequences changed when their passes from the cut ran again\n", .{ repeats, 2 * n });
            std.debug.print("{s} train-pack: passes from a kept prefix repeat to the bit\n", .{if (repeats == 0) "PASS" else "FAIL"});
            if (repeats != 0) return error.Unrepeatable;
        }

        /// TF_PACK_DUMP: the packed target pass's losses and pushes written there, for another build's to compare.
        fn dump(gpa: std.mem.Allocator, io: std.Io, seen: []const Seen) !void {
            const path = platform.c.getenv("TF_PACK_DUMP") orelse return;
            var bytes: std.ArrayList(u8) = .empty;
            for (seen) |x| {
                try bytes.appendSlice(gpa, std.mem.asBytes(&x.loss));
                try bytes.appendSlice(gpa, std.mem.sliceAsBytes(x.push));
            }
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = std.mem.span(path), .data = bytes.items });
        }

        /// `seqs` in as few passes as hold them.
        fn packed_(gpa: std.mem.Allocator, t: *Trainer, seqs: []const Seq, mode: train.Mode, layers: []const usize, out: []Seen) !void {
            var at: usize = 0;
            while (at < seqs.len) {
                const n = fits(t, seqs[at..]);
                try runPass(gpa, t, seqs[at..][0..n], mode, layers, out[at..][0..n]);
                at += n;
            }
        }

        /// How many sequences from the first one pass holds.
        fn fits(t: *Trainer, seqs: []const Seq) usize {
            var r: usize = 0;
            var n: usize = 0;
            while (n < seqs.len and n < train.max_seqs) : (n += 1) {
                r += seqs[n].ids.len - 1 - seqs[n].skip;
                if (r > t.pack and n > 0) break;
            }
            return n;
        }

        /// One pass over `seqs` in `mode`, each one's loss, pushes and keys (its rows after any skipped) kept in `out`.
        fn runPass(gpa: std.mem.Allocator, t: *Trainer, seqs: []const Seq, mode: train.Mode, layers: []const usize, out: []Seen) !void {
            var got: [train.max_seqs]train.Result = undefined;
            for (seqs, 0..) |s, i| for (s.nudge.layers, 0..) |mask, k| {
                var bits = mask;
                while (bits != 0) : (bits &= bits - 1) @memset(t.deltaAt(i, k, @ctz(bits)), 0);
            };
            try t.pass(seqs, mode, .{ .decide = 0, .whole = true }, got[0..seqs.len]);
            for (seqs, out, 0..) |s, *o, i| {
                var pushes: std.ArrayList(f32) = .empty;
                var keys: std.ArrayList(u16) = .empty;
                if (mode == .target) for (s.nudge.layers, 0..) |mask, k| {
                    var bits = mask;
                    while (bits != 0) : (bits &= bits - 1) try pushes.appendSlice(gpa, t.pushAt(i, k, @ctz(bits)));
                };
                if (mode != .loss) for (layers) |l| for (s.skip..s.ids.len - 1) |r| try keys.appendSlice(gpa, t.normed(l, t.spans[i].row(r)));
                o.* = .{ .loss = got[i].loss, .recalled = got[i].recalled, .push = pushes.items, .keys = keys.items };
            }
        }

        /// `s` with its tokens rotated by one: another text of the same length, rows and nudges.
        fn turned(gpa: std.mem.Allocator, s: Seq) !Seq {
            const ids = try gpa.alloc(u32, s.ids.len);
            for (ids, 0..) |*v, r| v.* = s.ids[(r + 1) % s.ids.len];
            var o = s;
            o.ids = ids;
            return o;
        }

        /// A whole pass's keys from cut sequence `c`'s first row run on, at each of `layers` pocket layers.
        fn tail(gpa: std.mem.Allocator, x: Seen, c: Seq, layers: usize) !Seen {
            if (x.keys.len == 0) return x;
            const per = x.keys.len / layers; // [layer][row][d]
            const d = per / (c.ids.len - 1);
            var o = x;
            o.keys = try gpa.alloc(u16, layers * (c.ids.len - 1 - c.skip) * d);
            for (0..layers) |l| @memcpy(o.keys[l * (per - c.skip * d) ..][0 .. per - c.skip * d], x.keys[l * per + c.skip * d ..][0 .. per - c.skip * d]);
            return o;
        }

        fn rows(seqs: []const Seq) usize {
            var n: usize = 0;
            for (seqs) |s| n += s.ids.len - 1;
            return n;
        }
    };
}

fn same(a: Seen, b: Seen) bool {
    return @as(u32, @bitCast(a.loss)) == @as(u32, @bitCast(b.loss)) and std.mem.eql(u8, std.mem.sliceAsBytes(a.push), std.mem.sliceAsBytes(b.push)) and std.mem.eql(u16, a.keys, b.keys);
}

fn now(io: std.Io) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds())) / 1e9;
}

/// How two passes' results differ: losses, pushes (each vector, and all as one), keys (each row's vector).
const Diff = struct {
    loss_rel: std.ArrayList(f32) = .empty,
    loss_abs: f32 = 0,
    worst: [2]f32 = .{ 0, 0 }, // the two losses furthest apart relatively
    recalled: usize = 0,
    push_rel: std.ArrayList(f32) = .empty, // each vector's |a - b| over |a|
    cosine: std.ArrayList(f64) = .empty,
    all: [4]f64 = @splat(0), // every push as one vector: a.a, b.b, a.b, |a - b|^2
    key_rel: std.ArrayList(f32) = .empty, // each row's key vector's |a - b| over |a|
    keys: usize = 0,
    keys_far: usize = 0, // values more than one bf16 step apart

    fn add(w: *Diff, a: Seen, b: Seen) void {
        const gpa = std.heap.c_allocator;
        const rel = @abs(a.loss - b.loss) / @max(@abs(a.loss), 1e-6);
        if (w.loss_rel.items.len == 0 or rel > std.mem.max(f32, w.loss_rel.items)) w.worst = .{ a.loss, b.loss };
        w.loss_rel.append(gpa, rel) catch {};
        w.loss_abs = @max(w.loss_abs, @abs(a.loss - b.loss));
        w.recalled += @intFromBool(a.recalled != b.recalled);
        const d = 2688;
        var at: usize = 0;
        while (at + d <= a.push.len) : (at += d) {
            var s: [4]f64 = @splat(0);
            for (a.push[at..][0..d], b.push[at..][0..d]) |x, y| sum(&s, x, y);
            w.push_rel.append(gpa, @floatCast(if (s[0] > 0) @sqrt(s[3] / s[0]) else 0)) catch {};
            w.cosine.append(gpa, if (s[0] > 0 and s[1] > 0) s[2] / @sqrt(s[0] * s[1]) else 1) catch {};
            for (&w.all, s) |*x, y| x.* += y;
        }
        at = 0;
        while (at + d <= a.keys.len) : (at += d) {
            var s: [4]f64 = @splat(0);
            for (a.keys[at..][0..d], b.keys[at..][0..d]) |x, y| {
                sum(&s, bf(x), bf(y));
                const ux: i32 = @intCast(x & 0x7fff);
                const uy: i32 = @intCast(y & 0x7fff);
                w.keys_far += @intFromBool(x != y and ((x ^ y) & 0x8000 != 0 or @abs(ux - uy) > 1));
            }
            w.keys += d;
            w.key_rel.append(gpa, @floatCast(if (s[0] > 0) @sqrt(s[3] / s[0]) else 0)) catch {};
        }
    }

    fn print(w: *Diff) void {
        const n = w.loss_rel.items.len;
        if (n > 0) std.debug.print("  losses: {d}, relative difference median {e:.2}, largest {e:.2} ({d:.5} against {d:.5}); largest absolute {e:.2}; recalled differing {d}\n", .{ n, median(w.loss_rel.items), std.mem.max(f32, w.loss_rel.items), w.worst[0], w.worst[1], w.loss_abs, w.recalled });
        const m = w.push_rel.items.len;
        if (m > 0) {
            std.mem.sort(f64, w.cosine.items, {}, std.sort.asc(f64));
            const a = w.all;
            std.debug.print("  pushes: {d} vectors, relative difference median {e:.2}, worst {e:.2}; cosine median {d:.6}, least {d:.6}; all as one vector: relative difference {e:.2}, cosine {d:.6}\n", .{ m, median(w.push_rel.items), std.mem.max(f32, w.push_rel.items), w.cosine.items[m / 2], w.cosine.items[0], @sqrt(a[3] / @max(a[0], 1e-30)), a[2] / @sqrt(@max(a[0] * a[1], 1e-30)) });
        }
        if (w.key_rel.items.len > 0) std.debug.print("  keys: {d} row vectors, relative difference median {e:.2}, worst {e:.2}; {d} of {d} values more than one bf16 step apart\n", .{ w.key_rel.items.len, median(w.key_rel.items), std.mem.max(f32, w.key_rel.items), w.keys_far, w.keys });
        inline for (.{ "loss_rel", "push_rel", "cosine", "key_rel" }) |f| @field(w, f).deinit(std.heap.c_allocator);
    }
};

/// a.a, b.b, a.b and |a - b|^2 summed in `s`.
fn sum(s: *[4]f64, x: f32, y: f32) void {
    s[0] += x * x;
    s[1] += y * y;
    s[2] += x * y;
    s[3] += (x - y) * (x - y);
}

fn median(xs: []f32) f32 {
    std.mem.sort(f32, xs, {}, std.sort.asc(f32));
    return xs[xs.len / 2];
}

fn bf(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}
