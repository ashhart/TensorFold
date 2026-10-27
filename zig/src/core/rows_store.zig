//! Conversation states on disk (--learn-conversations): rows in blocks states share, each state's own parts in a file.
const std = @import("std");
const platform = @import("sys");
const builtin = @import("builtin");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
pub const image = @import("rows_image.zig");
pub const chain = @import("rows_chain.zig");
pub const file = @import("rows_file.zig");
pub const write = @import("rows_write.zig");
pub const read = @import("rows_read.zig");
const dirs = @import("lanes").learned_dirs;
const Admission = @import("lanes").learned_disk.Admission;
const Allocator = std.mem.Allocator;
const Name = chain.Name;
const Span = modes.Span;

/// Tokens a block holds: 256 rows of every row part (the 27B's keys and values: 16 MiB).
pub const block_tokens = 256;

/// What a family gives the store: its kept states' image as parts, and copies out of and into kept states.
pub const Image = struct {
    /// A state's image at `at`, its parts in order, appended to `out`.
    parts: *const fn (ptr: *anyopaque, at: u32, out: *std.ArrayList(image.Part), a: Allocator) anyerror!void,
    /// Copy into owned `out`; all GPU producers and copies must complete before returning.
    gather: *const fn (ptr: *anyopaque, saved: pc.Saved, ranges: []const image.Range, out: []u8) anyerror!void,
    /// A new kept state of `at` tokens, its image filled by `scatter` and then restored as any other.
    blank: *const fn (ptr: *anyopaque, at: u32) anyerror!pc.Saved,
    /// Copy `in`, back to back, into `ranges` of `saved`'s image.
    scatter: *const fn (ptr: *anyopaque, saved: pc.Saved, ranges: []const image.Range, in: []const u8) anyerror!void,
    /// Whether `saved`'s image can still be read whole (null: always).
    whole: ?*const fn (ptr: *anyopaque, saved: pc.Saved) bool = null,
};

/// A state on disk: its name and position, what it read, its file's bytes and its blocks' (each counted whole).
pub const State = struct {
    name: Name,
    at: u32,
    tokens: []u32, // position plus lookahead
    spans: []Span, // decoded spans through `at`
    starts: []u32, // a planned family's chunk starts below `at`
    own: u64,
    rows: u64,
    used: u64,
    shared: bool,
};

/// A block file's bytes, and how many states name it.
pub const Block = struct { bytes: u64, refs: u32 = 0 };

pub const Counts = struct { written: u64 = 0, read: u64 = 0, reads: u64 = 0, writes: u64 = 0, skipped: u64 = 0, failed: u64 = 0, evicted: u64 = 0, verified: u64 = 0, differed: u64 = 0 };

/// --spill-gib: off (every keep written behind), evicted (evictions and close), finish (both: evictions finish).
pub const Spill = enum { off, evicted, finish };

pub const Options = struct {
    root: []const u8,
    identity: u64, // the family's identity (weights, arithmetic, build); the store's directory is its own hash of it
    cap: u64, // --learn-gib: bytes under the root, every identity together
    floor: u64, // --learn-min-free-gib
    daily: u64, // --learn-daily-gib: bytes written a day under the root
    lookahead: u32 = 0,
    planned: bool = false,
    spill: Spill = .off,
    sync_writes: bool = false,
    staging_bytes: u64 = 1 << 30,
    slots: ?usize = null, // --checkpoint-slots: the RAM store's states at most, beside this one (null: the family's)
};

/// The store under `o`, the RAM store's disk: `o.slots` bounds what it keeps in RAM.
pub fn attach(gpa: Allocator, store: *pc.Store, o: Options) !*Rows {
    const r = try Rows.open(gpa, store.family, o);
    store.disk = r;
    if (o.slots) |n| store.most = n;
    return r;
}

/// Where --spill-gib's states live: $HOME/.cache/tensorfold/spilled, beside the learned states' root.
pub fn spillRoot(a: Allocator) ![]const u8 {
    const home = platform.c.getenv("HOME") orelse return error.RowsDir;
    return std.fmt.allocPrint(a, "{s}/.cache/tensorfold/spilled", .{std.mem.span(home)});
}

pub const Rows = struct {
    gpa: Allocator,
    family: pc.Snapshots, // the kept states' family (its drop frees a state read back that failed)
    img: *const Image,
    root: []u8,
    dir: [:0]u8,
    keying: chain.Keying,
    cap: u64,
    daily: u64,
    admission: Admission = .{},
    states: std.ArrayList(State) = .empty,
    blocks: std.AutoHashMapUnmanaged(chain.Stem, Block) = .empty,
    queue: std.ArrayList(write.Queued) = .empty, // kept states waiting to be written, newest last (written first)
    job: ?*write.Job = null, // the synchronous writer's borrowed entry
    stage: file.Stage = &.{},
    clock: u64 = 0,
    read_rate: f64 = 2e9, // bytes a second, measured as states are read
    prefill_rate: f64 = 0, // the family's prompt tokens a second (0: none measured yet)
    counts: Counts = .{},
    turn: u64 = 0, // bytes written since the last report line
    verify: bool = false, // TF_IMPRINT_VERIFY=1: each state written is read back and compared with the RAM state
    spill: Spill = .off,
    file_io: file.IO = .{},
    worker: ?*write.io.Worker = null,
    staging_bytes: u64 = 1 << 30,
    sync_writes: bool = false,
    stopped: bool = false,
    flush_clock: ?u64 = null, // a close protects states touched after this clock from older queued saves
    next_job: u64 = 0,
    failures: write.Failures = .{},
    fetch: ?*@import("rows_prefetch.zig").Prefetch = null, // states read ahead for queued requests (a lane host's)

    /// The store under `o.root` for `o.identity` (made 0700 when missing); its files are read, unreferenced blocks go.
    pub fn open(gpa: Allocator, family: pc.Snapshots, o: Options) !*Rows {
        const r = try init(gpa, family, o);
        errdefer r.deinit();
        try makePath(gpa, r.dir);
        try r.scan();
        r.settle();
        r.stamp();
        r.sync_writes = o.sync_writes or (if (platform.c.getenv("TENSORFOLD_ROWS_SYNC")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false);
        r.staging_bytes = o.staging_bytes;
        r.next_job = @truncate(@as(u128, @intCast(write.nowNs())));
        r.verify = if (platform.c.getenv("TF_IMPRINT_VERIFY")) |v| v[0] == '1' else false;
        return r;
    }

    /// A store holding only its names, nothing read: from here `deinit` alone owns `r` and its strings.
    fn init(gpa: Allocator, family: pc.Snapshots, o: Options) !*Rows {
        const img = family.vtable.rows orelse return error.RowsUnsupported;
        const r = try gpa.create(Rows);
        errdefer gpa.destroy(r);
        var h = std.hash.Wyhash.init(0x726f7773);
        h.update(std.mem.asBytes(&o.identity));
        const id = h.final();
        const root = try gpa.dupe(u8, o.root);
        errdefer gpa.free(root);
        const dir = try std.fmt.allocPrintSentinel(gpa, "{s}/{x:0>16}", .{ o.root, id }, 0);
        r.* = .{ .gpa = gpa, .family = family, .img = img, .root = root, .dir = dir, .keying = .{ .identity = id, .block = block_tokens, .lookahead = o.lookahead, .planned = o.planned }, .cap = o.cap, .daily = o.daily, .admission = .{ .floor = o.floor }, .spill = o.spill };
        return r;
    }

    pub fn deinit(r: *Rows) void {
        write.stop(r, write.nowNs());
        if (r.fetch) |f| f.destroy();
        if (r.job) |j| write.abort(r, j);
        for (r.queue.items) |q| r.gpa.free(q.starts);
        for (r.states.items) |*s| freeState(r.gpa, s);
        r.states.deinit(r.gpa);
        r.blocks.deinit(r.gpa);
        r.queue.deinit(r.gpa);
        if (r.stage.len > 0) r.gpa.free(r.stage);
        r.gpa.free(r.dir);
        r.gpa.free(r.root);
        r.gpa.destroy(r);
    }

    fn freeState(gpa: Allocator, s: *State) void {
        gpa.free(s.tokens);
        gpa.free(s.spans);
        gpa.free(s.starts);
    }

    /// A staging buffer of at least `bytes` (kept between steps).
    pub fn staged(r: *Rows, bytes: u64) !file.Stage {
        if (r.stage.len >= bytes) return r.stage;
        if (r.stage.len > 0) r.gpa.free(r.stage);
        r.stage = &.{};
        r.stage = try file.stage(r.gpa, bytes);
        return r.stage;
    }

    /// A prompt pass of `tokens` in `ns` moves the rate a disk read is weighed against (under 1,024 tokens: none).
    pub fn prefilled(r: *Rows, tokens: usize, ns: i96) void {
        if (tokens < 1024 or ns <= 0) return;
        const rate = @as(f64, @floatFromInt(tokens)) / (@as(f64, @floatFromInt(ns)) / 1e9);
        r.prefill_rate = if (r.prefill_rate > 0) 0.5 * r.prefill_rate + 0.5 * rate else rate;
    }

    /// Every byte the store holds: its states' files and its blocks.
    pub fn held(r: *const Rows) u64 {
        var n: u64 = 0;
        for (r.states.items) |s| n += s.own;
        var it = r.blocks.valueIterator();
        while (it.next()) |b| n += b.bytes;
        return n;
    }

    /// A state's image parts at `at`, from its family (a's memory).
    pub fn parts(r: *const Rows, a: Allocator, at: u32) ![]image.Part {
        var list: std.ArrayList(image.Part) = .empty;
        errdefer list.deinit(a);
        try r.img.parts(r.family.ptr, at, &list, a);
        if (!image.valid(list.items, at, r.keying.block)) return error.RowsLayout;
        return list.toOwnedSlice(a);
    }

    /// A state's whole blocks (names, groups, bytes), in `out`.
    pub fn wantsOf(r: *const Rows, a: Allocator, s: *const State, out: *std.ArrayList(write.Want)) !void {
        const ps = try r.parts(a, s.at);
        defer a.free(ps);
        try r.blockNames(a, ps, s.tokens, s.spans, s.starts, s.at, out);
    }

    pub fn find(r: *const Rows, name: Name) ?usize {
        for (r.states.items, 0..) |s, i| if (std.mem.eql(u8, &s.name, &name)) return i;
        return null;
    }

    /// The names of a state's whole blocks, every group, in `out` (a's memory).
    pub fn blockNames(r: *const Rows, a: Allocator, layout: []const image.Part, tokens: []const u32, spans: []const Span, starts: []const u32, at: u32, out: *std.ArrayList(write.Want)) !void {
        var gbuf: [8]u32 = undefined;
        for (try image.groups(layout, &gbuf)) |g| {
            const w = image.wholeBlocks(at, r.keying.block, g);
            if (w.end == 0) continue;
            const names = try a.alloc(Name, w.end);
            defer a.free(names);
            const planned_starts = try withAt(a, starts, at, r.keying.planned);
            defer a.free(planned_starts);
            chain.blocks(r.keying, tokens, spans, planned_starts, at, g, w.end, names);
            for (names[w.first..w.end], w.first..) |n, k| try out.append(a, .{ .name = n, .window = g, .k = @intCast(k), .bytes = image.blockBytes(layout, r.keying.block, g) + @sizeOf(file.Trailer) });
        }
    }

    /// A state's block names named again, each block's refs moved by `delta`.
    pub fn ref(r: *Rows, wants: []const write.Want, delta: i2) void {
        for (wants) |w| if (r.blocks.getPtr(chain.stem(w.name))) |b| {
            if (delta > 0) b.refs += 1 else b.refs -|= 1;
        };
    }

    /// State `i` forgotten: its file, and every block no other state names.
    pub fn remove(r: *Rows, i: usize) void {
        write.fence(r);
        var s = r.states.orderedRemove(i);
        defer freeState(r.gpa, &s);
        unlinkName(r.dir, s.name, ".st");
        var wants: std.ArrayList(write.Want) = .empty;
        defer wants.deinit(r.gpa);
        r.wantsOf(r.gpa, &s, &wants) catch {};
        r.ref(wants.items, -1);
        r.collect();
    }

    /// Blocks no state names are deleted (a write that lost its entry, or the last state that named them).
    pub fn collect(r: *Rows) void {
        var gone: std.ArrayList(chain.Stem) = .empty;
        defer gone.deinit(r.gpa);
        var it = r.blocks.iterator();
        while (it.next()) |kv| if (kv.value_ptr.refs == 0 and !write.wanted(r, kv.key_ptr.*)) gone.append(r.gpa, kv.key_ptr.*) catch {};
        if (gone.items.len > 0) write.fence(r);
        for (gone.items) |n| {
            unlinkName(r.dir, n, ".blk");
            _ = r.blocks.remove(n);
        }
    }

    /// The disk state reaching furthest into `prompt` past `longer_than`: its tokens a prefix, spans and starts equal.
    pub fn best(r: *const Rows, prompt: []const u32, starts: []const u32, spans: []const Span, longer_than: u32) ?usize {
        var out: ?usize = null;
        for (r.states.items, 0..) |*s, i| {
            if (s.at <= longer_than) continue;
            if (out != null and s.at <= r.states.items[out.?].at) continue;
            if (!r.matches(s, prompt, starts, spans)) continue;
            out = i;
        }
        return out;
    }

    pub fn matches(r: *const Rows, s: *const State, prompt: []const u32, starts: []const u32, spans: []const Span) bool {
        if (s.tokens.len > prompt.len or s.at >= prompt.len) return false;
        if (!std.mem.eql(u32, prompt[0..s.tokens.len], s.tokens) or !modes.equal(s.spans, spans, s.at)) return false;
        return !r.keying.planned or sameBelow(starts, s.starts, s.at);
    }

    /// State `i` was used now: the newest for eviction.
    pub fn touch(r: *Rows, i: usize) void {
        r.clock += 1;
        r.states.items[i].used = r.clock;
    }

    /// Room for `bytes` more under the cap: other identities first (least recently opened), then this one's by value.
    pub fn room(r: *Rows, bytes: u64) bool {
        const others = dirs.totalOthers(r.root, r.dir) catch return false;
        var over = (others + r.held() + bytes) -| r.cap;
        if (over == 0) return true;
        const Old = struct { id: u64, used: u64 };
        var olds: std.ArrayList(Old) = .empty;
        defer olds.deinit(r.gpa);
        var cursor: ?u64 = null;
        while (dirs.next(r.root, r.dir, cursor) catch null) |id| {
            olds.append(r.gpa, .{ .id = id, .used = dirs.used(r.root, id) catch 0 }) catch return false;
            cursor = id;
        }
        std.mem.sort(Old, olds.items, {}, struct {
            fn older(_: void, x: Old, y: Old) bool {
                return x.used < y.used;
            }
        }.older);
        for (olds.items) |o| {
            if (over == 0) break;
            const b = dirs.otherBytes(r.root, r.dir, o.id) catch continue;
            write.fence(r);
            dirs.removeOther(r.root, r.dir, o.id) catch continue;
            over -|= b;
        }
        while (over > 0) {
            const i = r.victim() orelse return false;
            const before = r.held();
            r.remove(i);
            over -|= before - r.held();
            r.counts.evicted += 1;
        }
        return true;
    }

    /// The state worth least: tokens saved per byte only it holds (its file, blocks no other names), over 1 + its age.
    fn victim(r: *const Rows) ?usize {
        var out: ?usize = null;
        if (r.spill == .evicted) { // --spill-gib alone: the least recently used first
            for (r.states.items, 0..) |s, i| {
                if (r.flush_clock) |cut| if (s.used > cut) continue;
                if (out == null or s.used < r.states.items[out.?].used) out = i;
            }
            return out;
        }
        var worst: f64 = std.math.inf(f64);
        var wants: std.ArrayList(write.Want) = .empty;
        defer wants.deinit(r.gpa);
        for (r.states.items, 0..) |*s, i| {
            if (r.flush_clock) |cut| if (s.used > cut) continue;
            wants.clearRetainingCapacity();
            r.wantsOf(r.gpa, s, &wants) catch {};
            var only = s.own;
            for (wants.items) |w| if (r.blocks.get(chain.stem(w.name))) |b| {
                if (b.refs <= 1) only += b.bytes;
            };
            const age: f64 = @floatFromInt(r.clock -| s.used);
            const v = @as(f64, @floatFromInt(s.at)) / @as(f64, @floatFromInt(@max(only, 1))) / (1 + age);
            if (v < worst) {
                worst = v;
                out = i;
            }
        }
        return out;
    }

    /// Whether `bytes` more fit today's write budget (root/written: the day and its bytes, every identity together).
    pub fn budget(r: *const Rows, bytes: u64) bool {
        const w = r.written();
        return w.bytes + bytes <= r.daily;
    }

    const Day = struct { day: u64, bytes: u64 };

    fn written(r: *const Rows) Day {
        const today = dayNow();
        var buf: [1100]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&buf, "{s}/written", .{r.root}, 0) catch return .{ .day = today, .bytes = std.math.maxInt(u64) };
        const fd = platform.c.open(p, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(platform.c.mode_t, 0));
        if (fd < 0) return .{ .day = today, .bytes = 0 };
        defer _ = platform.c.close(fd);
        var d: Day = undefined;
        if (platform.c.pread(fd, std.mem.asBytes(&d), @sizeOf(Day), 0) != @sizeOf(Day) or d.day != today) return .{ .day = today, .bytes = 0 };
        return d;
    }

    /// `bytes` written now count against today's budget.
    pub fn spent(r: *Rows, bytes: u64) void {
        r.counts.written += bytes;
        r.turn += bytes;
        var d = r.written();
        d.bytes +|= bytes;
        var buf: [1100]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&buf, "{s}/written", .{r.root}, 0) catch return;
        const fd = platform.c.open(p, .{ .ACCMODE = .WRONLY, .CREAT = true, .NOFOLLOW = true }, @as(platform.c.mode_t, 0o600));
        if (fd < 0) return;
        defer _ = platform.c.close(fd);
        _ = platform.c.pwrite(fd, std.mem.asBytes(&d), @sizeOf(Day), 0);
    }

    /// Every state and block of this identity forgotten (DELETE /v1/imprint); the queue too. The bytes freed.
    pub fn forgetAll(r: *Rows) u64 {
        write.fence(r);
        if (r.job) |j| write.abort(r, j);
        for (r.queue.items) |q| r.gpa.free(q.starts);
        r.queue.clearRetainingCapacity();
        const before = r.held();
        for (r.states.items) |*s| {
            unlinkName(r.dir, s.name, ".st");
            freeState(r.gpa, s);
        }
        r.states.clearRetainingCapacity();
        var it = r.blocks.keyIterator();
        while (it.next()) |n| unlinkName(r.dir, n.*, ".blk");
        r.blocks.clearRetainingCapacity();
        return before;
    }

    /// The files under the store's directory: states read back and checked by name, blocks by size; leftovers go.
    fn scan(r: *Rows) !void {
        const d = platform.c.opendir(r.dir) orelse return error.RowsDir;
        defer _ = platform.c.closedir(d);
        while (platform.c.readdir(d)) |ent| {
            if (ent.type != platform.c.DT.REG) continue;
            const name = std.mem.sliceTo(&ent.name, 0);
            var buf: [1200]u8 = undefined;
            const p = std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ r.dir, name }, 0) catch continue;
            if (std.mem.endsWith(u8, name, ".part")) {
                _ = platform.c.unlink(p);
                continue;
            }
            const n = parseName(name) orelse continue;
            if (std.mem.endsWith(u8, name, ".blk")) {
                const bytes = dirs.fileBytes(p) catch continue;
                try r.blocks.put(r.gpa, n, .{ .bytes = bytes });
            } else if (std.mem.endsWith(u8, name, ".st")) {
                r.readState(p) catch {
                    _ = platform.c.unlink(p);
                };
            }
        }
    }

    /// A state file's metadata checked against its name, kept for lookups (its payload is checked when read).
    fn readState(r: *Rows, p: [:0]const u8) !void {
        var o = try file.open(p, .state, null);
        defer o.close();
        const m = try file.meta(&o, r.gpa);
        defer r.gpa.free(m);
        if (m.len < 12) return error.RowsRead;
        const counts = std.mem.bytesAsSlice(u32, m[0..12]);
        const nt: usize = counts[0];
        const ns: usize = counts[1];
        const nst: usize = counts[2];
        if (m.len != 12 + 4 * (nt + 2 * ns + nst) or nt == 0) return error.RowsRead;
        var s: State = .{ .name = o.trailer.name, .at = o.trailer.at, .tokens = &.{}, .spans = &.{}, .starts = &.{}, .own = o.trailer.payload + o.trailer.meta + @sizeOf(file.Trailer), .rows = 0, .used = 0, .shared = o.trailer.flags & 1 != 0 };
        errdefer freeState(r.gpa, &s);
        s.tokens = try r.gpa.dupe(u32, @alignCast(std.mem.bytesAsSlice(u32, m[12..][0 .. 4 * nt])));
        s.spans = try r.gpa.dupe(Span, @alignCast(std.mem.bytesAsSlice(Span, m[12 + 4 * nt ..][0 .. 8 * ns])));
        s.starts = try r.gpa.dupe(u32, @alignCast(std.mem.bytesAsSlice(u32, m[12 + 4 * nt + 8 * ns ..][0 .. 4 * nst])));
        if (s.at == 0 or s.tokens.len != s.at + r.keying.lookahead or o.trailer.block != r.keying.block) return error.RowsRead;
        if (!std.mem.eql(u8, &chain.state(r.keying, s.tokens, s.spans, s.starts, s.at), &s.name)) return error.RowsRead;
        modes.validate(s.spans) catch return error.RowsRead;
        r.clock += 1;
        s.used = r.clock;
        try r.states.append(r.gpa, s);
    }

    /// Each state's blocks counted; a state missing one, or that its family cannot lay out, goes; then idle blocks.
    fn settle(r: *Rows) void {
        var i: usize = 0;
        while (i < r.states.items.len) {
            const s = &r.states.items[i];
            var wants: std.ArrayList(write.Want) = .empty;
            defer wants.deinit(r.gpa);
            const whole = if (r.wantsOf(r.gpa, s, &wants)) |_| for (wants.items) |w| {
                if (!r.blocks.contains(chain.stem(w.name))) break false;
            } else true else |_| false;
            if (!whole) {
                var gone = r.states.orderedRemove(i);
                unlinkName(r.dir, gone.name, ".st");
                freeState(r.gpa, &gone);
                continue;
            }
            r.ref(wants.items, 1);
            s.rows = 0;
            for (wants.items) |w| s.rows += w.bytes;
            i += 1;
        }
        r.collect();
    }

    /// This identity was opened (and wrote): the newest under the root for every identity's sweep.
    pub fn stamp(r: *const Rows) void {
        var path_buf: [1100]u8 = undefined;
        const clock_path = std.fmt.bufPrintSentinel(&path_buf, "{s}/clock", .{r.root}, 0) catch return;
        const now = readWord(clock_path) +| 1;
        writeWord(clock_path, now);
        var used: [1100]u8 = undefined;
        writeWord(std.fmt.bufPrintSentinel(&used, "{s}/used", .{r.dir}, 0) catch return, now);
    }
};

/// Whether a prompt's chunk starts below `at` are the state's, and `at` is one of them.
fn sameBelow(starts: []const u32, kept: []const u32, at: u32) bool {
    var n: usize = 0;
    while (n < starts.len and starts[n] < at) n += 1;
    return std.mem.eql(u32, starts[0..n], kept) and std.mem.indexOfScalar(u32, starts, at) != null;
}

/// A planned state's starts with its own position last: its blocks' last chunk ends by it.
pub fn withAt(a: Allocator, starts: []const u32, at: u32, planned: bool) ![]u32 {
    if (!planned) return a.alloc(u32, 0);
    var out = try a.alloc(u32, starts.len + 1);
    @memcpy(out[0..starts.len], starts);
    out[starts.len] = at;
    return out;
}

fn dayNow() u64 {
    var t: platform.c.timespec = undefined;
    if (platform.c.clock_gettime(platform.c.CLOCK.REALTIME, &t) != 0 or t.sec < 0) return 0;
    return @as(u64, @intCast(t.sec)) / 86400;
}

/// A file's stem: the first 16 bytes of its name in hex (32 digits), then `.blk` or `.st`.
fn parseName(name: []const u8) ?chain.Stem {
    if (name.len < 33 or name[32] != '.') return null;
    var n: chain.Stem = undefined;
    _ = std.fmt.hexToBytes(&n, name[0..32]) catch return null;
    return n;
}

/// The file of name `n` (a Name or a Stem) with extension `ext` removed (a missing one is fine).
pub fn unlinkName(dir: []const u8, n: anytype, ext: []const u8) void {
    var buf: [1200]u8 = undefined;
    const p = file.path(&buf, dir, &chain.hex(n), ext, false) catch return;
    _ = platform.c.unlink(p);
}

fn readWord(p: [:0]const u8) u64 {
    const fd = platform.c.open(p, .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
    if (fd < 0) return 0;
    defer _ = platform.c.close(fd);
    var w: u64 = 0;
    return if (platform.c.pread(fd, std.mem.asBytes(&w), 8, 0) == 8) w else 0;
}

fn writeWord(p: [:0]const u8, w: u64) void {
    var buf: [1200]u8 = undefined;
    const part = std.fmt.bufPrintSentinel(&buf, "{s}.part", .{p}, 0) catch return;
    const fd = platform.c.open(part, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .NOFOLLOW = true }, @as(platform.c.mode_t, 0o600));
    if (fd < 0) return;
    defer _ = platform.c.close(fd);
    if (platform.c.write(fd, std.mem.asBytes(&w), 8) != 8) {
        _ = platform.c.unlink(part);
        return;
    }
    _ = platform.c.rename(part, p);
}

/// `dir` and its parents, 0700.
fn makePath(gpa: Allocator, dir: [:0]const u8) !void {
    const tmp = try gpa.dupeSentinel(u8, dir, 0);
    defer gpa.free(tmp);
    for (tmp[1..], 1..) |ch, i| if (ch == '/') {
        tmp[i] = 0;
        _ = platform.c.mkdir(tmp.ptr, 0o700);
        tmp[i] = '/';
    };
    const rc = platform.c.mkdir(tmp.ptr, 0o700);
    if (rc != 0 and platform.c.errno(rc) != .EXIST) return error.RowsDir;
}

test {
    _ = image;
    _ = chain;
    _ = file;
    _ = @import("rows_test.zig");
    _ = @import("rows_spill_test.zig");
    _ = @import("rows_async_test.zig");
    _ = @import("rows_close_test.zig");
    _ = @import("rows_select_test.zig");
    _ = @import("rows_recall_select_test.zig");
    _ = @import("rows_pool_select_test.zig");
    _ = @import("rows_prefetch.zig");
}
