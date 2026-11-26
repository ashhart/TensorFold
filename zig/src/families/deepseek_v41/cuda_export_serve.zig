//! Hand-offs exported over MKVH (`--mcdma-produce LINK`) and/or written to files (`TF_DSV41_HANDOFF_DUMP=DIR`).
const std = @import("std");
const builtin = @import("builtin");
const fabric = @import("fabric");
const cx = @import("cuda_export.zig");
const file = @import("handoff_file.zig");
const producer = fabric.producer;
const log = std.log.scoped(.dsv41);

pub const Options = struct {
    /// mcdma-rpcd's link name (null: no link; the exports go to files only)
    link: ?[]const u8 = null,
    /// a directory the exports are written under (null: none written)
    dump: ?[]const u8 = null,
    /// the checkpoint id both ends give a hand-off
    model: []const u8,
    /// the most export bytes a link holds at once (TF_PRODUCE_HELD_GIB)
    held_bytes: u64 = 32 << 30,
    ttl_ns: u64 = 120 * std.time.ns_per_s,
    /// KV_SPLIT: this producer is shard `shard` of `shards` (each rank exports its own pool pages on its own link)
    shard: u32 = 0,
    shards: u32 = 1,
    /// serve held exports over the server's own HTTP port (`--mcdma-produce http`), same cap and expiry as a link
    http: bool = false,
};

/// What `save` hands `exported`: the engine's state, read back.
pub const Saved = struct { cap: cx.Captured, read_ns: u64 = 0 };

/// The engine's side of a streamed export (cuda_export_stream.zig): fed the prompt's progress, stopped before release.
pub const Feeder = struct {
    ptr: *anyopaque,
    progress: *const fn (ptr: *anyopaque, pos: u64) void,
    /// waits for the segments queued; whether every paged part is converted
    drain: *const fn (ptr: *anyopaque) bool,
    stop: *const fn (ptr: *anyopaque) void,
};

/// One export the producer holds; a reading stream pins it, so expiry or release only marks it `gone` until it closes.
pub const Held = struct { built: *cx.Built, name: [32]u8, pins: u32 = 0, gone: bool = false, feeder: ?Feeder = null };

pub const Produce = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    source: cx.Source,
    opts: Options,
    box: fabric.mailbox.Mailbox = undefined,
    reg: ?fabric.registration.Registration = null, // MODE poll: the listen daemon sends staged replies only to a service
    table: producer.Table,
    serving: producer.Serving = undefined,
    thread: ?std.Thread = null,
    held: std.AutoHashMapUnmanaged([16]u8, *Held) = .empty,
    lock: std.atomic.Mutex = .unlocked, // `held`: the engine thread adds, the serving thread releases
    streams: std.atomic.Value(u32) = .init(0), // reads under way (HTTP): the producer stops after them
    building: std.atomic.Value(u32) = .init(0), // exports converting on their own threads (`exportedBg`)
    pending: std.AutoHashMapUnmanaged([16]u8, void) = .empty, // their ids, under `lock`: a reader waits for them
    dump_fault: ?DumpStep = null, // a test's seam: the dump step that fails

    /// The most exports converting at once; one more converts on the caller's thread.
    pub const max_building: u32 = 2;

    /// The steps of a dump after its tokens are written, each a place a test makes it fail.
    pub const DumpStep = enum { create, write, flush, rename };

    /// A producer over `source`: attaches the link when `opts.link` is set, makes the dump dir when `opts.dump` is.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, source: cx.Source, opts: Options, problem: *[]const u8) !*Produce {
        const p = try gpa.create(Produce);
        errdefer gpa.destroy(p);
        p.* = .{ .gpa = gpa, .io = io, .source = source, .opts = opts, .table = .{ .gpa = gpa, .ttl_ns = opts.ttl_ns, .clock = fabric.words.nowNs, .cap_bytes = opts.held_bytes } };
        errdefer p.table.deinit();
        if (opts.dump) |dir| std.Io.Dir.cwd().createDirPath(io, dir) catch |err| {
            problem.* = "TF_DSV41_HANDOFF_DUMP: the directory cannot be made";
            return err;
        };
        if (opts.http) log.info("handoff: exports are served over HTTP, GET /v1/handoff/<id> on this server's port", .{});
        const link = opts.link orelse return p;
        if (builtin.os.tag != .linux) {
            problem.* = "--mcdma-produce runs on Linux, beside mcdma-rpcd's listen daemon";
            return error.UnsupportedPlatform;
        }
        p.box = fabric.mailbox.Mailbox.openListen(link) catch |err| {
            problem.* = "--mcdma-produce: no mcdma-rpcd listen daemon serves that link (start `mcdma-rpcd listen` first)";
            return err;
        };
        errdefer p.box.close();
        p.reg = fabric.registration.Registration.attach(link) catch |err| {
            problem.* = "--mcdma-produce: the listen daemon refused the service (another holds the link, or it is not running)";
            return err;
        };
        errdefer if (p.reg) |*r| r.close();
        log.info("mcdma handoff: attached to the listen daemon on link {s}", .{link});
        p.serving.init(gpa, &p.box, fabric.words.native, &p.table, opts.model);
        p.serving.responder.tp_rank = opts.shard;
        p.serving.responder.tp_size = opts.shards;
        errdefer p.serving.deinit();
        p.thread = try std.Thread.spawn(.{}, run, .{p});
        log.info("mcdma handoff: exporting DeepSeek-V4.1 states as MKVH protocol 2 on link {s}", .{link});
        return p;
    }

    /// Stops serving and frees every export still held; the server's socket timeouts bound a read under way.
    pub fn stop(p: *Produce) void {
        if (p.thread) |t| {
            p.serving.stop.store(true, .release);
            t.join();
        }
        while (p.streams.load(.acquire) > 0 or p.building.load(.acquire) > 0) std.Io.sleep(p.io, .fromMilliseconds(1), .awake) catch {};
        p.pending.deinit(p.gpa);
        var it = p.held.valueIterator();
        while (it.next()) |h| release(p, h.*);
        p.held.deinit(p.gpa);
        if (p.thread != null) {
            if (p.reg) |*r| r.close();
            p.serving.deinit();
            p.box.close();
        }
        p.table.deinit();
        p.gpa.destroy(p);
    }

    fn run(p: *Produce) void {
        var done: std.ArrayList([]const u8) = .empty;
        defer done.deinit(p.gpa);
        var since: u64 = 0;
        while (!p.serving.stop.load(.acquire)) {
            p.table.expire() catch {}; // exports age out and are freed while the link is down too
            p.table.takeFinished(&done) catch {};
            for (done.items) |name| reap(p, name);
            done.clearRetainingCapacity();
            if (!p.attached(&since)) continue;
            // a new connection's HELLO zeroes the request word and its client counts from 1 again
            if (fabric.words.load(p.box.wordAt(fabric.layout.request_word)) == 0) p.serving.service.last = 0;
            if (p.serving.service.next(std.time.ns_per_ms) catch return) |req|
                p.serving.responder.handle(req.seq, req.payload) catch |err| p.serving.responder.failed(req.seq, err) catch return;
        }
    }

    /// Registered with the listen daemon; a dropped registration (the link went down) is retried each second.
    fn attached(p: *Produce, since: *u64) bool {
        if (p.reg) |*r| {
            if (r.alive()) return true;
            r.close();
            p.reg = null;
            log.warn("mcdma handoff: the listen daemon dropped the service on link {s}; attaching again", .{p.opts.link.?});
        }
        const now = fabric.words.nowNs();
        if (now - since.* < std.time.ns_per_s) {
            fabric.words.pause(std.time.ns_per_ms);
            return false;
        }
        since.* = now;
        p.reg = fabric.registration.Registration.attach(p.opts.link.?) catch return false;
        log.info("mcdma handoff: attached to the listen daemon on link {s}", .{p.opts.link.?});
        return true;
    }

    fn reap(p: *Produce, name: []const u8) void {
        var id: [16]u8 = undefined;
        _ = std.fmt.hexToBytes(&id, name) catch return; // a failed handoff's entry holds no export
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        const got = p.held.fetchRemove(id);
        const reading = if (got) |kv| kv.value.pins > 0 else false;
        if (reading) got.?.value.gone = true; // the last read to close frees it
        p.lock.unlock();
        if (p.thread != null) _ = p.serving.responder.checksums.remove(id); // CLOSE removes it; an expired export's would stay
        if (got) |kv| if (!reading) release(p, kv.value);
    }

    fn release(p: *Produce, h: *Held) void {
        if (h.feeder) |f| f.stop(f.ptr); // before its parts go: the converting thread writes them
        h.built.deinit(p.gpa); // the frames the responder planned at OPEN go with it
        p.gpa.destroy(h);
    }

    /// Frees exports past their time and those the decoder released (a link's serving thread runs this each ms).
    fn sweep(p: *Produce) void {
        var done: std.ArrayList([]const u8) = .empty;
        defer done.deinit(p.gpa);
        p.table.expire() catch {};
        p.table.takeFinished(&done) catch {};
        for (done.items) |name| reap(p, name);
    }

    /// The longest a reader waits for an export that is still converting.
    const read_wait_ms: u32 = 120_000;

    fn isPending(p: *Produce, id: [16]u8) bool {
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        defer p.lock.unlock();
        return p.pending.contains(id);
    }

    /// One decoder's read of a held export (`GET /v1/handoff/<id>`): replies to OPEN and PULL; pinned until `close`.
    pub const Reading = struct {
        p: *Produce,
        held: *Held,
        replies: *file.Replies,

        /// the stream's length in bytes, known before the first
        pub fn size(r: *const Reading) u64 {
            return r.replies.size;
        }

        pub fn next(r: *Reading) !?[]const u8 {
            return r.replies.next();
        }

        pub fn close(r: *Reading) void {
            const p = r.p;
            const held = r.held;
            r.replies.deinit();
            p.gpa.destroy(r);
            p.unlease(held);
        }
    };

    /// The held export under `id`. `NotHeld`: never held, released or expired. `Refused`: it failed (`why` says why).
    pub fn read(p: *Produce, id: [16]u8, why: *[]const u8) error{ NotHeld, Refused, OutOfMemory, Unreadable }!*Reading {
        p.sweep();
        // an export still converting on its own thread is awaited, not reported missing
        var waited_ms: u32 = 0;
        while (p.isPending(id) and waited_ms < read_wait_ms) : (waited_ms += 1) std.Io.sleep(p.io, .fromMilliseconds(1), .awake) catch {};
        const failed_why: ?[]const u8 = if (p.table.get(id)) |e| (if (e == .failed) e.failed else null) else null;
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        const held = p.held.get(id) orelse {
            p.lock.unlock();
            why.* = failed_why orelse return error.NotHeld;
            return error.Refused;
        };
        held.pins += 1;
        p.lock.unlock();
        _ = p.streams.fetchAdd(1, .acq_rel);
        errdefer p.unlease(held);
        const r = try p.gpa.create(Reading);
        errdefer p.gpa.destroy(r);
        const replies = file.Replies.open(p.gpa, &held.built.exp, p.opts.model, p.opts.shard, p.opts.shards) catch |err| return if (err == error.OutOfMemory) error.OutOfMemory else error.Unreadable;
        r.* = .{ .p = p, .held = held, .replies = replies };
        return r;
    }

    fn unlease(p: *Produce, h: *Held) void {
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        h.pins -= 1;
        const last = h.pins == 0 and h.gone;
        p.lock.unlock();
        if (last) release(p, h);
        _ = p.streams.fetchSub(1, .acq_rel);
    }

    /// The decoder is done with `id` (`DELETE /v1/handoff/<id>`): its export is released now. False when none was held.
    pub fn discard(p: *Produce, id: [16]u8) bool {
        if (p.table.get(id) == null) return false;
        p.table.finish(id) catch return false;
        p.sweep();
        return true;
    }

    /// The engine's state at `at` tokens, read back (the lane host's `snapshots.save` at the cut).
    pub fn save(p: *Produce, at: u64) !*Saved {
        const t0 = fabric.words.nowNs();
        const cap = try p.source.take(p.source.ptr, p.gpa, at);
        const s = p.gpa.create(Saved) catch |err| {
            p.source.release(cap.ctx);
            return err;
        };
        s.* = .{ .cap = cap, .read_ns = fabric.words.nowNs() - t0 };
        return s;
    }

    pub fn drop(p: *Produce, saved: *Saved) void {
        p.source.release(saved.cap.ctx);
        p.gpa.destroy(saved);
    }

    /// `saved` (now the producer's) is the state after `tokens`, the prompt but its last token: exported under `id`.
    pub fn exported(p: *Produce, id: [16]u8, tokens: []const u32, saved: *Saved) void {
        defer p.drop(saved);
        p.record(id, tokens, saved.cap.state, saved.read_ns) catch |err| p.failed(id, @errorName(err));
    }

    /// `exported` with the conversion on its own thread (a file-only producer or a full set of converters: inline).
    pub fn exportedBg(p: *Produce, id: [16]u8, tokens: []const u32, saved: *Saved) void {
        if (p.lease(id)) |h| return p.exportedStream(h, id, tokens, saved);
        if ((p.opts.link == null and !p.opts.http) or p.building.load(.acquire) >= max_building) return p.exported(id, tokens, saved);
        const own = p.gpa.dupe(u32, tokens) catch return p.exported(id, tokens, saved);
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        const put = p.pending.put(p.gpa, id, {});
        p.lock.unlock();
        put catch {
            p.gpa.free(own);
            return p.exported(id, tokens, saved);
        };
        _ = p.building.fetchAdd(1, .acq_rel);
        const t = std.Thread.spawn(.{}, convert, .{ p, id, own, saved }) catch {
            p.finishBuild(id);
            p.gpa.free(own);
            return p.exported(id, tokens, saved);
        };
        t.detach();
    }

    /// A streamed export under `id` (`cx.skeleton`'s): announced now, its frames served as `feeder` converts them.
    pub fn beginStream(p: *Produce, id: [16]u8, built: *cx.Built, feeder: Feeder) !void {
        if (p.opts.link == null) return error.NoLink;
        const held = try p.gpa.create(Held);
        errdefer p.gpa.destroy(held);
        held.* = .{ .built = built, .name = undefined, .feeder = feeder };
        _ = std.fmt.bufPrint(&held.name, "{x}", .{&id}) catch unreachable;
        try p.admit(id, held, false); // the engine never leases a refused export
    }

    /// The streamed export `id`, pinned (null: none), so a release waits for `unlease`.
    fn lease(p: *Produce, id: [16]u8) ?*Held {
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        defer p.lock.unlock();
        const h = p.held.get(id) orelse return null;
        if (h.feeder == null) return null;
        h.pins += 1;
        _ = p.streams.fetchAdd(1, .acq_rel); // as a reading: `stop` waits for it, `unlease` ends it
        return h;
    }

    /// The engine thread: the prompt of streamed export `id` stands at `pos`.
    pub fn feedStream(p: *Produce, id: [16]u8, pos: u64) void {
        const h = p.lease(id) orelse return;
        h.feeder.?.progress(h.feeder.?.ptr, pos);
        p.unlease(h);
    }

    /// Streamed export `id`: its queued segments done, and whether every paged part is converted (null: not streamed).
    pub fn drainStream(p: *Produce, id: [16]u8) ?bool {
        const h = p.lease(id) orelse return null;
        defer p.unlease(h);
        return h.feeder.?.drain(h.feeder.?.ptr);
    }

    /// The cut of streamed export `id`: the parts left converted from `saved`, and the export is whole.
    fn exportedStream(p: *Produce, h: *Held, id: [16]u8, tokens: []const u32, saved: *Saved) void {
        defer p.unlease(h);
        defer p.drop(saved);
        _ = h.feeder.?.drain(h.feeder.?.ptr); // no segment is converting while the cut converts what is left
        const t0 = fabric.words.nowNs();
        cx.complete(h.built, saved.cap.state) catch |err| return p.failed(id, @errorName(err));
        const ms = (fabric.words.nowNs() - t0) / std.time.ns_per_ms;
        log.info("handoff {x}: {d} tokens, {d} parts, {d} MiB of bf16 rows streamed; read back {d} ms, completed {d} ms", .{ &id, tokens.len, h.built.parts.len, h.built.bytes >> 20, saved.read_ns / std.time.ns_per_ms, ms });
        if (p.opts.dump != null) p.dump(h.built, tokens) catch |err| log.warn("handoff {x}: not written to a file ({t})", .{ &id, err });
    }

    /// Streamed export `h` (pinned by the caller) fails now: feeder stopped, bytes off the cap, OPEN/PULL told `why`.
    fn failStream(p: *Produce, h: *Held, id: [16]u8, why: []const u8) void {
        h.built.exp.progress.?.fail(why); // a PULL already past the table's lookup answers with the failure
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        const feeder = h.feeder;
        h.feeder = null; // no lease reaches it again, and `release` has nothing left to stop
        p.lock.unlock();
        if (feeder) |f| f.stop(f.ptr);
        p.table.fail(id, why) catch {}; // the serving thread frees the export once it reaps its request
    }

    fn convert(p: *Produce, id: [16]u8, tokens: []u32, saved: *Saved) void {
        p.exported(id, tokens, saved);
        p.gpa.free(tokens);
        p.finishBuild(id);
    }

    fn finishBuild(p: *Produce, id: [16]u8) void {
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        _ = p.pending.remove(id);
        p.lock.unlock();
        _ = p.building.fetchSub(1, .acq_rel);
    }

    fn record(p: *Produce, id: [16]u8, tokens: []const u32, st: cx.State, read_ns: u64) !void {
        if (p.opts.http) p.sweep(); // what expired or was released makes room under the cap
        const t0 = fabric.words.nowNs();
        const built = try cx.build(p.gpa, id, tokens, st);
        const build_ns = fabric.words.nowNs() - t0;
        var kept = false;
        defer if (!kept) built.deinit(p.gpa);
        var frames: u64 = 0;
        for (built.exp.layers) |l| frames += l.parts.len;
        log.info("handoff {x}: {d} tokens, {d} parts, {d} MiB of bf16 rows; read back {d} ms, built {d} ms", .{ &id, tokens.len, frames, built.bytes >> 20, read_ns / std.time.ns_per_ms, build_ns / std.time.ns_per_ms });
        if (p.opts.link == null and !p.opts.http) {
            if (p.opts.dump != null) try p.dump(built, tokens);
            return;
        }
        const held = try p.gpa.create(Held);
        errdefer p.gpa.destroy(held);
        held.* = .{ .built = built, .name = undefined };
        _ = std.fmt.bufPrint(&held.name, "{x}", .{&id}) catch unreachable;
        try p.admit(id, held, p.opts.dump != null);
        kept = true;
        if (p.opts.dump == null) return;
        // written once admitted, so a refused export leaves no file; the pin keeps it while it is written
        defer p.unlease(held);
        p.dump(built, tokens) catch |err| log.warn("handoff {x}: not written to a file ({t})", .{ &id, err });
    }

    /// Admitted under the cap and published in one step: no reader pins a refused export. `pinned`: one for the caller.
    fn admit(p: *Produce, id: [16]u8, held: *Held, pinned: bool) !void {
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        defer p.lock.unlock();
        if (p.held.contains(id)) return error.HandoffHeld; // a reused id: the export already held stays
        try p.held.ensureUnusedCapacity(p.gpa, 1);
        try p.table.addSized(id, &held.name, .{ .ready = &held.built.exp }, held.built.bytes);
        p.held.putAssumeCapacityNoClobber(id, held);
        if (!pinned) return;
        held.pins += 1;
        _ = p.streams.fetchAdd(1, .acq_rel); // as a reading: `stop` waits for the file
    }

    /// `<dir>/<id>.mkvh2`, written whole under a `.part` name and renamed after `<id>.tokens.json`: present = complete.
    fn dump(p: *Produce, built: *cx.Built, tokens: []const u32) !void {
        const dir = p.opts.dump.?;
        const cwd = std.Io.Dir.cwd();
        var name_buf: [32 + file.extension.len]u8 = undefined;
        const name = file.fileName(&name_buf, built.exp.handoff);
        const final = try std.fs.path.join(p.gpa, &.{ dir, name });
        defer p.gpa.free(final);
        const part = try std.fmt.allocPrint(p.gpa, "{s}.part", .{final});
        defer p.gpa.free(part);
        const ids = try std.fmt.allocPrint(p.gpa, "{s}/{x}.tokens.json", .{ dir, &built.exp.handoff });
        defer p.gpa.free(ids);
        var json: std.Io.Writer.Allocating = .init(p.gpa);
        defer json.deinit();
        try std.json.Stringify.value(tokens, .{}, &json.writer);
        // an earlier export under a reused id keeps its files; a failed attempt removes only what it wrote
        if (cwd.access(p.io, final, .{})) |_| return error.HandoffFileExists else |err| if (err != error.FileNotFound) return err;
        const t = cwd.createFile(p.io, ids, .{ .exclusive = true }) catch |err| return if (err == error.PathAlreadyExists) error.HandoffFileExists else err;
        errdefer cwd.deleteFile(p.io, ids) catch {};
        {
            defer t.close(p.io);
            try t.writeStreamingAll(p.io, json.written());
        }
        try p.faultAt(.create);
        const f = try cwd.createFile(p.io, part, .{});
        errdefer cwd.deleteFile(p.io, part) catch {};
        {
            defer f.close(p.io);
            var buf: [64 << 10]u8 = undefined;
            var w = f.writerStreaming(p.io, &buf);
            try p.faultAt(.write);
            try file.write(p.gpa, &w.interface, &built.exp, p.opts.model, p.opts.shard, p.opts.shards);
            try p.faultAt(.flush);
            try w.interface.flush();
        }
        try p.faultAt(.rename);
        try cwd.rename(part, cwd, final, p.io);
        log.info("handoff {x}: written to {s} (model {s})", .{ &built.exp.handoff, final, p.opts.model });
    }

    fn faultAt(p: *const Produce, step: DumpStep) error{DumpFault}!void {
        if (p.dump_fault == step) return error.DumpFault;
    }

    /// No state will come for `id`: the decoder is told why (a link), and the reason lands in `<id>.error` (a dump).
    pub fn failed(p: *Produce, id: [16]u8, reason: []const u8) void {
        const why = if (std.mem.eql(u8, reason, "HeldExportsFull")) "the producer already holds TF_PRODUCE_HELD_GIB of exports; it refuses more until the decoder closes some" else reason;
        log.warn("mcdma handoff {x}: no export ({s})", .{ &id, why });
        if (p.opts.dump) |dir| {
            if (std.fmt.allocPrint(p.gpa, "{s}/{x}.error", .{ dir, &id })) |path| {
                defer p.gpa.free(path);
                std.Io.Dir.cwd().writeFile(p.io, .{ .sub_path = path, .data = why }) catch {};
            } else |_| {}
        }
        if (p.lease(id)) |h| {
            defer p.unlease(h);
            return p.failStream(h, id, why);
        }
        if (p.opts.link != null or p.opts.http) p.table.add(id, "failed", .{ .failed = why }) catch {};
    }
};
