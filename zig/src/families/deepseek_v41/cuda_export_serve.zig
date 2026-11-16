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
    /// serve held exports over the server's own HTTP port (`--mcdma-produce http`), same cap and expiry as a link
    http: bool = false,
};

/// What `save` hands `exported`: the engine's state, read back.
pub const Saved = struct { cap: cx.Captured };

/// One export the producer holds; a reading stream pins it, so expiry or release only marks it `gone` until it closes.
pub const Held = struct { built: *cx.Built, name: [32]u8, pins: u32 = 0, gone: bool = false };

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
        while (p.streams.load(.acquire) > 0) std.Io.sleep(p.io, .fromMilliseconds(1), .awake) catch {};
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
        const replies = file.Replies.open(p.gpa, &held.built.exp, p.opts.model, 0, 1) catch |err| return if (err == error.OutOfMemory) error.OutOfMemory else error.Unreadable;
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
        const cap = try p.source.take(p.source.ptr, p.gpa, at);
        const s = p.gpa.create(Saved) catch |err| {
            p.source.release(cap.ctx);
            return err;
        };
        s.* = .{ .cap = cap };
        return s;
    }

    pub fn drop(p: *Produce, saved: *Saved) void {
        p.source.release(saved.cap.ctx);
        p.gpa.destroy(saved);
    }

    /// `saved` (now the producer's) is the state after `tokens`, the prompt but its last token: exported under `id`.
    pub fn exported(p: *Produce, id: [16]u8, tokens: []const u32, saved: *Saved) void {
        defer p.drop(saved);
        p.record(id, tokens, saved.cap.state) catch |err| p.failed(id, @errorName(err));
    }

    fn record(p: *Produce, id: [16]u8, tokens: []const u32, st: cx.State) !void {
        if (p.opts.http) p.sweep(); // what expired or was released makes room under the cap
        const built = try cx.build(p.gpa, id, tokens, st);
        var kept = false;
        defer if (!kept) built.deinit(p.gpa);
        var frames: u64 = 0;
        for (built.exp.layers) |l| frames += l.parts.len;
        log.info("handoff {x}: {d} tokens, {d} parts, {d} MiB of bf16 rows", .{ &id, tokens.len, frames, built.bytes >> 20 });
        if (p.opts.dump != null) try p.dump(built, tokens);
        if (p.opts.link == null and !p.opts.http) return;
        const held = try p.gpa.create(Held);
        errdefer p.gpa.destroy(held);
        held.* = .{ .built = built, .name = undefined };
        _ = std.fmt.bufPrint(&held.name, "{x}", .{&id}) catch unreachable;
        {
            while (!p.lock.tryLock()) std.atomic.spinLoopHint();
            defer p.lock.unlock();
            const slot = try p.held.getOrPut(p.gpa, id);
            if (slot.found_existing) return error.HandoffHeld; // a reused id: the export already held stays
            slot.value_ptr.* = held;
        }
        p.table.addSized(id, &held.name, .{ .ready = &built.exp }, built.bytes) catch |err| {
            while (!p.lock.tryLock()) std.atomic.spinLoopHint();
            _ = p.held.remove(id);
            p.lock.unlock();
            return err;
        };
        kept = true;
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
        {
            var json: std.Io.Writer.Allocating = .init(p.gpa);
            defer json.deinit();
            try std.json.Stringify.value(tokens, .{}, &json.writer);
            try cwd.writeFile(p.io, .{ .sub_path = ids, .data = json.written() });
        }
        {
            var f = try cwd.createFile(p.io, part, .{});
            defer f.close(p.io);
            var buf: [64 << 10]u8 = undefined;
            var w = f.writerStreaming(p.io, &buf);
            try file.write(p.gpa, &w.interface, &built.exp, p.opts.model, 0, 1);
            try w.interface.flush();
        }
        try cwd.rename(part, cwd, final, p.io);
        log.info("handoff {x}: written to {s} (model {s})", .{ &built.exp.handoff, final, p.opts.model });
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
        if (p.opts.link != null or p.opts.http) p.table.add(id, "failed", .{ .failed = why }) catch {};
    }
};
