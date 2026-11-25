//! Native zero-dependency Zig OpenAI HTTP/SSE & `/v1/stream-parallel` server for TensorFold mlx_tf.
//!
//! Endpoints:
//!   GET  /health
//!   GET  /v1/models
//!   POST /v1/chat/completions  (streaming SSE + non-streaming JSON, stop sequences, penalties, stream_options)
//!   POST /v1/completions       (streaming SSE + non-streaming JSON, stop sequences, penalties, stream_options)
//!   POST /v1/stream-parallel   (zero-copy prefix-forked lockstep multi-stream wave, SSE + JSON)
const std = @import("std");
const mtl = @import("metal4.zig");
const model_mod = @import("model.zig");
const tok_mod = @import("tokenizer.zig");
const spec_mod = @import("speculate.zig");
const draft_mod = @import("drafter.zig");
const cross_mod = @import("cross_draft.zig");
const samp_mod = @import("sampler.zig");
const batch_mod = @import("batch.zig");
const pcache_mod = @import("prefix_cache.zig");
const vision_mod = @import("vision.zig");

const Model = model_mod.Model;
const Tokenizer = tok_mod.Tokenizer;
const nowMs = model_mod.nowMs;

// Minimal POSIX socket declarations (macOS arm64 ABI)
const SockAddrIn = extern struct {
    sin_len: u8 = @sizeOf(SockAddrIn),
    sin_family: u8 = 2, // AF_INET
    sin_port: u16,
    sin_addr: u32,
    sin_zero: [8]u8 = @splat(0),
};

extern "c" fn socket(domain: c_int, sock_type: c_int, protocol: c_int) c_int;
extern "c" fn setsockopt(fd: c_int, level: c_int, optname: c_int, optval: *const anyopaque, optlen: u32) c_int;
extern "c" fn bind(fd: c_int, addr: *const SockAddrIn, len: u32) c_int;
extern "c" fn listen(fd: c_int, backlog: c_int) c_int;
extern "c" fn accept(fd: c_int, addr: ?*anyopaque, len: ?*u32) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, count: usize) isize;
extern "c" fn write(fd: c_int, buf: [*]const u8, count: usize) isize;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn signal(sig: c_int, handler: usize) usize;

const AF_INET: c_int = 2;
const SOCK_STREAM: c_int = 1;
const SOL_SOCKET: c_int = 0xffff;
const SO_REUSEADDR: c_int = 0x0004;
const SO_NOSIGPIPE: c_int = 0x1022;
const IPPROTO_TCP: c_int = 6;
const TCP_NODELAY: c_int = 1;
const SIGPIPE: c_int = 13;
const SIG_IGN: usize = 1;

pub const DraftKind = enum { none, dflash2, cross_model };

const StopMatcher = struct {
    stops: [8][]const u8 = undefined,
    count: usize = 0,
    max_len: usize = 0,

    fn parse(obj: std.json.ObjectMap) StopMatcher {
        var sm = StopMatcher{};
        const sv = obj.get("stop") orelse return sm;
        switch (sv) {
            .string => |s| {
                if (s.len > 0) {
                    sm.stops[0] = s;
                    sm.count = 1;
                    sm.max_len = s.len;
                }
            },
            .array => |arr| {
                for (arr.items) |item| {
                    if (sm.count >= 8) break;
                    if (item == .string and item.string.len > 0) {
                        sm.stops[sm.count] = item.string;
                        sm.count += 1;
                        if (item.string.len > sm.max_len) sm.max_len = item.string.len;
                    }
                }
            },
            else => {},
        }
        return sm;
    }

    /// Return earliest index where any stop sequence occurs in `text`, or null.
    fn findEarliest(self: StopMatcher, text: []const u8) ?usize {
        var best: ?usize = null;
        for (self.stops[0..self.count]) |st| {
            if (std.mem.indexOf(u8, text, st)) |pos| {
                if (best == null or pos < best.?) best = pos;
            }
        }
        return best;
    }
};

pub const ServerEngine = struct {
    a: std.mem.Allocator,
    ctx: *mtl.Context,
    model: *Model,
    tok: *Tokenizer,
    model_name: []const u8,
    draft_kind: DraftKind = .none,
    dflash_dec: ?*draft_mod.HybridDFlashDecoder = null,
    cross_dec: ?*cross_mod.CrossModelDecoder = null,
    copy_dec: spec_mod.AdaptiveSpecDecoder,
    prefix_cache: ?pcache_mod.HybridPrefixCache = null,
    fork_engine: ?batch_mod.ForkBatchEngine = null,
    vision: ?*vision_mod.VisionEncoder = null,
    max_streams: usize = 16,
    max_suffix: u32 = 2048,

    pub fn init(
        a: std.mem.Allocator,
        ctx: *mtl.Context,
        model: *Model,
        tok: *Tokenizer,
        model_name: []const u8,
        dflash_dec: ?*draft_mod.HybridDFlashDecoder,
        cross_dec: ?*cross_mod.CrossModelDecoder,
        vision: ?*vision_mod.VisionEncoder,
        max_streams: usize,
    ) !ServerEngine {
        var im_end_id: u32 = 0;
        if (tok.encode(a, tok.turnEndToken())) |ids| {
            defer a.free(ids);
            if (ids.len == 1) im_end_id = ids[0];
        } else |_| {}

        const cache_slots: usize = if (model.w.total_bytes > 8 * 1024 * 1024 * 1024) 4 else 8;
        const pcache: ?pcache_mod.HybridPrefixCache = if (model.c.arch == .nemotron_h)
            null
        else
            (pcache_mod.HybridPrefixCache.init(a, model, cache_slots, im_end_id) catch null);

        return .{
            .a = a,
            .ctx = ctx,
            .model = model,
            .tok = tok,
            .model_name = model_name,
            .draft_kind = if (dflash_dec != null) .dflash2 else if (cross_dec != null) .cross_model else .none,
            .dflash_dec = dflash_dec,
            .cross_dec = cross_dec,
            .copy_dec = try spec_mod.AdaptiveSpecDecoder.init(a, 127),
            .prefix_cache = pcache,
            .vision = vision,
            .max_streams = @max(1, @min(max_streams, batch_mod.max_streams)),
        };
    }

    pub fn deinit(self: *ServerEngine) void {
        if (self.prefix_cache) |*pc| pc.deinit();
        if (self.fork_engine) |*fe| fe.deinit();
        self.copy_dec.deinit();
    }

    fn ensureForkEngine(self: *ServerEngine) !*batch_mod.ForkBatchEngine {
        if (self.fork_engine == null) {
            self.fork_engine = try batch_mod.ForkBatchEngine.init(self.a, self.model, self.max_streams, self.max_suffix);
        }
        return &self.fork_engine.?;
    }

    pub fn serve(self: *ServerEngine, port: u16) !void {
        return self.serveHost("127.0.0.1", port);
    }

    pub fn serveHost(self: *ServerEngine, host: []const u8, port: u16) !void {
        _ = signal(SIGPIPE, SIG_IGN);

        const srv_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (srv_fd < 0) return error.SocketFailed;
        defer _ = close(srv_fd);

        const one: c_int = 1;
        _ = setsockopt(srv_fd, SOL_SOCKET, SO_REUSEADDR, &one, @sizeOf(c_int));

        const ipv4 = parseIpv4(host) orelse 0x7F000001;
        const addr = SockAddrIn{
            .sin_port = std.mem.nativeToBig(u16, port),
            .sin_addr = std.mem.nativeToBig(u32, ipv4),
        };
        if (bind(srv_fd, &addr, @sizeOf(SockAddrIn)) < 0) return error.BindFailed;
        if (listen(srv_fd, 64) < 0) return error.ListenFailed;

        std.debug.print("TensorFold mlx_tf HTTP server listening on http://{s}:{d} (model={s}, arch={s}, chat={s}, vision={}, draft={s}, m5_tensor_ops={}, max_context={d}, max_streams={d})\n", .{
            host, port, self.model_name, @tagName(self.model.c.arch), @tagName(self.tok.chat_style), self.vision != null, @tagName(self.draft_kind), self.ctx.has_tensor_ops, self.model.max_context, self.max_streams,
        });

        while (true) {
            const client_fd = accept(srv_fd, null, null);
            if (client_fd < 0) continue;
            _ = setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, @sizeOf(c_int));
            _ = setsockopt(client_fd, SOL_SOCKET, SO_NOSIGPIPE, &one, @sizeOf(c_int));
            self.handleClient(client_fd) catch |err| {
                std.debug.print("HTTP request error: {}\n", .{err});
                sendError(client_fd, 500, "internal_error", "internal server error");
            };
            _ = close(client_fd);
        }
    }

    fn handleClient(self: *ServerEngine, fd: c_int) !void {
        var req_buf: std.ArrayList(u8) = .empty;
        defer req_buf.deinit(self.a);

        var chunk: [8192]u8 = undefined;
        var header_end: ?usize = null;
        var content_len: usize = 0;

        while (true) {
            const n = read(fd, &chunk, chunk.len);
            if (n <= 0) break;
            try req_buf.appendSlice(self.a, chunk[0..@intCast(n)]);
            if (header_end == null) {
                if (std.mem.indexOf(u8, req_buf.items, "\r\n\r\n")) |pos| {
                    header_end = pos + 4;
                    content_len = parseContentLength(req_buf.items[0..pos]);
                }
            }
            if (header_end) |he| {
                if (req_buf.items.len >= he + content_len) break;
            }
            if (req_buf.items.len > 16 * 1024 * 1024) return error.RequestTooLarge;
        }

        const he = header_end orelse return;
        const header_text = req_buf.items[0..he];
        const body = req_buf.items[he..@min(req_buf.items.len, he + content_len)];

        const line_end = std.mem.indexOf(u8, header_text, "\r\n") orelse return;
        const first_line = header_text[0..line_end];
        var parts = std.mem.splitScalar(u8, first_line, ' ');
        const method = parts.next() orelse return;
        const raw_path = parts.next() orelse return;
        const path = if (std.mem.indexOfScalar(u8, raw_path, '?')) |q| raw_path[0..q] else raw_path;

        if (std.mem.eql(u8, method, "OPTIONS")) {
            _ = writeAll(fd, "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nConnection: close\r\n\r\n");
            return;
        }

        if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/health")) {
            const txt = try std.fmt.allocPrint(self.a, "{{\"status\":\"ok\",\"engine\":\"tensorfold-mlx_tf\",\"version\":\"0.4.0\",\"device\":\"{s}\",\"arch\":\"{s}\",\"chat_style\":\"{s}\",\"vision\":{},\"m5_tensor_ops\":{},\"model\":\"{s}\",\"drafter\":\"{s}\",\"max_context\":{d},\"max_streams\":{d}}}", .{
                self.ctx.deviceName(), @tagName(self.model.c.arch), @tagName(self.tok.chat_style), self.vision != null, self.ctx.has_tensor_ops, self.model_name, @tagName(self.draft_kind), self.model.max_context, self.max_streams,
            });
            defer self.a.free(txt);
            sendJson(fd, 200, txt);
            return;
        }

        if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/v1/models")) {
            const txt = try std.fmt.allocPrint(self.a, "{{\"object\":\"list\",\"data\":[{{\"id\":\"{s}\",\"object\":\"model\",\"owned_by\":\"tensorfold\",\"arch\":\"{s}\",\"vision\":{},\"context_length\":{d}}}]}}", .{
                self.model_name, @tagName(self.model.c.arch), self.vision != null, self.model.max_context,
            });
            defer self.a.free(txt);
            sendJson(fd, 200, txt);
            return;
        }

        if (std.mem.eql(u8, method, "POST") and (std.mem.eql(u8, path, "/v1/chat/completions") or std.mem.eql(u8, path, "/v1/completions"))) {
            const is_chat = std.mem.eql(u8, path, "/v1/chat/completions");
            try self.handleCompletion(fd, body, is_chat);
            return;
        }

        if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/stream-parallel")) {
            if (self.model.c.arch == .nemotron_h) {
                sendError(fd, 400, "invalid_request_error", "stream-parallel is currently supported on hybrid GDN and dense architectures");
                return;
            }
            try self.handleStreamParallel(fd, body);
            return;
        }

        sendError(fd, 404, "not_found_error", "endpoint not found");
    }

    fn handleCompletion(self: *ServerEngine, fd: c_int, body: []const u8, is_chat: bool) !void {
        const parsed = try std.json.parseFromSlice(std.json.Value, self.a, body, .{});
        defer parsed.deinit();
        const obj = parsed.value.object;

        const max_tokens: usize = getJsonUsize(obj, "max_tokens", getJsonUsize(obj, "max_completion_tokens", 256));
        const stream = getJsonBool(obj, "stream", false);
        var include_usage = true;
        if (obj.get("stream_options")) |so| {
            if (so == .object) include_usage = getJsonBool(so.object, "include_usage", true);
        }
        var enable_thinking = false;
        if (obj.get("chat_template_kwargs")) |ctk| {
            if (ctk == .object) enable_thinking = getJsonBool(ctk.object, "enable_thinking", false);
        }
        const rep_pen = getJsonF32(obj, "repetition_penalty", 1.0);
        const freq_pen = getJsonF32(obj, "frequency_penalty", 0.0);
        const pres_pen = getJsonF32(obj, "presence_penalty", 0.0);
        const has_penalties = (rep_pen != 1.0 and rep_pen > 0.0) or freq_pen != 0.0 or pres_pen != 0.0;

        const sampling = samp_mod.SamplingParams{
            .temperature = getJsonF32(obj, "temperature", 0.0),
            .top_p = getJsonF32(obj, "top_p", 1.0),
            .top_k = @intCast(getJsonUsize(obj, "top_k", 0)),
            .min_p = getJsonF32(obj, "min_p", 0.0),
            .seed = @intCast(getJsonUsize(obj, "seed", 42)),
            .repetition_penalty = if (rep_pen > 0.0) rep_pen else 1.0,
            .frequency_penalty = freq_pen,
            .presence_penalty = pres_pen,
        };
        const stop_matcher = StopMatcher.parse(obj);

        var all_vis_embs: std.ArrayList(u16) = .empty;
        defer all_vis_embs.deinit(self.a);
        var has_images = false;

        const prompt_ids: []u32 = if (is_chat) blk: {
            const msgs_val = obj.get("messages") orelse return sendJson(fd, 400, "{\"error\":\"missing messages\"}");
            // Check if any message contains multimodal content arrays with image_url/image parts
            for (msgs_val.array.items) |item| {
                if (item.object.get("content")) |cv| {
                    if (cv == .array) {
                        for (cv.array.items) |part| {
                            if (part == .object) {
                                const pt = if (part.object.get("type")) |tv| (if (tv == .string) tv.string else "") else "";
                                if (std.mem.eql(u8, pt, "image_url") or std.mem.eql(u8, pt, "image")) {
                                    has_images = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if (!has_images) {
                var msgs = try self.a.alloc(Tokenizer.ChatMessage, msgs_val.array.items.len);
                defer self.a.free(msgs);
                var tmp_strs: std.ArrayList([]u8) = .empty;
                defer {
                    for (tmp_strs.items) |s| self.a.free(s);
                    tmp_strs.deinit(self.a);
                }
                for (msgs_val.array.items, 0..) |item, i| {
                    const mo = item.object;
                    const role = mo.get("role").?.string;
                    var content: []const u8 = "";
                    if (mo.get("content")) |cv| {
                        if (cv == .string) {
                            content = cv.string;
                        } else if (cv == .array) {
                            var txt_acc: std.ArrayList(u8) = .empty;
                            errdefer txt_acc.deinit(self.a);
                            for (cv.array.items) |part| {
                                if (part == .object) {
                                    if (part.object.get("text")) |tv| {
                                        if (tv == .string) try txt_acc.appendSlice(self.a, tv.string);
                                    }
                                }
                            }
                            const owned = try txt_acc.toOwnedSlice(self.a);
                            try tmp_strs.append(self.a, owned);
                            content = owned;
                        }
                    }
                    msgs[i] = .{ .role = role, .content = content };
                }
                const rendered = try self.tok.formatChatStyleMessages(self.a, msgs, true, enable_thinking);
                defer self.a.free(rendered);
                break :blk try self.tok.encode(self.a, rendered);
            }

            // Multimodal chat prompt construction (Qwen3.5-VL & Gemma 3-VL)
            const ve = self.vision orelse return sendError(fd, 400, "invalid_request_error", "model does not include a vision encoder");
            var ids_acc: std.ArrayList(u32) = .empty;
            errdefer ids_acc.deinit(self.a);
            const is_gemma = (ve.c.kind == .gemma3_siglip);
            if (is_gemma) {
                const bos = try self.tok.encode(self.a, "<bos>");
                defer self.a.free(bos);
                try ids_acc.appendSlice(self.a, bos);
            }
            for (msgs_val.array.items) |item| {
                const mo = item.object;
                const role = mo.get("role").?.string;
                const hdr = if (is_gemma)
                    try std.fmt.allocPrint(self.a, "<start_of_turn>{s}\n", .{if (std.mem.eql(u8, role, "assistant")) "model" else role})
                else
                    try std.fmt.allocPrint(self.a, "<|im_start|>{s}\n", .{role});
                defer self.a.free(hdr);
                const h_ids = try self.tok.encode(self.a, hdr);
                defer self.a.free(h_ids);
                try ids_acc.appendSlice(self.a, h_ids);

                if (mo.get("content")) |cv| {
                    if (cv == .string) {
                        const t_ids = try self.tok.encode(self.a, cv.string);
                        defer self.a.free(t_ids);
                        try ids_acc.appendSlice(self.a, t_ids);
                    } else if (cv == .array) {
                        for (cv.array.items) |part| {
                            if (part != .object) continue;
                            const po = part.object;
                            const pt = if (po.get("type")) |tv| (if (tv == .string) tv.string else "") else "";
                            if (std.mem.eql(u8, pt, "image_url") or std.mem.eql(u8, pt, "image")) {
                                var url_opt: ?[]const u8 = null;
                                if (po.get("image_url")) |iu| {
                                    if (iu == .object) {
                                        if (iu.object.get("url")) |uv| if (uv == .string) {
                                            url_opt = uv.string;
                                        };
                                    } else if (iu == .string) {
                                        url_opt = iu.string;
                                    }
                                } else if (po.get("image")) |iv| {
                                    if (iv == .string) url_opt = iv.string;
                                }
                                if (url_opt) |url| {
                                    const enc_img = try ve.encodeImage(self.a, url);
                                    defer enc_img.deinit(self.a);
                                    try all_vis_embs.appendSlice(self.a, enc_img.embs);
                                    try ids_acc.append(self.a, ve.c.vision_start_token_id);
                                    for (0..enc_img.num_tokens) |_| try ids_acc.append(self.a, ve.c.image_token_id);
                                    try ids_acc.append(self.a, ve.c.vision_end_token_id);
                                }
                            } else if (po.get("text")) |tv| {
                                if (tv == .string and tv.string.len > 0) {
                                    const t_ids = try self.tok.encode(self.a, tv.string);
                                    defer self.a.free(t_ids);
                                    try ids_acc.appendSlice(self.a, t_ids);
                                }
                            }
                        }
                    }
                }
                const ftr = if (is_gemma) "<end_of_turn>\n" else "<|im_end|>\n";
                const f_ids = try self.tok.encode(self.a, ftr);
                defer self.a.free(f_ids);
                try ids_acc.appendSlice(self.a, f_ids);
            }
            const gen_hdr = if (is_gemma)
                "<start_of_turn>model\n"
            else if (enable_thinking)
                "<|im_start|>assistant\n<think>\n"
            else
                "<|im_start|>assistant\n<think>\n\n</think>\n\n";
            const g_ids = try self.tok.encode(self.a, gen_hdr);
            defer self.a.free(g_ids);
            try ids_acc.appendSlice(self.a, g_ids);
            break :blk try ids_acc.toOwnedSlice(self.a);
        } else blk: {
            const p = obj.get("prompt") orelse return sendError(fd, 400, "invalid_request_error", "missing prompt");
            break :blk try self.tok.encode(self.a, p.string);
        };
        defer self.a.free(prompt_ids);

        if (prompt_ids.len == 0) {
            sendError(fd, 400, "invalid_request_error", "empty prompt");
            return;
        }
        if (prompt_ids.len >= self.model.max_context) {
            const msg = try std.fmt.allocPrint(self.a, "prompt length ({d} tokens) exceeds max_context ({d})", .{ prompt_ids.len, self.model.max_context });
            defer self.a.free(msg);
            sendError(fd, 400, "invalid_request_error", msg);
            return;
        }
        const avail_tokens: usize = @as(usize, self.model.max_context) - prompt_ids.len;
        const max_tokens_clamped: usize = @min(max_tokens, avail_tokens);

        const SseCtx = struct {
            a: std.mem.Allocator,
            fd: c_int,
            tok: *Tokenizer,
            model_name: []const u8,
            is_chat: bool,
            stops: StopMatcher,
            pending: std.ArrayList(u8) = .empty,
            hit_stop: bool = false,

            fn deinit(s: *@This()) void {
                s.pending.deinit(s.a);
            }

            fn writeDelta(s: *@This(), text: []const u8) void {
                if (text.len == 0) return;
                var buf: std.ArrayList(u8) = .empty;
                defer buf.deinit(s.a);
                if (s.is_chat) {
                    buf.appendSlice(s.a, "data: {\"id\":\"mlx_tf\",\"object\":\"chat.completion.chunk\",\"model\":\"") catch return;
                    appendJsonEscaped(s.a, &buf, s.model_name) catch return;
                    buf.appendSlice(s.a, "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"") catch return;
                    appendJsonEscaped(s.a, &buf, text) catch return;
                    buf.appendSlice(s.a, "\"},\"finish_reason\":null}]}\n\n") catch return;
                } else {
                    buf.appendSlice(s.a, "data: {\"id\":\"mlx_tf\",\"object\":\"text_completion\",\"model\":\"") catch return;
                    appendJsonEscaped(s.a, &buf, s.model_name) catch return;
                    buf.appendSlice(s.a, "\",\"choices\":[{\"index\":0,\"text\":\"") catch return;
                    appendJsonEscaped(s.a, &buf, text) catch return;
                    buf.appendSlice(s.a, "\",\"finish_reason\":null}]}\n\n") catch return;
                }
                if (!writeAll(s.fd, buf.items)) s.hit_stop = true;
            }

            fn emit(raw: ?*anyopaque, tok_id: u32) void {
                const s: *@This() = @ptrCast(@alignCast(raw.?));
                if (s.hit_stop) return;
                const piece = s.tok.decodeToken(tok_id);
                if (s.stops.count == 0) {
                    s.writeDelta(piece);
                    return;
                }
                s.pending.appendSlice(s.a, piece) catch return;
                if (s.stops.findEarliest(s.pending.items)) |pos| {
                    s.writeDelta(s.pending.items[0..pos]);
                    s.pending.clearRetainingCapacity();
                    s.hit_stop = true;
                    return;
                }
                const hold = s.stops.max_len - 1;
                if (s.pending.items.len > hold) {
                    const flush_n = s.pending.items.len - hold;
                    s.writeDelta(s.pending.items[0..flush_n]);
                    std.mem.copyForwards(u8, s.pending.items[0..hold], s.pending.items[flush_n..]);
                    s.pending.items.len = hold;
                }
            }

            fn flushRemainder(s: *@This()) void {
                if (!s.hit_stop and s.pending.items.len > 0) {
                    s.writeDelta(s.pending.items);
                    s.pending.clearRetainingCapacity();
                }
            }
        };

        var sse_ctx = SseCtx{ .a = self.a, .fd = fd, .tok = self.tok, .model_name = self.model_name, .is_chat = is_chat, .stops = stop_matcher };
        defer sse_ctx.deinit();
        if (stream) {
            _ = writeAll(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
        }

        var out_ids: std.ArrayList(u32) = .empty;
        defer out_ids.deinit(self.a);
        const cb: ?*const fn (?*anyopaque, u32) void = if (stream) SseCtx.emit else null;

        var stats = spec_mod.SpecStats{};
        var cached_tokens: usize = 0;
        var hit_stop_seq = false;

        // Use token-by-token loop whenever images, stop sequences, penalties, Nemotron-H, or sampling without cross-draft are active
        const use_direct_loop = has_images or (self.model.c.arch == .nemotron_h) or has_penalties or (stop_matcher.count > 0) or
            (sampling.temperature > 1e-6 and self.cross_dec == null);

        if (!use_direct_loop and sampling.temperature <= 1e-6) {
            if (self.dflash_dec) |dd| {
                if (self.prefix_cache) |*pc| pc.invalidate();
                stats = try dd.generate(prompt_ids, max_tokens_clamped, true, &out_ids, self.a, cb, &sse_ctx);
            } else if (self.cross_dec) |cd| {
                if (self.prefix_cache) |*pc| pc.invalidate();
                stats = try cd.generate(prompt_ids, max_tokens_clamped, true, &out_ids, self.a, cb, &sse_ctx);
            } else if (self.prefix_cache) |*pc| {
                const pr = try pc.prefillCached(prompt_ids, 128, false);
                cached_tokens = pr.cached_tokens;
                stats = try self.copy_dec.generateFromPrefilled(self.model, prompt_ids, pr.next_tok, max_tokens_clamped, true, &out_ids, self.a, cb, &sse_ctx);
                try pc.commitGenerated(out_ids.items, 0);
            } else {
                self.model.reset();
                stats = try self.copy_dec.generate(self.model, prompt_ids, max_tokens_clamped, true, &out_ids, self.a, cb, &sse_ctx);
            }
        } else if (!use_direct_loop and self.cross_dec != null) {
            if (self.prefix_cache) |*pc| pc.invalidate();
            stats = try self.cross_dec.?.generateSampled(prompt_ids, max_tokens_clamped, true, sampling, &out_ids, self.a, cb, &sse_ctx);
        } else {
            // Direct prefill + token loop supporting penalties, stop sequences, sampling, Nemotron-H, and prefix cache!
            var rng = samp_mod.Rng.init(sampling.seed);
            const need_logits = (sampling.temperature > 1e-6) or has_penalties;
            var counts = std.AutoHashMap(u32, u32).init(self.a);
            defer counts.deinit();
            if (has_penalties) {
                for (prompt_ids) |pid| {
                    const gop = try counts.getOrPut(pid);
                    gop.value_ptr.* = if (gop.found_existing) gop.value_ptr.* + 1 else 1;
                }
            }

            var next_tok: u32 = 0;
            if (has_images) {
                if (self.prefix_cache) |*pc| pc.invalidate();
                self.model.reset();
                next_tok = try self.model.prefillWithEmbeddings(prompt_ids, all_vis_embs.items, self.vision.?.c.image_token_id, 128);
            } else if (self.prefix_cache) |*pc| {
                const pr = try pc.prefillCached(prompt_ids, 128, need_logits);
                cached_tokens = pr.cached_tokens;
                next_tok = pr.next_tok;
            } else {
                self.model.reset();
                next_tok = try self.model.prefill(prompt_ids, if (self.model.c.arch == .nemotron_h) 64 else 128);
            }
            if (need_logits) {
                next_tok = try self.model.sampleRowWithPenalties(0, sampling, &rng, if (has_penalties) &counts else null);
            }

            var running_text: std.ArrayList(u8) = .empty;
            defer running_text.deinit(self.a);
            const td0 = nowMs();
            while (stats.tokens_generated < max_tokens_clamped) {
                if (self.model.c.isEos(next_tok)) break;
                if (has_penalties) {
                    const gop = try counts.getOrPut(next_tok);
                    gop.value_ptr.* = if (gop.found_existing) gop.value_ptr.* + 1 else 1;
                }
                try out_ids.append(self.a, next_tok);
                if (cb) |f| f(&sse_ctx, next_tok);
                stats.tokens_generated += 1;
                stats.rounds += 1;
                if (sse_ctx.hit_stop) {
                    hit_stop_seq = true;
                    break;
                }
                if (!stream and stop_matcher.count > 0) {
                    try running_text.appendSlice(self.a, self.tok.decodeToken(next_tok));
                    if (stop_matcher.findEarliest(running_text.items) != null) {
                        hit_stop_seq = true;
                        break;
                    }
                }
                if (stats.tokens_generated >= max_tokens_clamped) break;
                if (need_logits) {
                    _ = try self.model.stepWithLogits(next_tok);
                    next_tok = try self.model.sampleRowWithPenalties(0, sampling, &rng, if (has_penalties) &counts else null);
                } else {
                    next_tok = try self.model.step(next_tok);
                }
            }
            stats.decode_ms = nowMs() - td0;
            if (!has_images) {
                if (self.prefix_cache) |*pc| try pc.commitGenerated(out_ids.items, 0);
            }
        }

        const finish_reason: []const u8 = if (hit_stop_seq or stats.tokens_generated < max_tokens_clamped) "stop" else "length";

        if (stream) {
            sse_ctx.flushRemainder();
            const obj_kind = if (is_chat) "chat.completion.chunk" else "text_completion";
            if (include_usage) {
                const done_chunk = try std.fmt.allocPrint(self.a, "data: {{\"id\":\"mlx_tf\",\"object\":\"{s}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{s}\"}}],\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d},\"prompt_tokens_details\":{{\"cached_tokens\":{d}}},\"tok_per_sec\":{d:.1}}}}}\n\ndata: [DONE]\n\n", .{
                    obj_kind,
                    finish_reason,
                    prompt_ids.len,
                    stats.tokens_generated,
                    prompt_ids.len + stats.tokens_generated,
                    cached_tokens,
                    stats.tokPerSec(),
                });
                defer self.a.free(done_chunk);
                _ = writeAll(fd, done_chunk);
            } else {
                const done_chunk = try std.fmt.allocPrint(self.a, "data: {{\"id\":\"mlx_tf\",\"object\":\"{s}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{s}\"}}]}}\n\ndata: [DONE]\n\n", .{
                    obj_kind, finish_reason,
                });
                defer self.a.free(done_chunk);
                _ = writeAll(fd, done_chunk);
            }
            return;
        }

        // Assemble full response text (and trim at first stop sequence if triggered)
        var text_buf: std.ArrayList(u8) = .empty;
        defer text_buf.deinit(self.a);
        for (out_ids.items) |tid| try text_buf.appendSlice(self.a, self.tok.decodeToken(tid));
        var final_text = text_buf.items;
        if (stop_matcher.count > 0) {
            if (stop_matcher.findEarliest(final_text)) |pos| final_text = final_text[0..pos];
        }

        var json_buf: std.ArrayList(u8) = .empty;
        defer json_buf.deinit(self.a);
        if (is_chat) {
            try json_buf.appendSlice(self.a, "{\"id\":\"mlx_tf\",\"object\":\"chat.completion\",\"model\":\"");
            try appendJsonEscaped(self.a, &json_buf, self.model_name);
            try json_buf.appendSlice(self.a, "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"");
            try appendJsonEscaped(self.a, &json_buf, final_text);
            try json_buf.appendSlice(self.a, "\"},\"finish_reason\":\"");
            try json_buf.appendSlice(self.a, finish_reason);
            try json_buf.appendSlice(self.a, "\"}]");
        } else {
            try json_buf.appendSlice(self.a, "{\"id\":\"mlx_tf\",\"object\":\"text_completion\",\"model\":\"");
            try appendJsonEscaped(self.a, &json_buf, self.model_name);
            try json_buf.appendSlice(self.a, "\",\"choices\":[{\"index\":0,\"text\":\"");
            try appendJsonEscaped(self.a, &json_buf, final_text);
            try json_buf.appendSlice(self.a, "\",\"finish_reason\":\"");
            try json_buf.appendSlice(self.a, finish_reason);
            try json_buf.appendSlice(self.a, "\"}]");
        }
        const usage = try std.fmt.allocPrint(self.a, ",\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d},\"prompt_tokens_details\":{{\"cached_tokens\":{d}}},\"tok_per_sec\":{d:.1},\"tok_per_round\":{d:.2}}}}}", .{
            prompt_ids.len, stats.tokens_generated, prompt_ids.len + stats.tokens_generated, cached_tokens, stats.tokPerSec(), stats.avgTokensPerRound(),
        });
        defer self.a.free(usage);
        try json_buf.appendSlice(self.a, usage);
        sendJson(fd, 200, json_buf.items);
    }

    fn handleStreamParallel(self: *ServerEngine, fd: c_int, body: []const u8) !void {
        const parsed = try std.json.parseFromSlice(std.json.Value, self.a, body, .{});
        defer parsed.deinit();
        const obj = parsed.value.object;

        const stream_sse = getJsonBool(obj, "stream", false);
        const enable_thinking = getJsonBool(obj, "enable_thinking", false);
        const default_max = getJsonUsize(obj, "max_tokens", 256);
        const default_temp = getJsonF32(obj, "temperature", 0.0);

        const streams_val = obj.get("streams") orelse return sendError(fd, 400, "invalid_request_error", "missing streams array");
        const n = streams_val.array.items.len;
        if (n == 0 or n > self.max_streams) {
            return sendError(fd, 400, "invalid_request_error", "streams count must be between 1 and max_streams");
        }

        const shared_text = if (obj.get("shared_messages")) |sm| blk: {
            var msgs = try self.a.alloc(Tokenizer.ChatMessage, sm.array.items.len);
            defer self.a.free(msgs);
            for (sm.array.items, 0..) |item, i| {
                const mo = item.object;
                msgs[i] = .{
                    .role = mo.get("role").?.string,
                    .content = if (mo.get("content")) |cv| (if (cv == .string) cv.string else "") else "",
                };
            }
            break :blk try self.tok.formatChatStyleMessages(self.a, msgs, false, enable_thinking);
        } else if (obj.get("shared_prompt")) |sp|
            try self.a.dupe(u8, sp.string)
        else
            try self.a.dupe(u8, "");
        defer self.a.free(shared_text);

        const shared_ids = try self.tok.encode(self.a, shared_text);
        defer self.a.free(shared_ids);

        var specs = try self.a.alloc(batch_mod.StreamSpec, n);
        defer self.a.free(specs);
        var stream_labels = try self.a.alloc([]const u8, n);
        defer self.a.free(stream_labels);
        var delta_allocs = try self.a.alloc([]u32, n);
        defer {
            for (delta_allocs) |d| self.a.free(d);
            self.a.free(delta_allocs);
        }

        for (streams_val.array.items, 0..) |item, i| {
            const so = item.object;
            stream_labels[i] = if (so.get("id")) |idv| idv.string else "stream";
            const d_text = if (so.get("delta_messages")) |dm| blk: {
                var msgs = try self.a.alloc(Tokenizer.ChatMessage, dm.array.items.len);
                defer self.a.free(msgs);
                for (dm.array.items, 0..) |mi, k| {
                    const mo = mi.object;
                    msgs[k] = .{
                        .role = mo.get("role").?.string,
                        .content = if (mo.get("content")) |cv| (if (cv == .string) cv.string else "") else "",
                    };
                }
                break :blk try self.tok.formatChatStyleMessages(self.a, msgs, true, enable_thinking);
            } else if (so.get("delta_prompt")) |dp|
                try self.a.dupe(u8, dp.string)
            else
                try self.a.dupe(u8, "");
            defer self.a.free(d_text);

            delta_allocs[i] = try self.tok.encode(self.a, d_text);
            specs[i] = .{
                .delta_tokens = delta_allocs[i],
                .max_tokens = getJsonUsize(so, "max_tokens", default_max),
                .sampling = .{
                    .temperature = getJsonF32(so, "temperature", default_temp),
                    .top_p = getJsonF32(so, "top_p", 1.0),
                    .top_k = @intCast(getJsonUsize(so, "top_k", 0)),
                    .min_p = getJsonF32(so, "min_p", 0.0),
                    .seed = @intCast(getJsonUsize(so, "seed", 42 + i)),
                },
            };
        }

        if (self.prefix_cache) |*pc| pc.invalidate();
        const fe = try self.ensureForkEngine();

        const WaveSseCtx = struct {
            a: std.mem.Allocator,
            fd: c_int,
            tok: *Tokenizer,
            labels: []const []const u8,
            fn emit(raw: ?*anyopaque, sidx: usize, tok_id: u32) void {
                const s: *@This() = @ptrCast(@alignCast(raw.?));
                const piece = s.tok.decodeToken(tok_id);
                var buf: std.ArrayList(u8) = .empty;
                defer buf.deinit(s.a);
                const hdr = std.fmt.allocPrint(s.a, "data: {{\"stream_index\":{d},\"id\":\"", .{sidx}) catch return;
                defer s.a.free(hdr);
                buf.appendSlice(s.a, hdr) catch return;
                appendJsonEscaped(s.a, &buf, s.labels[sidx]) catch return;
                buf.appendSlice(s.a, "\",\"delta\":\"") catch return;
                appendJsonEscaped(s.a, &buf, piece) catch return;
                buf.appendSlice(s.a, "\"}\n\n") catch return;
                _ = writeAll(s.fd, buf.items);
            }
        };

        var ws_ctx = WaveSseCtx{ .a = self.a, .fd = fd, .tok = self.tok, .labels = stream_labels };
        if (stream_sse) {
            _ = writeAll(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
        }

        const out_lists = try self.a.alloc(std.ArrayList(u32), n);
        for (out_lists) |*l| l.* = .empty;
        defer {
            for (out_lists) |*l| l.deinit(self.a);
            self.a.free(out_lists);
        }

        const cb: ?batch_mod.StreamTokenCallback = if (stream_sse) WaveSseCtx.emit else null;
        const stats = try fe.runWave(shared_ids, specs, out_lists, self.a, cb, &ws_ctx);

        if (stream_sse) {
            const summary = try std.fmt.allocPrint(self.a, "data: {{\"done\":true,\"num_streams\":{d},\"prefix_tokens\":{d},\"cached_tokens\":{d},\"total_generated\":{d},\"decode_tok_per_sec\":{d:.1},\"e2e_tok_per_sec\":{d:.1}}}\n\ndata: [DONE]\n\n", .{
                stats.num_streams, stats.prefix_tokens, stats.cached_tokens, stats.total_generated, stats.decodeTokPerSec(), stats.e2eTokPerSec(),
            });
            defer self.a.free(summary);
            _ = writeAll(fd, summary);
            return;
        }

        var json_buf: std.ArrayList(u8) = .empty;
        defer json_buf.deinit(self.a);
        try json_buf.appendSlice(self.a, "{\"object\":\"stream_parallel.wave\",\"streams\":[");
        for (0..n) |i| {
            if (i > 0) try json_buf.append(self.a, ',');
            var tb: std.ArrayList(u8) = .empty;
            defer tb.deinit(self.a);
            for (out_lists[i].items) |tid| try tb.appendSlice(self.a, self.tok.decodeToken(tid));

            const pre = try std.fmt.allocPrint(self.a, "{{\"index\":{d},\"id\":\"", .{i});
            defer self.a.free(pre);
            try json_buf.appendSlice(self.a, pre);
            try appendJsonEscaped(self.a, &json_buf, stream_labels[i]);
            try json_buf.appendSlice(self.a, "\",\"text\":\"");
            try appendJsonEscaped(self.a, &json_buf, tb.items);
            const post = try std.fmt.allocPrint(self.a, "\",\"completion_tokens\":{d}}}", .{stats.per_stream_tokens[i]});
            defer self.a.free(post);
            try json_buf.appendSlice(self.a, post);
        }
        const foot = try std.fmt.allocPrint(self.a, "],\"usage\":{{\"num_streams\":{d},\"prefix_tokens\":{d},\"cached_tokens\":{d},\"total_generated\":{d},\"prefill_ms\":{d:.2},\"fork_ms\":{d:.2},\"delta_prefill_ms\":{d:.2},\"decode_ms\":{d:.2},\"decode_tok_per_sec\":{d:.1},\"e2e_tok_per_sec\":{d:.1}}}}}", .{
            stats.num_streams,
            stats.prefix_tokens,
            stats.cached_tokens,
            stats.total_generated,
            stats.prefill_ms,
            stats.fork_ms,
            stats.delta_prefill_ms,
            stats.decode_ms,
            stats.decodeTokPerSec(),
            stats.e2eTokPerSec(),
        });
        defer self.a.free(foot);
        try json_buf.appendSlice(self.a, foot);
        sendJson(fd, 200, json_buf.items);
    }
};

fn parseContentLength(headers: []const u8) usize {
    var it = std.mem.splitSequence(u8, headers, "\r\n");
    while (it.next()) |line| {
        if (line.len >= 15 and std.ascii.eqlIgnoreCase(line[0..15], "content-length:")) {
            const val = std.mem.trim(u8, line[15..], " \t");
            return std.fmt.parseInt(usize, val, 10) catch 0;
        }
    }
    return 0;
}

fn parseIpv4(host: []const u8) ?u32 {
    if (std.mem.eql(u8, host, "localhost")) return 0x7F000001;
    var parts = std.mem.splitScalar(u8, host, '.');
    var out: u32 = 0;
    var count: usize = 0;
    while (parts.next()) |p| {
        if (count >= 4) return null;
        const v = std.fmt.parseInt(u8, p, 10) catch return null;
        out = (out << 8) | @as(u32, v);
        count += 1;
    }
    return if (count == 4) out else null;
}

fn writeAll(fd: c_int, data: []const u8) bool {
    var off: usize = 0;
    while (off < data.len) {
        const n = write(fd, data.ptr + off, data.len - off);
        if (n <= 0) return false;
        off += @intCast(n);
    }
    return true;
}

fn sendError(fd: c_int, status: u16, err_type: []const u8, msg: []const u8) void {
    var buf: [512]u8 = undefined;
    const body = std.fmt.bufPrint(&buf, "{{\"error\":{{\"message\":\"{s}\",\"type\":\"{s}\"}}}}", .{ msg, err_type }) catch
        "{\"error\":{\"message\":\"request error\",\"type\":\"invalid_request_error\"}}";
    sendJson(fd, status, body);
}

fn sendJson(fd: c_int, status: u16, body: []const u8) void {
    var hdr_buf: [256]u8 = undefined;
    const status_txt = switch (status) {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        else => "Internal Server Error",
    };
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 {d} {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", .{
        status, status_txt, body.len,
    }) catch return;
    _ = writeAll(fd, hdr);
    _ = writeAll(fd, body);
}

fn appendJsonEscaped(a: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    for (s) |ch| {
        switch (ch) {
            '"' => try out.appendSlice(a, "\\\""),
            '\\' => try out.appendSlice(a, "\\\\"),
            '\n' => try out.appendSlice(a, "\\n"),
            '\r' => try out.appendSlice(a, "\\r"),
            '\t' => try out.appendSlice(a, "\\t"),
            0x00...0x08, 0x0B, 0x0C, 0x0E...0x1F => {
                var esc: [6]u8 = undefined;
                const e = std.fmt.bufPrint(&esc, "\\u{x:0>4}", .{ch}) catch unreachable;
                try out.appendSlice(a, e);
            },
            else => try out.append(a, ch),
        }
    }
}

fn getJsonUsize(obj: std.json.ObjectMap, key: []const u8, fallback: usize) usize {
    const v = obj.get(key) orelse return fallback;
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else fallback,
        else => fallback,
    };
}

fn getJsonF32(obj: std.json.ObjectMap, key: []const u8, fallback: f32) f32 {
    const v = obj.get(key) orelse return fallback;
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => fallback,
    };
}

fn getJsonBool(obj: std.json.ObjectMap, key: []const u8, fallback: bool) bool {
    const v = obj.get(key) orelse return fallback;
    return switch (v) {
        .bool => |b| b,
        else => fallback,
    };
}
