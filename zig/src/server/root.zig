//! The native HTTP server's surface: what main and the test servers build a server from.
pub const json = @import("json");
pub const errors = @import("errors.zig");
pub const auth = @import("auth.zig");
pub const cli = @import("cli.zig");
pub const log = @import("log.zig");
pub const live = @import("live.zig");
pub const model_text = @import("model_text.zig");
pub const hf_text = @import("hf_text.zig");
pub const listener = @import("listener.zig");
pub const server = @import("server.zig");
pub const serve = @import("serve.zig");
pub const hub = @import("hub.zig");
pub const tool_specs = @import("tool_specs.zig");
pub const Server = server.Server;
pub const Config = server.Config;

test {
    _ = @import("embedding_protocol_test.zig");
    _ = @import("embedding_service_test.zig");
    _ = @import("http_limits_test.zig");
    _ = @import("done_times_test.zig");
    _ = @import("drain_test.zig");
    _ = @import("keepalive_test.zig");
    @import("std").testing.refAllDecls(@This());
    _ = @import("json");
    _ = @import("std_options.zig");
    _ = @import("fields.zig");
    _ = @import("reply_text.zig");
    _ = @import("tool_parse.zig");
    _ = @import("tool_stream.zig");
    _ = @import("tool_params.zig");
    _ = @import("tool_grammar.zig");
    _ = @import("tool_required_test.zig");
    _ = @import("grammar.zig");
    _ = @import("pyrepr.zig");
    _ = @import("metrics.zig");
    _ = @import("http_conn.zig");
    _ = @import("auth.zig");
    _ = @import("cli.zig");
    _ = @import("cli_test.zig");
    _ = @import("live.zig");
    _ = @import("listener.zig");
    _ = @import("hub.zig");
    _ = @import("chunk_plan.zig");
    _ = @import("status_routes.zig");
    _ = @import("speech.zig");
    if (comptime @import("builtin").os.tag != .windows) _ = @import("speech_stream_test.zig"); // over a Unix socket pair
    _ = @import("drama_prompt.zig");
    _ = @import("health_test.zig");
    _ = @import("memory_routes.zig");
    _ = @import("process_memory.zig");
    _ = @import("log.zig");
    _ = @import("request_log.zig");
    _ = @import("messages.zig");
    _ = @import("images.zig");
    _ = @import("images_video_marker_test.zig");
    _ = @import("image_key.zig");
    _ = @import("image_pieces.zig");
    _ = @import("media_fetch.zig");
    _ = @import("videos.zig");
    _ = @import("stream_preflight_test.zig");
    _ = @import("stream_dsml_test.zig");
    _ = @import("logprobs_test.zig");
    _ = @import("thinking_reserve_test.zig");
    _ = @import("late_system_test.zig");
    _ = @import("compact_test.zig");
    _ = @import("request_log.zig");
    _ = @import("completion_prompt_test.zig");
    _ = @import("hf_eos_http_test.zig");
    _ = @import("server_init_test.zig");
    _ = @import("decisions.zig");
    _ = @import("decisions_test.zig");
    _ = @import("systemone.zig");
    _ = @import("systemone_test.zig");
    _ = @import("slide_context_test.zig");
    // both run over Unix socket pairs
    if (comptime @import("builtin").os.tag != .windows) {
        _ = @import("slide_stream_test.zig");
        _ = @import("slide_save_stream_test.zig");
    }
    _ = @import("slide_writer_test.zig");
    _ = @import("slide.zig");
    _ = @import("slide_cli.zig");
    _ = @import("slide_key.zig");
    _ = @import("slide_graph.zig");
    _ = @import("slide_graph_delete.zig");
    _ = @import("slide_graph_test.zig");
    _ = @import("slide_recheck.zig");
    _ = @import("slide_model.zig");
    _ = @import("slide_lesson.zig");
    _ = @import("slide_words.zig");
    _ = @import("slide_found.zig");
    _ = @import("slide_untold.zig");
    _ = @import("slide_replace.zig");
    _ = @import("slide_answers.zig");
    _ = @import("slide_recheck.zig");
    _ = @import("slide_forget.zig");
    _ = @import("slide_forget_test.zig");
    _ = @import("slide_supplied.zig");
    _ = @import("slide_questions_test.zig");
    _ = @import("slide_whom.zig");
    _ = @import("slide_subject.zig");
    _ = @import("slide_span.zig");
    _ = @import("slide_route.zig");
    _ = @import("slide_side.zig");
    _ = @import("slide_take.zig");
    _ = @import("slide_teacher.zig");
    _ = @import("slide_pick.zig");
    _ = @import("slide_context.zig");
    _ = @import("slide_talks.zig");
    _ = @import("startup.zig");
    _ = @import("responses_include_test.zig");
    _ = @import("cortheon/cortheon.zig");
    _ = @import("cortheon/cortheon_test.zig");
    _ = @import("recall_routes.zig");
    // the benchmark command's serve child runs under POSIX signals and waits; Windows refuses the command
    if (comptime @import("builtin").os.tag != .windows) {
        _ = @import("benchmark_suite.zig");
        _ = @import("benchmark_check.zig");
        _ = @import("benchmark_receipt_test.zig");
        _ = @import("benchmark_stop_test.zig");
        _ = @import("benchmark_sse_test.zig");
        _ = @import("benchmark_stream_test.zig");
        _ = @import("benchmark_server_test.zig");
        _ = @import("benchmark_facts_test.zig");
        _ = @import("benchmark_rocm_test.zig");
        _ = @import("benchmark_manifest.zig");
        _ = @import("benchmark_drafts.zig");
        _ = @import("benchmark_upload.zig");
        _ = @import("benchmark_upload_test.zig");
        _ = @import("benchmark_args.zig");
        _ = @import("benchmark_cli_test.zig");
        _ = @import("benchmark_run_test.zig");
        _ = @import("benchmark_early_test.zig");
        _ = @import("benchmark_publication_test.zig");
        _ = @import("benchmark_chat_only_test.zig");
        _ = @import("benchmark_config_test.zig");
    }
}

test {
    _ = @import("slide_retention_test.zig");
}

test {
    _ = @import("slide_save_failure_test.zig");
}

test {
    _ = @import("slide_subject_guard_test.zig");
    _ = @import("server_recovery_test.zig");
}

test {
    _ = @import("cortheon/gate_test.zig");
    _ = @import("cortheon/gate_http_test.zig");
}
