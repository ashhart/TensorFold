// A Nemotron page holds [kv_heads][TF_PAGE_TOKENS][hd] bf16 of one cache (keys or values): head h's rows in a run.
inline ulong nemo_paged_row(device const uint* table, uint pos, ulong page_bytes, uint h, uint hd) {
  return tf_page_offset(table, pos, page_bytes, ulong(hd) * 2) + ulong(h) * TF_PAGE_TOKENS * hd * 2;
}
