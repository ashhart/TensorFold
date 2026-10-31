## Exactness
| numerics | serve_check | accepted/drafted (served) | token-hash audit | serve_cancel | rewind (cache on vs off) | one chunk hash |
|---|---|---|---|---|---|---|
| exact | 15 ok, 0 fail | 1285/1688 | 11 groups, 0 differ | 3 ok, 0 fail | 0 differ (rc 0), up to 10240 cached tokens | yes |

## Prompt-chunk seams (`generate --chat --max 64 --hash`)
| numerics | --chunk | prompt tokens | logits hash | rc |
|---|---|---|---|---|
| exact | 512 | 10368 | c8f6888043c972e0 | 0 |
| exact | 2048 | 10368 | c8f6888043c972e0 | 0 |
| exact | 8192 | 10368 | c8f6888043c972e0 | 0 |
| exact | 20480 | 10368 | c8f6888043c972e0 | 0 |

Seams exercised (prompt longer than 8,192 and shorter than the largest chunk): exact True.
