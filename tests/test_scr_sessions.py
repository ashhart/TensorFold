import os
import unittest

os.environ.setdefault("TENSORFOLD_SCR_ENABLED", "0")

from tensorfold.families.qwen3_5.cuda.scr.config import Config
from tensorfold.families.qwen3_5.cuda.scr.sessions import Session, SessionStore, Snapshot


def cfg(**over):
    c = Config()
    c.fastdiff = False
    for k, v in over.items():
        setattr(c, k, v)
    return c


def snap(ids, bytes_=1000):
    return Snapshot(ids=list(ids), pos=len(ids), kv=[], rec=[], conv=[], bytes=bytes_)


class TestMatch(unittest.TestCase):
    def test_continuation_matches(self):
        c = cfg(match_min_tokens=10, match_min_ratio=0.5)
        st = SessionStore(c)
        s, created = st.session_for(list(range(1000)))
        self.assertTrue(created)
        s.snap = snap(range(1000))
        sid = st.match(list(range(1000)) + [7, 7, 7])
        self.assertEqual(sid, s.sid)

    def test_unrelated_rejected(self):
        c = cfg(match_min_tokens=10, match_min_ratio=0.5)
        st = SessionStore(c)
        s, _ = st.session_for(list(range(1000)))
        s.snap = snap(range(1000))
        self.assertIsNone(st.match(list(range(5000, 1500, -1))))

    def test_ambiguity_margin(self):
        # a weak slotted match loses to no one; but a clearly better bare
        # (snapshot-less) candidate beyond the margin aborts matching entirely
        c = cfg(match_min_tokens=10, match_min_ratio=0.5, match_ambig_margin=0.15)
        st = SessionStore(c)
        new = [1] * 100 + [2] * 50
        A = [1] * 100 + [3] * 50                            # slotted: r_all = 0.667
        B = [1] * 100 + [2] * 49 + [9]                      # bare:    r_all = 0.993
        st.sessions["a"] = Session(sid="a", ids=A,
                                   snap=Snapshot(ids=A, pos=len(A), kv=[], rec=[], conv=[], bytes=1))
        self.assertEqual(st.match(new), "a")                # a alone: wins
        st.sessions["b"] = Session(sid="b", ids=B, snap=None)   # the bare twin
        self.assertIsNone(st.match(new))                    # 0.993 > 0.667 + 0.15

    def test_lru_count_cap(self):
        c = cfg(match_min_tokens=10, max_sessions=3)
        st = SessionStore(c)
        for i in range(5):
            s, _ = st.session_for([i] * 50 + list(range(100)))
            st.refresh(s.sid, snap([i] * 50 + list(range(100))))
        self.assertLessEqual(len(st.sessions), 3)


class TestBudget(unittest.TestCase):
    def test_byte_budget_evicts_oldest(self):
        c = cfg(match_min_tokens=10, side_gib=0.0000004)    # ~430 bytes: evicts everything
        st = SessionStore(c)
        s1, _ = st.session_for(list(range(100)))
        st.refresh(s1.sid, snap(range(100), bytes_=1024))
        s2, _ = st.session_for([9] * 10 + list(range(100)))
        st.refresh(s2.sid, snap([9] * 10 + list(range(100)), bytes_=2048))
        self.assertLessEqual(st.held_bytes(), int(c.side_gib * 1024 ** 3) + 2048)
        self.assertEqual(st.stats["evicted"] >= 1, True)


class TestRefresh(unittest.TestCase):
    def test_last_writer_wins_and_ids_update(self):
        c = cfg(match_min_tokens=10)
        st = SessionStore(c)
        s, _ = st.session_for(list(range(100)))
        st.refresh(s.sid, snap(list(range(100)) + [1, 2], bytes_=10))
        self.assertEqual(s.ids, list(range(100)) + [1, 2])
        self.assertEqual(s.snap.pos, 102)
        # next turn matches against the committed ids now
        sid = st.match(list(range(100)) + [1, 2] + [3])
        self.assertEqual(sid, s.sid)


if __name__ == "__main__":
    unittest.main()