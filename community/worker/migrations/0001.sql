PRAGMA foreign_keys = ON;

CREATE TABLE publishers (
  id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  token_sha256 TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  revoked_at TEXT,
  daily_quota INTEGER NOT NULL DEFAULT 20 CHECK(daily_quota BETWEEN 1 AND 1000)
);

CREATE TABLE daily_usage (
  publisher_id TEXT NOT NULL REFERENCES publishers(id),
  day TEXT NOT NULL,
  uploads INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(publisher_id, day)
);

CREATE TABLE submissions (
  id TEXT PRIMARY KEY,
  publisher_id TEXT NOT NULL REFERENCES publishers(id),
  run_id TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  received_at TEXT NOT NULL,
  received_day TEXT NOT NULL,
  approved_at TEXT,
  status TEXT NOT NULL CHECK(status IN ('pending', 'approved', 'withdrawn')),
  model_repo TEXT NOT NULL,
  backend TEXT NOT NULL CHECK(backend IN ('mlx', 'cuda')),
  gpu_label TEXT NOT NULL,
  protocol_complete INTEGER NOT NULL CHECK(protocol_complete IN (0, 1)),
  receipt_json TEXT,
  summary_json TEXT,
  UNIQUE(publisher_id, run_id)
);

CREATE INDEX submissions_public ON submissions(status, received_at DESC, id DESC);
CREATE INDEX submissions_filters ON submissions(status, model_repo, backend, gpu_label);
CREATE INDEX submissions_publisher ON submissions(publisher_id, received_at DESC);

CREATE TRIGGER submissions_quota BEFORE INSERT ON submissions
BEGIN
  SELECT CASE WHEN COALESCE((
    SELECT uploads FROM daily_usage
    WHERE publisher_id = NEW.publisher_id AND day = NEW.received_day
  ), 0) >= (SELECT daily_quota FROM publishers WHERE id = NEW.publisher_id)
  THEN RAISE(ABORT, 'daily_quota_exceeded') END;
  SELECT CASE WHEN (SELECT revoked_at FROM publishers WHERE id = NEW.publisher_id) IS NOT NULL
  THEN RAISE(ABORT, 'publisher_revoked') END;
END;

CREATE TRIGGER submissions_count AFTER INSERT ON submissions
BEGIN
  INSERT INTO daily_usage(publisher_id, day, uploads)
  VALUES(NEW.publisher_id, NEW.received_day, 1)
  ON CONFLICT(publisher_id, day) DO UPDATE SET uploads = uploads + 1;
END;
