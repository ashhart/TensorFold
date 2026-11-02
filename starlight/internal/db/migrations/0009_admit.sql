-- One statement per admission and settlement; a refusal raises SL001 (a limit: kind, key) or SL002 (no slot) and rolls back.

CREATE INDEX admission_queue_rank ON admission_queue (model_id, base_rank);

-- A waiting request's rank: its class plus 100 for each ageing step since it queued, at most one class, on the database clock.
CREATE FUNCTION starlight_rank(base int, since timestamptz, step float8) RETURNS int LANGUAGE sql STABLE AS $$
    SELECT base + least(10, greatest(0, floor(extract(epoch FROM now() - since) / step)))::int * 100
$$;

-- Each rank group follower needs an active node, seen in 90 s and cleared for the class; PL/pgSQL caches its plan.
CREATE FUNCTION starlight_members_eligible(replica uuid, class int) RETURNS boolean LANGUAGE plpgsql STABLE AS $$
BEGIN
    RETURN NOT EXISTS (SELECT 1 FROM replica_members m JOIN nodes n ON n.id = m.node_id WHERE m.replica_id = replica
        AND (n.status = 'active' AND n.last_seen_at > now() - interval '90 seconds'
            AND array_position(ARRAY['public', 'internal', 'confidential', 'restricted'], n.max_classification) - 1 >= class) IS NOT TRUE);
END $$;

-- The amount each key|window counts, micros for cost and tokens for the rest; PL/pgSQL, as SQL bodies replan per call.
CREATE FUNCTION starlight_amounts(keys text[], tokens bigint, micros bigint) RETURNS bigint[] LANGUAGE plpgsql IMMUTABLE AS $$
BEGIN
    RETURN ARRAY(SELECT CASE WHEN k LIKE 'cost:%' THEN micros ELSE tokens END FROM unnest(keys) WITH ORDINALITY AS u(k, i) ORDER BY i);
END $$;

-- Rows lock in the order admission takes them, key bytewise then window, so settling never deadlocks with admitting.
CREATE FUNCTION starlight_settle_counters(keys text[], res bigint[], act bigint[]) RETURNS void LANGUAGE plpgsql AS $$
DECLARE
    c record;
BEGIN
    FOR c IN SELECT k, w, sum(r)::bigint AS r, sum(a)::bigint AS a FROM (SELECT split_part(x, '|', 1) AS k, split_part(x, '|', 2)::timestamptz AS w, r, a
            FROM unnest(keys, res, act) AS u(x, r, a) WHERE position('|' IN x) > 0) s GROUP BY k, w ORDER BY k COLLATE "C", w LOOP
        IF c.k LIKE 'tpm:%' THEN
            UPDATE counters SET used = greatest(0, used - c.r + c.a) WHERE key = c.k AND window_start = c.w;
        ELSE
            UPDATE counters SET reserved = greatest(0, reserved - c.r), used = used + c.a WHERE key = c.k AND window_start = c.w;
        END IF;
    END LOOP;
END $$;

CREATE FUNCTION starlight_admit(r jsonb) RETURNS void LANGUAGE plpgsql AS $$
DECLARE
    s jsonb;
    c jsonb;
    n bigint;
    v_used bigint;
    v_res bigint;
    v_slots int;
    v_free int;
    v_blocked int;
    v_target uuid;
    v_route boolean := r ? 'route';
    v_class int := (r->>'class')::int;
    v_ageing float8 := (r->>'ageing')::float8;
    v_at timestamptz := coalesce((r->>'at')::timestamptz, now());
    v_rank int;
BEGIN
    IF (r->>'tokens')::bigint < 0 OR (r->>'micros')::bigint < 0 OR jsonb_path_exists(r->'counters', '$[*] ? (@.amount < 0)') THEN
        RAISE EXCEPTION USING ERRCODE = 'invalid_parameter_value', MESSAGE = 'negative amount';
    END IF;
    v_rank := starlight_rank((r->>'base')::int, v_at, v_ageing);
    FOR s IN SELECT jsonb_array_elements(r->'scopes') LOOP
        PERFORM pg_advisory_xact_lock(hashtextextended('starlight:scope:' || (s->>'key'), 0));
        SELECT count(*) INTO n FROM leases WHERE scopes @> ARRAY[s->>'key'] AND heartbeat_at > now() - interval '60 seconds';
        IF n >= (s->>'limit')::bigint THEN
            RAISE EXCEPTION USING ERRCODE = 'SL001', MESSAGE = 'concurrency', DETAIL = s->>'key';
        END IF;
    END LOOP;
    FOR c IN SELECT jsonb_array_elements(r->'counters') LOOP
        IF (c->>'reserve')::boolean THEN
            INSERT INTO counters AS t (key, window_start, used, reserved) VALUES (c->>'key', (c->>'window')::timestamptz, 0, (c->>'amount')::bigint)
            ON CONFLICT (key, window_start) DO UPDATE SET reserved = t.reserved + excluded.reserved RETURNING t.used, t.reserved INTO v_used, v_res;
        ELSE
            INSERT INTO counters AS t (key, window_start, used, reserved) VALUES (c->>'key', (c->>'window')::timestamptz, (c->>'amount')::bigint, 0)
            ON CONFLICT (key, window_start) DO UPDATE SET used = t.used + excluded.used RETURNING t.used, t.reserved INTO v_used, v_res;
        END IF;
        IF v_used + v_res > (c->>'limit')::bigint THEN
            RAISE EXCEPTION USING ERRCODE = 'SL001', MESSAGE = CASE WHEN (c->>'reserve')::boolean THEN 'budget' ELSE 'rate' END, DETAIL = c->>'key';
        END IF;
    END LOOP;
    IF v_route THEN
        v_target := (r->>'route')::uuid;
        PERFORM pg_advisory_xact_lock(hashtextextended('starlight:target:' || v_target::text, 0));
        SELECT rt.max_concurrency INTO v_slots FROM provider_routes rt JOIN providers pv ON pv.id = rt.provider_id
        WHERE rt.id = v_target AND rt.enabled AND pv.enabled
            AND array_position(ARRAY['public', 'internal', 'confidential', 'restricted'], pv.max_classification) - 1 >= v_class;
        IF NOT FOUND THEN
            RAISE EXCEPTION USING ERRCODE = 'SL002', MESSAGE = 'gone';
        END IF;
        SELECT count(*) INTO n FROM leases WHERE route_id = v_target AND heartbeat_at > now() - interval '60 seconds';
    ELSE
        v_target := (r->>'replica')::uuid;
        PERFORM pg_advisory_xact_lock(hashtextextended('starlight:target:' || v_target::text, 0));
        SELECT p.slots INTO v_slots FROM replicas p JOIN deployments d ON d.id = p.deployment_id JOIN nodes nd ON nd.id = p.node_id
        WHERE p.id = v_target AND p.desired = 'running' AND p.state = 'ready' AND p.deployment_revision = d.revision AND d.status = 'active'
            AND nd.status = 'active' AND nd.last_seen_at > now() - interval '90 seconds'
            AND array_position(ARRAY['public', 'internal', 'confidential', 'restricted'], nd.max_classification) - 1 >= v_class
            AND starlight_members_eligible(p.id, v_class);
        IF NOT FOUND THEN
            RAISE EXCEPTION USING ERRCODE = 'SL002', MESSAGE = 'gone';
        END IF;
        SELECT count(*) INTO n FROM leases WHERE replica_id = v_target AND heartbeat_at > now() - interval '60 seconds';
    END IF;
    v_free := v_slots - n;
    IF v_free <= 0 THEN
        RAISE EXCEPTION USING ERRCODE = 'SL002', MESSAGE = 'full';
    END IF;
    SELECT count(*) INTO v_blocked FROM (
        (SELECT 1 FROM admission_queue q WHERE q.model_id = (r->>'model')::uuid AND q.id <> (r->>'queue')::uuid
            AND q.enqueued_at < v_at AND q.candidates && ARRAY[v_target]
            AND q.heartbeat_at > now() - interval '30 seconds' AND q.deadline > now()
            AND starlight_rank(q.base_rank, q.enqueued_at, v_ageing) >= v_rank
            ORDER BY q.enqueued_at LIMIT v_free)
        UNION ALL
        (SELECT 1 FROM admission_queue q WHERE q.model_id = (r->>'model')::uuid AND q.id <> (r->>'queue')::uuid
            AND q.enqueued_at >= v_at AND q.base_rank > (r->>'base')::int AND q.candidates && ARRAY[v_target]
            AND q.heartbeat_at > now() - interval '30 seconds' AND q.deadline > now()
            AND starlight_rank(q.base_rank, q.enqueued_at, v_ageing) > v_rank
            LIMIT v_free)) b;
    IF v_blocked >= v_free THEN
        RAISE EXCEPTION USING ERRCODE = 'SL002', MESSAGE = 'outranked';
    END IF;
    INSERT INTO leases (id, request_id, gateway_id, model_id, replica_id, route_id, scopes, budget_keys, reserved_tokens, reserved_micros,
        priority, deadline, principal_type, principal_id, project_id, client)
    VALUES ((r->>'id')::uuid, r->>'request', (r->>'gateway')::uuid, (r->>'model')::uuid,
        CASE WHEN v_route THEN NULL ELSE v_target END, CASE WHEN v_route THEN v_target END,
        ARRAY(SELECT jsonb_array_elements_text(r->'scope_keys')), ARRAY(SELECT jsonb_array_elements_text(r->'budget_keys')),
        (r->>'tokens')::bigint, (r->>'micros')::bigint, r->>'priority', (r->>'deadline')::timestamptz,
        r->>'principal_type', nullif(r->>'principal_id', '')::uuid, nullif(r->>'project_id', '')::uuid, r->>'client');
    IF (r->>'queued')::boolean THEN
        DELETE FROM admission_queue WHERE id = (r->>'queue')::uuid;
    END IF;
END $$;

-- Ends a lease, replacing its reservations with what it used and waking waiters; a lease already gone changes nothing.
CREATE FUNCTION starlight_release(lease uuid, tokens bigint, micros bigint) RETURNS void LANGUAGE plpgsql AS $$
DECLARE
    v_model uuid;
    v_keys text[];
    v_rt bigint;
    v_rm bigint;
    v_res bigint[];
    v_act bigint[];
BEGIN
    DELETE FROM leases WHERE id = lease RETURNING model_id, budget_keys, reserved_tokens, reserved_micros INTO v_model, v_keys, v_rt, v_rm;
    IF FOUND THEN
        SELECT coalesce(array_agg(CASE WHEN k LIKE 'cost:%' THEN v_rm ELSE v_rt END ORDER BY i), '{}'),
            coalesce(array_agg(CASE WHEN k LIKE 'cost:%' THEN micros ELSE tokens END ORDER BY i), '{}')
            INTO v_res, v_act FROM unnest(v_keys) WITH ORDINALITY AS u(k, i);
        PERFORM starlight_settle_counters(v_keys, v_res, v_act);
        PERFORM pg_notify('starlight_capacity', v_model::text);
    END IF;
END $$;

CREATE FUNCTION starlight_settle(r jsonb) RETURNS void LANGUAGE plpgsql AS $$
BEGIN
    PERFORM starlight_release((r->>'lease')::uuid, (r->>'tokens')::bigint, (r->>'micros')::bigint);
    INSERT INTO usage_events (id, created_at, request_id, principal_type, principal_id, user_id, project_id, api_key_id, task_id, model_id,
        model_name, route_kind, route_id, node_id, protocol, status, outcome, prompt_tokens, completion_tokens, cached_tokens, cost_micros,
        queue_ms, ttft_ms, duration_ms, classification, priority, client)
    VALUES ((r->>'usage')::uuid, now(), r->>'request', r->>'principal_type', (r->>'principal_id')::uuid, nullif(r->>'user_id', '')::uuid,
        (r->>'project_id')::uuid, nullif(r->>'api_key_id', '')::uuid, nullif(r->>'task_id', '')::uuid, (r->>'model')::uuid, r->>'model_name',
        r->>'route_kind', nullif(r->>'route_id', '')::uuid, nullif(r->>'node_id', '')::uuid, r->>'protocol', (r->>'status')::int, r->>'outcome',
        (r->>'prompt')::bigint, (r->>'completion')::bigint, (r->>'cached')::bigint, (r->>'cost')::bigint, (r->>'queue_ms')::int,
        (r->>'ttft_ms')::int, (r->>'duration_ms')::int, r->>'classification', r->>'priority', r->>'client');
    IF r ? 'replica' THEN
        UPDATE replicas SET last_request_at = now() WHERE id = (r->>'replica')::uuid
            AND (last_request_at IS NULL OR last_request_at < now() - interval '10 seconds');
    END IF;
END $$;
