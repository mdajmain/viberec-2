"""Step 13: who uses VibeRec and what they do (Adil, 2026-10-10).

Every visitor is a profile:
- guest:     id "g_<32 hex>", random, kept in a cookie by the web app (no IP or fingerprint);
- member:    id "m_<32 hex>", linked to a Clerk account (the database stores Clerk's user id, never emails/passwords);
- anonymous: id "a_<32 hex>", what a profile becomes after "Forget me".
All history hangs off the profile id with ON UPDATE CASCADE, so renaming a profile moves its whole history:
- sign-up merges a guest's history into the member profile;
- "Forget me" renames the profile to a fresh anonymous id and drops the Clerk link: the reactions stay for training,
  but nothing points to the person any more.
"""
import json
import re
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    profile_id text PRIMARY KEY, kind text NOT NULL CHECK (kind IN ('guest', 'member', 'anonymous')),
    clerk_user_id text UNIQUE, created timestamptz DEFAULT now(), last_seen timestamptz DEFAULT now());
CREATE TABLE IF NOT EXISTS reactions (
    id bigserial PRIMARY KEY,
    profile_id text NOT NULL REFERENCES profiles ON UPDATE CASCADE ON DELETE CASCADE,
    movie_id integer NOT NULL REFERENCES movies,
    verdict text NOT NULL CHECK (verdict IN ('like', 'dislike', 'unseen', 'not_this_vibe')),
    source text NOT NULL CHECK (source IN ('quiz', 'search', 'feed')), context_id text, created timestamptz DEFAULT now());
CREATE INDEX IF NOT EXISTS reactions_profile ON reactions (profile_id, created);
CREATE TABLE IF NOT EXISTS searches (
    search_id text PRIMARY KEY,
    profile_id text REFERENCES profiles ON UPDATE CASCADE ON DELETE CASCADE,
    prompt text NOT NULL, understood jsonb, created timestamptz DEFAULT now());
CREATE INDEX IF NOT EXISTS searches_profile ON searches (profile_id, created);
CREATE TABLE IF NOT EXISTS shown (
    id bigserial PRIMARY KEY,
    profile_id text REFERENCES profiles ON UPDATE CASCADE ON DELETE CASCADE,
    context text NOT NULL CHECK (context IN ('search', 'feed')), context_id text NOT NULL,
    movie_id integer NOT NULL, position smallint NOT NULL, design text, created timestamptz DEFAULT now());
CREATE INDEX IF NOT EXISTS shown_context ON shown (context_id);
CREATE TABLE IF NOT EXISTS taste_profiles (
    profile_id text PRIMARY KEY REFERENCES profiles ON UPDATE CASCADE ON DELETE CASCADE,
    taste vector({n_taste}), vibe vector({n_vibe}), n_likes integer, n_reactions integer, summary jsonb,
    updated timestamptz DEFAULT now());
"""
GUEST_ID = re.compile(r"^g_[0-9a-f]{32}$")
VERDICTS = ("like", "dislike", "unseen", "not_this_vibe")
SOURCES = ("quiz", "search", "feed")


def new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex}"


def create_schema(conn, n_taste=128, n_vibe=41):
    with conn.cursor() as cur:
        cur.execute(SCHEMA.format(n_taste=n_taste, n_vibe=n_vibe))
    conn.commit()


def _one(conn, sql, params=()):
    r = conn.execute(sql, params).fetchone()
    return r[0] if r else None


# ---------------------------------------------------------------------- profiles
def ensure_guest(conn, guest_id):
    """The guest profile for a cookie id (created on first sight). Refuses ids that are not "g_" + 32 hex."""
    if not GUEST_ID.match(guest_id or ""):
        raise ValueError("not a guest id")
    conn.execute("INSERT INTO profiles (profile_id, kind) VALUES (%s, 'guest') "
                 "ON CONFLICT (profile_id) DO UPDATE SET last_seen = now()", (guest_id,))
    conn.commit()
    return guest_id


def sign_in(conn, clerk_user_id, guest_id=None):
    """The member profile for a Clerk user, created on first sign-in. A guest history from this browser is merged in:
    renamed into the new member profile, or moved into an existing one."""
    if not clerk_user_id:
        raise ValueError("no Clerk user id")
    member = _one(conn, "SELECT profile_id FROM profiles WHERE clerk_user_id = %s", (clerk_user_id,))
    guest = guest_id if guest_id and _one(conn, "SELECT 1 FROM profiles WHERE profile_id = %s AND kind = 'guest'",
                                          (guest_id,)) else None
    if member is None and guest is not None:                       # first sign-in from this browser: the guest becomes the member
        member = new_id("m")
        conn.execute("UPDATE profiles SET profile_id = %s, kind = 'member', clerk_user_id = %s, last_seen = now() "
                     "WHERE profile_id = %s", (member, clerk_user_id, guest))
    elif member is None:
        member = new_id("m")
        conn.execute("INSERT INTO profiles (profile_id, kind, clerk_user_id) VALUES (%s, 'member', %s)", (member, clerk_user_id))
    elif guest is not None:                                        # already a member: move the guest's history over
        for table in ("reactions", "searches", "shown"):
            conn.execute(f"UPDATE {table} SET profile_id = %s WHERE profile_id = %s", (member, guest))
        conn.execute("DELETE FROM profiles WHERE profile_id = %s", (guest,))
    conn.execute("UPDATE profiles SET last_seen = now() WHERE profile_id = %s", (member,))
    conn.commit()
    return member


def kind_of(conn, profile_id):
    return _one(conn, "SELECT kind FROM profiles WHERE profile_id = %s", (profile_id,))


def forget(conn, profile_id):
    """"Forget me": the profile (and, through ON UPDATE CASCADE, all its history) moves to a fresh anonymous id and
    loses its Clerk link and taste profile. Returns the anonymous id, which is never shown to anyone."""
    anon = new_id("a")
    n = conn.execute("UPDATE profiles SET profile_id = %s, kind = 'anonymous', clerk_user_id = NULL WHERE profile_id = %s",
                     (anon, profile_id)).rowcount
    conn.execute("DELETE FROM taste_profiles WHERE profile_id = %s", (anon,))
    conn.commit()
    if not n:
        raise KeyError("no such profile")
    return anon


# ---------------------------------------------------------------------- history
def add_reaction(conn, profile_id, movie_id, verdict, source, context_id=None):
    if verdict not in VERDICTS or source not in SOURCES:
        raise ValueError(f"verdict must be one of {VERDICTS}, source one of {SOURCES}")
    conn.execute("INSERT INTO reactions (profile_id, movie_id, verdict, source, context_id) VALUES (%s, %s, %s, %s, %s)",
                 (profile_id, int(movie_id), verdict, source, context_id))
    conn.commit()


def answers(conn, profile_id):
    """{movie_id: the person's latest verdict on it}."""
    rows = conn.execute("SELECT DISTINCT ON (movie_id) movie_id, verdict FROM reactions WHERE profile_id = %s "
                        "ORDER BY movie_id, created DESC, id DESC", (profile_id,)).fetchall()
    return {int(m): v for m, v in rows}


def log_search(conn, profile_id, prompt, understood):
    sid = new_id("s")
    conn.execute("INSERT INTO searches (search_id, profile_id, prompt, understood) VALUES (%s, %s, %s, %s)",
                 (sid, profile_id, prompt, json.dumps(understood)))
    conn.commit()
    return sid


def log_shown(conn, profile_id, context, context_id, movie_ids, designs=None):
    designs = designs or [None] * len(movie_ids)
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO shown (profile_id, context, context_id, movie_id, position, design) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        [(profile_id, context, context_id, int(m), i + 1, d) for i, (m, d) in enumerate(zip(movie_ids, designs))])
    conn.commit()


def searches_since(conn, profile_id, minutes=60):
    return _one(conn, "SELECT count(*) FROM searches WHERE profile_id = %s AND created > now() - make_interval(mins => %s)",
                (profile_id, minutes))


def save_taste(conn, profile_id, taste, vibe, n_likes, n_reactions, summary):
    vec = lambda x: None if x is None else "[" + ",".join(f"{float(v):.6g}" for v in x) + "]"
    conn.execute("INSERT INTO taste_profiles (profile_id, taste, vibe, n_likes, n_reactions, summary, updated) "
                 "VALUES (%s, %s, %s, %s, %s, %s, now()) ON CONFLICT (profile_id) DO UPDATE SET taste = EXCLUDED.taste, "
                 "vibe = EXCLUDED.vibe, n_likes = EXCLUDED.n_likes, n_reactions = EXCLUDED.n_reactions, "
                 "summary = EXCLUDED.summary, updated = now()",
                 (profile_id, vec(taste), vec(vibe), int(n_likes), int(n_reactions), json.dumps(summary)))
    conn.commit()


def history(conn, profile_id, limit=50):
    """The person's searches (newest first, each with the films shown and their reactions) and all reactions."""
    searches = conn.execute("SELECT search_id, prompt, understood, created FROM searches WHERE profile_id = %s "
                            "ORDER BY created DESC LIMIT %s", (profile_id, limit)).fetchall()
    shown = conn.execute("SELECT s.context_id, s.movie_id, s.position, m.display_title, m.year, m.poster_path "
                         "FROM shown s JOIN movies m USING (movie_id) WHERE s.profile_id = %s AND s.context = 'search' "
                         "ORDER BY s.context_id, s.position", (profile_id,)).fetchall()
    by_search = {}
    for cid, mid, pos, title, year, poster in shown:
        by_search.setdefault(cid, []).append({"movie_id": mid, "position": pos, "title": title, "year": year, "poster_path": poster})
    reactions = conn.execute("SELECT r.movie_id, m.display_title, m.year, m.poster_path, r.verdict, r.source, r.created "
                             "FROM reactions r JOIN movies m USING (movie_id) WHERE r.profile_id = %s "
                             "ORDER BY r.created DESC, r.id DESC", (profile_id,)).fetchall()
    return {"searches": [{"search_id": sid, "prompt": p, "understood": u, "created": c.isoformat(),
                          "results": by_search.get(sid, [])} for sid, p, u, c in searches],
            "reactions": [{"movie_id": m, "title": t, "year": y, "poster_path": pp, "verdict": v, "source": s,
                           "created": c.isoformat()} for m, t, y, pp, v, s, c in reactions]}
