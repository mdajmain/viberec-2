"""User-data layer against a throwaway database (viberec_test in the local Docker Postgres); skipped without it."""
import pytest

psycopg = pytest.importorskip("psycopg")
from src.viberec import userdata as ud

ADMIN = "postgresql://viberec:viberec-local-only@localhost:5433/viberec"
TEST = "postgresql://viberec:viberec-local-only@localhost:5433/viberec_test"


@pytest.fixture
def conn():
    try:
        admin = psycopg.connect(ADMIN, autocommit=True, connect_timeout=3)
    except Exception:
        pytest.skip("local Postgres not running")
    if not admin.execute("SELECT 1 FROM pg_database WHERE datname = 'viberec_test'").fetchone():
        admin.execute("CREATE DATABASE viberec_test")
    admin.close()
    c = psycopg.connect(TEST)
    c.execute("DROP TABLE IF EXISTS shown, searches, reactions, taste_profiles, profiles, movies CASCADE")
    c.execute("CREATE EXTENSION IF NOT EXISTS vector")
    c.execute("CREATE TABLE movies (movie_id integer PRIMARY KEY, display_title text, year integer, poster_path text)")
    c.execute("INSERT INTO movies VALUES (1, 'Hitch (2005)', 2005, '/h.jpg'), (2, 'Up (2009)', 2009, NULL), "
              "(3, 'Saw (2004)', 2004, NULL)")
    c.commit()
    ud.create_schema(c, n_taste=3, n_vibe=2)
    yield c
    c.close()


def test_guest_ids_are_checked_and_idempotent(conn):
    with pytest.raises(ValueError):
        ud.ensure_guest(conn, "g_not-hex")
    g = ud.new_id("g")
    assert ud.ensure_guest(conn, g) == g and ud.ensure_guest(conn, g) == g
    assert ud.kind_of(conn, g) == "guest"


def test_latest_reaction_wins(conn):
    g = ud.ensure_guest(conn, ud.new_id("g"))
    ud.add_reaction(conn, g, 1, "unseen", "quiz")
    ud.add_reaction(conn, g, 1, "like", "feed")
    ud.add_reaction(conn, g, 3, "dislike", "quiz")
    assert ud.answers(conn, g) == {1: "like", 3: "dislike"}
    with pytest.raises(ValueError):
        ud.add_reaction(conn, g, 2, "love", "quiz")


def test_first_sign_in_turns_the_guest_into_the_member(conn):
    g = ud.ensure_guest(conn, ud.new_id("g"))
    ud.add_reaction(conn, g, 1, "like", "quiz")
    sid = ud.log_search(conn, g, "like Hitch", {"sample_movies": ["Hitch (2005)"]})
    ud.log_shown(conn, g, "search", sid, [2, 3], ["fusion", "two_step"])
    m = ud.sign_in(conn, "user_abc", g)
    assert m.startswith("m_") and ud.kind_of(conn, m) == "member" and ud.kind_of(conn, g) is None
    assert ud.answers(conn, m) == {1: "like"}
    h = ud.history(conn, m)
    assert h["searches"][0]["prompt"] == "like Hitch" and [r["movie_id"] for r in h["searches"][0]["results"]] == [2, 3]
    assert ud.sign_in(conn, "user_abc") == m                       # same account later: same profile


def test_guest_history_merges_into_an_existing_member(conn):
    m = ud.sign_in(conn, "user_xyz")
    ud.add_reaction(conn, m, 1, "like", "quiz")
    g = ud.ensure_guest(conn, ud.new_id("g"))
    ud.add_reaction(conn, g, 2, "like", "feed")
    assert ud.sign_in(conn, "user_xyz", g) == m
    assert ud.answers(conn, m) == {1: "like", 2: "like"} and ud.kind_of(conn, g) is None


def test_forget_me_keeps_reactions_but_not_the_person(conn):
    m = ud.sign_in(conn, "user_forget")
    ud.add_reaction(conn, m, 1, "like", "quiz")
    ud.save_taste(conn, m, [0.1, 0.2, 0.3], [1.0, -1.0], 1, 1, {"likes": 1})
    anon = ud.forget(conn, m)
    assert anon.startswith("a_") and ud.kind_of(conn, m) is None and ud.kind_of(conn, anon) == "anonymous"
    assert ud.answers(conn, anon) == {1: "like"}                   # the training signal stays
    assert conn.execute("SELECT clerk_user_id FROM profiles WHERE profile_id = %s", (anon,)).fetchone()[0] is None
    assert conn.execute("SELECT count(*) FROM taste_profiles WHERE profile_id IN (%s, %s)", (m, anon)).fetchone()[0] == 0
    assert ud.sign_in(conn, "user_forget") != anon                 # signing in again starts a fresh profile


def test_search_counter(conn):
    g = ud.ensure_guest(conn, ud.new_id("g"))
    for _ in range(3):
        ud.log_search(conn, g, "x", {})
    assert ud.searches_since(conn, g, 60) == 3
