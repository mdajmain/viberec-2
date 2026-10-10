"""LLM vibe tagging through Bedrock's Converse API with a forced tool (structured JSON answer) (Step 5).

One request per movie. The answer has: 0-3 scores for the Genome-checkable attributes; light_heavy, pacing
(1-5) and ending, on the same scales as Adil's hand labels (data/labels/vibe_labels_v2.xlsx); a few
categorical fields from PROJECT.md §7; and a 100-150 word vibe paragraph for the vibe vector.
"""
import json

from .genome import NOT_GENOME_CHECKABLE, VIBE_MAP

ENDINGS = ["happy", "bittersweet", "sad", "open"]
ARCS = ["rags_to_riches", "riches_to_rags", "man_in_a_hole", "icarus", "cinderella", "oedipus", "flat", "unclear"]
LIFE_STAGES = ["child", "teen", "student", "young_adult", "adult", "parent", "middle_aged", "elderly", "mixed", "unclear"]
SCALE = "0 = not present, 1 = slightly, 2 = clearly present, 3 = central to the movie"

ATTRIBUTES = {a: tags for d in VIBE_MAP.values() for a, tags in d.items() if a not in NOT_GENOME_CHECKABLE}

SYSTEM = """You describe how movies FEEL, for a vibe-based recommender. Base every judgment on the text provided \
(plot, critics' reception, keywords). If the text does not support a judgment, use the lowest level. \
Do not reward fame or quality: a famous movie is not automatically 'heartwarming' or 'demanding'."""


THREE = ["light_heavy", "pacing", "ending_type"]          # Adil's three hand-labelled vibes


def tool_spec(compact=False, keep=None):
    """Bedrock Converse tool definition; the model is forced to call it, so the answer is always this JSON.
    compact=True asks only for the 41 vibe scores and the three hand-labelled vibes (about half the output tokens).
    keep=[fields] asks only for those fields (keep=THREE: the three hand-labelled vibes, no 41-vibe list)."""
    attr_props = {a: {"type": "integer", "enum": [0, 1, 2, 3], "description": f"{', '.join(tags)}. {SCALE}"}
                  for a, tags in ATTRIBUTES.items()}
    schema = {
        "type": "object",
        "properties": {
            "attributes": {"type": "object", "properties": attr_props, "required": list(attr_props)},
            "light_heavy": {"type": "integer", "enum": [1, 2, 3, 4, 5],
                            "description": "1 = heavy: demanding or draining, you have to be in the mood; "
                                           "5 = light: easy and feel-good, you would put it on any time"},
            "pacing": {"type": "integer", "enum": [1, 2, 3, 4, 5],
                       "description": "1 = slow: lingering and quiet, long scenes; 5 = fast: always moving, quick scenes"},
            "ending_type": {"type": "string", "enum": ENDINGS,
                            "description": "happy = things work out; bittersweet = something won and something lost; "
                                           "sad = ends in loss or tragedy; open = ambiguous or unresolved"},
            "emotional_arc": {"type": "string", "enum": ARCS,
                              "description": "shape of the protagonist's fortunes: rags_to_riches (rise), riches_to_rags (fall), "
                                             "man_in_a_hole (fall then rise), icarus (rise then fall), cinderella (rise, fall, rise), "
                                             "oedipus (fall, rise, fall)"},
            "protagonist_life_stage": {"type": "string", "enum": LIFE_STAGES},
            "protagonist_job": {"type": "string", "description": "main character's job in 1-3 words, or 'none' / 'unclear'"},
            "vibe_paragraph": {"type": "string",
                               "description": "100-150 words on how the movie FEELS to watch: tone, mood, pacing, emotional journey, "
                                              "setting atmosphere, who it suits and when to watch it. No plot spoilers beyond the premise."},
        },
        "required": ["attributes", "light_heavy", "pacing", "ending_type", "emotional_arc",
                     "protagonist_life_stage", "protagonist_job", "vibe_paragraph"],
    }
    if compact or keep:
        keep = keep or ["attributes", "light_heavy", "pacing", "ending_type"]
        schema = {"type": "object", "properties": {k: schema["properties"][k] for k in keep}, "required": keep}
    return {"tools": [{"toolSpec": {"name": "record_vibe", "description": "Record the movie's vibe.",
                                    "inputSchema": {"json": schema}}}],
            "toolChoice": {"tool": {"name": "record_vibe"}}}


def build_prompt(title, year, genres, plot, reception=None, keywords=None, max_plot_words=1200, max_reception_words=400):
    """The user message. Long sections are cut to bound cost; the cut is marked."""
    def cut(text, n):
        words = text.split()
        return text if len(words) <= n else " ".join(words[:n]) + " [...]"
    parts = [f"Movie: {title} ({year})", f"Genres: {genres.replace('|', ', ')}", f"\nPlot:\n{cut(plot, max_plot_words)}"]
    if reception:
        parts.append(f"\nCritics' reception:\n{cut(reception, max_reception_words)}")
    if keywords:
        parts.append(f"\nKeywords: {', '.join(keywords)}")
    parts.append("\nRecord this movie's vibe with the record_vibe tool.")
    return "\n".join(parts)


def call(client, model_id, prompt, max_tokens=1500):
    """One Converse request. Returns (answer dict, usage dict)."""
    resp = client.converse(modelId=model_id, system=[{"text": SYSTEM}],
                           messages=[{"role": "user", "content": [{"text": prompt}]}],
                           toolConfig=tool_spec(), inferenceConfig={"maxTokens": max_tokens, "temperature": 0})
    blocks = [b["toolUse"]["input"] for b in resp["output"]["message"]["content"] if "toolUse" in b]
    if not blocks:
        raise ValueError(f"no tool call in the answer (stopReason={resp.get('stopReason')})")
    return blocks[0], resp["usage"]


def check_answer(answer):
    """Problems with an answer (empty list = valid): missing attributes or out-of-range values."""
    problems = [f"missing attribute {a}" for a in ATTRIBUTES if a not in answer.get("attributes", {})]
    problems += [f"{a}={v} not in 0-3" for a, v in answer.get("attributes", {}).items() if v not in (0, 1, 2, 3)]
    for k in ("light_heavy", "pacing"):
        if answer.get(k) not in (1, 2, 3, 4, 5):
            problems.append(f"{k}={answer.get(k)} not in 1-5")
    if answer.get("ending_type") not in ENDINGS:
        problems.append(f"ending_type={answer.get('ending_type')} not one of {ENDINGS}")
    words = len(str(answer.get("vibe_paragraph", "")).split())
    if not 60 <= words <= 250:
        problems.append(f"vibe_paragraph has {words} words")
    return problems


def to_json(answer):
    return json.dumps(answer, ensure_ascii=False)


def call_guarded(client, guard, model_id, prompt, compact=True, max_tokens=900, keep=None):
    """One tagging request through the budget guard (llm_budget). Returns (answer dict, usage dict)."""
    from .llm_budget import guarded_converse
    resp = guarded_converse(client, guard, "vibe tagging", model_id, [{"role": "user", "content": [{"text": prompt}]}],
                            {"maxTokens": max_tokens, "temperature": 0}, system=[{"text": SYSTEM}],
                            toolConfig=tool_spec(compact, keep))
    blocks = [b["toolUse"]["input"] for b in resp["output"]["message"]["content"] if "toolUse" in b]
    if not blocks:
        raise ValueError(f"no tool call in the answer (stopReason={resp.get('stopReason')})")
    return blocks[0], resp["usage"]


def check_compact(answer):
    """Problems with a compact answer (empty list = valid)."""
    return [p for p in check_answer({**answer, "vibe_paragraph": "word " * 100}) if not p.startswith("vibe_paragraph")]


def check_three(answer):
    """Problems with a three-vibe answer (empty list = valid)."""
    problems = [f"{k}={answer.get(k)} not in 1-5" for k in ("light_heavy", "pacing") if answer.get(k) not in (1, 2, 3, 4, 5)]
    if answer.get("ending_type") not in ENDINGS:
        problems.append(f"ending_type={answer.get('ending_type')} not one of {ENDINGS}")
    return problems
