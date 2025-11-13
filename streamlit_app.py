# streamlit_app.py
# -------------------------------------------------------
# Discover Title Lab (UI)
# - Works with engine/scorer.py (global model)
# - GPT for generation (one-call multi-angle)
# - Exclamation-ban + basic risk/hallucination checks
# - Scores displayed 0..100 via score_titles()

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from engine.scorer import score_titles  # external scorer

# ---------------------------
# Init
# ---------------------------
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4.1")  # default model for generation

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set. Put it in your .env and restart.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Discover Title Lab", page_icon="📰", layout="wide")

# --- Simple password gate ---
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("DISCOVER SMASHER 4000")
    st.caption("With great power, comes great responsibility. - Uncle Ben")
    pwd = st.text_input("Password", type="password")
    if st.button("Enter"):
        if APP_PASSWORD and pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# ---------------------------
# Constants / Helpers
# ---------------------------
ANGLES = [
    "Authority + Number + Emotion + Controversy",
    "Explainer / How It Works",
    "Winners & Losers / Meta Shift",
    "Patch/Update Impact",
    "Guide / Best Builds",
    "Beginners’ Tips",
    "Hard Truth / Unpopular Opinion",
    "Data / Stats Lead",
    "Exclusive / First Look (ethical)",
    "Speculation (grounded)",
    "Curiosity",
]

BANNED_PHRASES = [
    r"\byou won'?t believe\b",
    r"\bshocking\b",
    r"\bclick here\b",
    r"\bviral\b",
    r"\bmust[- ]see\b",
]

# Precompiled regex for speed
RE_LINE_BULLET = re.compile(r"^\s*[-*\d\.\)]\s*")
RE_ANGLE_PREFIX = re.compile(r"^\[([^\]]+)\]\s*(.+)$")
RE_EMOJI = re.compile(
    r"[\U0001F600-\U0001F64F"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF]"
)
RE_WORDS = re.compile(r"\b\w+'\w+|\w+\b")
RE_BANNED = [re.compile(pat) for pat in BANNED_PHRASES]


def supports_temperature(model: str) -> bool:
    """
    We only pass 'temperature' for non-5.x models.
    """
    m = model.lower()
    if "gpt-5" in m:
        return False
    return True


def try_parse_json(text: str) -> List[Dict[str, Any]]:
    """
    Accepts JSON array OR newline bullets — returns list of {title, angle}
    """
    text = text.strip()

    # Attempt JSON parse first
    try:
        data = json.loads(text)
        # Object: maybe {"results":[...]} or {"headlines":[...]}
        if isinstance(data, dict):
            if "results" in data:
                data = data["results"]
            elif "headlines" in data:
                data = data["headlines"]
        if isinstance(data, list):
            out: List[Dict[str, Any]] = []
            for item in data:
                if isinstance(item, dict) and "title" in item:
                    angle = item.get("angle", "")
                    out.append(
                        {
                            "title": str(item["title"]).strip(),
                            "angle": str(angle).strip(),
                        }
                    )
                elif isinstance(item, str):
                    out.append({"title": item.strip(), "angle": ""})
            return out
    except Exception:
        pass

    # Fallback: parse list-like text lines
    titles: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = RE_LINE_BULLET.sub("", line).strip()
        if not line:
            continue
        m = RE_ANGLE_PREFIX.match(line)
        if m:
            titles.append({"title": m.group(2).strip(), "angle": m.group(1).strip()})
        else:
            titles.append({"title": line, "angle": ""})
    return titles


def assess_risk(title: str) -> Tuple[str, List[str]]:
    """
    Simple “risk” heuristic for clickbait/hallucination flags.
    Returns ("OK" or "Risky", reasons[])
    """
    reasons: List[str] = []

    if "!" in title:
        reasons.append("exclamation mark")

    low = title.lower()
    for pat in RE_BANNED:
        if pat.search(low):
            reasons.append("clickbait phrase")

    n_chars = len(title)
    if n_chars < 60:
        reasons.append("very short (<60 chars)")
    if n_chars > 130:
        reasons.append("very long (>130 chars)")

    if RE_EMOJI.search(title):
        reasons.append("emoji")

    if reasons:
        return "Risky", reasons
    return "OK", []


def enforce_no_exclamation(title: str) -> str:
    return title.replace("!", "")


def words_count(s: str) -> int:
    return len(RE_WORDS.findall(s))


def per_angle_counts(total: int, k: int) -> List[int]:
    """
    Split 'total' across k angles, distributing remainders.
    """
    base = total // k
    rem = total % k
    return [base + (1 if i < rem else 0) for i in range(k)]


# ---------------------------
# LLM Generation
# ---------------------------
SYSTEM = """You are an expert Google Discover headline optimizer for a large
gaming and entertainment news site.

Your ONLY job is to write feed-first headlines that
perform as well as possible in the Google Discover feed.

Assume:
- Headlines appear in a mobile swipe feed beside a thumbnail image.
- Readers are semi-familiar with the game/franchise but may not know the
  specific update, drama, build, or leak.
- The goal is maximum qualified engagement and sustained Discover visibility,
  NOT short-term clickbait spikes.

Global rules (obey ALL of them):
- NEVER invent or assume dates, years, seasons, timelines, or release windows.
- Only include a year or date in a headline if it is explicitly stated
  in the story brief. If the brief does not mention a date/year, the
  headline MUST NOT contain one.
- Put the PRIMARY ENTITY (game / franchise / platform / studio) within the
  FIRST 35 CHARACTERS whenever possible.
- Prefer 70–110 characters. Absolute hard limits:
  - never shorter than 55 characters
  - never longer than 115 characters
- Every headline must clearly communicate the payoff:
  guide, ranking, patch impact, drama explained, buff/nerf, meta shift, etc.
  No vague curiosity gaps.
- Use concrete details: numbers, named entities, patch names, dates or seasons
  ONLY when they add clarity or urgency and are present in the brief.
- Avoid clickbait and manipulation:
  - no “you won’t believe”, “shocking”, “insane”, “mind-blowing”,
    “must-see”, “click here”, emoji, or ALL CAPS.
- Avoid misleading or speculative claims. If something is uncertain, phrase it
  conditionally (“could”, “might”, “appears to”, “so far”).
- Avoid ambiguous pronouns when the subject might be unclear
  (“this change”, “they”, “it”) — name the thing.
- Assume the image already shows the game; don’t waste characters repeating
  obvious visual information unless it adds new meaning.
- Vary headline structures across a batch so 10–20 options are not near
  duplicates. Mix:
  - colon formats
  - how-to / explainer
  - rankings or tier lists
  - X vs Y comparisons
  - patch/nerf/buff impact
  - winners & losers
  - data/usage stats framing
- Respect the publisher’s reputation: headlines must feel authoritative,
  knowledgeable, and aligned with editorial standards.

Output requirements:
- Return JSON ONLY, no prose.
- Either:
  - an object: {"results":[{"title":"...","angle":"..."}]}
  - OR a raw JSON array: [{"title":"...","angle":"..."}].
"""

ANGLE_GUIDANCE = {
    "Authority + Number + Emotion + Controversy": (
        "Fuse authority sources / figure, specific numerical detail, emotional stake, "
        "and grounded controversy (no sensationalism)."
    ),
    "Explainer / How It Works": (
        "Lead with clarity and payoff—explain the mechanism or impact succinctly."
    ),
    "Winners & Losers / Meta Shift": (
        "Highlight shifts, buffs/nerfs, clear winner/loser framing, avoid opinionated exaggeration."
    ),
    "Patch/Update Impact": (
        "Center on patch/update changes and what that means—concrete features, timelines, outcomes."
    ),
    "Guide / Best Builds": "Actionable builds/loadouts with explicit benefits—be precise.",
    "Beginners’ Tips": (
        "Accessible, welcoming, practical advice for new or returning players."
    ),
    "Hard Truth / Unpopular Opinion": (
        "Present a defensible critique grounded in facts; avoid inflammatory tone."
    ),
    "Data / Stats Lead": "Open with verifiable figures, comparisons, trend lines.",
    "Exclusive / First Look (ethical)": (
        "If not truly exclusive, reframe as 'early look' or 'preview'—never mislead."
    ),
    "Speculation (grounded)": (
        "Use conditional phrasing with support (leaks, patterns, official hints) and avoid certainty."
    ),
    "Curiosity": "Use a curious approach for maximum engagement, but not clickbait.",
}


def build_user_prompt_multi(brief: str, angles: List[str], counts: List[int]) -> str:
    """
    Ask the model to produce all angles in one JSON object, with strong
    Google Discover bias.
    """
    angle_explanations = ""
    for a in angles:
        angle_explanations += f"- {a}: {ANGLE_GUIDANCE.get(a, '')}\n"

    pairs = "\n".join([f"- {a} :: {c}" for a, c in zip(angles, counts)])

    return f"""
ARTICLE BRIEF (editor-written):
{brief.strip()}

PUBLISHER CONTEXT:
- Large gaming & entertainment news outlet with strong topical authority
  across major franchises.
- Audience mix: returning readers + new players who see the card in Discover.
- Vertical: news, guides, builds, rankings, meta shifts, patches, leaks,
  beginner explainers, opinion pieces, and long-term guides.

GOAL:
- Generate multiple headline candidates that maximize the probability of
  being surfaced, clicked, and sustained in Google Discover, while staying
  factual and non-clickbait.

ANGLES & COUNTS (angle :: requested_count):
{pairs}

ANGLE EXPLANATIONS:
{angle_explanations}

STRICT REQUIREMENTS (do not violate):
- Every result MUST include an "angle" field that exactly matches one of the
  angles listed above.
- Total number of titles MUST equal the total requested count.
- No exclamation marks.
- 55–115 characters per title; aim mostly for 70–110.
- Put the main entity (game / franchise / platform / studio) as early as
  natural (ideally within the first 35 characters).
- Make the payoff explicit (what the article actually delivers).
- Avoid banned phrases and low-quality tricks:
  "you won't believe", "shocking", "insane", "mind-blowing", "must see",
  "click here", emoji, ALL CAPS words, or fake urgency.
- Do not fabricate numbers, leaks, quotes, or promises.
- NEVER invent dates, years, seasons, or timelines that are not present
  in the article brief.

RETURN FORMAT (VERY IMPORTANT):
Return VALID JSON only, in this exact schema:

{{
  "results": [
    {{"title": "string", "angle": "one of the provided angles"}},
    ...
  ]
}}

If you cannot produce valid results, return:
{{"results": []}}.
"""


def _parse_and_enforce(
    content: str, default_angle: Optional[str] = None
) -> List[Dict[str, Any]]:
    parsed = try_parse_json(content)
    if not parsed:
        parsed = [
            {"title": line, "angle": default_angle or ""}
            for line in content.splitlines()
            if line.strip()
        ]
    for item in parsed:
        item["title"] = enforce_no_exclamation(item.get("title", ""))
    return parsed


def generate_titles(
    brief: str,
    angles: List[str],
    total_count: int,
    model: str,
) -> List[Dict[str, Any]]:
    """
    One-call generation: ask the model for all angles at once.
    Enforces per-angle counts and exclamation ban.
    """
    if not angles:
        angles = [ANGLES[0]]

    counts = per_angle_counts(total_count, len(angles))

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_prompt_multi(brief, angles, counts)},
    ]

    kwargs: Dict[str, Any] = dict(model=model, messages=messages)
    if supports_temperature(model):
        kwargs["temperature"] = 0.9

    with st.spinner("Contacting model and generating titles..."):
        resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ""

    items: List[Dict[str, Any]] = []
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            raw = data.get("results")
            if isinstance(raw, list):
                items = raw
        elif isinstance(data, list):
            items = data
    except Exception:
        pass

    if not items:
        items = _parse_and_enforce(content)

    normalized: List[Dict[str, Any]] = []
    for it in items:
        t = str(it.get("title", "")).strip()
        a = str(it.get("angle", "")).strip() or angles[0]
        if not t:
            continue
        normalized.append({"title": enforce_no_exclamation(t), "angle": a})

    desired = {a: c for a, c in zip(angles, counts)}
    used_counts = {a: 0 for a in angles}
    out: List[Dict[str, Any]] = []

    # First pass respecting requested angles
    for it in normalized:
        a = it["angle"]
        if a not in desired:
            a = angles[0]
        if used_counts[a] < desired[a]:
            out.append({"title": it["title"], "angle": a})
            used_counts[a] += 1

    # Second pass: top up if short
    if len(out) < total_count:
        for it in normalized:
            if len(out) >= total_count:
                break
            if any(o["title"] == it["title"] for o in out):
                continue
            out.append(
                {"title": it["title"], "angle": it.get("angle", angles[0]) or angles[0]}
            )

    return out[:total_count]


# ---------------------------
# UI
# ---------------------------
st.title("DISCOVER SMASHER 4000")
st.caption("Generate ethically high-CTR titles and score them with your local model.")

with st.sidebar:
    st.subheader("Generation")
    default_brief = (
        "Add a brief containing an authority, a number and an emotion/controversy."
    )
    brief = st.text_area(
        "Story brief",
        value=default_brief,
        height=160,
        help=(
            "Give enough context for an editor: who/what/why now, constraints, audience."
        ),
    )

    angles_selected = st.multiselect(
        "Angle(s)",
        options=ANGLES,
        default=[
            "Authority + Number + Emotion + Controversy",
            "Explainer / How It Works",
            "Patch/Update Impact",
            "Data / Stats Lead",
        ],
        help="Pick 1+ angles; the app divides the requested count across them.",
    )

    count = st.number_input(
        "How many titles in total?", min_value=4, max_value=60, value=16, step=2
    )

    gen_model = st.text_input(
        "OpenAI model",
        value=GEN_MODEL,
        help=(
            "Your generation model (default gpt-4.1). "
            "Temperature is only passed for models that support it."
        ),
    )

    st.markdown("---")
    go = st.button("Generate & Score", type="primary", use_container_width=True)

if go:
    if not brief.strip():
        st.error("Please provide a brief.")
        st.stop()

    # Generate
    gen_titles = generate_titles(brief, angles_selected, int(count), gen_model)

    if not gen_titles:
        st.warning(
            "No titles were generated. Try again or adjust your angles/brief/model."
        )
        st.stop()

    # Score (0..100)
    raw_list = [t["title"] for t in gen_titles]
    scored_df = score_titles(raw_list)  # expects DataFrame with columns: title, pwin

    if not isinstance(scored_df, pd.DataFrame) or scored_df.empty:
        st.error("Scoring returned no data. Retrain your model, then try again.")
        st.stop()

    # Merge angle and derived meta
    df = pd.DataFrame(gen_titles)
    df = df.merge(scored_df, on="title", how="left", copy=False)
    df["score"] = (df["pwin"].fillna(0) * 100).round(1)
    df["chars"] = df["title"].str.len()
    df["words"] = df["title"].apply(words_count)

    # Vectorized risk calculation
    def _risk_flags(series: pd.Series) -> pd.DataFrame:
        s = series.astype(str)
        exclam = s.str.contains("!", regex=False)
        short = s.str.len().lt(60)
        long_ = s.str.len().gt(130)
        emoji = s.apply(lambda x: bool(RE_EMOJI.search(x)))
        clickbait = pd.Series(False, index=s.index)
        for pat in RE_BANNED:
            clickbait |= s.str.contains(pat)

        labels = pd.Series("OK", index=s.index)
        reasons = pd.Series("", index=s.index)

        risky = exclam | short | long_ | emoji | clickbait
        labels[risky] = "Risky"

        def add_reason(mask, text, current):
            cur = current.copy()
            to_update = mask & (cur != "")
            cur[to_update] = cur[to_update] + ", " + text
            to_update = mask & (cur == "")
            cur[to_update] = text
            return cur

        reasons = add_reason(exclam, "exclamation mark", reasons)
        reasons = add_reason(clickbait, "clickbait phrase", reasons)
        reasons = add_reason(short, "very short (<60 chars)", reasons)
        reasons = add_reason(long_, "very long (>130 chars)", reasons)
        reasons = add_reason(emoji, "emoji", reasons)

        reasons = reasons.fillna("")
        return pd.DataFrame({"risk": labels, "risk_reasons": reasons})

    df = pd.concat([df, _risk_flags(df["title"])], axis=1)

    # Sort by score desc
    df = df.sort_values("score", ascending=False, kind="mergesort").reset_index(
        drop=True
    )

    # ---------- Top Picks UI ----------
    st.subheader("Top Picks")
    top_container = st.container()
    rows = df.head(8).to_dict(orient="records")

    with top_container:
        for i, row in enumerate(rows, start=1):

            # Thin separator line + spacing
            st.markdown(
                """
                <div style="
                    padding: 12px 0px 6px 0px;
                    border-bottom: 1px solid rgba(255,255,255,0.12);
                    margin-bottom: 14px;
                ">
                """,
                unsafe_allow_html=True,
            )

            colL, colR = st.columns([0.78, 0.22])

            # LEFT SIDE: Title + meta
            with colL:
                st.markdown(
                    f"<b>{i}. {row['title']}</b>", unsafe_allow_html=True
                )
                st.caption(
                    f"Angle: {row.get('angle','—')} • "
                    f"{int(row['chars'])} chars • {int(row['words'])} words"
                )

            # RIGHT SIDE: Score + emoji band
            with colR:
                s = float(row["score"])
                if s >= 70:
                    q = "🟢 High"
                elif s >= 50:
                    q = "🟡 Medium"
                else:
                    q = "🟠 Low"

                st.metric("Score", f"{s:.1f}/100")
                st.caption(q)

            # Risk warning
            if row["risk"] == "Risky":
                st.warning(
                    f"Risk: {row['risk_reasons'] or 'Potential clickbait/formatting issues.'}"
                )

            st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Full table ----------
    with st.expander("See all results"):
        show_cols = [
            "title",
            "angle",
            "score",
            "pwin",
            "risk",
            "risk_reasons",
            "chars",
            "words",
        ]
        present = [c for c in show_cols if c in df.columns]
        st.dataframe(df[present], use_container_width=True)

    # ---------- JSON export ----------
    with st.expander("Copy JSON"):
        payload = [
            {
                "title": r["title"],
                "angle": r.get("angle", ""),
                "score": float(r["score"]),
            }
            for _, r in df.iterrows()
        ]
        st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")

else:
    st.info(
        "Fill the brief, pick angles, choose how many titles to generate, then click **Generate & Score**."
    )