# streamlit_app.py
# -------------------------------------------------------
# Discover Title Lab (UI)
# - Works with engine/scorer.py (global model)
# - GPT-5 for generation (temperature forced to 1)
# - Multi-angle generation, count control, progress bar
# - Exclamation-ban + basic risk/hallucination checks
# - Scores displayed 0..100 via score_titles()

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from engine.scorer import score_titles  # expects the new scorer you just installed

# ---------------------------
# Init
# ---------------------------
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-5")  # default to GPT-5 as you requested

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set. Put it in your .env and restart.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Helpers
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
    "Curiosity"
]

def supports_temperature(model: str) -> bool:
    """
    GPT-5 chat.completions typically lock temp=1.
    We only pass 'temperature' for older 4.x models or other permissive models.
    """
    m = model.lower()
    if "gpt-5" in m:
        return False
    # allow for 4.x families
    return True

def try_parse_json(text: str) -> List[Dict[str, Any]]:
    """
    Accepts JSON array OR newline bullets — returns list of {title, angle}
    """
    text = text.strip()
    # Attempt JSON array first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            out = []
            for item in data:
                if isinstance(item, dict) and "title" in item:
                    angle = item.get("angle", "")
                    out.append({"title": str(item["title"]).strip(), "angle": str(angle).strip()})
                elif isinstance(item, str):
                    out.append({"title": item.strip(), "angle": ""})
            return out
    except Exception:
        pass

    # Fallback: parse list-like text lines
    titles: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[-*\d\.\)]\s*", "", line).strip()
        if not line:
            continue
        # split if angle prefix pattern like "[Angle] Title"
        m = re.match(r"^\[([^\]]+)\]\s*(.+)$", line)
        if m:
            titles.append({"title": m.group(2).strip(), "angle": m.group(1).strip()})
        else:
            titles.append({"title": line, "angle": ""})
    return titles

BANNED_PHRASES = [
    r"\byou won'?t believe\b",
    r"\bshocking\b",
    r"\bclick here\b",
    r"\bviral\b",
    r"\bmust[- ]see\b",
]

def assess_risk(title: str) -> Tuple[str, List[str]]:
    """
    Simple “risk” heuristic for clickbait/hallucination flags.
    Returns ("OK" or "Risky", reasons[])
    """
    reasons = []

    # Ban exclamation marks completely per your requirement
    if "!" in title:
        reasons.append("exclamation mark")

    # Clickbait phrases
    low = title.lower()
    for pat in BANNED_PHRASES:
        if re.search(pat, low):
            reasons.append("clickbait phrase")

    # Overlong/too-short for Discover (you set 90–105 sweet spot)
    n_chars = len(title)
    if n_chars < 60:
        reasons.append("very short (<60 chars)")
    if n_chars > 130:
        reasons.append("very long (>130 chars)")

    # Emojis
    if re.search(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
                 r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]", title):
        reasons.append("emoji")

    if reasons:
        return "Risky", reasons
    return "OK", []

def enforce_no_exclamation(title: str) -> str:
    return title.replace("!", "")

def words_count(s: str) -> int:
    return len(re.findall(r"\b\w+'\w+|\w+\b", s))

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
SYSTEM = """You are a Google Discover headline generator for a gaming/entertainment publisher.
Produce titles that have the highest chance to be shown on Google Discover.
Prefer 90–105 characters for Discover. Include curiosity, authority source or figure, specific numerical detail, emotion, or grounded controversy when appropriate.
You must create scroll stopping titles, but not click baity. We need maximum engagement.
Fact check the numbers, we must not mislead users.
Return JSON ONLY as an array of objects: [{"title": "...", "angle": "..."}]."""

ANGLE_GUIDANCE = {
    "Authority + Number + Emotion + Controversy":
        "Fuse authority sources / figure, specific numerical detail, emotional stake, and grounded controversy (no sensationalism).",
    "Explainer / How It Works":
        "Lead with clarity and payoff—explain the mechanism or impact succinctly.",
    "Winners & Losers / Meta Shift":
        "Highlight shifts, buffs/nerfs, clear winner/loser framing, avoid opinionated exaggeration.",
    "Patch/Update Impact":
        "Center on patch/update changes and what that means—concrete features, timelines, outcomes.",
    "Guide / Best Builds":
        "Actionable builds/loadouts with explicit benefits—be precise.",
    "Beginners’ Tips":
        "Accessible, welcoming, practical advice for new or returning players.",
    "Hard Truth / Unpopular Opinion":
        "Present a defensible critique grounded in facts; avoid inflammatory tone.",
    "Data / Stats Lead":
        "Open with verifiable figures, comparisons, trend lines.",
    "Exclusive / First Look (ethical)":
        "If not truly exclusive, reframe as 'early look' or 'preview'—never mislead.",
    "Speculation (grounded)":
        "Use conditional phrasing with support (leaks, patterns, official hints) and avoid certainty.",
    "Curiosity":
        "Use a curious approach for maximum engagement, but not clickbait."
}

def build_user_prompt(brief: str, angle: str, count: int) -> str:
    gl = ANGLE_GUIDANCE.get(angle, "")
    return f"""
BRIEF:
{brief.strip()}

ANGLE:
{angle} — {gl}

REQUIREMENTS:
- Return exactly {count} diverse titles.
- No exclamation marks.
- Maximize visibility for Google Discover.
- Prefer 90–105 characters; never exceed 130.
- Ethical wording; no hype or unverifiable claims.
- JSON array only: [{{"title": "...", "angle": "{angle}"}}]
"""

def generate_for_angle(brief: str, angle: str, count: int, model: str) -> List[Dict[str, Any]]:
    """
    Generate {count} titles for one angle; parse and return list of {title, angle}.
    Auto-omits 'temperature' for GPT-5 (temp=1).
    """
    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user_prompt(brief, angle, count)},
        ],
    )
    # Only pass temperature for permissive models
    if supports_temperature(model):
        kwargs["temperature"] = 0.9

    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ""

    parsed = try_parse_json(content)
    # Fallback if parsing failed: split by lines
    if not parsed:
        parsed = [{"title": line, "angle": angle} for line in content.splitlines() if line.strip()]

    # Enforce exclamation ban
    for item in parsed:
        item["title"] = enforce_no_exclamation(item.get("title", ""))

    # Truncate to requested count
    return parsed[:count]

def generate_titles(brief: str, angles: List[str], total_count: int, model: str) -> List[Dict[str, Any]]:
    """
    Split total_count across selected angles; stream progress.
    """
    if not angles:
        angles = [ANGLES[0]]
    counts = per_angle_counts(total_count, len(angles))

    out: List[Dict[str, Any]] = []
    progress = st.progress(0)
    steps = max(1, len(angles) + 1)
    done = 0

    for angle, n in zip(angles, counts):
        if n <= 0:
            done += 1
            progress.progress(min(1.0, done / steps))
            continue
        with st.spinner(f"Generating {n} titles for: {angle}"):
            chunk = generate_for_angle(brief, angle, n, model)
        out.extend(chunk)
        done += 1
        progress.progress(min(1.0, done / steps))

    progress.progress(1.0)
    return out

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Discover Title Lab", page_icon="📰", layout="wide")
st.title("DISCOVER SMASHER 4000")
st.caption("Generate ethically high-CTR titles and score them with your local model.")

with st.sidebar:
    st.subheader("Generation")
    default_brief = "Add a brief containing an authority, a number and an emotion/controversy."
    brief = st.text_area("Story brief", value=default_brief, height=160,
                         help="Give enough context for an editor: who/what/why now, constraints, audience.")
    angles_selected = st.multiselect("Angle(s)", options=ANGLES, default=[
        "Authority + Number + Emotion + Controversy",
        "Explainer / How It Works",
        "Patch/Update Impact",
        "Data / Stats Lead",
    ], help="Pick 1+ angles; the app divides the requested count across them.")
    count = st.number_input("How many titles in total?", min_value=4, max_value=60, value=16, step=2)
    gen_model = st.text_input("OpenAI model", value=GEN_MODEL,
                              help="Your generation model (default gpt-5). Temperature is only passed for models that support it.")
    st.markdown("---")
    go = st.button("Generate & Score", type="primary", use_container_width=True)

if go:
    if not brief.strip():
        st.error("Please provide a brief.")
        st.stop()

    # Generate
    gen_titles = generate_titles(brief, angles_selected, int(count), gen_model)

    if not gen_titles:
        st.warning("No titles were generated. Try again or adjust your angles/brief.")
        st.stop()

    # Score (0..100)
    raw_list = [t["title"] for t in gen_titles]
    scored_df = score_titles(raw_list)  # returns DataFrame with columns: title, pwin

    if not isinstance(scored_df, pd.DataFrame) or scored_df.empty:
        st.error("Scoring returned no data. Retrain your model, then try again.")
        st.stop()

    # Merge angle and derived meta
    df = pd.DataFrame(gen_titles)  # ensures we keep 'angle' column
    df = df.merge(scored_df, on="title", how="left")
    df["score"] = (df["pwin"].fillna(0) * 100).round(1)
    df["chars"] = df["title"].map(len)
    df["words"] = df["title"].map(words_count)

    risks: List[str] = []
    risk_labels: List[str] = []
    for t in df["title"]:
        label, reasons = assess_risk(t)
        risk_labels.append(label)
        risks.append(", ".join(reasons))
    df["risk"] = risk_labels
    df["risk_reasons"] = risks

    # Sort by score desc
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Display
    st.subheader("Top Picks")
    for i, row in df.head(8).iterrows():
        colL, colR = st.columns([0.78, 0.22])
        with colL:
            st.markdown(f"**{i+1}. {row['title']}**")
            st.caption(f"Angle: {row.get('angle','') or '—'}  •  {int(row['chars'])} chars, {int(row['words'])} words")
        with colR:
            # Quality badge: simple banding
            s = float(row["score"])
            if s >= 70:
                q = "✅ High"
            elif s >= 50:
                q = "🟡 Medium"
            else:
                q = "🟠 Low"
            st.metric("Score", f"{s:.1f}/100", help="Model-estimated win probability × 100")
            st.caption(q)

        # Risk pill
        if row["risk"] == "Risky":
            st.warning(f"Risk: {row['risk_reasons'] or 'Potential clickbait/formatting issues.'}")

        st.divider()

    # Full table
    with st.expander("See all results"):
        show_cols = ["title", "angle", "score", "pwin", "risk", "risk_reasons", "chars", "words"]
        present = [c for c in show_cols if c in df.columns]
        st.dataframe(df[present], use_container_width=True)

    # Copyable JSON
    with st.expander("Copy JSON"):
        payload = [{"title": r["title"], "angle": r.get("angle", ""), "score": float(r["score"])} for _, r in df.iterrows()]
        st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")

else:
    st.info("Fill the brief, pick angles, choose how many titles to generate, then click **Generate & Score**.")