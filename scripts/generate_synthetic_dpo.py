#!/usr/bin/env python3
"""
Generate synthetic DPO preference pairs for Trace Layer 2 policy shaping.

Mirrors generate_synthetic_sft.py — template-driven, seeded RNG, balanced
output. Produces (prompt, chosen, rejected) preference pairs targeting the
three Trace failure axes plus a coaching-voice axis and distractors.

Five families:
  A — over_flatten        (~28%): aspiration-looking inputs with real prospective signal
  B — add_coaching        (~28%): rejected adds prescriptive/coaching framing
  C — mis_tier_mixed      (~28%): episodic event + prospective commitment combined
  D — conditional_commitment (~9%): conditional but specific commitment
  E — schema_drift           (~8%): unambiguous tier vs cleanly-wrong tier (distractors)

Output: JSONL records of the form
  {"prompt": "...", "chosen": "...", "rejected": "...", "metadata": {...}}

Usage:
  # Train set (default seed=42)
  python scripts/generate_synthetic_dpo.py --n 80 --seed 42 \\
      --output data/processed/dpo/train.jsonl

  # Validation set (different seed for non-overlapping cases)
  python scripts/generate_synthetic_dpo.py --n 12 --seed 99 \\
      --output data/processed/dpo/valid.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

# ── Constants ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a Trace memory extraction engine. Transform the raw trace input into a "
    "structured Trace-style memory record. Output valid JSON only. Do not add advice, "
    "coaching, prescriptive framing, or interpretation. Witness and structure; do not "
    "suggest or evaluate."
)

REQUIRED_FIELDS = (
    "memory_tier",
    "content_summary",
    "stated_intent",
    "emotional_valence",
    "topic_cluster",
    "timestamp",
    "source",
    "source_id",
    "channel",
)

VALID_TIERS = {"episodic", "semantic", "procedural", "prospective"}
VALID_VALENCES = {"positive", "neutral", "negative", "mixed"}

# Family proportions — must sum to ≤ 1.0; remainder rounded into the largest bucket
FAMILY_PROPORTIONS = {
    "over_flatten": 25 / 90,
    "add_coaching": 25 / 90,
    "mis_tier_mixed": 25 / 90,
    "conditional_commitment": 8 / 90,
    "schema_drift": 7 / 90,
}

# ── Lexicons ──────────────────────────────────────────────────────────────────

PEOPLE = [
    "David", "Claire", "Marcus", "Jordan", "Sarah", "Alex", "Priya", "Diego",
    "Emma", "Liam", "Mia", "Noah", "Aisha", "Ben", "Zoe", "Chen",
    "Ravi", "Kira", "Ivan", "Olivia", "Theo", "Maya", "Ezra", "Lila",
]

DEFERRAL_DURATIONS = [
    "weeks", "almost a month", "over a month", "two months",
    "longer than I want to admit", "ages", "the better part of a quarter",
]

SHORT_TIMEFRAMES = [
    "this week", "by end of week", "before Friday", "by Friday",
    "this Sunday", "before next Monday", "in the next few days",
]

WORK_DELIVERABLES = [
    ("retrospective write-up", "team", "engineering / project management"),
    ("design doc", "Jordan", "engineering / design"),
    ("post-mortem", "incident review channel", "engineering / incidents"),
    ("migration guide", "the customer success team", "engineering / documentation"),
    ("runbook update", "the on-call rotation", "engineering / operations"),
    ("API documentation", "developer relations", "engineering / documentation"),
    ("benchmark report", "leadership", "engineering / performance"),
    ("RFC draft", "the architecture group", "engineering / design"),
    ("review notes", "Diego", "engineering / collaboration"),
    ("hiring rubric", "Priya", "management / hiring"),
]

OVERDUE_HEALTH_TASKS = [
    ("schedule my annual physical", "health / preventive care",
     "I keep putting off scheduling my annual physical. It's been over a year."),
    ("get the dental cleaning rebooked", "health / dental",
     "I've been postponing the dental cleaning that I rescheduled twice already."),
    ("renew my prescription", "health / medication",
     "My prescription has been in renewal limbo for two weeks."),
    ("book the eye exam", "health / vision",
     "Haven't had an eye exam in three years and I keep saying I'll book one."),
]

PERSONAL_RELATIONSHIP_TASKS = [
    ("send a real message to {person} about the {topic}", "relationships / friendship",
     "I've been meaning to write to {person} about the {topic}. Every time I open the chat I close it again."),
    ("call {person} back about the {topic}", "relationships / family",
     "{person} left me a voicemail about the {topic} {duration} ago and I still haven't returned it."),
    ("follow up with {person} about the mentorship", "relationships / mentorship",
     "{person} mentioned the mentorship thing twice and I keep saying I'll follow up."),
]

PERSONAL_TOPICS = [
    "wedding", "the trip", "the new house", "the project", "the conference",
    "the side gig", "the holidays", "the move",
]

EPISODIC_EVENTS = [
    ("Shipped the v2.{n}.{m} release today after {weeks} weeks of work",
     "engineering / releases", "v2.{n}.{m}"),
    ("Wrapped up the customer onboarding cycle this afternoon",
     "engineering / onboarding", "customer onboarding"),
    ("Demoed the new dashboard to {stakeholder} today",
     "engineering / demos", "dashboard demo"),
    ("Closed out the {project} migration this afternoon",
     "engineering / migration", "{project} migration"),
    ("Delivered the {deliverable} to the working group today",
     "engineering / delivery", "{deliverable} delivery"),
]

EPISODIC_PROJECTS = ["billing", "auth", "search", "ingest", "analytics"]
EPISODIC_STAKEHOLDERS = ["the leadership team", "the design partners", "the product council"]
EPISODIC_DELIVERABLES = ["proposal", "evaluation", "spec", "audit report"]

CHANNELS_BY_CONTEXT = {
    "work": "work",
    "personal": "personal",
    "health": "personal",
    "learning": "personal",
}

# ── Date helpers ──────────────────────────────────────────────────────────────

BASE_DATE = date(2026, 4, 15)


def pick_date(rng: random.Random, window_days: int = 28) -> str:
    """Pick a date in a ±window/2 window around BASE_DATE."""
    offset = rng.randint(-window_days // 2, window_days // 2)
    d = BASE_DATE + timedelta(days=offset)
    return d.isoformat()


def case_id_for(seed: int, family: str, idx: int, prompt_text: str) -> str:
    h = hashlib.sha1(f"{seed}|{family}|{idx}|{prompt_text}".encode()).hexdigest()[:8]
    return f"dpo_{family[:4]}_{idx:03d}_{h}"


def trace_source_id(timestamp: str, family: str, idx: int) -> str:
    yyyymmdd = timestamp.replace("-", "")
    return f"trace_{yyyymmdd}_{family[:3]}{idx:03d}"


# ── Record assembly ───────────────────────────────────────────────────────────

def build_record(
    *,
    tier: str,
    summary: str,
    intent: str | None,
    valence: str,
    topic: str,
    timestamp: str,
    source_id: str,
    channel: str,
    source: str = "daily_trace",
) -> dict:
    return {
        "content_summary": summary,
        "memory_tier": tier,
        "emotional_valence": valence,
        "stated_intent": intent,
        "topic_cluster": topic,
        "timestamp": timestamp,
        "source": source,
        "source_id": source_id,
        "channel": channel,
    }


def build_pair_record(
    *,
    family: str,
    target_tier: str,
    seed: int,
    idx: int,
    user_input: str,
    chosen: dict,
    rejected: dict,
) -> dict:
    prompt = f"{SYSTEM_PROMPT}\n\nInput: {user_input}"
    return {
        "prompt": prompt,
        "chosen": json.dumps(chosen, ensure_ascii=False),
        "rejected": json.dumps(rejected, ensure_ascii=False),
        "metadata": {
            "axis": family,
            "target_tier": target_tier,
            "generation_seed": seed,
            "case_id": case_id_for(seed, family, idx, user_input),
        },
    }


# ── Family A — over_flatten ───────────────────────────────────────────────────
# Inputs that look like aspirations but contain a real prospective signal.
# Chosen: prospective with stated_intent.
# Rejected: semantic, stated_intent=null, collapsed to "general pattern".

def _a_persona_followup(rng: random.Random, seed: int, idx: int) -> dict:
    person = rng.choice(PEOPLE)
    duration = rng.choice(DEFERRAL_DURATIONS)
    timeframe = rng.choice(SHORT_TIMEFRAMES)
    avoid = rng.choice([
        "open my messages", "see their name in my inbox",
        "think about it during a quiet moment", "scroll past their last text",
    ])
    user_input = (
        f"I've been saying I'll reach out to {person} for {duration} now. "
        f"Every time I {avoid} I think about it and then close the app. "
        f"I should really just do it {timeframe}."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "over_flatten", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Repeated deferral of reaching out to {person} over {duration}; "
            f"self-stated intention to follow through {timeframe} despite the avoidance pattern."
        ),
        intent=f"Reach out to {person} {timeframe}.",
        valence="neutral",
        topic="relationships / networking",
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected = build_record(
        tier="semantic",
        summary=(
            f"Has a long-running pattern of avoiding outreach to {person}; "
            f"reflects on the avoidance without committing to action."
        ),
        intent=None,
        valence="neutral",
        topic="relationships / patterns of avoidance",
        timestamp=ts, source_id=sid, channel="personal",
    )
    return build_pair_record(
        family="over_flatten", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _a_overdue_health(rng: random.Random, seed: int, idx: int) -> dict:
    action, topic, lead_in = rng.choice(OVERDUE_HEALTH_TASKS)
    timeframe = rng.choice(SHORT_TIMEFRAMES)
    user_input = (
        f"{lead_in} I'm going to {action} {timeframe} — no more deferring."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "over_flatten", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Long-deferred health task; explicit self-commitment to {action} {timeframe}."
        ),
        intent=f"{action.capitalize()} {timeframe}.",
        valence="neutral",
        topic=topic,
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected = build_record(
        tier="semantic",
        summary=(
            f"Recognizes a pattern of deferring health admin tasks; "
            f"aware of the avoidance without a concrete plan."
        ),
        intent=None,
        valence="neutral",
        topic="self-knowledge / health avoidance",
        timestamp=ts, source_id=sid, channel="personal",
    )
    return build_pair_record(
        family="over_flatten", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _a_writing_commitment(rng: random.Random, seed: int, idx: int) -> dict:
    artifact = rng.choice([
        "the blog post draft", "the conference abstract", "the proposal",
        "the cover letter", "the workshop submission",
    ])
    duration = rng.choice(DEFERRAL_DURATIONS)
    timeframe = rng.choice(SHORT_TIMEFRAMES)
    user_input = (
        f"{artifact.capitalize()} has been sitting unfinished for {duration}. "
        f"Every weekend I tell myself I'll get to it. I'm finally going to send it {timeframe}."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "over_flatten", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"{artifact.capitalize()} has been deferred for {duration}; "
            f"explicit self-commitment to send {timeframe}."
        ),
        intent=f"Send {artifact} {timeframe}.",
        valence="neutral",
        topic="writing / commitments",
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected = build_record(
        tier="semantic",
        summary=(
            f"Long-running pattern of weekend procrastination on writing tasks; "
            f"awareness without a specific commitment."
        ),
        intent=None,
        valence="neutral",
        topic="self-knowledge / procrastination",
        timestamp=ts, source_id=sid, channel="personal",
    )
    return build_pair_record(
        family="over_flatten", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _a_conversation_avoidance(rng: random.Random, seed: int, idx: int) -> dict:
    target = rng.choice(["my manager", "Sam", "the lead", "Maya", "the design partner"])
    topic_phrase = rng.choice([
        "the project timeline", "the team capacity issue", "the scope creep",
        "the role re-leveling", "the deliverable slip",
    ])
    timeframe = rng.choice(SHORT_TIMEFRAMES)
    user_input = (
        f"I need to have the conversation with {target} about {topic_phrase}. "
        f"I've been avoiding it but it's becoming a blocker. I'll do it {timeframe}."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "over_flatten", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Self-identified need to discuss {topic_phrase} with {target} {timeframe}; "
            f"acknowledges prior avoidance and growing impact."
        ),
        intent=f"Have a conversation with {target} about {topic_phrase} {timeframe}.",
        valence="neutral",
        topic="work / communication",
        timestamp=ts, source_id=sid, channel="work",
    )
    rejected = build_record(
        tier="semantic",
        summary=(
            f"Has a long-running pattern of avoiding hard conversations; "
            f"aware that avoidance creates downstream blockers."
        ),
        intent=None,
        valence="neutral",
        topic="self-knowledge / conflict avoidance",
        timestamp=ts, source_id=sid, channel="work",
    )
    return build_pair_record(
        family="over_flatten", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _a_admin_task(rng: random.Random, seed: int, idx: int) -> dict:
    task, topic = rng.choice([
        ("update my LinkedIn", "career / admin"),
        ("file the expense reports", "work / admin"),
        ("renew my passport", "personal / admin"),
        ("submit the quarterly review form", "career / performance review"),
        ("clean up my browser bookmarks", "productivity / admin"),
    ])
    duration = rng.choice(DEFERRAL_DURATIONS)
    timeframe = rng.choice(SHORT_TIMEFRAMES)
    user_input = (
        f"I keep meaning to {task}. It's been on my list for {duration}. "
        f"I'm just going to sit down and do it {timeframe}."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "over_flatten", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Admin task — {task} — has been deferred for {duration}; "
            f"explicit self-commitment to complete {timeframe}."
        ),
        intent=f"{task.capitalize()} {timeframe}.",
        valence="neutral",
        topic=topic,
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected = build_record(
        tier="semantic",
        summary=(
            f"General self-pattern of deferring administrative tasks for extended periods; "
            f"reflective rather than directive."
        ),
        intent=None,
        valence="neutral",
        topic="self-knowledge / admin avoidance",
        timestamp=ts, source_id=sid, channel="personal",
    )
    return build_pair_record(
        family="over_flatten", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _a_repeated_promise(rng: random.Random, seed: int, idx: int) -> dict:
    person = rng.choice(PEOPLE)
    promise = rng.choice([
        "send the photos from the trip",
        "share the recipe",
        "introduce them to my mentor",
        "send the book I mentioned",
    ])
    timeframe = rng.choice(SHORT_TIMEFRAMES)
    user_input = (
        f"I told {person} I'd {promise} weeks ago. They haven't pushed but I know it's outstanding. "
        f"Going to actually do it {timeframe}."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "over_flatten", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Outstanding promise to {person} (to {promise}) made weeks ago; "
            f"explicit self-commitment to follow through {timeframe}."
        ),
        intent=f"{promise.capitalize()} for {person} {timeframe}.",
        valence="neutral",
        topic="relationships / commitments",
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected = build_record(
        tier="semantic",
        summary=(
            f"Self-aware pattern of letting small promises sit unfulfilled; "
            f"trusts that the recipient won't push, which enables further deferral."
        ),
        intent=None,
        valence="neutral",
        topic="self-knowledge / commitment patterns",
        timestamp=ts, source_id=sid, channel="personal",
    )
    return build_pair_record(
        family="over_flatten", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _a_learning_intent(rng: random.Random, seed: int, idx: int) -> dict:
    course = rng.choice([
        "the half-finished course on distributed systems",
        "the language-learning app I paid for",
        "the certification prep material",
        "the security training module",
    ])
    timeframe = rng.choice(SHORT_TIMEFRAMES)
    user_input = (
        f"I've had {course} sitting open for months and only completed two sessions. "
        f"Going to block evening time and finish at least one chapter {timeframe}."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "over_flatten", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Long-deferred study material ({course}); "
            f"self-commitment to block evening time and finish a chapter {timeframe}."
        ),
        intent=f"Block evening time and finish one chapter of {course} {timeframe}.",
        valence="neutral",
        topic="learning / commitments",
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected = build_record(
        tier="semantic",
        summary=(
            f"Tendency to start learning material without finishing; "
            f"awareness of the pattern without a specific re-engagement plan."
        ),
        intent=None,
        valence="neutral",
        topic="self-knowledge / learning patterns",
        timestamp=ts, source_id=sid, channel="personal",
    )
    return build_pair_record(
        family="over_flatten", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _a_creative_project(rng: random.Random, seed: int, idx: int) -> dict:
    project = rng.choice([
        "the side project I started in January",
        "the photo book for the trip",
        "the homepage redesign for my personal site",
        "the playlist I've been curating",
    ])
    timeframe = rng.choice(SHORT_TIMEFRAMES)
    user_input = (
        f"{project.capitalize()} has been 80% done for ages. I keep telling myself I'll wrap it up. "
        f"I'm going to push through and finish {timeframe}."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "over_flatten", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"{project.capitalize()} has been near-completion for an extended period; "
            f"self-commitment to finish {timeframe}."
        ),
        intent=f"Finish {project} {timeframe}.",
        valence="neutral",
        topic="creative / commitments",
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected = build_record(
        tier="semantic",
        summary=(
            f"Pattern of leaving creative projects at 80% completion; "
            f"acknowledged but framed as ongoing rather than action-imminent."
        ),
        intent=None,
        valence="neutral",
        topic="self-knowledge / completion avoidance",
        timestamp=ts, source_id=sid, channel="personal",
    )
    return build_pair_record(
        family="over_flatten", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


FAMILY_A: list[Callable[[random.Random, int, int], dict]] = [
    _a_persona_followup,
    _a_overdue_health,
    _a_writing_commitment,
    _a_conversation_avoidance,
    _a_admin_task,
    _a_repeated_promise,
    _a_learning_intent,
    _a_creative_project,
]


# ── Family B — add_coaching ───────────────────────────────────────────────────
# Cross all four tiers. Chosen: faithful witness. Rejected: same tier + advice
# inserted into content_summary, often fabricates stated_intent from the advice.

def _b_episodic_struggle(rng: random.Random, seed: int, idx: int) -> dict:
    activity = rng.choice([
        ("workout", "missed another workout this week. Third week in a row now",
         "Keep telling myself it's because of the project crunch, but I know I'm also avoiding it",
         "health / exercise"),
        ("morning", "skipped the morning routine again today",
         "Each time I tell myself it's just one day off but the streak is broken",
         "habits / routines"),
        ("journaling", "missed journaling for the fifth day running",
         "I open the app and then close it without writing",
         "habits / reflection"),
    ])
    label, primary, follow_up, topic = activity
    user_input = f"{primary.capitalize()}. {follow_up}."
    ts = pick_date(rng)
    sid = trace_source_id(ts, "add_coaching", idx)
    chosen = build_record(
        tier="episodic",
        summary=f"{primary.capitalize()}. Acknowledges both an external explanation and an avoidance component.",
        intent=None,
        valence="negative",
        topic=topic,
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected = build_record(
        tier="episodic",
        summary=(
            f"{primary.capitalize()}. Should consider scheduling {label} sessions in advance "
            f"and addressing the underlying avoidance behavior. Blocking calendar time may help."
        ),
        intent=f"Restart {label} routine.",
        valence="negative",
        topic=topic,
        timestamp=ts, source_id=sid, channel="personal",
    )
    return build_pair_record(
        family="add_coaching", target_tier="episodic",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _b_semantic_pattern(rng: random.Random, seed: int, idx: int) -> dict:
    pattern, pattern_noun, topic = rng.choice([
        ("over-explain when I'm uncertain", "over-explaining when uncertain", "self-knowledge / communication"),
        ("avoid conflict until it accumulates", "avoiding conflict until it accumulates", "self-knowledge / conflict"),
        ("under-charge for my work", "under-charging for my work", "self-knowledge / pricing"),
        ("say yes to too many small requests", "saying yes to too many small requests", "self-knowledge / boundaries"),
    ])
    user_input = (
        f"I've noticed I {pattern}. It's a long-running pattern. "
        f"Awareness alone hasn't really changed the behavior."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "add_coaching", idx)
    chosen = build_record(
        tier="semantic",
        summary=(
            f"Long-running self-pattern: tendency to {pattern}. "
            f"Awareness has not yet translated into behavioral change."
        ),
        intent=None,
        valence="neutral",
        topic=topic,
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected = build_record(
        tier="semantic",
        summary=(
            f"Self-pattern: tendency to {pattern}. To break this, consider deliberate practice "
            f"and accountability check-ins; a coach or therapist may help convert awareness into change."
        ),
        intent=f"Address my pattern of {pattern_noun} through deliberate practice.",
        valence="neutral",
        topic=topic,
        timestamp=ts, source_id=sid, channel="personal",
    )
    return build_pair_record(
        family="add_coaching", target_tier="semantic",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _b_procedural_learning(rng: random.Random, seed: int, idx: int) -> dict:
    technique, topic = rng.choice([
        ("explaining the problem to a non-expert before debugging",
         "engineering / problem-solving"),
        ("running tests on a 3-row sample before touching the real dataset",
         "engineering / data / testing"),
        ("writing the rollback script before the migration",
         "engineering / databases"),
    ])
    user_input = (
        f"Found that {technique} consistently catches issues earlier. "
        f"This is a reliable pattern now."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "add_coaching", idx)
    chosen = build_record(
        tier="procedural",
        summary=(
            f"Reliable practice: {technique}. Identified as a consistent pattern."
        ),
        intent=None,
        valence="positive",
        topic=topic,
        timestamp=ts, source_id=sid, channel="work",
    )
    rejected = build_record(
        tier="procedural",
        summary=(
            f"Reliable practice: {technique}. To make this a habit, consider documenting it "
            f"in your team wiki and adding it to your code review checklist."
        ),
        intent=f"Adopt {technique} as a standard practice across the team.",
        valence="positive",
        topic=topic,
        timestamp=ts, source_id=sid, channel="work",
    )
    return build_pair_record(
        family="add_coaching", target_tier="procedural",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _b_emotional_venting(rng: random.Random, seed: int, idx: int) -> dict:
    setup = rng.choice([
        ("Ate way too much sugar this week and feel terrible.",
         "Energy is low and I know food is part of it.",
         "negative", "health / nutrition"),
        ("Feeling really burned out this week.",
         "Everything takes more effort than it should.",
         "negative", "health / mental health"),
        ("Had another argument with my partner about chores.",
         "Same recurring topic. We never seem to land it.",
         "negative", "relationships / household"),
    ])
    line1, line2, valence, topic = setup
    user_input = f"{line1} {line2}"
    ts = pick_date(rng)
    sid = trace_source_id(ts, "add_coaching", idx)
    chosen = build_record(
        tier="episodic",
        summary=f"{line1} {line2}",
        intent=None,
        valence=valence,
        topic=topic,
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected = build_record(
        tier="episodic",
        summary=(
            f"{line1} {line2} Consider talking to a professional and trying small "
            f"behavioral changes; addressing the underlying causes early may help."
        ),
        intent="Address underlying causes through behavioral changes and professional support.",
        valence=valence,
        topic=topic,
        timestamp=ts, source_id=sid, channel="personal",
    )
    return build_pair_record(
        family="add_coaching", target_tier="episodic",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _b_career_uncertainty(rng: random.Random, seed: int, idx: int) -> dict:
    setup = rng.choice([
        ("undercharging clients but don't know how to raise rates",
         "career / pricing"),
        ("ready for a bigger role but the path isn't obvious",
         "career / progression"),
        ("at a plateau in my current job — not sure if I should push or move on",
         "career / decisions"),
    ])
    framing, topic = setup
    user_input = f"I think I'm {framing}."
    ts = pick_date(rng)
    sid = trace_source_id(ts, "add_coaching", idx)
    chosen = build_record(
        tier="semantic",
        summary=f"Reflects on {framing}. No specific plan or commitment stated.",
        intent=None,
        valence="neutral",
        topic=topic,
        timestamp=ts, source_id=sid, channel="work",
    )
    rejected = build_record(
        tier="semantic",
        summary=(
            f"Reflects on {framing}. To make progress, consider benchmarking against peers, "
            f"having career conversations with a mentor, and setting concrete milestones."
        ),
        intent=f"Address career uncertainty by benchmarking and setting milestones.",
        valence="neutral",
        topic=topic,
        timestamp=ts, source_id=sid, channel="work",
    )
    return build_pair_record(
        family="add_coaching", target_tier="semantic",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _b_prospective_with_followthrough_advice(rng: random.Random, seed: int, idx: int) -> dict:
    person = rng.choice(PEOPLE)
    deliverable, _, topic = rng.choice(WORK_DELIVERABLES)
    timeframe = rng.choice(SHORT_TIMEFRAMES)
    user_input = (
        f"Told {person} I'd have the {deliverable} ready {timeframe}. "
        f"It's on the calendar."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "add_coaching", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Committed to delivering the {deliverable} to {person} {timeframe}. "
            f"Scheduled on the calendar."
        ),
        intent=f"Deliver the {deliverable} to {person} {timeframe}.",
        valence="neutral",
        topic=topic,
        timestamp=ts, source_id=sid, channel="work",
    )
    rejected = build_record(
        tier="prospective",
        summary=(
            f"Committed to delivering the {deliverable} to {person} {timeframe}. "
            f"To follow through reliably, consider blocking 2 hours of focused time "
            f"in the morning and removing distractions."
        ),
        intent=(
            f"Deliver the {deliverable} to {person} {timeframe} by blocking 2 hours "
            f"of focused morning time and minimizing distractions."
        ),
        valence="neutral",
        topic=topic,
        timestamp=ts, source_id=sid, channel="work",
    )
    return build_pair_record(
        family="add_coaching", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _b_recurring_friction(rng: random.Random, seed: int, idx: int) -> dict:
    setup = rng.choice([
        ("I keep getting into the same argument with my partner about chores",
         "relationships / household"),
        ("I keep getting blocked by the same review bottleneck on every PR",
         "engineering / process"),
        ("I keep losing the first 30 minutes of my morning to email triage",
         "productivity / habits"),
    ])
    framing, topic = setup
    user_input = f"{framing}. Same script every time."
    ts = pick_date(rng)
    sid = trace_source_id(ts, "add_coaching", idx)
    chosen = build_record(
        tier="semantic",
        summary=f"Recurring friction pattern: {framing}.",
        intent=None,
        valence="negative",
        topic=topic,
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected = build_record(
        tier="semantic",
        summary=(
            f"Recurring friction pattern: {framing}. Consider scheduling a structured "
            f"conversation, surfacing root causes, and trying a behavioral experiment for two weeks."
        ),
        intent="Run a 2-week behavioral experiment to break the recurring friction pattern.",
        valence="negative",
        topic=topic,
        timestamp=ts, source_id=sid, channel="personal",
    )
    return build_pair_record(
        family="add_coaching", target_tier="semantic",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _b_late_night_self_critique(rng: random.Random, seed: int, idx: int) -> dict:
    setup = rng.choice([
        ("Procrastinated on the budget review again, third week running.",
         "personal / finance"),
        ("Stayed up too late doom-scrolling. Again.",
         "habits / sleep"),
        ("Skipped the 1:1 prep yet again — winged it instead.",
         "work / preparation"),
    ])
    framing, topic = setup
    user_input = framing
    ts = pick_date(rng)
    sid = trace_source_id(ts, "add_coaching", idx)
    chosen = build_record(
        tier="episodic",
        summary=framing.rstrip("."),
        intent=None,
        valence="negative",
        topic=topic,
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected = build_record(
        tier="episodic",
        summary=(
            f"{framing.rstrip('.')}. Consider setting a hard cut-off time and "
            f"using website blockers; building accountability with a peer may help."
        ),
        intent="Set a hard cut-off time and use website blockers to break the pattern.",
        valence="negative",
        topic=topic,
        timestamp=ts, source_id=sid, channel="personal",
    )
    return build_pair_record(
        family="add_coaching", target_tier="episodic",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


FAMILY_B: list[Callable[[random.Random, int, int], dict]] = [
    _b_episodic_struggle,
    _b_semantic_pattern,
    _b_procedural_learning,
    _b_emotional_venting,
    _b_career_uncertainty,
    _b_prospective_with_followthrough_advice,
    _b_recurring_friction,
    _b_late_night_self_critique,
]


# ── Family C — mis_tier_mixed ─────────────────────────────────────────────────
# Episodic event + prospective commitment. Chosen: prospective (commitment is
# the dominant signal). Rejected: episodic (picks the surface event, drops
# the commitment).

def _c_ship_plus_doc(rng: random.Random, seed: int, idx: int) -> dict:
    n = rng.randint(2, 4)
    m = rng.randint(0, 9)
    weeks = rng.randint(2, 6)
    deliverable = rng.choice(["migration guide", "release notes", "rollout plan"])
    stakeholder = rng.choice(["the customer success team", "the developer relations team", "the partners"])
    timeframe = rng.choice(["by next Monday", "by end of week", "by Wednesday"])
    user_input = (
        f"Shipped the v{n}.{m} release today after {weeks} weeks of work. "
        f"Felt good to get it across the line. I've promised {stakeholder} I'll have "
        f"the {deliverable} written and sent to them {timeframe}."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "mis_tier_mixed", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Shipped v{n}.{m} after {weeks} weeks; on the back of that, committed to writing "
            f"and delivering the {deliverable} to {stakeholder} {timeframe}."
        ),
        intent=f"Write and deliver the {deliverable} to {stakeholder} {timeframe}.",
        valence="positive",
        topic="engineering / documentation / releases",
        timestamp=ts, source_id=sid, channel="work",
    )
    rejected = build_record(
        tier="episodic",
        summary=(
            f"Successfully shipped v{n}.{m} release after {weeks} weeks of work; "
            f"deployment went smoothly."
        ),
        intent=None,
        valence="positive",
        topic="engineering / releases",
        timestamp=ts, source_id=sid, channel="work",
    )
    return build_pair_record(
        family="mis_tier_mixed", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _c_meeting_plus_followup(rng: random.Random, seed: int, idx: int) -> dict:
    person = rng.choice(PEOPLE)
    topic_phrase = rng.choice([
        "the roadmap", "the org changes", "the reorg implications",
        "the staffing plan", "the launch readiness",
    ])
    deliverable = rng.choice([
        "a one-pager summarizing the options",
        "the revised proposal",
        "the slides for the readout",
    ])
    timeframe = rng.choice(["by Thursday", "by end of week", "by tomorrow afternoon"])
    user_input = (
        f"Had a really productive meeting with {person} about {topic_phrase} this morning. "
        f"Walked out aligned on direction. I committed to sending {person} {deliverable} {timeframe}."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "mis_tier_mixed", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Following an aligned meeting with {person} about {topic_phrase}, committed to sending "
            f"{deliverable} {timeframe}."
        ),
        intent=f"Send {person} {deliverable} {timeframe}.",
        valence="positive",
        topic="work / collaboration / commitments",
        timestamp=ts, source_id=sid, channel="work",
    )
    rejected = build_record(
        tier="episodic",
        summary=(
            f"Productive meeting with {person} about {topic_phrase}; ended aligned on direction."
        ),
        intent=None,
        valence="positive",
        topic="work / meetings",
        timestamp=ts, source_id=sid, channel="work",
    )
    return build_pair_record(
        family="mis_tier_mixed", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _c_demo_plus_followup(rng: random.Random, seed: int, idx: int) -> dict:
    audience = rng.choice(["leadership", "the product council", "the design partners", "the customer advisory board"])
    deliverable = rng.choice([
        "a write-up of the trade-offs",
        "the updated benchmark numbers",
        "a costed plan for the rollout",
    ])
    timeframe = rng.choice(["by Friday", "by next Tuesday", "by end of week"])
    user_input = (
        f"Demoed the new pipeline to {audience} today. Got asked good questions. "
        f"Agreed to follow up with {deliverable} {timeframe}."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "mis_tier_mixed", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Demoed the new pipeline to {audience}; agreed to follow up with {deliverable} {timeframe}."
        ),
        intent=f"Follow up with {audience} by sending {deliverable} {timeframe}.",
        valence="positive",
        topic="engineering / demos / commitments",
        timestamp=ts, source_id=sid, channel="work",
    )
    rejected = build_record(
        tier="episodic",
        summary=(
            f"Demoed the new pipeline to {audience}; received good questions."
        ),
        intent=None,
        valence="positive",
        topic="engineering / demos",
        timestamp=ts, source_id=sid, channel="work",
    )
    return build_pair_record(
        family="mis_tier_mixed", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _c_event_plus_writeup(rng: random.Random, seed: int, idx: int) -> dict:
    event = rng.choice([
        "the offsite", "the conference", "the workshop",
        "the customer summit", "the strategy day",
    ])
    audience = rng.choice(["the team", "leadership", "the wider org"])
    timeframe = rng.choice(["by next Monday", "by Wednesday", "by end of week"])
    user_input = (
        f"Got back from {event} late last night. Surprisingly useful. "
        f"I told {audience} I'd send a debrief write-up {timeframe}."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "mis_tier_mixed", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Returned from {event}; committed to sending a debrief write-up to {audience} {timeframe}."
        ),
        intent=f"Send {event} debrief write-up to {audience} {timeframe}.",
        valence="positive",
        topic="work / writing / commitments",
        timestamp=ts, source_id=sid, channel="work",
    )
    rejected = build_record(
        tier="episodic",
        summary=(
            f"Returned from {event}; experience described as surprisingly useful."
        ),
        intent=None,
        valence="positive",
        topic="work / events",
        timestamp=ts, source_id=sid, channel="work",
    )
    return build_pair_record(
        family="mis_tier_mixed", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _c_close_out_plus_handoff(rng: random.Random, seed: int, idx: int) -> dict:
    project = rng.choice(EPISODIC_PROJECTS)
    successor = rng.choice(PEOPLE)
    timeframe = rng.choice(["by Thursday", "by end of week", "by next Monday"])
    user_input = (
        f"Closed out the {project} migration this afternoon. "
        f"I told {successor} I'd write a complete handoff doc {timeframe}."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "mis_tier_mixed", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Closed the {project} migration; committed to writing a complete handoff doc "
            f"for {successor} {timeframe}."
        ),
        intent=f"Write {project} handoff doc for {successor} {timeframe}.",
        valence="positive",
        topic="engineering / documentation / handoff",
        timestamp=ts, source_id=sid, channel="work",
    )
    rejected = build_record(
        tier="episodic",
        summary=f"Closed out the {project} migration this afternoon.",
        intent=None,
        valence="positive",
        topic="engineering / migration",
        timestamp=ts, source_id=sid, channel="work",
    )
    return build_pair_record(
        family="mis_tier_mixed", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _c_one_on_one_plus_action(rng: random.Random, seed: int, idx: int) -> dict:
    person = rng.choice(PEOPLE)
    action = rng.choice([
        "send them a list of three growth opportunities",
        "share the framework I mentioned",
        "book a follow-up to walk through their proposal",
    ])
    timeframe = rng.choice(["by Friday", "before our next 1:1", "by end of week"])
    user_input = (
        f"Had a really good 1:1 with {person} today. "
        f"They were honest about what's not working. "
        f"I committed to {action} {timeframe}."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "mis_tier_mixed", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Strong 1:1 with {person}; "
            f"committed to {action} {timeframe}."
        ),
        intent=f"{action.capitalize()} for {person} {timeframe}.",
        valence="positive",
        topic="management / 1:1s / commitments",
        timestamp=ts, source_id=sid, channel="work",
    )
    rejected = build_record(
        tier="episodic",
        summary=(
            f"Productive 1:1 with {person}; honest discussion about what's not working."
        ),
        intent=None,
        valence="positive",
        topic="management / 1:1s",
        timestamp=ts, source_id=sid, channel="work",
    )
    return build_pair_record(
        family="mis_tier_mixed", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _c_incident_plus_postmortem(rng: random.Random, seed: int, idx: int) -> dict:
    when = rng.choice(["this morning", "overnight", "yesterday", "this afternoon"])
    timeframe = rng.choice(["by tomorrow EOD", "by Wednesday", "by Friday"])
    user_input = (
        f"Resolved the incident from {when} after about three hours. "
        f"Promised to have the post-mortem draft circulated {timeframe}."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "mis_tier_mixed", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Resolved the {when} incident in ~3 hours; committed to circulating a post-mortem "
            f"draft {timeframe}."
        ),
        intent=f"Circulate post-mortem draft {timeframe}.",
        valence="neutral",
        topic="engineering / incidents / documentation",
        timestamp=ts, source_id=sid, channel="work",
    )
    rejected = build_record(
        tier="episodic",
        summary=f"Resolved the {when} incident after about three hours.",
        intent=None,
        valence="neutral",
        topic="engineering / incidents",
        timestamp=ts, source_id=sid, channel="work",
    )
    return build_pair_record(
        family="mis_tier_mixed", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _c_milestone_plus_reachout(rng: random.Random, seed: int, idx: int) -> dict:
    milestone = rng.choice([
        "passed the certification today",
        "got the offer for the new role",
        "got accepted to the workshop",
        "had the paper accepted",
    ])
    person = rng.choice(PEOPLE)
    timeframe = rng.choice(["by tonight", "by tomorrow", "before the end of the week"])
    user_input = (
        f"{milestone.capitalize()}. Felt amazing. "
        f"I want to write a real thank-you to {person} who supported me through this — going to send it {timeframe}."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "mis_tier_mixed", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"{milestone.capitalize()}; committed to sending a thank-you message to {person} {timeframe}."
        ),
        intent=f"Send a thank-you message to {person} {timeframe}.",
        valence="positive",
        topic="relationships / gratitude / commitments",
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected = build_record(
        tier="episodic",
        summary=f"{milestone.capitalize()}; emotionally significant.",
        intent=None,
        valence="positive",
        topic="career / milestones",
        timestamp=ts, source_id=sid, channel="personal",
    )
    return build_pair_record(
        family="mis_tier_mixed", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


FAMILY_C: list[Callable[[random.Random, int, int], dict]] = [
    _c_ship_plus_doc,
    _c_meeting_plus_followup,
    _c_demo_plus_followup,
    _c_event_plus_writeup,
    _c_close_out_plus_handoff,
    _c_one_on_one_plus_action,
    _c_incident_plus_postmortem,
    _c_milestone_plus_reachout,
]


# ── Family D — conditional_commitment ─────────────────────────────────────────
# Conditional but specific commitment. Chosen: prospective with conditional intent.
# Rejected: semantic (treats conditional as no-action-yet) OR prospective with
# stated_intent: null.

def _d_sprint_then_off(rng: random.Random, seed: int, idx: int) -> dict:
    completion = rng.choice([
        "wrap the current sprint by Thursday",
        "land the migration this week",
        "close the open priority-one bug",
    ])
    reward = rng.choice([
        "take Friday afternoon completely off — no laptop, just read and walk",
        "take a long weekend with the phone in airplane mode",
        "block Friday for a real rest day",
    ])
    user_input = f"If I can {completion}, I'm going to {reward}. I need it."
    ts = pick_date(rng)
    sid = trace_source_id(ts, "conditional_commitment", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Conditional commitment: {reward}, contingent on completing the milestone — "
            f"{completion}."
        ),
        intent=f"{reward.capitalize()}, contingent on {completion}.",
        valence="positive",
        topic="work-life balance / rest",
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected_choice = rng.choice(["semantic", "intent_null"])
    if rejected_choice == "semantic":
        rejected = build_record(
            tier="semantic",
            summary=(
                f"Reflects on the desire to rest after intense work; "
                f"frames as aspirational rather than committed."
            ),
            intent=None,
            valence="neutral",
            topic="self-knowledge / rest needs",
            timestamp=ts, source_id=sid, channel="personal",
        )
    else:
        rejected = build_record(
            tier="prospective",
            summary=(
                f"Plans to {reward} if the upcoming milestone is completed."
            ),
            intent=None,
            valence="positive",
            topic="work-life balance / rest",
            timestamp=ts, source_id=sid, channel="personal",
        )
    return build_pair_record(
        family="conditional_commitment", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _d_funding_then_hire(rng: random.Random, seed: int, idx: int) -> dict:
    condition = rng.choice([
        "the funding round closes",
        "the budget gets approved next quarter",
        "the headcount request goes through",
    ])
    action = rng.choice([
        "open the senior engineer role I've been holding back on",
        "kick off the hiring loop for the platform team",
        "post the role and start reaching out to the shortlist",
    ])
    user_input = f"Once {condition}, I'm going to {action}. The shortlist is already in my notes."
    ts = pick_date(rng)
    sid = trace_source_id(ts, "conditional_commitment", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Conditional commitment: {action}, contingent on {condition}. "
            f"Shortlist already prepared."
        ),
        intent=f"{action.capitalize()} once {condition}.",
        valence="neutral",
        topic="management / hiring",
        timestamp=ts, source_id=sid, channel="work",
    )
    rejected_choice = rng.choice(["semantic", "intent_null"])
    if rejected_choice == "semantic":
        rejected = build_record(
            tier="semantic",
            summary=(
                f"Has a general intention to expand the team when funding allows; "
                f"shortlist exists but no triggering event yet."
            ),
            intent=None,
            valence="neutral",
            topic="management / aspirations",
            timestamp=ts, source_id=sid, channel="work",
        )
    else:
        rejected = build_record(
            tier="prospective",
            summary=(
                f"Plans to expand the team contingent on funding; shortlist already prepared."
            ),
            intent=None,
            valence="neutral",
            topic="management / hiring",
            timestamp=ts, source_id=sid, channel="work",
        )
    return build_pair_record(
        family="conditional_commitment", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _d_review_then_publish(rng: random.Random, seed: int, idx: int) -> dict:
    condition = rng.choice([
        "Diego signs off on the draft",
        "the legal review comes back clean",
        "the security team clears it",
    ])
    action = rng.choice([
        "publish the post on Friday morning",
        "ship the announcement next Tuesday",
        "send the announcement to the customer list end of week",
    ])
    user_input = f"Assuming {condition}, I'll {action}. Already drafted the announcement."
    ts = pick_date(rng)
    sid = trace_source_id(ts, "conditional_commitment", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Conditional commitment: {action}, contingent on {condition}. "
            f"Announcement already drafted."
        ),
        intent=f"{action.capitalize()} once {condition}.",
        valence="neutral",
        topic="communications / launches",
        timestamp=ts, source_id=sid, channel="work",
    )
    rejected_choice = rng.choice(["semantic", "intent_null"])
    if rejected_choice == "semantic":
        rejected = build_record(
            tier="semantic",
            summary=(
                f"Holds the general intention to publish once review processes clear; "
                f"depends on multiple gating reviews."
            ),
            intent=None,
            valence="neutral",
            topic="communications / process",
            timestamp=ts, source_id=sid, channel="work",
        )
    else:
        rejected = build_record(
            tier="prospective",
            summary=f"Plans to publish the announcement once review clears.",
            intent=None,
            valence="neutral",
            topic="communications / launches",
            timestamp=ts, source_id=sid, channel="work",
        )
    return build_pair_record(
        family="conditional_commitment", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _d_outcome_then_celebrate(rng: random.Random, seed: int, idx: int) -> dict:
    condition = rng.choice([
        "the proposal lands well",
        "the certification exam goes well",
        "we hit the launch metric",
    ])
    action = rng.choice([
        "take the team out for a proper dinner",
        "book the cabin weekend I've been pricing",
        "buy myself the camera I've been eyeing",
    ])
    user_input = (
        f"If {condition} this Friday, I'm going to {action}. "
        f"Already have a placeholder on the calendar."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "conditional_commitment", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Conditional commitment: {action} contingent on {condition} on Friday. "
            f"Placeholder on the calendar."
        ),
        intent=f"{action.capitalize()}, contingent on {condition} this Friday.",
        valence="positive",
        topic="celebration / commitments",
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected_choice = rng.choice(["semantic", "intent_null"])
    if rejected_choice == "semantic":
        rejected = build_record(
            tier="semantic",
            summary=(
                f"Has an aspirational reward in mind contingent on a positive outcome later this week."
            ),
            intent=None,
            valence="positive",
            topic="self-knowledge / motivation",
            timestamp=ts, source_id=sid, channel="personal",
        )
    else:
        rejected = build_record(
            tier="prospective",
            summary=f"Plans to celebrate if {condition} this Friday.",
            intent=None,
            valence="positive",
            topic="celebration / commitments",
            timestamp=ts, source_id=sid, channel="personal",
        )
    return build_pair_record(
        family="conditional_commitment", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _d_health_then_change(rng: random.Random, seed: int, idx: int) -> dict:
    condition = rng.choice([
        "this week's bloodwork comes back clean",
        "the physical therapist signs off",
        "I hit two more weeks of consistent sleep",
    ])
    action = rng.choice([
        "start the new training plan on Monday",
        "ramp the running schedule back up",
        "add the strength sessions back in",
    ])
    user_input = (
        f"Assuming {condition}, I'm going to {action}. The plan is already saved on my phone."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "conditional_commitment", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Conditional commitment: {action}, contingent on {condition}. "
            f"Plan saved and ready."
        ),
        intent=f"{action.capitalize()}, contingent on {condition}.",
        valence="positive",
        topic="health / training",
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected_choice = rng.choice(["semantic", "intent_null"])
    if rejected_choice == "semantic":
        rejected = build_record(
            tier="semantic",
            summary=(
                f"Has a general intention to resume training once health markers stabilize."
            ),
            intent=None,
            valence="neutral",
            topic="health / aspirations",
            timestamp=ts, source_id=sid, channel="personal",
        )
    else:
        rejected = build_record(
            tier="prospective",
            summary=f"Plans to {action} once {condition}.",
            intent=None,
            valence="positive",
            topic="health / training",
            timestamp=ts, source_id=sid, channel="personal",
        )
    return build_pair_record(
        family="conditional_commitment", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _d_logistics_then_book(rng: random.Random, seed: int, idx: int) -> dict:
    condition = rng.choice([
        "I get the time-off approval back",
        "the project ships on schedule",
        "the conflict on my calendar resolves",
    ])
    action = rng.choice([
        "book the flights for the trip in June",
        "book the offsite for the team",
        "book the surgery I've been delaying",
    ])
    user_input = (
        f"If {condition} by Wednesday, I'll {action}. "
        f"Quotes are sitting in my drafts."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "conditional_commitment", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            f"Conditional commitment: {action}, contingent on {condition} by Wednesday. "
            f"Quotes already drafted."
        ),
        intent=f"{action.capitalize()} contingent on {condition} by Wednesday.",
        valence="neutral",
        topic="logistics / commitments",
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected_choice = rng.choice(["semantic", "intent_null"])
    if rejected_choice == "semantic":
        rejected = build_record(
            tier="semantic",
            summary=(
                f"Has an aspirational booking in mind contingent on logistics resolving."
            ),
            intent=None,
            valence="neutral",
            topic="logistics / aspirations",
            timestamp=ts, source_id=sid, channel="personal",
        )
    else:
        rejected = build_record(
            tier="prospective",
            summary=f"Plans to {action} contingent on logistics.",
            intent=None,
            valence="neutral",
            topic="logistics / commitments",
            timestamp=ts, source_id=sid, channel="personal",
        )
    return build_pair_record(
        family="conditional_commitment", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


FAMILY_D: list[Callable[[random.Random, int, int], dict]] = [
    _d_sprint_then_off,
    _d_funding_then_hire,
    _d_review_then_publish,
    _d_outcome_then_celebrate,
    _d_health_then_change,
    _d_logistics_then_book,
]


# ── Family E — schema_drift / tier_swap distractors ───────────────────────────
# Cleanly correct vs cleanly wrong tier on UNAMBIGUOUS inputs. Prevents DPO
# from over-shifting toward boundary tiers.

def _e_obvious_episodic(rng: random.Random, seed: int, idx: int) -> dict:
    user_input = (
        "Spent two hours fixing the off-by-one bug in the metrics aggregation today. "
        "Tests pass, deployed to staging."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "schema_drift", idx)
    chosen = build_record(
        tier="episodic",
        summary="Fixed an off-by-one bug in metrics aggregation; ~2 hours; deployed to staging with tests passing.",
        intent=None, valence="positive",
        topic="engineering / debugging",
        timestamp=ts, source_id=sid, channel="work",
    )
    rejected = build_record(
        tier="procedural",
        summary="Reliable bug-fix workflow: identify off-by-one error, test, deploy to staging.",
        intent=None, valence="positive",
        topic="engineering / debugging / process",
        timestamp=ts, source_id=sid, channel="work",
    )
    return build_pair_record(
        family="schema_drift", target_tier="episodic",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _e_obvious_procedural(rng: random.Random, seed: int, idx: int) -> dict:
    user_input = (
        "Learned that the most reliable way to debug flaky tests is to run them 50 times in a loop "
        "with the same seed before assuming they're really flaky. Found three real bugs that way."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "schema_drift", idx)
    chosen = build_record(
        tier="procedural",
        summary=(
            "Reliable flaky-test debugging: run 50 iterations with a fixed seed before "
            "concluding flakiness. Surfaced three real bugs."
        ),
        intent=None, valence="positive",
        topic="engineering / testing / debugging",
        timestamp=ts, source_id=sid, channel="work",
    )
    rejected = build_record(
        tier="episodic",
        summary="Today, ran flaky tests 50 times in a loop and found three real bugs.",
        intent=None, valence="positive",
        topic="engineering / testing",
        timestamp=ts, source_id=sid, channel="work",
    )
    return build_pair_record(
        family="schema_drift", target_tier="procedural",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _e_obvious_semantic(rng: random.Random, seed: int, idx: int) -> dict:
    user_input = (
        "I trust people more when they tell me what they don't know than when they project "
        "confidence. Epistemic humility is one of the few traits I weigh heavily in collaborators."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "schema_drift", idx)
    chosen = build_record(
        tier="semantic",
        summary=(
            "Core value: epistemic humility is a primary trust signal in collaborators. "
            "Confidence projection without acknowledged unknowns reduces trust."
        ),
        intent=None, valence="neutral",
        topic="self-knowledge / values / collaboration",
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected = build_record(
        tier="episodic",
        summary="Reflected today on what makes collaborators trustworthy.",
        intent=None, valence="neutral",
        topic="reflection / values",
        timestamp=ts, source_id=sid, channel="personal",
    )
    return build_pair_record(
        family="schema_drift", target_tier="semantic",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _e_obvious_prospective(rng: random.Random, seed: int, idx: int) -> dict:
    user_input = (
        "Submitting the grant proposal draft to the committee by Friday 5pm. "
        "Firm external deadline, no extensions."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "schema_drift", idx)
    chosen = build_record(
        tier="prospective",
        summary=(
            "Firm external deadline: grant proposal draft due to the committee by Friday 5pm. "
            "No extensions available."
        ),
        intent="Submit grant proposal draft to the committee by Friday 5pm.",
        valence="neutral",
        topic="work / grants / deadlines",
        timestamp=ts, source_id=sid, channel="work",
    )
    rejected = build_record(
        tier="semantic",
        summary=(
            "Holds a recurring belief that external deadlines should be honored without extension."
        ),
        intent=None, valence="neutral",
        topic="self-knowledge / work values",
        timestamp=ts, source_id=sid, channel="work",
    )
    return build_pair_record(
        family="schema_drift", target_tier="prospective",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _e_obvious_episodic_2(rng: random.Random, seed: int, idx: int) -> dict:
    user_input = (
        "Ran a 10K this morning. First time at that distance in over a year. "
        "Pace was slower than I'd like but the body held up."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "schema_drift", idx)
    chosen = build_record(
        tier="episodic",
        summary=(
            "Ran a 10K — first at that distance in over a year. Pace below target but no physical issues."
        ),
        intent=None, valence="mixed",
        topic="health / exercise / running",
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected = build_record(
        tier="semantic",
        summary="Identifies as a runner with a particular relationship to pace and endurance.",
        intent=None, valence="neutral",
        topic="self-knowledge / running",
        timestamp=ts, source_id=sid, channel="personal",
    )
    return build_pair_record(
        family="schema_drift", target_tier="episodic",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _e_obvious_procedural_2(rng: random.Random, seed: int, idx: int) -> dict:
    user_input = (
        "I always start a code review by reading the tests before the implementation. "
        "Catches design issues a lot faster than going top-to-bottom."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "schema_drift", idx)
    chosen = build_record(
        tier="procedural",
        summary=(
            "Code review practice: read tests before implementation. Surfaces design issues "
            "more efficiently than linear top-to-bottom reading."
        ),
        intent=None, valence="positive",
        topic="engineering / code review",
        timestamp=ts, source_id=sid, channel="work",
    )
    rejected = build_record(
        tier="episodic",
        summary="Did a code review today by reading tests first.",
        intent=None, valence="positive",
        topic="engineering / code review",
        timestamp=ts, source_id=sid, channel="work",
    )
    return build_pair_record(
        family="schema_drift", target_tier="procedural",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


def _e_obvious_semantic_2(rng: random.Random, seed: int, idx: int) -> dict:
    user_input = (
        "I do my best thinking in long uninterrupted blocks. Quality of decisions drops sharply "
        "under time pressure. This has been consistent across years."
    )
    ts = pick_date(rng)
    sid = trace_source_id(ts, "schema_drift", idx)
    chosen = build_record(
        tier="semantic",
        summary=(
            "Self-knowledge: best thinking happens in long uninterrupted blocks; "
            "decision quality degrades sharply under time pressure. Consistent over years."
        ),
        intent=None, valence="neutral",
        topic="self-knowledge / cognitive style",
        timestamp=ts, source_id=sid, channel="personal",
    )
    rejected = build_record(
        tier="episodic",
        summary="Today, reflected on my preferred working conditions.",
        intent=None, valence="neutral",
        topic="reflection / work style",
        timestamp=ts, source_id=sid, channel="personal",
    )
    return build_pair_record(
        family="schema_drift", target_tier="semantic",
        seed=seed, idx=idx,
        user_input=user_input, chosen=chosen, rejected=rejected,
    )


FAMILY_E: list[Callable[[random.Random, int, int], dict]] = [
    _e_obvious_episodic,
    _e_obvious_procedural,
    _e_obvious_semantic,
    _e_obvious_prospective,
    _e_obvious_episodic_2,
    _e_obvious_procedural_2,
    _e_obvious_semantic_2,
]


FAMILIES = {
    "over_flatten": FAMILY_A,
    "add_coaching": FAMILY_B,
    "mis_tier_mixed": FAMILY_C,
    "conditional_commitment": FAMILY_D,
    "schema_drift": FAMILY_E,
}


# ── Validation ────────────────────────────────────────────────────────────────

def parse_record_field(record: dict, field: str) -> tuple[bool, str]:
    """Parse the chosen or rejected field as JSON; verify required keys."""
    raw = record.get(field, "")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"{field} is not valid JSON: {e}"
    missing = [f for f in REQUIRED_FIELDS if f not in parsed]
    if missing:
        return False, f"{field} missing required fields: {missing}"
    if parsed["memory_tier"] not in VALID_TIERS:
        return False, f"{field} has invalid memory_tier: {parsed['memory_tier']}"
    if parsed["emotional_valence"] not in VALID_VALENCES:
        return False, f"{field} has invalid emotional_valence: {parsed['emotional_valence']}"
    return True, ""


def validate_record(record: dict) -> list[str]:
    """Return a list of validation errors for a single record (empty = clean)."""
    errors: list[str] = []
    for required_top_level in ("prompt", "chosen", "rejected", "metadata"):
        if required_top_level not in record:
            errors.append(f"missing top-level field: {required_top_level}")
    if errors:
        return errors

    chosen_ok, chosen_msg = parse_record_field(record, "chosen")
    if not chosen_ok:
        errors.append(chosen_msg)
    rejected_ok, rejected_msg = parse_record_field(record, "rejected")
    if not rejected_ok:
        errors.append(rejected_msg)

    if chosen_ok and rejected_ok:
        chosen = json.loads(record["chosen"])
        rejected = json.loads(record["rejected"])
        axis = record["metadata"].get("axis")
        if axis in ("over_flatten", "add_coaching", "mis_tier_mixed", "conditional_commitment", "schema_drift"):
            differs_on = (
                chosen["memory_tier"] != rejected["memory_tier"]
                or chosen["stated_intent"] != rejected["stated_intent"]
                or chosen["content_summary"] != rejected["content_summary"]
            )
            if not differs_on:
                errors.append(
                    f"chosen and rejected do not differ on memory_tier, stated_intent, "
                    f"or content_summary (axis: {axis})"
                )

    md = record.get("metadata", {})
    for k in ("axis", "target_tier", "generation_seed", "case_id"):
        if k not in md:
            errors.append(f"metadata missing field: {k}")

    return errors


# ── Generation ────────────────────────────────────────────────────────────────

def compute_family_targets(n: int) -> dict[str, int]:
    """Allocate n pairs across families using FAMILY_PROPORTIONS, rounding fairly."""
    floats = {fam: n * prop for fam, prop in FAMILY_PROPORTIONS.items()}
    base = {fam: int(v) for fam, v in floats.items()}
    remainder = n - sum(base.values())
    fractions = sorted(
        ((floats[fam] - base[fam], fam) for fam in floats),
        reverse=True,
    )
    for _, fam in fractions[:remainder]:
        base[fam] += 1
    return base


def generate(n: int, seed: int) -> list[dict]:
    targets = compute_family_targets(n)
    master_rng = random.Random(seed)

    records: list[dict] = []
    seen_prompts: set[str] = set()

    for family, count in targets.items():
        templates = FAMILIES[family]
        family_rng = random.Random(master_rng.randint(0, 2**31 - 1))
        # Round-robin through templates with per-instance RNG seeded from family_rng.
        for i in range(count):
            template_fn = templates[i % len(templates)]
            instance_seed = family_rng.randint(0, 2**31 - 1)
            instance_rng = random.Random(instance_seed)
            # Up to 6 attempts to produce a unique-prompt record.
            for attempt in range(6):
                record = template_fn(instance_rng, seed, len(records))
                if record["prompt"] not in seen_prompts:
                    break
            else:
                # Couldn't dedup after 6 tries; accept the duplicate (validator will flag).
                pass
            seen_prompts.add(record["prompt"])
            records.append(record)

    return records


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic DPO preference pairs for Trace Layer 2."
    )
    parser.add_argument("--n", type=int, default=80,
                        help="Total number of pairs to generate (default: 80).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default: 42).")
    parser.add_argument("--output", default="data/processed/dpo/train.jsonl",
                        help="Output JSONL path.")
    parser.add_argument("--validation-fraction", type=float, default=0.0,
                        help="If > 0, split into train/valid; valid path is "
                             "derived by replacing 'train' with 'valid' in --output. Default: 0.0.")
    args = parser.parse_args()

    if args.n < 5:
        print("ERROR: --n must be at least 5 (one per family minimum).", file=sys.stderr)
        sys.exit(2)

    targets = compute_family_targets(args.n)
    print(f"Generating {args.n} DPO pairs (seed={args.seed})")
    print("Family targets:")
    for fam, count in targets.items():
        print(f"  {fam:<28} {count}")

    records = generate(args.n, args.seed)

    # Split if requested
    if args.validation_fraction > 0:
        valid_count = max(1, int(round(len(records) * args.validation_fraction)))
        train_records = records[:-valid_count]
        valid_records = records[-valid_count:]
        train_path = Path(args.output)
        valid_path = Path(str(train_path).replace("train", "valid"))
        write_jsonl(train_records, train_path)
        write_jsonl(valid_records, valid_path)
        print(f"\nWrote {len(train_records)} train records to {train_path}")
        print(f"Wrote {len(valid_records)} valid records to {valid_path}")
    else:
        out = Path(args.output)
        write_jsonl(records, out)
        print(f"\nWrote {len(records)} records to {out}")

    # Inline smoke validation
    print("\nRunning inline validation...")
    errs = 0
    for i, rec in enumerate(records):
        rec_errs = validate_record(rec)
        if rec_errs:
            errs += 1
            print(f"  [{i}] ({rec['metadata'].get('case_id', '?')})")
            for e in rec_errs:
                print(f"      - {e}")
    if errs == 0:
        print("  OK: all records validate.")
    else:
        print(f"  FAIL: {errs} records failed validation.")
        sys.exit(1)

    # Distribution summary
    fam_counts = Counter(rec["metadata"]["axis"] for rec in records)
    print("\nFinal family distribution:")
    for fam in FAMILY_PROPORTIONS:
        print(f"  {fam:<28} {fam_counts.get(fam, 0)} (target {targets[fam]})")
    print()


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
