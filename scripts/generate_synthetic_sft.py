#!/usr/bin/env python3
"""
Generate synthetic SFT training data for Trace Layer 2 extraction.

No model or GPU required — examples are constructed from templates.
Output is a JSONL file in mlx-lm chat format, ready for prepare_data.py.

Coverage: 100 examples — 25 per tier (balanced)
  - 25 episodic
  - 25 semantic
  - 25 procedural
  - 25 prospective  (including aspiration-vs-commitment boundary cases)

Usage:
  python scripts/generate_synthetic_sft.py
  python scripts/generate_synthetic_sft.py --output data/sft_train_100.jsonl --shuffle --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a Trace memory extraction engine. Transform the raw trace input into a "
    "structured Trace-style memory record. Output valid JSON only. Do not add advice, "
    "coaching, prescriptive framing, or interpretation. Witness and structure; do not "
    "suggest or evaluate."
)


def make_example(user_input: str, record: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": json.dumps(record, ensure_ascii=False)},
        ]
    }


# ── Episodic examples (15) ────────────────────────────────────────────────────

EPISODIC = [
    make_example(
        "Spent most of yesterday afternoon debugging a subtle race condition in the job scheduler. "
        "Turned out the lock was being acquired in the wrong order in two goroutines. Fixed it, tests pass.",
        {"content_summary": "Debugged and resolved a race condition in the job scheduler caused by incorrect lock acquisition order across two goroutines.",
         "memory_tier": "episodic", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / debugging", "timestamp": "2025-10-14",
         "source": "daily_trace", "source_id": "trace_20251014_001", "channel": "work"},
    ),
    make_example(
        "Had a really productive 3-hour deep work session on the data pipeline refactor. "
        "Got the schema migration logic working end to end. Felt focused the whole time.",
        {"content_summary": "Completed a focused 3-hour deep work session on the data pipeline schema migration, achieving end-to-end functionality.",
         "memory_tier": "episodic", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / data pipeline", "timestamp": "2025-10-15",
         "source": "daily_trace", "source_id": "trace_20251015_001", "channel": "work"},
    ),
    make_example(
        "Got pulled into an unplanned two-hour strategy meeting that derailed my entire afternoon. "
        "Left feeling frustrated and behind on sprint work. Nobody had an agenda.",
        {"content_summary": "Unplanned two-hour strategy meeting disrupted the afternoon, leaving sprint work delayed. No agenda was prepared.",
         "memory_tier": "episodic", "emotional_valence": "negative", "stated_intent": None,
         "topic_cluster": "work / meetings", "timestamp": "2025-10-16",
         "source": "daily_trace", "source_id": "trace_20251016_001", "channel": "work"},
    ),
    make_example(
        "Finished reading 'The Pragmatic Programmer' last night. Really liked the section on "
        "orthogonality — it reframed how I think about module boundaries. Took about three weeks on and off.",
        {"content_summary": "Finished reading The Pragmatic Programmer; found the orthogonality chapter particularly useful for thinking about module boundaries.",
         "memory_tier": "episodic", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "learning / books / software engineering", "timestamp": "2025-10-17",
         "source": "daily_trace", "source_id": "trace_20251017_001", "channel": "personal"},
    ),
    make_example(
        "Paired with Alex on the auth refactor today. We got the token refresh logic working "
        "cleanly after two frustrating false starts. Alex caught a subtle edge case I'd missed in the expiry check.",
        {"content_summary": "Paired programming session with Alex on the auth refactor; resolved token refresh logic after two failed attempts, with Alex identifying an expiry window edge case.",
         "memory_tier": "episodic", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / collaboration", "timestamp": "2025-10-18",
         "source": "daily_trace", "source_id": "trace_20251018_001", "channel": "work"},
    ),
    make_example(
        "Had a difficult conversation with my manager today about the promotion timeline. "
        "Felt like my concerns weren't fully heard. Left the meeting uncertain about next steps.",
        {"content_summary": "Difficult conversation with manager about promotion timeline; concerns felt unheard and next steps remain unclear.",
         "memory_tier": "episodic", "emotional_valence": "negative", "stated_intent": None,
         "topic_cluster": "career / management", "timestamp": "2025-10-19",
         "source": "daily_trace", "source_id": "trace_20251019_001", "channel": "work"},
    ),
    make_example(
        "Shipped the v2.1 release today after three weeks of work. The deployment went smoothly "
        "with no issues. Felt good to get it across the line.",
        {"content_summary": "Successfully shipped v2.1 release after three weeks; deployment was smooth with no issues.",
         "memory_tier": "episodic", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / releases", "timestamp": "2025-10-20",
         "source": "daily_trace", "source_id": "trace_20251020_001", "channel": "work"},
    ),
    make_example(
        "Went for a 45-minute run this morning before work. First run in three weeks. "
        "Felt harder than expected but good to be moving again.",
        {"content_summary": "First run in three weeks — 45 minutes before work. Felt physically demanding but positive overall.",
         "memory_tier": "episodic", "emotional_valence": "mixed", "stated_intent": None,
         "topic_cluster": "health / exercise", "timestamp": "2025-10-21",
         "source": "daily_trace", "source_id": "trace_20251021_001", "channel": "personal"},
    ),
    make_example(
        "Missed the team standup again today — third time this month. Got absorbed in a deep "
        "work session and lost track of time. Felt embarrassed when I saw the Slack messages.",
        {"content_summary": "Missed the team standup for the third time this month due to being absorbed in a deep work session; felt embarrassed afterwards.",
         "memory_tier": "episodic", "emotional_valence": "negative", "stated_intent": None,
         "topic_cluster": "work / communication", "timestamp": "2025-10-22",
         "source": "daily_trace", "source_id": "trace_20251022_001", "channel": "work"},
    ),
    make_example(
        "Had a great coffee catch-up with Nadia today. She's pivoting to product management "
        "after five years in engineering. Her reasoning made a lot of sense.",
        {"content_summary": "Informal catch-up with Nadia who is transitioning from engineering to product management after five years.",
         "memory_tier": "episodic", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "relationships / career conversations", "timestamp": "2025-10-23",
         "source": "daily_trace", "source_id": "trace_20251023_001", "channel": "personal"},
    ),
    make_example(
        "Finally cleared the 47 unread emails that had been building up for two weeks. "
        "Most of them were noise but there were three important threads I'd missed.",
        {"content_summary": "Cleared 47 accumulated unread emails; identified three important missed threads among mostly low-signal messages.",
         "memory_tier": "episodic", "emotional_valence": "mixed", "stated_intent": None,
         "topic_cluster": "work / communication / inbox", "timestamp": "2025-10-24",
         "source": "daily_trace", "source_id": "trace_20251024_001", "channel": "work"},
    ),
    make_example(
        "Presented the MLX training pipeline prototype to the team today. The demo went well — "
        "good engagement and three concrete follow-up questions from people who were clearly thinking about it.",
        {"content_summary": "Presented the MLX training pipeline prototype to the team; well-received with concrete follow-up questions demonstrating engagement.",
         "memory_tier": "episodic", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / presentations", "timestamp": "2025-10-25",
         "source": "daily_trace", "source_id": "trace_20251025_001", "channel": "work"},
    ),
    make_example(
        "Took the afternoon off to visit my parents. First time in six weeks. "
        "Felt overdue — good to catch up but also noticed how much I'd been avoiding it.",
        {"content_summary": "First visit to parents in six weeks, taken as an afternoon off. Mixed awareness of the overdue nature of the visit.",
         "memory_tier": "episodic", "emotional_valence": "mixed", "stated_intent": None,
         "topic_cluster": "relationships / family", "timestamp": "2025-10-26",
         "source": "daily_trace", "source_id": "trace_20251026_001", "channel": "personal"},
    ),
    make_example(
        "Completed the quarterly self-review today. Took about two hours. "
        "Harder than I expected to articulate the impact of the work I've done. "
        "Ended up rewriting the summary section three times.",
        {"content_summary": "Completed two-hour quarterly self-review; struggled to articulate work impact clearly, rewriting the summary section three times.",
         "memory_tier": "episodic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "career / performance review", "timestamp": "2025-10-27",
         "source": "daily_trace", "source_id": "trace_20251027_001", "channel": "work"},
    ),
    make_example(
        "Watched a 90-minute talk on distributed consensus algorithms tonight. "
        "The section on Raft leader election finally clicked for me. Feel like I now have "
        "a solid mental model of why log compaction works the way it does.",
        {"content_summary": "Watched a 90-minute talk on distributed consensus; Raft leader election and log compaction semantics now clearly understood.",
         "memory_tier": "episodic", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "learning / distributed systems", "timestamp": "2025-10-28",
         "source": "daily_trace", "source_id": "trace_20251028_001", "channel": "personal"},
    ),
]

# ── Semantic examples (15) ────────────────────────────────────────────────────

SEMANTIC = [
    make_example(
        "I've noticed over the past year that I do my clearest thinking in the first 90 minutes "
        "after waking, before checking messages. Afternoons are better for routine tasks.",
        {"content_summary": "Personal insight: peak cognitive clarity is in the first 90 minutes post-waking; afternoons suit routine work better.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "self-knowledge / work patterns", "timestamp": "2025-10-15",
         "source": "daily_trace", "source_id": "trace_20251015_002", "channel": "personal"},
    ),
    make_example(
        "I've been vaguely thinking about switching to a standing desk setup. Read a few "
        "articles about posture and long-term health. Not sure if it's worth the cost right now — no concrete plans.",
        {"content_summary": "Vague interest in transitioning to a standing desk, informed by reading about posture and health. No plan or timeline stated.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "health / work environment", "timestamp": "2025-10-16",
         "source": "daily_trace", "source_id": "trace_20251016_002", "channel": "personal"},
    ),
    make_example(
        "I'd love to get back into regular meditation at some point. Used to do it consistently "
        "a few years ago. No idea when I'd actually restart — life feels too packed right now.",
        {"content_summary": "Nostalgic interest in resuming meditation practice; previously consistent but currently not practicing. No timeline or plan stated.",
         "memory_tier": "semantic", "emotional_valence": "mixed", "stated_intent": None,
         "topic_cluster": "health / habits / mindfulness", "timestamp": "2025-10-17",
         "source": "daily_trace", "source_id": "trace_20251017_002", "channel": "personal"},
    ),
    make_example(
        "I think I might want to move toward more infrastructure-focused work at some point. "
        "The systems design problems are what I find most energizing. But I'm not actively looking to change anything right now.",
        {"content_summary": "Directional preference toward infrastructure and systems design work; energized by those problems. No active career change intended.",
         "memory_tier": "semantic", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "career / self-knowledge / engineering", "timestamp": "2025-10-18",
         "source": "daily_trace", "source_id": "trace_20251018_002", "channel": "work"},
    ),
    make_example(
        "I've been thinking I should probably write more consistently — maybe a daily note habit "
        "would help me think better. Not sure when I'd actually start though.",
        {"content_summary": "Vague belief that a consistent daily writing habit would improve thinking. No concrete start plan.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "habits / writing / reflection", "timestamp": "2025-10-19",
         "source": "daily_trace", "source_id": "trace_20251019_002", "channel": "personal"},
    ),
    make_example(
        "I generally work better with a lot of context before jumping into a problem — "
        "I've noticed I make worse decisions when I'm rushed. Deep work in long uninterrupted blocks is where I do my best thinking.",
        {"content_summary": "Self-knowledge: works better with full context and long uninterrupted blocks; quality degrades under time pressure.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "self-knowledge / cognitive style", "timestamp": "2025-10-20",
         "source": "daily_trace", "source_id": "trace_20251020_002", "channel": "personal"},
    ),
    make_example(
        "I care a lot about code clarity over cleverness. I've learned that I regret 'clever' "
        "solutions within six months when I have to read them again.",
        {"content_summary": "Personal engineering value: prefers code clarity over cleverness; past experience with clever solutions has consistently led to regret.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "engineering / values / code quality", "timestamp": "2025-10-21",
         "source": "daily_trace", "source_id": "trace_20251021_002", "channel": "work"},
    ),
    make_example(
        "I've always been better at systems thinking than at detailed execution. "
        "Planning and architecture come naturally; following through on the tedious parts takes real effort.",
        {"content_summary": "Self-awareness: strong in systems thinking and architecture; sustained execution on tedious tasks requires deliberate effort.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "self-knowledge / work style", "timestamp": "2025-10-22",
         "source": "daily_trace", "source_id": "trace_20251022_002", "channel": "personal"},
    ),
    make_example(
        "At some point I'd like to build something completely on my own — a small product "
        "with real users. Not sure what it would be or when I'd have time.",
        {"content_summary": "Long-term aspiration to build an independent product with real users. No idea, timeline, or plan is present.",
         "memory_tier": "semantic", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "career / entrepreneurship / aspirations", "timestamp": "2025-10-23",
         "source": "daily_trace", "source_id": "trace_20251023_002", "channel": "personal"},
    ),
    make_example(
        "I've noticed that I tend to over-explain things when I'm uncertain. It's a way of "
        "filling space when I don't fully trust my own position.",
        {"content_summary": "Self-pattern identified: over-explaining correlates with internal uncertainty, used as a coping mechanism.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "self-knowledge / communication", "timestamp": "2025-10-24",
         "source": "daily_trace", "source_id": "trace_20251024_002", "channel": "personal"},
    ),
    make_example(
        "I find it much easier to stay motivated on work that has a clear external artifact "
        "at the end — a shipped feature, a published document, something tangible. "
        "Open-ended exploratory work drains me more than it should.",
        {"content_summary": "Motivation pattern: driven by tangible outputs (shipped features, documents); open-ended exploratory work is disproportionately draining.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "self-knowledge / motivation", "timestamp": "2025-10-25",
         "source": "daily_trace", "source_id": "trace_20251025_002", "channel": "work"},
    ),
    make_example(
        "I've been thinking about learning Japanese for years. Never got past the basics. "
        "Probably won't unless something changes. Just an ongoing background interest.",
        {"content_summary": "Long-standing background interest in learning Japanese; has never progressed beyond basics and no change in trajectory is anticipated.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "learning / languages / aspirations", "timestamp": "2025-10-26",
         "source": "daily_trace", "source_id": "trace_20251026_002", "channel": "personal"},
    ),
    make_example(
        "I've come to realize I trust people more when they tell me what they don't know "
        "than when they project confidence. Epistemic humility matters a lot to me in collaborators.",
        {"content_summary": "Core value identified: epistemic humility is a key trust signal in collaborators; confident projection of unknown knowledge reduces trust.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "self-knowledge / values / collaboration", "timestamp": "2025-10-27",
         "source": "daily_trace", "source_id": "trace_20251027_002", "channel": "personal"},
    ),
    make_example(
        "Conflict avoidance is a pattern I've noticed in myself for a long time. "
        "I tend to let things simmer rather than raise them early. I know it causes more "
        "problems downstream but it's a hard habit to break.",
        {"content_summary": "Long-recognized self-pattern of conflict avoidance: preference to let issues accumulate rather than address them proactively, despite awareness of downstream costs.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "self-knowledge / communication / habits", "timestamp": "2025-10-28",
         "source": "daily_trace", "source_id": "trace_20251028_002", "channel": "personal"},
    ),
    make_example(
        "I've always thought I'd write a technical blog at some point — sharing what I've "
        "learned about distributed systems. No urgency. Just something I keep imagining.",
        {"content_summary": "Background aspiration to write a technical blog about distributed systems; no urgency, timeline, or concrete step contemplated.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "writing / engineering / aspirations", "timestamp": "2025-10-29",
         "source": "daily_trace", "source_id": "trace_20251029_001", "channel": "personal"},
    ),
]

# ── Procedural examples (10) ──────────────────────────────────────────────────

PROCEDURAL = [
    make_example(
        "Finally figured out the right pattern for handling concurrent map access in Go: "
        "acquire a write lock only when mutating, use RLock for reads, and never hold the lock "
        "across a goroutine boundary. This is solid mental scaffolding now.",
        {"content_summary": "Internalized Go concurrency pattern: write lock only on mutation, RLock for reads, never hold lock across goroutine boundaries.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / Go / concurrency", "timestamp": "2025-10-15",
         "source": "daily_trace", "source_id": "trace_20251015_003", "channel": "work"},
    ),
    make_example(
        "Developed a reliable workflow for complex debugging sessions: first reproduce the "
        "issue in isolation, then add structured logging before reaching for a debugger, "
        "then binary-search the call stack. This sequence consistently works.",
        {"content_summary": "Established personal debugging workflow: isolate → structured logging → binary search call stack. Consistently effective.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / debugging / process", "timestamp": "2025-10-16",
         "source": "daily_trace", "source_id": "trace_20251016_003", "channel": "work"},
    ),
    make_example(
        "Learned that the most effective way to give design feedback is to lead with the "
        "question I'm trying to answer rather than the solution I have in mind. "
        "Keeps the discussion open instead of triggering defensiveness.",
        {"content_summary": "Design feedback technique: open with the question, not the proposed solution. Reduces defensiveness and keeps the discussion exploratory.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "communication / feedback / design", "timestamp": "2025-10-17",
         "source": "daily_trace", "source_id": "trace_20251017_003", "channel": "work"},
    ),
    make_example(
        "Got a reliable system for keeping my inbox under control: process to zero once a day "
        "at 3pm, use snooze for anything that needs action later, and archive aggressively. "
        "This has worked consistently for two months now.",
        {"content_summary": "Proven inbox management system: daily 3pm processing, snooze for deferred actions, aggressive archiving. Two months of consistent success.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "productivity / email management", "timestamp": "2025-10-18",
         "source": "daily_trace", "source_id": "trace_20251018_003", "channel": "work"},
    ),
    make_example(
        "Found a good approach for writing technical RFCs: start with the problem statement "
        "and constraints before any solution, explicitly call out alternatives considered "
        "and why they were rejected. This structure gets much better review feedback.",
        {"content_summary": "RFC writing structure: problem statement and constraints first, then solution, with explicit alternatives and rejection rationale. Generates better review feedback.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / writing / design process", "timestamp": "2025-10-19",
         "source": "daily_trace", "source_id": "trace_20251019_003", "channel": "work"},
    ),
    make_example(
        "Figured out the right way to run a retrospective that actually produces action items: "
        "time-box the venting phase strictly to 10 minutes, then spend 20 minutes on root "
        "cause for the single biggest issue, and end with one specific owned action.",
        {"content_summary": "Effective retrospective format: 10-minute venting, 20-minute root cause on single top issue, one owned action item. Produces actionable outcomes.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "management / process / retrospectives", "timestamp": "2025-10-20",
         "source": "daily_trace", "source_id": "trace_20251020_003", "channel": "work"},
    ),
    make_example(
        "When I'm stuck on a problem, the most reliable unsticking technique is to explain "
        "it out loud to someone who knows nothing about it. The act of teaching always "
        "reveals the assumption I was getting wrong.",
        {"content_summary": "Reliable unsticking technique: explain the problem to a non-expert aloud. The teaching process consistently surfaces incorrect assumptions.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "problem-solving / learning", "timestamp": "2025-10-21",
         "source": "daily_trace", "source_id": "trace_20251021_003", "channel": "personal"},
    ),
    make_example(
        "Developed my approach for code review: read the PR description first, then read "
        "the tests before the implementation, then trace one happy path manually. "
        "Catches design issues much faster than reading top to bottom.",
        {"content_summary": "Personal code review workflow: description → tests → implementation → manual happy path trace. More effective at catching design issues than linear reading.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / code review", "timestamp": "2025-10-22",
         "source": "daily_trace", "source_id": "trace_20251022_003", "channel": "work"},
    ),
    make_example(
        "The way I prep for difficult conversations now: write out the concern in one sentence, "
        "then write out the best case for the other person's position, then write what "
        "I actually want from the conversation. Does the same work as rehearsing but faster.",
        {"content_summary": "Difficult conversation prep technique: one-sentence concern statement, best case for other position, desired outcome. Effective alternative to rehearsing.",
         "memory_tier": "procedural", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "communication / conflict / preparation", "timestamp": "2025-10-23",
         "source": "daily_trace", "source_id": "trace_20251023_003", "channel": "personal"},
    ),
    make_example(
        "I've learned to start every new project by writing the README first — "
        "specifically the 'how to use this' section. If I can't explain the interface "
        "before building it, I don't understand the problem well enough yet.",
        {"content_summary": "Project development practice: write the README usage section first. Inability to explain the interface upfront signals insufficient problem understanding.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / design / documentation", "timestamp": "2025-10-24",
         "source": "daily_trace", "source_id": "trace_20251024_003", "channel": "work"},
    ),
]

# ── Prospective examples (25, including 10 aspiration-vs-commitment cases) ────

PROSPECTIVE = [
    # Clear commitments with external accountability
    make_example(
        "I have to send the project retrospective write-up to the team by Thursday. "
        "I promised Marcus I'd have it done. Need to block 2 hours tomorrow morning for it.",
        {"content_summary": "Committed to delivering a project retrospective write-up to the team by Thursday; explicitly promised to Marcus. Blocking time tomorrow morning.",
         "memory_tier": "prospective", "emotional_valence": "neutral",
         "stated_intent": "Send project retrospective write-up to the team by Thursday.",
         "topic_cluster": "project management / communication", "timestamp": "2025-10-15",
         "source": "daily_trace", "source_id": "trace_20251015_004", "channel": "work"},
    ),
    make_example(
        "Told Jordan I'd review the design docs for the onboarding rewrite by tomorrow "
        "morning before the team sync. She's counting on my feedback to finalize the scope.",
        {"content_summary": "Explicit commitment to review Jordan's onboarding rewrite design docs before tomorrow's team sync. Social accountability: Jordan is waiting.",
         "memory_tier": "prospective", "emotional_valence": "neutral",
         "stated_intent": "Review Jordan's onboarding rewrite design docs before tomorrow's team sync.",
         "topic_cluster": "work / design / collaboration", "timestamp": "2025-10-16",
         "source": "daily_trace", "source_id": "trace_20251016_004", "channel": "work"},
    ),
    make_example(
        "Need to submit the grant proposal draft to the committee by Friday 5pm. "
        "This is a firm external deadline — no extensions available.",
        {"content_summary": "Firm external deadline: grant proposal draft must be submitted to committee by Friday 5pm. No extensions available.",
         "memory_tier": "prospective", "emotional_valence": "neutral",
         "stated_intent": "Submit grant proposal draft to committee by Friday 5pm.",
         "topic_cluster": "work / grants / deadlines", "timestamp": "2025-10-17",
         "source": "daily_trace", "source_id": "trace_20251017_004", "channel": "work"},
    ),
    make_example(
        "Told Sarah I'd review her pull request before noon today. "
        "Just saw her Slack message asking if I'm still on it.",
        {"content_summary": "Committed to reviewing Sarah's pull request before noon; social follow-up received via Slack confirming the accountability.",
         "memory_tier": "prospective", "emotional_valence": "neutral",
         "stated_intent": "Review Sarah's pull request before noon today.",
         "topic_cluster": "engineering / collaboration", "timestamp": "2025-10-19",
         "source": "daily_trace", "source_id": "trace_20251019_004", "channel": "work"},
    ),
    make_example(
        "Made a clear decision today: I'm going to stop accepting ad-hoc Slack requests during "
        "deep work hours. Starting tomorrow, I'll only check Slack at 10am and 3pm. Already set the status.",
        {"content_summary": "Explicit self-commitment: restrict Slack to 10am and 3pm daily starting tomorrow; decline ad-hoc requests during deep work. Status already set.",
         "memory_tier": "prospective", "emotional_valence": "positive",
         "stated_intent": "Restrict Slack checks to 10am and 3pm daily starting tomorrow.",
         "topic_cluster": "productivity / communication habits", "timestamp": "2025-10-20",
         "source": "daily_trace", "source_id": "trace_20251020_004", "channel": "work"},
    ),
    make_example(
        "Committed to writing the migration guide for v2.1 and delivering it to the "
        "customer success team by next Monday. Promised them this in the release call.",
        {"content_summary": "Committed to writing and delivering v2.1 migration guide to customer success team by next Monday; commitment made on the release call.",
         "memory_tier": "prospective", "emotional_valence": "neutral",
         "stated_intent": "Write and deliver v2.1 migration guide to customer success team by next Monday.",
         "topic_cluster": "engineering / documentation / releases", "timestamp": "2025-10-21",
         "source": "daily_trace", "source_id": "trace_20251021_004", "channel": "work"},
    ),
    make_example(
        "Need to schedule the onboarding sync with the two new engineers this week. "
        "They've been waiting four days and I keep deferring it. Will do it before end of day tomorrow.",
        {"content_summary": "Explicit intention to schedule onboarding sync with two new engineers before end of day tomorrow; has been deferred four days already.",
         "memory_tier": "prospective", "emotional_valence": "neutral",
         "stated_intent": "Schedule onboarding sync with new engineers before end of day tomorrow.",
         "topic_cluster": "management / onboarding", "timestamp": "2025-10-22",
         "source": "daily_trace", "source_id": "trace_20251022_004", "channel": "work"},
    ),
    make_example(
        "I need to have the conversation with my manager about the project timeline by end "
        "of this week. I've been avoiding it but it's becoming a blocker.",
        {"content_summary": "Self-identified need to have a project timeline conversation with manager by end of week; acknowledges prior avoidance.",
         "memory_tier": "prospective", "emotional_valence": "neutral",
         "stated_intent": "Have a conversation with manager about the project timeline by end of this week.",
         "topic_cluster": "work / management / communication", "timestamp": "2025-10-23",
         "source": "daily_trace", "source_id": "trace_20251023_004", "channel": "work"},
    ),
    make_example(
        "Called my dad this morning and promised I'd come visit next weekend. "
        "I've been saying 'soon' for too long. This time it's on the calendar.",
        {"content_summary": "Explicit commitment to visit father next weekend; verbal promise made during a call and already on the calendar.",
         "memory_tier": "prospective", "emotional_valence": "positive",
         "stated_intent": "Visit father next weekend (confirmed by phone and added to calendar).",
         "topic_cluster": "relationships / family", "timestamp": "2025-10-24",
         "source": "daily_trace", "source_id": "trace_20251024_004", "channel": "personal"},
    ),
    make_example(
        "Decided I'm going to present a 15-minute lightning talk at the next team all-hands "
        "on the MLX pipeline work. Spoke to the organizer about it — she's adding me to the agenda.",
        {"content_summary": "Committed to presenting a 15-minute lightning talk on the MLX pipeline at the next team all-hands; organizer confirmed and adding to agenda.",
         "memory_tier": "prospective", "emotional_valence": "positive",
         "stated_intent": "Present a 15-minute lightning talk on the MLX pipeline at the next team all-hands.",
         "topic_cluster": "engineering / presentations", "timestamp": "2025-10-25",
         "source": "daily_trace", "source_id": "trace_20251025_004", "channel": "work"},
    ),
    # Conditional / contingent commitments (still prospective)
    make_example(
        "If I can wrap the current sprint by Thursday, I'm going to take Friday afternoon "
        "completely off — no laptop, just read and walk. I need it.",
        {"content_summary": "Contingent commitment: take Friday afternoon off (no laptop, reading and walking) if the sprint is completed by Thursday.",
         "memory_tier": "prospective", "emotional_valence": "positive",
         "stated_intent": "Take Friday afternoon off to read and walk, contingent on completing sprint by Thursday.",
         "topic_cluster": "work-life balance / rest", "timestamp": "2025-10-26",
         "source": "daily_trace", "source_id": "trace_20251026_004", "channel": "personal"},
    ),
    make_example(
        "Once this project ships, I'm taking a proper week off. No half-measures — "
        "out of office, phone on airplane mode, the whole thing. I've promised myself this.",
        {"content_summary": "Self-promise to take a full week off (out-of-office, phone on airplane mode) once the current project ships.",
         "memory_tier": "prospective", "emotional_valence": "positive",
         "stated_intent": "Take a full week off with complete disconnection after the current project ships.",
         "topic_cluster": "work-life balance / rest", "timestamp": "2025-10-27",
         "source": "daily_trace", "source_id": "trace_20251027_004", "channel": "personal"},
    ),
    # Weak-signal prospective (aspiration-vs-commitment boundary — prospective side)
    make_example(
        "I've been saying I'll reach out to David for weeks now. Every time I open my messages "
        "I think about it and then close them. I should really just do it this week.",
        {"content_summary": "Repeated deferral of reaching out to David over several weeks; self-stated intention to act this week despite previous avoidance.",
         "memory_tier": "prospective", "emotional_valence": "neutral",
         "stated_intent": "Reach out to David this week.",
         "topic_cluster": "relationships / networking", "timestamp": "2025-10-28",
         "source": "daily_trace", "source_id": "trace_20251028_004", "channel": "personal"},
    ),
    make_example(
        "I keep putting off scheduling my annual physical. It's been over a year. "
        "I'm going to book the appointment this week — no more deferring.",
        {"content_summary": "Annual physical overdue by over a year; explicit self-commitment to book the appointment this week.",
         "memory_tier": "prospective", "emotional_valence": "neutral",
         "stated_intent": "Book annual physical appointment this week.",
         "topic_cluster": "health / preventive care", "timestamp": "2025-10-29",
         "source": "daily_trace", "source_id": "trace_20251029_002", "channel": "personal"},
    ),
    make_example(
        "I've been meaning to reach out to Claire about the mentorship opportunity. "
        "She mentioned it twice and I keep saying I'll follow up. I really should just send the message this week.",
        {"content_summary": "Repeated deferral of following up with Claire on a mentorship opportunity she has mentioned twice; self-directed intention to send a message this week.",
         "memory_tier": "prospective", "emotional_valence": "neutral",
         "stated_intent": "Reach out to Claire about the mentorship opportunity this week.",
         "topic_cluster": "relationships / mentorship", "timestamp": "2025-10-30",
         "source": "daily_trace", "source_id": "trace_20251030_001", "channel": "personal"},
    ),
    # Aspiration-vs-commitment boundary cases — SEMANTIC side (NOT prospective)
    make_example(
        "I've been thinking about maybe learning Rust at some point. Heard it's worth it "
        "for systems programming. Not planning to do anything about it anytime soon.",
        {"content_summary": "Background interest in learning Rust for systems programming; explicitly no current plan or timeline.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "learning / programming languages", "timestamp": "2025-10-16",
         "source": "daily_trace", "source_id": "trace_20251016_005", "channel": "personal"},
    ),
    make_example(
        "I'd love to travel somewhere I've never been this year — maybe Southeast Asia. "
        "Just an idea I keep coming back to. Nothing concrete.",
        {"content_summary": "Recurring travel aspiration toward Southeast Asia; no concrete plan, timeline, or logistics.",
         "memory_tier": "semantic", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "travel / aspirations", "timestamp": "2025-10-17",
         "source": "daily_trace", "source_id": "trace_20251017_005", "channel": "personal"},
    ),
    make_example(
        "Sometimes I think about going back to do a part-time master's degree in CS. "
        "No real urgency — just something I imagine when I'm frustrated with knowledge gaps.",
        {"content_summary": "Occasional fantasy of pursuing a part-time CS master's degree when frustrated by knowledge gaps; no urgency or actionable intent.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "career / education / aspirations", "timestamp": "2025-10-18",
         "source": "daily_trace", "source_id": "trace_20251018_005", "channel": "personal"},
    ),
    make_example(
        "I keep thinking it would be nice to get better at public speaking. "
        "I'm not particularly bad at it but I could be much better. "
        "Maybe I'll look into a course someday.",
        {"content_summary": "Vague aspiration to improve public speaking skills; acknowledges room for improvement but no plan or timeline beyond a speculative 'someday'.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "skills / communication / aspirations", "timestamp": "2025-10-19",
         "source": "daily_trace", "source_id": "trace_20251019_005", "channel": "personal"},
    ),
    make_example(
        "I've been wanting to cook more at home instead of ordering out so much. "
        "It would save money and probably be healthier. I think about it often "
        "but don't do anything differently.",
        {"content_summary": "Recurring desire to cook more at home for financial and health reasons; aware of the aspiration but behavior hasn't changed.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "lifestyle / habits / health", "timestamp": "2025-10-20",
         "source": "daily_trace", "source_id": "trace_20251020_005", "channel": "personal"},
    ),
    make_example(
        "At some point I'd like to open source one of my internal tools. The config "
        "validator especially seems like something others would find useful. "
        "No concrete plans — just an idea.",
        {"content_summary": "Background idea to open source an internal config validation tool; believes it would be useful to others but no concrete plan exists.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "engineering / open source / aspirations", "timestamp": "2025-10-21",
         "source": "daily_trace", "source_id": "trace_20251021_005", "channel": "work"},
    ),
    make_example(
        "I keep meaning to set up a proper home office — better monitor, standing desk, "
        "actual cable management. Keep not doing it because it feels like a big project.",
        {"content_summary": "Ongoing intention to set up a proper home office (monitor, standing desk, cable management); consistently deferred due to perceived scope.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "lifestyle / workspace / aspirations", "timestamp": "2025-10-22",
         "source": "daily_trace", "source_id": "trace_20251022_005", "channel": "personal"},
    ),
    make_example(
        "I've been telling myself I'll get better at saying no to low-priority requests "
        "for about a year now. Still working on it. Awareness hasn't translated into behavior yet.",
        {"content_summary": "Year-long awareness of a pattern of over-accepting low-priority requests; behavioral change has not yet followed the self-awareness.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "self-knowledge / habits / boundaries", "timestamp": "2025-10-23",
         "source": "daily_trace", "source_id": "trace_20251023_005", "channel": "personal"},
    ),
    make_example(
        "Promised myself I'd start journaling daily after that productivity article I read. "
        "That was three months ago. Still haven't started consistently. "
        "Not sure it's really going to happen.",
        {"content_summary": "Self-promise to journal daily made three months ago following a productivity article; not yet followed through and now expressing doubt it will happen.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "habits / reflection / aspirations", "timestamp": "2025-10-24",
         "source": "daily_trace", "source_id": "trace_20251024_005", "channel": "personal"},
    ),
    make_example(
        "I think I need to get better at delegating. I know intellectually that I hold on "
        "to too much myself. But putting it into practice is a different matter — "
        "no specific plan for how to change.",
        {"content_summary": "Intellectual recognition of over-delegation avoidance pattern; no specific plan or behavioral commitment to change.",
         "memory_tier": "semantic", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "self-knowledge / management / habits", "timestamp": "2025-10-25",
         "source": "daily_trace", "source_id": "trace_20251025_005", "channel": "work"},
    ),
]


# ── Additional episodic examples (10, to reach 25 total) ─────────────────────

EPISODIC_EXTRA = [
    make_example(
        "Gave my first conference talk today at PyCon. About 200 people in the room. "
        "Felt nervous for the first five minutes then found my rhythm. Good questions from the audience afterward.",
        {"content_summary": "First conference talk at PyCon; ~200 attendees, initial nerves gave way to fluency, received engaged audience questions.",
         "memory_tier": "episodic", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "career / public speaking", "timestamp": "2025-11-01",
         "source": "daily_trace", "source_id": "trace_20251101_001", "channel": "work"},
    ),
    make_example(
        "Had a really rough onboarding session with the new contractor today. "
        "Communication was unclear from the start and we wasted two hours on the wrong codebase section.",
        {"content_summary": "Difficult onboarding session with new contractor; miscommunication led to two hours spent on the wrong codebase section.",
         "memory_tier": "episodic", "emotional_valence": "negative", "stated_intent": None,
         "topic_cluster": "work / onboarding / communication", "timestamp": "2025-11-02",
         "source": "daily_trace", "source_id": "trace_20251102_001", "channel": "work"},
    ),
    make_example(
        "Reached 100 days of consistent morning journaling today. Never expected to stick with it this long. "
        "Feels like a real habit now, not an effort.",
        {"content_summary": "100-day milestone of consistent morning journaling achieved; now feels habitual rather than effortful.",
         "memory_tier": "episodic", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "habits / reflection / milestones", "timestamp": "2025-11-03",
         "source": "daily_trace", "source_id": "trace_20251103_001", "channel": "personal"},
    ),
    make_example(
        "Accidentally deleted the wrong branch in production. Spent two hours recovering from the backup. "
        "No data lost but deeply embarrassing. Filed an incident report.",
        {"content_summary": "Accidentally deleted a production branch; two-hour recovery from backup with no data loss. Incident report filed.",
         "memory_tier": "episodic", "emotional_valence": "negative", "stated_intent": None,
         "topic_cluster": "engineering / incidents / git", "timestamp": "2025-11-04",
         "source": "daily_trace", "source_id": "trace_20251104_001", "channel": "work"},
    ),
    make_example(
        "Had a surprisingly good 1:1 with my skip-level today. She asked what was blocking me and actually listened. "
        "First time I've felt heard at that level in a while.",
        {"content_summary": "Positive skip-level 1:1; manager actively listened to blockers, first time feeling genuinely heard at that level in recent memory.",
         "memory_tier": "episodic", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "career / management / relationships", "timestamp": "2025-11-05",
         "source": "daily_trace", "source_id": "trace_20251105_001", "channel": "work"},
    ),
    make_example(
        "Hiked the Marin Headlands loop today — about 12 miles. First proper long hike in over a year. "
        "Body held up better than expected. Good to be outside for that long.",
        {"content_summary": "12-mile Marin Headlands hike completed; first extended hike in over a year with better-than-expected physical performance.",
         "memory_tier": "episodic", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "health / exercise / outdoors", "timestamp": "2025-11-06",
         "source": "daily_trace", "source_id": "trace_20251106_001", "channel": "personal"},
    ),
    make_example(
        "The client rejected the proposal we spent four weeks on. They changed the spec mid-process "
        "and now want something completely different. Entire team is frustrated.",
        {"content_summary": "Four-week proposal rejected by client due to mid-process spec change; team morale low following the outcome.",
         "memory_tier": "episodic", "emotional_valence": "negative", "stated_intent": None,
         "topic_cluster": "work / client / projects", "timestamp": "2025-11-07",
         "source": "daily_trace", "source_id": "trace_20251107_001", "channel": "work"},
    ),
    make_example(
        "Passed my AWS Solutions Architect exam on the first attempt today. "
        "Studied for six weeks. Higher score than I expected — 874.",
        {"content_summary": "Passed AWS Solutions Architect certification exam on first attempt with a score of 874 after six weeks of study.",
         "memory_tier": "episodic", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "learning / certifications / cloud", "timestamp": "2025-11-08",
         "source": "daily_trace", "source_id": "trace_20251108_001", "channel": "work"},
    ),
    make_example(
        "Tried pair programming with the junior engineer on the cache invalidation logic. "
        "Slower than coding alone but she asked questions that caught two bugs I would have missed.",
        {"content_summary": "Pair programming session on cache invalidation with junior engineer; slower pace but two bugs caught by her questions that would have been missed solo.",
         "memory_tier": "episodic", "emotional_valence": "mixed", "stated_intent": None,
         "topic_cluster": "engineering / mentorship / collaboration", "timestamp": "2025-11-09",
         "source": "daily_trace", "source_id": "trace_20251109_001", "channel": "work"},
    ),
    make_example(
        "Submitted the resignation letter today. Four years at this company. "
        "Felt surreal hitting send. Warm but bittersweet.",
        {"content_summary": "Resignation letter submitted after four years at the company; emotionally surreal and bittersweet.",
         "memory_tier": "episodic", "emotional_valence": "mixed", "stated_intent": None,
         "topic_cluster": "career / transitions", "timestamp": "2025-11-10",
         "source": "daily_trace", "source_id": "trace_20251110_001", "channel": "work"},
    ),
]

# ── Additional procedural examples (15, to reach 25 total) ───────────────────

PROCEDURAL_EXTRA = [
    make_example(
        "Figured out how to reliably estimate engineering tasks: give a best-case, "
        "likely-case, and worst-case, always multiply the likely case by 1.5 for the "
        "external commitment. Two years of doing this has made my estimates accurate.",
        {"content_summary": "Reliable estimation practice: three-point estimate (best/likely/worst), externally commit to 1.5× the likely-case. Two years of validation.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / project management / estimation", "timestamp": "2025-11-01",
         "source": "daily_trace", "source_id": "trace_20251101_002", "channel": "work"},
    ),
    make_example(
        "The way I handle ambiguous requirements now: write out what I think they mean in one sentence, "
        "send it to the stakeholder as a question not a proposal, and don't build anything until I get a yes. "
        "Saves me rework every single time.",
        {"content_summary": "Ambiguous requirement handling: summarize interpretation as a question, await explicit confirmation before building. Consistently prevents rework.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / communication / requirements", "timestamp": "2025-11-02",
         "source": "daily_trace", "source_id": "trace_20251102_002", "channel": "work"},
    ),
    make_example(
        "I've found the most effective way to learn a new codebase: run the tests first, "
        "then read the entry point file, then trace the critical path for the main use case. "
        "Never start by reading README or docs.",
        {"content_summary": "Effective codebase onboarding sequence: run tests → read entry point → trace critical path for main use case. Docs last.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / learning / onboarding", "timestamp": "2025-11-03",
         "source": "daily_trace", "source_id": "trace_20251103_002", "channel": "work"},
    ),
    make_example(
        "The pattern I use for database migrations now: always write the rollback script before the migration, "
        "test both in staging, and never deploy the forward migration on a Friday. "
        "Has saved me twice.",
        {"content_summary": "Database migration practice: write rollback before migration, test both in staging, never deploy on Fridays. Prevented incidents twice.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / databases / deployment", "timestamp": "2025-11-04",
         "source": "daily_trace", "source_id": "trace_20251104_002", "channel": "work"},
    ),
    make_example(
        "The best way I've found to keep meetings productive: own the agenda, "
        "send it 24 hours before, explicitly label which items are decisions vs. updates vs. discussion, "
        "and end 5 minutes early to allow bio breaks.",
        {"content_summary": "Effective meeting facilitation: send agenda 24h ahead, label item types (decision/update/discussion), end 5 minutes early.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "management / communication / meetings", "timestamp": "2025-11-05",
         "source": "daily_trace", "source_id": "trace_20251105_002", "channel": "work"},
    ),
    make_example(
        "When I'm reviewing someone else's architecture, I've learned to ask 'what's the failure mode of this?' "
        "for each component before evaluating the happy path. Finds more real problems than "
        "trying to improve the design.",
        {"content_summary": "Architecture review technique: ask 'what's the failure mode?' for each component before evaluating the happy path. More effective than design critique.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / architecture / review", "timestamp": "2025-11-06",
         "source": "daily_trace", "source_id": "trace_20251106_002", "channel": "work"},
    ),
    make_example(
        "For writing proposals or technical documents, I now write the executive summary last "
        "but first in the document. The act of forcing a one-paragraph summary always reveals "
        "whether the underlying argument is sound.",
        {"content_summary": "Document writing technique: write executive summary last (but position first). Forces clarity check on the underlying argument's soundness.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "writing / communication / process", "timestamp": "2025-11-07",
         "source": "daily_trace", "source_id": "trace_20251107_002", "channel": "work"},
    ),
    make_example(
        "The way I handle being overwhelmed at work: list everything in my head on paper first "
        "(don't organize, just dump), then highlight the one thing that if done would most reduce the overwhelm. "
        "Do that one thing before anything else.",
        {"content_summary": "Overwhelm management technique: brain dump without organizing → identify one highest-leverage item → execute before anything else.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "productivity / stress management", "timestamp": "2025-11-08",
         "source": "daily_trace", "source_id": "trace_20251108_002", "channel": "personal"},
    ),
    make_example(
        "For async code reviews I've found: always leave one positive, specific comment for every three "
        "change requests. Not to be nice — it tells the author what to preserve. "
        "Review quality went up when I started doing this.",
        {"content_summary": "Code review practice: leave one specific positive comment per three change requests to signal what to preserve. Improves review quality.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / code review / communication", "timestamp": "2025-11-09",
         "source": "daily_trace", "source_id": "trace_20251109_002", "channel": "work"},
    ),
    make_example(
        "I've found that when I'm procrastinating on a task, it's almost always because "
        "I haven't defined the first physical action clearly enough. Once I can write "
        "'open file X and change line Y', I start immediately.",
        {"content_summary": "Procrastination insight: root cause is usually insufficient specificity of next action. Defining a concrete first step immediately removes the block.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "productivity / habits / self-knowledge", "timestamp": "2025-11-10",
         "source": "daily_trace", "source_id": "trace_20251110_002", "channel": "personal"},
    ),
    make_example(
        "The pattern for giving feedback to senior peers: ask if they want feedback first, "
        "then lead with what the work is trying to do (their goal), then offer the observation "
        "as a question not a judgment. Never skip step one.",
        {"content_summary": "Senior peer feedback approach: ask permission first, name their goal, offer observation as question not judgment.",
         "memory_tier": "procedural", "emotional_valence": "neutral", "stated_intent": None,
         "topic_cluster": "communication / feedback / relationships", "timestamp": "2025-11-11",
         "source": "daily_trace", "source_id": "trace_20251111_002", "channel": "work"},
    ),
    make_example(
        "I now always start any new data pipeline with an end-to-end test using a tiny sample — "
        "3 rows max — before touching the real dataset. Catches schema issues and wiring bugs "
        "in 2 minutes instead of 2 hours.",
        {"content_summary": "Data pipeline development practice: run end-to-end test on 3-row sample before touching real data. Catches schema and wiring issues in minutes vs. hours.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / data / testing", "timestamp": "2025-11-12",
         "source": "daily_trace", "source_id": "trace_20251112_002", "channel": "work"},
    ),
    make_example(
        "Learned that the right way to introduce process changes on a team is: "
        "pilot it on your own work first for two weeks, then share the outcome as data "
        "rather than a proposal. Much lower resistance than proposing upfront.",
        {"content_summary": "Process change adoption: self-pilot for two weeks first, then share outcome data instead of a proposal. Reduces team resistance significantly.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "management / team / change", "timestamp": "2025-11-13",
         "source": "daily_trace", "source_id": "trace_20251113_002", "channel": "work"},
    ),
    make_example(
        "The way I recover from decision paralysis on technical choices: "
        "write down what would be true if option A was right, and what would be true "
        "if option B was right. Usually one set of conditions is clearly more realistic.",
        {"content_summary": "Decision paralysis technique for technical choices: list preconditions for each option being correct. Usually one option's preconditions are more realistic.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / decision-making", "timestamp": "2025-11-14",
         "source": "daily_trace", "source_id": "trace_20251114_002", "channel": "work"},
    ),
    make_example(
        "When I want to understand a system quickly, I've learned to look for what happens "
        "when the system receives bad input before I read the normal flow. Error handling reveals "
        "the authors' assumptions better than any other part of the code.",
        {"content_summary": "Rapid system understanding technique: read error handling before normal flow. Error paths reveal authorial assumptions more transparently than happy paths.",
         "memory_tier": "procedural", "emotional_valence": "positive", "stated_intent": None,
         "topic_cluster": "engineering / code reading / debugging", "timestamp": "2025-11-15",
         "source": "daily_trace", "source_id": "trace_20251115_002", "channel": "work"},
    ),
]

# ── Additional prospective examples (10, to reach 25 total) ──────────────────

PROSPECTIVE_EXTRA = [
    make_example(
        "Set a hard deadline with myself: finish the API documentation by this Friday "
        "or block Monday to do it. Putting it in the calendar now.",
        {"content_summary": "Self-imposed deadline: complete API documentation by Friday, with Monday blocked as fallback. Added to calendar.",
         "memory_tier": "prospective", "emotional_valence": "neutral",
         "stated_intent": "Complete API documentation by Friday (or Monday at latest).",
         "topic_cluster": "engineering / documentation / planning", "timestamp": "2025-11-01",
         "source": "daily_trace", "source_id": "trace_20251101_003", "channel": "work"},
    ),
    make_example(
        "Got an email from the conference organizers — I agreed to submit an abstract by November 15th. "
        "It's on my calendar. The talk would be about the MLX pipeline work.",
        {"content_summary": "Agreed to submit conference abstract by November 15th; confirmed via email and on calendar. Topic: MLX pipeline.",
         "memory_tier": "prospective", "emotional_valence": "positive",
         "stated_intent": "Submit conference abstract by November 15th.",
         "topic_cluster": "engineering / conferences / writing", "timestamp": "2025-11-02",
         "source": "daily_trace", "source_id": "trace_20251102_003", "channel": "work"},
    ),
    make_example(
        "Committed to my therapist that I'd try the morning routine she suggested for two full weeks "
        "starting Monday. Told her I'd report back at our next session.",
        {"content_summary": "Commitment to therapist: try suggested morning routine for two weeks starting Monday; will report back at next session.",
         "memory_tier": "prospective", "emotional_valence": "neutral",
         "stated_intent": "Follow therapist's suggested morning routine for two weeks starting Monday.",
         "topic_cluster": "health / mental health / habits", "timestamp": "2025-11-03",
         "source": "daily_trace", "source_id": "trace_20251103_003", "channel": "personal"},
    ),
    make_example(
        "Decided I'm finally going to finish the distributed systems course I started in March. "
        "Going to block one hour every Tuesday and Thursday evening until it's done. "
        "Lectures blocked on the calendar starting this week.",
        {"content_summary": "Commitment to complete a distributed systems course; 1-hour blocks scheduled Tuesday and Thursday evenings, starting this week.",
         "memory_tier": "prospective", "emotional_valence": "positive",
         "stated_intent": "Complete distributed systems course by blocking 1 hour Tuesday and Thursday evenings.",
         "topic_cluster": "learning / distributed systems / habits", "timestamp": "2025-11-04",
         "source": "daily_trace", "source_id": "trace_20251104_003", "channel": "personal"},
    ),
    make_example(
        "The team agreed today: we're moving the staging deploy to Thursdays instead of Fridays. "
        "I'm responsible for updating the CI/CD config and the runbook. Targeting next week.",
        {"content_summary": "Team decision: staging deploys move from Fridays to Thursdays. Responsible for updating CI/CD config and runbook, targeting next week.",
         "memory_tier": "prospective", "emotional_valence": "neutral",
         "stated_intent": "Update CI/CD config and runbook for Thursday staging deploys by next week.",
         "topic_cluster": "engineering / CI/CD / process", "timestamp": "2025-11-05",
         "source": "daily_trace", "source_id": "trace_20251105_003", "channel": "work"},
    ),
    make_example(
        "Promised to write a post-mortem for the outage that happened last Tuesday. "
        "It's overdue. Will have a draft to the team by end of day Wednesday.",
        {"content_summary": "Overdue post-mortem for last Tuesday's outage; committed to delivering a draft to the team by end of day Wednesday.",
         "memory_tier": "prospective", "emotional_valence": "neutral",
         "stated_intent": "Deliver post-mortem draft for last Tuesday's outage by end of day Wednesday.",
         "topic_cluster": "engineering / incidents / documentation", "timestamp": "2025-11-06",
         "source": "daily_trace", "source_id": "trace_20251106_003", "channel": "work"},
    ),
    make_example(
        "Going to do a digital detox weekend starting Friday evening — no news, no social, "
        "no work Slack. Already told my partner and put my work status on DND.",
        {"content_summary": "Planned digital detox weekend starting Friday evening: no news, social media, or work Slack. Partner informed, work status set to DND.",
         "memory_tier": "prospective", "emotional_valence": "positive",
         "stated_intent": "Digital detox weekend starting Friday evening: no news, social media, or work communication.",
         "topic_cluster": "work-life balance / mental health / rest", "timestamp": "2025-11-07",
         "source": "daily_trace", "source_id": "trace_20251107_003", "channel": "personal"},
    ),
    make_example(
        "I need to have a direct conversation with Sam about the recurring lateness on deliverables. "
        "It's affecting the whole team and I keep avoiding it. Will do it before the sprint review Thursday.",
        {"content_summary": "Need to have a direct conversation with Sam about recurring delivery lateness; repeatedly avoided despite team impact. Self-committed to doing it before sprint review Thursday.",
         "memory_tier": "prospective", "emotional_valence": "neutral",
         "stated_intent": "Have a direct conversation with Sam about recurring delivery lateness before sprint review Thursday.",
         "topic_cluster": "management / communication / accountability", "timestamp": "2025-11-08",
         "source": "daily_trace", "source_id": "trace_20251108_003", "channel": "work"},
    ),
    make_example(
        "Registered for the half marathon in March. Non-refundable. I'm actually doing this. "
        "Training starts next Monday — found a 16-week plan that fits.",
        {"content_summary": "Registered for a non-refundable March half marathon; 16-week training plan identified, starting next Monday.",
         "memory_tier": "prospective", "emotional_valence": "positive",
         "stated_intent": "Train for and complete March half marathon; 16-week plan starts next Monday.",
         "topic_cluster": "health / exercise / goals", "timestamp": "2025-11-09",
         "source": "daily_trace", "source_id": "trace_20251109_003", "channel": "personal"},
    ),
    make_example(
        "Agreed with my co-author that we'll submit the paper draft to the workshop by December 1st. "
        "She's handling the related work section, I'm doing the experiments. We're meeting weekly.",
        {"content_summary": "Co-authored paper submission agreed for December 1st workshop deadline. Division: related work (co-author) and experiments (self). Weekly meetings in place.",
         "memory_tier": "prospective", "emotional_valence": "neutral",
         "stated_intent": "Submit paper draft to workshop by December 1st; responsible for experiments section.",
         "topic_cluster": "research / writing / collaboration", "timestamp": "2025-11-10",
         "source": "daily_trace", "source_id": "trace_20251110_003", "channel": "work"},
    ),
]

# ── Assembly ──────────────────────────────────────────────────────────────────

ALL_EXAMPLES = EPISODIC + EPISODIC_EXTRA + SEMANTIC + PROCEDURAL + PROCEDURAL_EXTRA + PROSPECTIVE + PROSPECTIVE_EXTRA


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic SFT training data for Trace Layer 2. "
            "No model required — examples are template-based."
        )
    )
    parser.add_argument(
        "--output",
        default="data/sft_train_100.jsonl",
        help="Output JSONL path (default: data/sft_train_100.jsonl).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle examples before writing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42).",
    )
    args = parser.parse_args()

    examples = ALL_EXAMPLES.copy()

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(examples)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Distribution summary
    from collections import Counter
    tiers = Counter()
    for ex in examples:
        for msg in ex["messages"]:
            if msg["role"] == "assistant":
                try:
                    rec = json.loads(msg["content"])
                    tiers[rec.get("memory_tier", "unknown")] += 1
                except Exception:
                    pass

    print(f"\nWrote {len(examples)} examples to {out_path}")
    print("\nTier distribution:")
    for tier in ["episodic", "semantic", "procedural", "prospective"]:
        print(f"  {tier:<15} {tiers.get(tier, 0)}")
    print()
    print("Next steps:")
    print(f"  python scripts/prepare_data.py --sft {args.output} --eval data/eval_gold.jsonl --seed 42")
    print("  python scripts/train_sft.py --config configs/sft_trace_qwen25_3b_v2.yaml")


if __name__ == "__main__":
    main()
