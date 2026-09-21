# Causal Video Production — Layer 2 Skill

## When to Use

Any generated-video production where objects must persist across clips: process
and recipe films, product builds, assembly, craft, and multi-shot narrative.
Whenever a later clip must contain *the same* glass, tool, garment or character
as an earlier one.

If the piece is a single clip with no cross-shot identity, this skill is
overhead — use the normal generation path.

## Why this exists

Reference analysis used to describe **visual states**: "blueberry mash",
"glass packed with ice". Handed a state with no precondition, a video model
composes a plausible tableau for that instant. The observed result was
ingredients teleporting, fruit changing identity between beats, and containers
appearing fully assembled with no one placing them.

Two independent models (Veo family and MiniMax Hailuo-3) produced the same
failures from the same plan, which is what ruled out model choice as the cause.
The defect was in the representation.

See `docs/experiments/2026-09-21-h3-causal-ab.md` for the A/B that established
the fix.

---

## A. Causal scene representation

Every act is written as a transition, never as a state:

```
PREVIOUS_STATE  what is on screen before the action, including contents and position
ACTION          a visually observable physical event, performed by a visible agent
RESULTING_STATE what the action leaves behind
```

**The ACTION must be observable.** "The glass now contains raspberries" is a
state. "The gloved hand tilts the second glass and raspberries fall into the
first" is an action. Only the second can be rendered without invention.

Never rely on:

- state-only descriptions
- implied transfers ("then the mixture is combined")
- objects appearing with no one introducing them
- objects disappearing with no one removing them
- ingredients changing type between beats
- identity changing between states

**Write identity into the prompt with deictic anchors.** "The same glass", "the
same gloved hand", "the berries already in it". Vague noun phrases ("a glass")
give the model permission to produce a different one.

### Elided actions

A reference will sometimes cut over an action. Do not invent footage to fill
it. Mark it `ELIDED_ACTION` and keep the before/after states rigorous, so a
viewer completes the gap. Eliding is legitimate; leaving the after-state with
no before is not.

---

## B. Cross-clip continuity

```
Act N final frame  ->  Act N+1 first_frame
```

Causal prompts fix causality *inside* a clip. They do not hold identity
*between* clips — in the A/B test the unchained arm drifted the working glass
through four different silhouettes across five acts. Chaining held one glass,
one board crop, one shadow angle and one camera height across all five.

Chain whatever applies: characters, clothing, props, tools, containers,
ingredients, environment, camera position, lighting, shadows, composition.

Implementation: `lib/causal_chain.py`. Pass `first_frame` (a path, URL or data
URI) to `openrouter_video`, or let `generate_chain()` handle extraction and
threading.

**A failed act breaks the chain.** There is no final frame to carry forward, so
`generate_chain` stops rather than silently emitting an unchained clip.

---

## C. Generation defaults

| Setting | Default | Notes |
|---|---|---|
| resolution | **1080x1920** | vertical social delivery target |
| aspect_ratio | **9:16** | |
| duration | **5s per causal act** | enough for precondition, motion, postcondition |
| audio | **off** | OpenMontage composes its own; audio-on tiers cost more |

**2K is an optional high-quality mode, never the default.** It roughly triples
the bill for pixels a 1080p timeline discards. Reach for it only when the
delivery target is genuinely above 1080p.

Some models support only one tier — Hailuo-3 is 2K-only and rejects 720p at
validation. Check the model page before assuming a resolution is available.

---

## D. Production structure

```
Reference / Idea
  -> Analysis                 (transition_sampling: true — see below)
  -> Causal Plan              (PREVIOUS_STATE -> ACTION -> RESULTING_STATE per act)
  -> Entity Bible             (schemas/artifacts/entity_bible.schema.json)
  -> Approval                 (human gate — cost is known before this point)
  -> Generate Act 1
  -> Extract final frame
  -> Generate Act 2 with first_frame
  -> Extract final frame
  -> ... repeat
  -> Continuity QA            (lib/continuity_qa.py, then agent review)
  -> Edit
  -> Compose
  -> Publish
```

### Generate complete actions; compress in the edit

**Causal completeness is a generation property. Cut rhythm is an edit
property.** Do not compress actions at generation time to hit a target runtime.

A 15s deliverable whose causal chain needs seven acts requires ~35s of
generated footage. Asking each clip to be both a complete action and a
1-second fragment produces neither — that conflation is what created the
original state-only plan. Generate whole actions, then cut the social duration
out of them.

### Analysis must sample transitions

Run `video_analyzer` with `transition_sampling: true` for reference-driven
work. The default single frame per scene only shows the state a cut lands on;
actions inside a scene are invisible, which is exactly how three real actions
went missing from the first analysis of this project.

---

## Entity Bible

Declare persistent entities once in an `entity_bible` artifact and cite them by
`entity_id` from scene plans and prompts. Each entry carries
`identity_constraints`, `continuity_constraints`, `allowed_changes` and
`forbidden_changes`.

`allowed_changes` matters as much as `forbidden_changes`: without it a glass
filling with liquid reads as a continuity break rather than the point of the
shot.

---

## Continuity QA

`lib/continuity_qa.py` stages evidence at every act boundary — last frame of N
beside first frame of N+1 — plus measured checks (resolution, duration, audio
streams) and a contact sheet.

**It does not score semantic continuity, and should not be extended to.**
Whether two glasses are the same glass is a question for a multimodal reviewer.
Every semantic dimension is returned as `NEEDS_AGENT_REVIEW`; a number there
would launder a guess.

---

## Checklist before spending credits

- [ ] Analysis run with `transition_sampling: true`
- [ ] Every act is PREVIOUS_STATE -> ACTION -> RESULTING_STATE
- [ ] Every ACTION is visually observable with a visible agent
- [ ] Every object has an introduction beat
- [ ] Elided actions marked, before/after preserved
- [ ] Entity bible written; prompts use deictic anchors
- [ ] 1080x1920 / 9:16 / 5s / audio off, unless justified
- [ ] Model supports the chosen resolution and duration
- [ ] Chain planned with `dry_run=True`; total cost known
- [ ] Human approval on the cost

## Related

- `lib/causal_chain.py` — chaining orchestration
- `lib/continuity_qa.py` — evidence layer
- `schemas/artifacts/entity_bible.schema.json`
- `.agents/skills/openrouter-video/SKILL.md` — wire-level payload shapes
- `skills/creative/video-gen-prompting.md` — 5-aspect prompt vocabulary
- `skills/meta/video-reference-analyst.md` — reference intake
