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

### The chain carries only what the handoff frame SHOWS

**A persistent entity whose identity depends on fine geometry or detail must
have its identifying features visible in the handoff frame, or be supplied as an
explicit reference image. First-frame chaining alone is not sufficient for
hidden or occluded fine geometry.**

Measured on 2026-09-21, acts A2-A4 of `berry-bramble-15s`: the glass, board,
hands, camera height, lighting and shadow held perfectly across four chained
acts, while the muddler became three different tools --

| Act | Head geometry |
|---|---|
| A2 | wide hollow head, bottom rim cut into a crown of ~8 triangular teeth |
| A3 | solid flat-bottomed puck, conical fillet, no teeth |
| A4 | cylinder with round holes drilled through the wall, no wide head |

Shaft diameter stayed at 27-30px throughout, which is why a silhouette check
passed it. The cause was not the chain failing: in **both** handoff frames the
muddler head was submerged in pulp, so its identifying geometry was never in the
pixels being handed forward. Each act invented a plausible end. A1 ended with no
muddler at all, so A2 invented the first one from the words "stainless-steel
muddler".

Practical rules:

- Before a cut, ask what each persistent entity actually LOOKS LIKE in the final
  frame. Occluded, submerged, out of frame, or motion-blurred means not carried.
- Either stage a beat that re-exposes the entity before the handoff, or attach a
  reference image.
- Describe the discriminating geometry in every act prompt ("toothed crown
  head"), not the category ("a muddler"). Free, and it narrows the invention.
- Fine detail - teeth, perforations, engraving, labels, logos - is the first
  thing to drift and the last thing a silhouette check catches.

### Make the identity visible at the handoff (planning rule)

**If an entity has fine-grained identity that must survive across clips, the
production planner should make that identity visibly observable in the handoff
frame wherever possible.** Stage the action so the distinguishing feature is
exposed, unoccluded and in focus at the moment the clip ends.

Naming the category is not enough. An entity described only by its class - a
"stainless-steel tool", "a delivery van", "her jacket" - leaves every
discriminating detail to be re-invented: the crown of teeth on a tool head, the
livery on a vehicle door, the patch on a sleeve. Plan a beat that shows the
feature, then hand over.

This is a **planning rule, not a guarantee from the model.** A visible feature
raises the odds the next act reproduces it; it does not promise it. Combine with
prompt constraints that name the feature, and with references where the model
supports them.

Applies uniformly to process subjects (tools, containers, ingredients, work
surfaces) and narrative subjects (characters, clothing, props, vehicles,
locations). Nothing in the implementation knows what kind of entity it is.

### What each mechanism is actually for

| Mechanism | What it does | What it is NOT |
|---|---|---|
| Prompt constraints | Name the discriminating feature in words. Always available, on every model. | Not a guarantee; text does not pin fine geometry. |
| `first_frame` | Conditions the OPENING frame. Carries everything visible in the handoff frame: composition, camera, lighting, object states. The causal chain default. | Cannot carry what the handoff frame does not show. |
| `last_frame` | One endpoint of an **interpolation** between two known frames. | **Not an object-identity conditioning primitive.** On `google/veo-3.1-lite` it is invalid on its own - the provider rejects it with "Frame interpolation requires both an input image and a last frame." |
| `input_references` | Identity anchors for what the handoff frame cannot carry - where a model supports them. | **Model-capability dependent and never to be assumed.** As of 2026-09-21 the OpenRouter video listing publishes no reference field for any model, so support is unestablished there. |

Nothing above is guaranteed by a model. Support is a property of the selected
endpoint, and the generation layer checks it rather than assuming it.

### Recommended path for Veo 3.1 Lite

```
Entity Bible
    -> Causal Planner
    -> Identity constraints in the prompt
    -> Entity-visible handoff planning
    -> first_frame chaining
    -> Veo 3.1 Lite
    -> Continuity QA
    -> Edit
```

**`last_frame` is not part of the normal causal chain.** Reach for it only when
a production explicitly wants an interpolation between two known endpoint
frames, and then supply both frames - `build_chain_payloads` refuses an act that
pins only its close. `input_references` stays out of this path until a model
advertises reference support.

### Who owns the entity bible

**One production video, one entity bible, and it is API-independent.** It
defines visual identity, identity constraints, continuity constraints,
allowed and forbidden changes, and reference assets when they exist. It says
nothing about `frame_images`, `input_references`, wire shapes or model names.

**The agent creates and maintains it** from the production plan and reference
material, as part of normal planning - the operator is not expected to author it
by hand. Surface it for human approval when the identity decisions are
consequential (a hero product, a recurring character) or when a constraint would
be expensive to discover late.

The generation layer then decides how each constraint can actually be expressed
for the selected model - see `lib.causal_chain.identity_mechanisms(model, caps)`,
which answers from a capability listing and treats anything unadvertised as
unavailable.

### The mechanism ladder (and who chooses)

```
causal prompt      fixes causality INSIDE a clip
prompt constraints name the discriminating feature in words
first_frame        pins the OPENING state; carries what is visible at the seam
last_frame         pins the CLOSING state, where supported by the model
identity refs      anchor what is NOT visible at the seam, where supported
```

**The entity bible says WHAT must stay the same; the generation layer decides
HOW.** The bible is production-scoped identity information - visual description,
identity and continuity constraints, allowed and forbidden changes, reference
images - and carries no API-specific behaviour. It does not know about
`frame_images`, `input_references`, wire shapes or which model is in use.

The generation layer reads the bible and picks the mechanism per act:

| Situation | Mechanism |
|---|---|
| Feature visible at the seam | first_frame chaining (already automatic) |
| The act must END in a specific state | `CausalAct.last_frame`, where the model advertises it |
| Feature occluded at the seam | `input_references`, where the model advertises it |
| Model supports neither, or as reinforcement | prompt constraints naming the feature, and a beat that exposes it |

Check `supported_frame_images` in `GET /api/v1/videos/models` before depending
on either frame pin: several models advertise `first_frame` only.

`CausalAct.reference_images` carries paths or URLs that are emitted as
`input_references`, and `CausalAct.last_frame` pins the closing frame. Both are
optional; an act that sets neither behaves exactly as before. Declare identity
once in the entity bible and attach it by citation:

```python
from lib.causal_chain import CausalAct, attach_entity_references

act = attach_entity_references(CausalAct(id="A3", prompt=...), bible, ["muddler"])
# act.reference_images == ("refs/muddler_toothed_head.jpg",)
```

**Model support is not guaranteed.** The unified OpenRouter schema accepts
`input_references` for every model; that is not the same as the upstream model
using it. Check `GET /api/v1/videos/models` for the model in question -- as of
2026-09-21 `google/veo-3.1-lite` advertises only
`supported_frame_images: [first_frame, last_frame]` and no reference capability
at all. Where references are unsupported, the fallback is to make the entity
visible at the handoff, or to move to a model that advertises support.

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
would launder a guess. **Do not invent an automated identity score.**

What continuity QA - machine evidence plus agent review - can legitimately
establish:

- state continuity across a boundary (what changed, what did not)
- object presence and absence
- consistency of *visible* geometry between compared frames
- frame-to-frame continuity of camera, framing, board, lighting
- measured facts: resolution, duration, frame rate, audio streams

What it must **not** claim:

- that an entity is identical when the evidence only shows it is *similar*
- that identity held where the feature was occluded, blurred or out of frame -
  absence of visible difference is not evidence of sameness
- any conclusion about frames that were never inspected. The 2026-09-21 tool
  drift passed every boundary check precisely because boundary frames match by
  construction while the drift happened mid-clip

State findings as what was observed, at what timestamps, at what resolution, and
say plainly when a question could not be answered from the evidence.

---

## Checklist before spending credits

- [ ] Analysis run with `transition_sampling: true`
- [ ] Every act is PREVIOUS_STATE -> ACTION -> RESULTING_STATE
- [ ] Every ACTION is visually observable with a visible agent
- [ ] Every object has an introduction beat
- [ ] Elided actions marked, before/after preserved
- [ ] Entity bible written; prompts use deictic anchors
- [ ] Every persistent entity is either unoccluded in each handoff frame OR has a
      reference image; fine-geometry entities name their discriminating detail in
      the prompt
- [ ] Reference support confirmed for the chosen model (GET /api/v1/videos/models),
      not assumed from the unified schema
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
