from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConceptProbeSpec:
    name: str
    description: str
    pairs: tuple[tuple[str, str], ...]


_DEFAULT_PROBE_SPECS: tuple[ConceptProbeSpec, ...] = (
    ConceptProbeSpec(
        name="negation",
        description="Sensitivity to explicit propositional negation.",
        pairs=(
            ("The cat is sleeping.", "The cat is not sleeping."),
            ("She agreed with the plan.", "She did not agree with the plan."),
            ("The door is open.", "The door is not open."),
            ("They arrived on time.", "They did not arrive on time."),
            ("He believes the claim.", "He does not believe the claim."),
            ("This sentence is correct.", "This sentence is not correct."),
        ),
    ),
    ConceptProbeSpec(
        name="conditionality",
        description="Sensitivity to if-then dependencies and counterfactual framing.",
        pairs=(
            ("If it rains, the match will stop.", "The match will stop."),
            ("If she calls, I will answer.", "I will answer."),
            ("If the key fits, the lock will open.", "The lock will open."),
            ("If demand rises, prices increase.", "Prices increase."),
            ("If they leave now, they arrive early.", "They arrive early."),
            ("If the battery dies, the screen turns off.", "The screen turns off."),
        ),
    ),
    ConceptProbeSpec(
        name="causation",
        description="Sensitivity to causal connectors and explanatory relations.",
        pairs=(
            ("The ground is wet because it rained.", "The ground is wet."),
            ("She smiled because the joke landed.", "She smiled."),
            ("The alarm rang because smoke filled the room.", "The alarm rang."),
            ("Traffic slowed because the bridge closed.", "Traffic slowed."),
            ("The glass broke because it fell.", "The glass broke."),
            ("He apologized because he was wrong.", "He apologized."),
        ),
    ),
    ConceptProbeSpec(
        name="quantification",
        description="Sensitivity to quantifiers such as all, some, none, and many.",
        pairs=(
            ("All of the lights are on.", "Some of the lights are on."),
            ("None of the files were copied.", "Some of the files were copied."),
            ("Every student passed the exam.", "A student passed the exam."),
            ("Most of the seats were empty.", "A seat was empty."),
            ("No errors were reported.", "Errors were reported."),
            ("Each box contains a label.", "A box contains a label."),
        ),
    ),
    ConceptProbeSpec(
        name="temporal_order",
        description="Sensitivity to ordering in time and sequence transitions.",
        pairs=(
            ("Before lunch, they held the meeting.", "After lunch, they held the meeting."),
            ("She finished the draft before she sent it.", "She sent the draft before she finished it."),
            ("The sun set before the lights came on.", "The lights came on before the sun set."),
            ("He stretched before he ran.", "He ran before he stretched."),
            ("The server rebooted after the update.", "The server rebooted before the update."),
            ("They celebrated after the game ended.", "They celebrated before the game ended."),
        ),
    ),
    ConceptProbeSpec(
        name="epistemic_modality",
        description="Sensitivity to certainty, uncertainty, and evidential stance.",
        pairs=(
            ("She will arrive soon.", "She might arrive soon."),
            ("The claim is true.", "The claim may be true."),
            ("He knows the answer.", "He might know the answer."),
            ("The package is lost.", "The package could be lost."),
            ("This method works.", "This method probably works."),
            ("They are responsible.", "They may be responsible."),
        ),
    ),
    ConceptProbeSpec(
        name="coordination",
        description="Sensitivity to conjunction, disjunction, and coordinated structure.",
        pairs=(
            ("Alice and Bob signed the letter.", "Alice or Bob signed the letter."),
            ("Tea and coffee are available.", "Tea or coffee is available."),
            ("The switch controls light and heat.", "The switch controls light or heat."),
            ("He can pause and restart the job.", "He can pause or restart the job."),
            ("The filter removes dust and smoke.", "The filter removes dust or smoke."),
            ("We need speed and accuracy.", "We need speed or accuracy."),
        ),
    ),
    ConceptProbeSpec(
        name="discourse_boundary",
        description="Sensitivity to discourse contrast and topic transition markers.",
        pairs=(
            ("The rollout was expensive. However, it succeeded.", "The rollout was expensive. It succeeded."),
            ("The draft is rough. Still, it is usable.", "The draft is rough. It is usable."),
            ("He missed the train. Nevertheless, he arrived early.", "He missed the train. He arrived early."),
            ("The task is hard. But the team can finish it.", "The task is hard. The team can finish it."),
            ("The results were noisy. Yet the trend was clear.", "The results were noisy. The trend was clear."),
            ("She was tired. Even so, she kept working.", "She was tired. She kept working."),
        ),
    ),
    ConceptProbeSpec(
        name="entity_tracking",
        description="Sensitivity to coreference and role assignment over short contexts.",
        pairs=(
            ("Alice thanked Bob because he helped her.", "Alice thanked Bob because she helped him."),
            ("Mira called Lena after she left the office.", "Mira called Lena after Lena left the office."),
            ("Jon passed the note to Eli before he sat down.", "Jon passed the note to Eli before Eli sat down."),
            ("Nina warned Tara when she saw the fire.", "Nina warned Tara when Tara saw the fire."),
            ("Omar met Luis after he finished work.", "Omar met Luis after Luis finished work."),
            ("Priya texted Maya because she was delayed.", "Priya texted Maya because Maya was delayed."),
        ),
    ),
)


def default_concept_probe_specs() -> list[ConceptProbeSpec]:
    return list(_DEFAULT_PROBE_SPECS)


def load_concept_probe_specs(spec: str) -> list[ConceptProbeSpec]:
    spec = (spec or "default").strip()
    if spec in {"", "default"}:
        return default_concept_probe_specs()
    if spec.lower() in {"none", "off", "disable", "disabled"}:
        return []

    path = Path(spec).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Concept probe JSON must be a list of objects")

    specs: list[ConceptProbeSpec] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each concept probe entry must be an object")
        name = str(item["name"])
        description = str(item.get("description", ""))
        raw_pairs = item.get("pairs")
        if not isinstance(raw_pairs, list) or not raw_pairs:
            raise ValueError(f"Concept probe {name!r} must include a non-empty pairs list")
        pairs: list[tuple[str, str]] = []
        for pair in raw_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"Concept probe {name!r} has malformed pair: {pair!r}")
            pairs.append((str(pair[0]), str(pair[1])))
        specs.append(ConceptProbeSpec(name=name, description=description, pairs=tuple(pairs)))
    return specs
