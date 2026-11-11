"""Opt-in Person Confidence output filter; no inference or persistence access."""

import math
import os
from collections import defaultdict


TRIAL_ENV = "HTTM_PERSON_TRIAL_MIN_CONF"
TRIAL_MIN_CONF = 0.35


def person_trial_min_conf_from_environment():
    value = os.environ.get(TRIAL_ENV)
    if value is None:
        return None
    if value == "0.35":
        return TRIAL_MIN_CONF
    raise ValueError(
        f"Invalid {TRIAL_ENV}={value!r}: only '0.35' is approved; "
        "unset the variable for the unchanged baseline."
    )


def _reject_removed_references(value, removed_refs, path):
    """Keep evidence verbatim or reject the frame; never rewrite ownership.

    Includes nested candidate lists and ROI/object provenance, whose person IDs
    are atomic strings in the existing JSON schema.
    """
    if isinstance(value, str) and value in removed_refs:
        raise ValueError(
            f"Person confidence trial: retained output at {path} references filtered "
            f"person {value!r}; cannot preserve behavior and return dangling references"
        )
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_removed_references(key, removed_refs, f"{path}.<key>")
            _reject_removed_references(child, removed_refs, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_removed_references(child, removed_refs, f"{path}[{index}]")


def filter_person_outputs(people, raw_people, min_conf=None):
    """Filter worker projections by the final merged COCO person confidence.

    Baseline is a true bypass, including absent confidence metadata. Candidate
    requires a unique output ref and exactly one raw person record for that ref.
    Kept dictionaries, behaviors, order and person refs are returned unchanged.
    """
    if min_conf is None:
        return people, []
    if isinstance(min_conf, bool) or min_conf != TRIAL_MIN_CONF:
        raise ValueError("Person confidence trial supports only min_conf=0.35")

    raw_by_ref = defaultdict(list)
    for raw in raw_people:
        if isinstance(raw, dict) and isinstance(raw.get("detection_id"), str):
            raw_by_ref[raw["detection_id"]].append(raw)
    kept, rejected, seen_refs = [], [], set()
    for person in people:
        ref = person.get("person_ref")
        if not isinstance(ref, str) or not ref.strip() or ref in seen_refs:
            raise ValueError(f"Person confidence trial: missing or ambiguous output person_ref {ref!r}")
        seen_refs.add(ref)
        matches = raw_by_ref[ref]
        if len(matches) != 1:
            raise ValueError(f"Person confidence trial: person_ref {ref!r} needs exactly one raw confidence record; got {len(matches)}")
        raw = matches[0]
        if raw.get("class_name", "person") != "person":
            raise ValueError(f"Person confidence trial: person_ref {ref!r} is not a person detection")
        confidence = raw.get("confidence")
        if (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError(f"Person confidence trial: missing or invalid confidence for person_ref {ref!r}")
        if confidence >= min_conf:
            kept.append(person)
        else:
            rejected.append({"person_ref": ref, "confidence": confidence,
                             "bbox": person.get("person_bbox_xyxy_clipped") or person.get("bbox_xyxy"),
                             "reason": "confidence_below_trial_threshold"})
    removed_refs = {row["person_ref"] for row in rejected}
    if removed_refs:
        for index, person in enumerate(kept):
            _reject_removed_references(person, removed_refs, f"people[{index}]")
    return kept, rejected
