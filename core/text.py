"""Text normalisation shared by ingestion and retrieval.

The problem this solves: TMDB stores "Bong Joon Ho", people (and LLMs) write
"Bong Joon-ho". An exact match returns nothing, and — the dangerous part — an
empty result from the graph is indistinguishable from "this director has no
films", so the failure is silent.

Names vary in three ways that carry no meaning:
    hyphens vs spaces     "Bong Joon-ho"   / "Bong Joon Ho"
    diacritics            "Fede Álvarez"   / "Fede Alvarez"
    punctuation & case    "Samuel L. Jackson" / "samuel l jackson"

We normalise ONCE at ingestion into `Person.name_normalized`, and normalise the
query the same way at lookup. Doing it at write time means the comparison is an
indexed equality check rather than a scan with string functions, and it lets us
strip accents properly in Python — Cypher has no portable way to do that without
APOC.
"""

from __future__ import annotations

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_name(value: str | None) -> str:
    """Fold a person's name to a comparable key.

    "Bong Joon-ho"      -> "bong joon ho"
    "Fede Álvarez"      -> "fede alvarez"
    "Samuel L. Jackson" -> "samuel l jackson"
    """
    if not value:
        return ""
    # NFKD splits "á" into "a" + combining accent; dropping the combining marks
    # leaves plain ASCII letters.
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _NON_ALNUM.sub(" ", stripped.lower()).strip()
