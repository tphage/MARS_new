"""Shared parsing utilities for LLM response processing."""

import re
from typing import List, Optional


def _get_parsing_config() -> dict:
    """Load parsing config from config.yaml."""
    try:
        from ..config import load_config
        return load_config().get("utils", {}).get("parsing", {})
    except Exception:
        return {}


def parse_to_list(content: str) -> List[str]:
    """Parse LLM response content into a list of strings.

    Handles numbered lists, bulleted lists, comma-separated values, and
    long single-paragraph responses.  Used by both ResearchManager and
    ResearchAssistant (and potentially other agents) so that list-parsing
    logic is maintained in one place.

    Args:
        content: Raw LLM response text.

    Returns:
        List of parsed string items (non-empty).
    """
    if not content or not isinstance(content, str):
        return [str(content)] if content else [""]

    content = content.strip()
    lines = content.split("\n")
    result: List[str] = []
    current_item: List[str] = []

    for line in lines:
        line = line.strip()

        if not line:
            if current_item:
                item_text = " ".join(current_item).strip()
                if item_text and len(item_text) >= 1:
                    result.append(item_text)
                current_item = []
            continue

        numbered_match = re.match(r"^(\d+)[.\)]\s+(.+)$", line)
        bullet_match = re.match(r"^[-*•]\s+(.+)$", line)

        if numbered_match or bullet_match:
            if current_item:
                item_text = " ".join(current_item).strip()
                if item_text and len(item_text) >= 1:
                    result.append(item_text)
                current_item = []

            item_text = numbered_match.group(2) if numbered_match else bullet_match.group(1)
            item_text = item_text.strip().strip("\"'")

            if item_text and len(item_text) >= 1:
                result.append(item_text)
        else:
            if line.startswith(("  ", "\t", "- ", "* ")) or not line[0].isupper():
                if current_item or result:
                    current_item.append(line)
                else:
                    item_text = line.strip("\"'")
                    if item_text and len(item_text) >= 1:
                        result.append(item_text)
            else:
                if current_item:
                    item_text = " ".join(current_item).strip()
                    if item_text and len(item_text) >= 1:
                        result.append(item_text)
                    current_item = []

                item_text = line.strip("\"'")
                if item_text and len(item_text) >= 1:
                    result.append(item_text)

    if current_item:
        item_text = " ".join(current_item).strip()
        if item_text and len(item_text) >= 1:
            result.append(item_text)

    # If no structured list found, try comma-separated
    if len(result) == 1 and "," in result[0]:
        result = [item.strip() for item in result[0].split(",") if item.strip()]

    # If still only one item and it's very long, try splitting by common delimiters
    threshold = _get_parsing_config().get("long_text_split_threshold", 200)
    if len(result) == 1 and len(result[0]) > threshold:
        for delimiter in ["\n\n", "; ", ". "]:
            if delimiter in result[0]:
                split_items = [item.strip() for item in result[0].split(delimiter) if item.strip()]
                if len(split_items) > 1:
                    result = split_items
                    break

    if not result:
        result = [content] if content else [""]

    return result


# ---------------------------------------------------------------------------
# Material-name cleaning
# ---------------------------------------------------------------------------

# Fragments that indicate the LLM echoed a prompt label rather than
# producing an actual material name.
_NAME_LABEL_PREFIXES = [
    "material name",
    "material:",
    "candidate:",
    "proposed material",
    "proposed candidate",
    "material candidate",
]


def clean_material_name(raw_name: Optional[str]) -> str:
    """Normalise a material name extracted from an LLM response.

    Handles the following artefacts commonly seen in raw LLM output:

    * Markdown formatting (``**bold**``, ``*italic*``, `` `code` ``)
    * Leading label echoes ("Material Name: …")
    * Surrounding quotes / brackets
    * Trailing punctuation (period, comma, semicolon)
    * Very long strings (truncated at the first sentence boundary ≤120 chars)
    * Embedded newlines
    * Unicode en/em-dashes normalised to ASCII hyphen

    Returns the cleaned name, or ``""`` if the input is empty / None.
    """
    if not raw_name or not isinstance(raw_name, str):
        return ""

    name = raw_name.strip()

    # Collapse newlines → space
    name = re.sub(r"[\r\n]+", " ", name)

    # Strip markdown bold/italic/code
    name = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", name)
    name = re.sub(r"`(.+?)`", r"\1", name)

    # Normalise dashes
    name = name.replace("\u2013", "-").replace("\u2014", "-")

    # Remove leading label echoes  ("Material Name: PEEK" → "PEEK")
    for prefix in _NAME_LABEL_PREFIXES:
        pattern = re.compile(r"^\s*" + re.escape(prefix) + r"\s*:?\s*", re.IGNORECASE)
        name = pattern.sub("", name)

    # Strip surrounding quotes and brackets
    name = name.strip("\"'""''«»[](){}")

    # Strip trailing punctuation that is unlikely to be part of a name
    name = re.sub(r"[.,;:!?]+$", "", name).strip()

    # If the name is extremely long, keep only up to the first sentence or max_material_name_length chars
    max_len = _get_parsing_config().get("max_material_name_length", 120)
    if len(name) > max_len:
        # Try to cut at the first sentence boundary
        m = re.search(r"[.!?]\s", name[: max_len + 1])
        if m:
            name = name[: m.start()].strip()
        else:
            name = name[:max_len].strip()

    # Final sanity check: if only whitespace or placeholder text remains, return empty
    if not name or name.lower() in ("unknown", "n/a", "none", "tbd"):
        return ""

    return name
