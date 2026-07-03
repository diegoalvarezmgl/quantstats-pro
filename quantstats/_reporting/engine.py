"""
Template rendering engine for section-based tearsheets.

The engine fills ``{{key}}`` placeholders from a fragments mapping and strips
any placeholder left unfilled, generalising the string-replace + regex cleanup
pattern that the legacy ``reports.html`` performs inline.
"""

import re as _regex

_PLACEHOLDER = _regex.compile(r"\{\{\s*(\w+)\s*\}\}")


def render_report(template_html: str, fragments: dict[str, str]) -> str:
    """
    Render a template by substituting ``{{key}}`` placeholders.

    Parameters
    ----------
    template_html : str
        Raw HTML template containing ``{{key}}`` placeholders.
    fragments : dict[str, str]
        Mapping of placeholder name to HTML/text fragment. Values are inserted
        verbatim. Keys not present in the template are ignored.

    Returns
    -------
    str
        The rendered HTML. Any placeholder without a matching key is removed.
    """

    def _sub(match: _regex.Match) -> str:
        key = match.group(1)
        value = fragments.get(key)
        return "" if value is None else str(value)

    return _PLACEHOLDER.sub(_sub, template_html)
