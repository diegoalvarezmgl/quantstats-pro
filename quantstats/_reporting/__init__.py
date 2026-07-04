"""
Section-based HTML reporting engine for QuantStats Pro.

This package powers the newer tearsheets (``reports.html_simple``,
``reports.html_montecarlo``, and ``reports.html_alpha_decay``) without
modifying the legacy monolithic ``reports.html`` path. The legacy report
keeps its bespoke rendering for backwards compatibility.
"""

from . import engine, sections

__all__ = ["engine", "sections"]
