"""tokengolf — a calibrated, abstaining local↔remote model cascade.

Per task: try the cheapest tier (local = free), estimate a calibrated confidence
that its answer is correct, and escalate to a pricier remote tier only when the
confidence is below the tier's threshold. Tiers are cost-ordered; only remote
tokens are scored. The threshold is chosen so accuracy clears a floor with the
fewest remote tokens (a risk–coverage operating point).

Built for AMD Developer Hackathon: ACT II, Track 1. The calibration subpackage is
reused verbatim (see ATTRIBUTION.md); the retarget onto local-vs-remote model
choice is the only original seam.
"""
