"""Task-specific teacher wrapper; the approved JSON and student contract stay fixed."""
from v7_grounded_contract_templates import SYSTEM as BASE_SYSTEM, SCHEMA, REPAIR

VERSION = "v7_spatial_def_failure_sequential_teacher_v1"
SYSTEM = BASE_SYSTEM + """

SPATIAL FAILURE-WINDOW GROUNDING
All requests in this run are SEQUENTIAL. The object registry and window facts are
teacher-only aids. Use the task-specific registry; do not transfer names or
initial layouts from another Spatial task. Describe the correct target and
destination and the physical change visible between PREVIOUS and CURRENT.

Reliable facts may establish that a grasp never formed, a held object moved
away from its destination, or a non-target object was handled while the target
stayed put. For an extraction obstruction, describe the attempted extraction
and the target remaining inside the drawer; do not invent a specific blocking
part, force, or secure grasp. If an aid is uncertain, preserve that uncertainty.

Do not disclose the event category, perturbation, human review, simulator
record, final episode result, or future recovery. Do not assume an object
listed in the registry appears in a particular local camera view. Future
GLOBAL images identify the task but cannot prove the CURRENT state. Keep the
approved JSON field prefixes and numerical rendering contract unchanged.
"""
