"""Review-only teacher wrapper; the approved v7 schema and renderer stay unchanged."""
from v7_grounded_contract_templates import SYSTEM as APPROVED_SYSTEM, SCHEMA, REPAIR, VERSION as BASE_VERSION

VERSION = 'v7_geometry_v2_review_teacher_v1'
SYSTEM = APPROVED_SYSTEM + '''

TASK-6 GEOMETRY REVIEW CONTEXT
The reference percentages are deterministic current geometric task advancement, not a probability of success. A score near 100 does not establish completed placement. A wrong placement near the plate may preserve substantial transport progress while the task remains incomplete. The current target and previous anchor are supplied for offline supervision; do not derive stage weights or claim the percentage is measured exactly from pixels.
Describe only what the supplied images support. A missed grasp may leave approach progress; a dropped bowl may lose holding and lifting progress without returning to the initial state; recovery requires visible evidence. Do not use future storyboard frames as evidence of the CURRENT state. If a numerical direction clearly conflicts with visible evidence, flag label_visual_conflict rather than inventing an action.
Do not mention the geometric formula, simulator, contact counts, trajectory outcome, or annotation procedure in explanation fields. Do not include failure categories or final outcome in the JSON explanation.
'''
