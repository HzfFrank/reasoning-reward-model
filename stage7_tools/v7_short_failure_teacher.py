"""Frozen short-window extension of the approved v7 teacher contract."""
from v7_spatial_def_teacher import SYSTEM as BASE_SYSTEM, SCHEMA
VERSION = 'v7_short_failure_teacher_v1'
SYSTEM = BASE_SYSTEM + '''

SHORT-WINDOW CONTRACT
The query spans only a few observations. Describe this local subphase, not an
entire failure episode. A settled dropped object is not necessarily falling
again; an empty-gripper aftermath is not necessarily a new grasp attempt.
The local subphase and endpoint state facts are teacher-only aids, not mandates
to invent visual evidence. If there is little visible change, describe the
persistent unsuccessful state accurately. Do not assume a failure requires a
negative numerical change. Use the supplied labels unchanged and flag only
clear substantial contradictions.
In progress.reason explicitly connect the local evidence in update with the
current task stage and remaining work in state. Use GLOBAL for object identity
and task context, without borrowing future success or recovery as current fact.
Object registry entries are identities, not guarantees that each object is
visible. Do not confuse the instructed target with a distractor.
Keep the approved prefixes, JSON schema and deterministic numerical renderer.
'''
