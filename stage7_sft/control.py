"""Pure sequential/from-zero state-machine logic used by inference tests."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class StubResult:
    ok: bool
    progress: int | None
    raw: str=''

def sequential_control(query_frames, generator, initial_progress=0):
    if not query_frames or any(query_frames[i] >= query_frames[i+1] for i in range(len(query_frames)-1)):
        raise ValueError('queries must be strictly increasing')
    previous_frame=0; previous_progress=initial_progress; records=[]
    for current in query_frames:
        result=generator(initial_frame=0,previous_frame=previous_frame,current_frame=current,previous_progress=previous_progress)
        records.append({'current':current,'previous_frame':previous_frame,'previous_progress':previous_progress,'result':result})
        if not result.ok: return records
        previous_frame=current; previous_progress=result.progress
    return records

def from_zero_control(query_frames, generator):
    records=[]
    for current in query_frames:
        result=generator(initial_frame=0,current_frame=current)
        records.append({'current':current,'result':result})
    return records
