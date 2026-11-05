#!/usr/bin/env python3
"""Unit tests use a test double and are not model-success evidence."""
from __future__ import annotations
from control import StubResult, sequential_control, from_zero_control

def main():
    calls=[]
    def seq(**kw):
        calls.append(kw); return StubResult(True, 11 if kw['current_frame']==4 else 22, '<valid>')
    rows=sequential_control([4,50],seq)
    assert rows[0]['previous_frame']==0 and rows[0]['previous_progress']==0
    assert rows[1]['previous_frame']==4 and rows[1]['previous_progress']==11
    assert calls[1]['previous_frame']==4 and calls[1]['previous_progress']==11
    failed=[]
    def bad(**kw): failed.append(kw); return StubResult(False,None,'bad')
    assert len(sequential_control([4,50],bad))==1 and len(failed)==1
    zero=[]
    def z(**kw): zero.append(kw); return StubResult(False,None,'bad') if kw['current_frame']==4 else StubResult(True,95,'ok')
    rows=from_zero_control([4,95],z)
    assert len(rows)==2 and all('previous_frame' not in c and 'previous_progress' not in c for c in zero)
    try: sequential_control([4,4],seq)
    except ValueError: pass
    else: raise AssertionError('duplicate query accepted')
    print({'status':'passed','kind':'test_double_control_logic','sequential_calls':len(calls),'from_zero_calls':len(zero)})
if __name__=='__main__': main()
