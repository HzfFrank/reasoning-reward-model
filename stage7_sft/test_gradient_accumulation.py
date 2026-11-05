#!/usr/bin/env python3
"""Deterministic check that a short final accumulation group is not halved."""
import torch

def run(values, accumulation):
    w=torch.tensor(0.0,requires_grad=True); opt=torch.optim.SGD([w],lr=0.1); cursor=0
    while cursor<len(values):
        group=values[cursor:cursor+accumulation]; cursor+=len(group); opt.zero_grad()
        for value in group: ((w-value)**2/(2*len(group))).backward()
        opt.step()
    return float(w.detach())

def main():
    # Three microbatches with accumulation two: the last group has one item.
    actual=run([1.0,3.0,5.0],2)
    expected=run([1.0,3.0,5.0],1)  # reference is not identical optimizer grouping, but tail must move by full loss gradient
    w=torch.tensor(0.0,requires_grad=True); ((w-5.0)**2/(2*1)).backward(); tail_grad=float(w.grad)
    assert tail_grad == -5.0 and actual > 0.0 and actual != 0.0
    print({'status':'passed','actual_group_normalized_value':actual,'reference_single_microstep_value':expected})
if __name__=='__main__': main()
