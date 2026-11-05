#!/usr/bin/env python3
"""Focused protocol checks that do not load model weights."""
from __future__ import annotations
import argparse, dataclasses, json
from pathlib import Path
from processing import V7Collator, encode_sample, load_processor
from protocol import load_samples, messages, parse_completion, validate_sample


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--model-path',required=True)
    args=ap.parse_args(); samples=load_samples(Path(args.data)); assert len(samples)==4
    processor=load_processor(args.model_path,16384)
    encoded=[encode_sample(s,processor,8192) for s in samples]
    collator=V7Collator(processor.tokenizer.pad_token_id)
    mixed=collator([encoded[0],encoded[-1]])
    assert mixed['input_ids'].shape[0]==2
    assert mixed['image_grid_thw'].shape[0]==42
    assert (mixed['labels'][0,:encoded[0]['prompt_tokens']]==-100).all()
    assert (mixed['labels'][1,:encoded[-1]['prompt_tokens']]==-100).all()
    for s,e in zip(samples,encoded):
        assert e['images']==(22 if s.mode=='sequential' else 20)
        assert e['tensors']['image_grid_thw'].shape[0]==e['images']
        assert parse_completion(s.completion)['ok']
        user=json.dumps(messages(s)[1],ensure_ascii=False)
        if s.mode=='from_zero': assert 'Previous task progress:' not in user and 'role=previous' not in user
        else: assert 'Previous task progress:' in user and 'role=previous' in user
    assert parse_completion('<think><update>a</update><state>b</state><progress>c</progress></think><answer>-2%</answer>')['progress']==-2
    assert parse_completion('<think><update>a</update><state>b</state><progress>c</progress></think><answer>104%</answer>')['out_of_range']
    assert not parse_completion('<progress>10%</progress>')['ok']
    assert not parse_completion('<think><update>a</update><state>b</state><progress>c</progress></think><answer>2.5%</answer>')['ok']
    zero=next(s for s in samples if s.mode=='from_zero')
    try: validate_sample(dataclasses.replace(zero,previous=0),verify_hash=False)
    except ValueError: pass
    else: raise AssertionError('from_zero accepted PREVIOUS')
    print(json.dumps({'samples':len(samples),'mixed_batch_images':int(mixed['image_grid_thw'].shape[0]),
                      'modes':[s.mode for s in samples],'parser_checks':'passed'}))
if __name__=='__main__': main()
