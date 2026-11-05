"""Qwen3-VL chat/vision processing with completion-only labels."""
from __future__ import annotations
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor
from protocol import Sample, messages

IGNORE_INDEX = -100


def load_processor(model_path: str, image_pixels: int):
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.image_processor.min_pixels = image_pixels
    processor.image_processor.max_pixels = image_pixels
    return processor


def encode_sample(sample: Sample, processor, max_seq_length: int, training: bool = True) -> dict:
    chat = messages(sample)
    prompt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    images = []
    for im in sample.images:
        with Image.open(Path(im['path'])) as image:
            images.append(image.convert('RGB'))
    batch = processor(text=[prompt], images=images, return_tensors='pt', padding=False, add_special_tokens=False)
    image_token = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    if (batch['input_ids'] == image_token).sum().item() == 0 or batch['pixel_values'].numel() == 0:
        raise ValueError('processor did not create visual tokens/features')
    if batch['image_grid_thw'].shape[0] != len(images):
        raise ValueError('image_grid_thw count differs from image positions')
    # The processor expands each image placeholder to its patch tokens. Never crop that expansion.
    prompt_ids = batch['input_ids'][0]
    completion_ids = []
    if training:
        completion_ids = processor.tokenizer.encode(sample.completion, add_special_tokens=False)
        completion_ids.append(processor.tokenizer.eos_token_id)
    total = prompt_ids.numel() + len(completion_ids)
    if total > max_seq_length:
        raise ValueError(f'{sample.sample_id}: {total} tokens exceed max_seq_length={max_seq_length}; reduce image_pixels or raise the limit')
    if training:
        batch['input_ids'] = torch.cat((prompt_ids, torch.tensor(completion_ids, dtype=torch.long))).unsqueeze(0)
        batch['attention_mask'] = torch.ones_like(batch['input_ids'])
        batch['labels'] = torch.cat((torch.full_like(prompt_ids, IGNORE_INDEX), torch.tensor(completion_ids, dtype=torch.long))).unsqueeze(0)
        if not torch.all(batch['labels'][0, :prompt_ids.numel()] == IGNORE_INDEX):
            raise AssertionError('prompt labels were not masked')
    return {'tensors': batch, 'prompt_tokens': int(prompt_ids.numel()), 'completion_tokens': len(completion_ids),
            'images': len(images), 'sample_id': sample.sample_id, 'mode': sample.mode}


class V7Collator:
    def __init__(self, pad_token_id: int): self.pad_token_id = pad_token_id

    def __call__(self, items: list[dict]) -> dict:
        if not items: raise ValueError('empty batch')
        tensors = [item['tensors'] for item in items]
        max_len = max(t['input_ids'].shape[1] for t in tensors)
        result = {}
        for key, pad in [('input_ids', self.pad_token_id), ('attention_mask', 0), ('labels', IGNORE_INDEX)]:
            if key not in tensors[0]: continue
            rows=[]
            for t in tensors:
                value=t[key][0]; rows.append(torch.nn.functional.pad(value, (0,max_len-value.numel()), value=pad))
            result[key]=torch.stack(rows)
        for key in tensors[0]:
            if key in result or key in ('input_ids','attention_mask','labels'): continue
            vals=[t[key] for t in tensors]
            if key in ('pixel_values','image_grid_thw'): result[key]=torch.cat(vals,dim=0)
            elif len(vals)==1: result[key]=vals[0]
            else: raise ValueError(f'unsupported batched processor field: {key}')
        return result
