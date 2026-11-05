#!/usr/bin/env python3
"""Full-parameter v7 SFT with Trainer/FSDP; train split only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import VERSION, TEMPLATE_SHA256
from protocol import load_samples


def atomic_json(path: Path, value: dict):
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.part')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--epochs', type=float, default=1.0)
    p.add_argument('--gradient-accumulation-steps', type=int, default=2)
    p.add_argument('--save-steps', type=int, default=100)
    p.add_argument('--resume', choices=['auto', 'never'], default='auto')
    return p.parse_args()


def protocol_manifest(a, samples, source_hash):
    modes = Counter(s.mode for s in samples)
    return {
        'kind': 'v7_full_parameter_sft', 'model_path': str(a.model_path.resolve()),
        'data_path': str(a.data.resolve()), 'data_sha256': source_hash,
        'prompt_version': VERSION, 'template_sha256': TEMPLATE_SHA256,
        'prompt_file_sha256': hashlib.sha256(Path(__file__).with_name('prompts.py').read_bytes()).hexdigest(),
        'split': 'train', 'sample_count': len(samples), 'modes': dict(modes),
        'image_pixels': 65536, 'max_seq_length': 8192,
        'global_sampling': 'round(i*(N-1)/7), i=0..7, paired main/wrist',
        'image_order': 'GLOBAL16, INITIAL2, optional PREVIOUS2, CURRENT2',
        'sequential_previous_progress_provenance': 'source_previous_progress teacher forcing',
        'output_protocol': '<think><update>...</update><state>...</state><progress>...</progress></think><answer>integer%</answer>',
        'trainable': 'full', 'precision': 'bf16', 'distributed': 'FSDP full_shard auto_wrap',
        'activation_checkpointing': True, 'optimizer': 'PyTorch fused AdamW',
        'learning_rate': 1e-5, 'betas': [0.9, 0.95], 'weight_decay': 0.1,
        'max_grad_norm': 1.0, 'lr_scheduler': 'constant',
        'per_device_batch_size': 1, 'gradient_accumulation_steps': a.gradient_accumulation_steps,
        'epochs': a.epochs, 'save_steps': a.save_steps,
        'early_checkpoint_step': 10,
    }


def main():
    a = parse_args()
    root = Path(__file__).resolve().parents[1]
    out = a.output_dir.resolve()
    from portable_paths import check_output
    check_output(out)
    if a.epochs <= 0 or a.gradient_accumulation_steps < 1 or a.save_steps < 1:
        raise ValueError('invalid epochs/accumulation/save interval')
    out.mkdir(parents=True, exist_ok=True)
    data_sha = hashlib.sha256(a.data.read_bytes()).hexdigest()
    existing = out / 'protocol_manifest.json'
    samples = load_samples(a.data, validate_images=a.preflight_only)
    if any(s.split != 'train' for s in samples):
        raise ValueError('formal SFT refuses validation/test rows')
    manifest = protocol_manifest(a, samples, data_sha)
    if existing.exists():
        if json.loads(existing.read_text()) != manifest:
            raise ValueError('existing run protocol/data/config differs; use a new output directory')
    elif a.preflight_only:
        atomic_json(existing, manifest)
    else:
        raise FileNotFoundError('run preflight on node 0 before distributed launch')
    if a.preflight_only:
        if not a.model_path.joinpath('config.json').is_file():
            raise FileNotFoundError('local model config missing')
        print(json.dumps({'status': 'preflight_pass', 'samples': len(samples),
                          'modes': manifest['modes'], 'data_sha256': data_sha,
                          'prompt_version': VERSION, 'image_paths_and_hashes_checked': True}))
        return

    import torch
    from torch.utils.data import Dataset
    from transformers import Qwen3VLForConditionalGeneration, Trainer, TrainerCallback, TrainingArguments
    from processing import V7Collator, encode_sample, load_processor

    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world < 2 or not torch.cuda.is_available():
        raise RuntimeError('distributed GPU training requires torchrun with at least two GPUs')
    processor = load_processor(str(a.model_path), 65536)

    class LazyDataset(Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self, i): return encode_sample(samples[i], processor, 8192)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(a.model_path), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa', low_cpu_mem_usage=True)
    model.config.use_cache = False
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total != trainable:
        raise RuntimeError('formal SFT requires all visual and language parameters trainable')

    class ReportCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 10:
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
                atomic_json(checkpoint / 'v7_protocol_manifest.json', manifest)

    args = TrainingArguments(
        output_dir=str(out / 'checkpoints'), num_train_epochs=a.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        learning_rate=1e-5, lr_scheduler_type='constant',
        adam_beta1=0.9, adam_beta2=0.95, weight_decay=0.1,
        max_grad_norm=1.0, optim='adamw_torch_fused', bf16=True,
        gradient_checkpointing=False,
        fsdp='full_shard auto_wrap',
        fsdp_config={'transformer_layer_cls_to_wrap': ['Qwen3VLTextDecoderLayer', 'Qwen3VLVisionBlock'],
                     'activation_checkpointing': True, 'offload_params': False,
                     'use_orig_params': True, 'limit_all_gathers': True},
        remove_unused_columns=False, dataloader_num_workers=0,
        save_strategy='steps', save_steps=a.save_steps, save_total_limit=2,
        eval_strategy='no', logging_steps=10, report_to=[],
        seed=7, data_seed=7, save_safetensors=True)
    trainer = Trainer(model=model, args=args, train_dataset=LazyDataset(),
                      data_collator=V7Collator(processor.tokenizer.pad_token_id),
                      processing_class=processor, callbacks=[ReportCallback()])
    checkpoints = sorted((out / 'checkpoints').glob('checkpoint-*'),
                         key=lambda p: int(p.name.split('-')[-1]))
    resume = str(checkpoints[-1]) if checkpoints and a.resume == 'auto' else None
    if checkpoints and a.resume == 'never':
        raise ValueError('checkpoints exist but --resume=never was requested')
    started = time.monotonic()
    result = trainer.train(resume_from_checkpoint=resume)
    if trainer.is_world_process_zero():
        atomic_json(out / 'train_summary.json', {
            'status': 'train_complete', 'optimizer_steps': result.global_step,
            'train_loss': result.training_loss, 'train_runtime_seconds': time.monotonic() - started,
            'world_size': world, 'total_parameters': total, 'trainable_parameters': trainable,
            'effective_batch_size': world * a.gradient_accumulation_steps,
            'resumed_from': resume, 'checkpoints_dir': str(out / 'checkpoints'),
        })
    # Trainer intentionally skips save_model() for a sharded FSDP state dict.
    # Switch only for the final export; periodic checkpoints stay sharded and resumable.
    plugin = trainer.accelerator.state.fsdp_plugin
    plugin.state_dict_config = None
    plugin.optim_state_dict_config = None
    plugin.set_state_dict_type('FULL_STATE_DICT')
    trainer.save_model(str(out / 'final_model'))
    if trainer.is_world_process_zero():
        processor.save_pretrained(out / 'final_model')
        atomic_json(out / 'final_model' / 'v7_protocol_manifest.json', manifest)
        print(json.dumps({'status': 'completed', 'global_step': result.global_step,
                          'final_model': str(out / 'final_model')}))

    # Keep every rank alive until rank 0 finishes exporting the final model.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
