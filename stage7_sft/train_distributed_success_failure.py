#!/usr/bin/env python3
"""One-epoch, full-parameter, ordered mixed v7 SFT using the proven Trainer/FSDP stack."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import VERSION, TEMPLATE_SHA256
from protocol import load_samples
from train_distributed import atomic_json


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment-dir', type=Path, required=True)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--resume', choices=['auto', 'never'], default='auto')
    p.add_argument('--world-size', type=int, default=8)
    p.add_argument('--max-optimizer-steps', type=int, default=919)
    p.add_argument('--stop-after-step', type=int,
                   help='Save and exit at this step; resume keeps the original total budget.')
    return p.parse_args()


def main():
    a = arguments()
    if a.world_size not in (2, 4, 8):
        raise ValueError('world size must be 2, 4 or 8; effective batch remains 16')
    if not 1 <= a.max_optimizer_steps <= 919:
        raise ValueError('max optimizer steps must be in [1, 919]')
    if a.stop_after_step is not None and not 1 <= a.stop_after_step < a.max_optimizer_steps:
        raise ValueError('stop-after-step must precede the total optimizer budget')
    accumulation = 16 // a.world_size
    root = Path(__file__).resolve().parents[1]
    run = a.experiment_dir.resolve()
    from portable_paths import check_output
    check_output(run)
    plan = json.loads((run / 'data/mix_manifest.json').read_text())
    data = run / 'data/mixed_train.jsonl'
    if sha(data) != plan['mixed_train_sha256'] or sha(run / 'data/exposure_plan.jsonl') != plan['exposure_plan_sha256']:
        raise ValueError('frozen mixed data or exposure plan changed')
    if not Path(plan['model_path']).is_absolute():
        raise ValueError('mix manifest requires an absolute model path')
    if plan['prompt_version'] != VERSION or plan['template_sha256'] != TEMPLATE_SHA256:
        raise ValueError('student prompt changed')
    if plan['physical_plan_length'] != 14704 or plan['optimizer_steps'] != 919 or plan['effective_batch_size'] != 16:
        raise ValueError('unexpected frozen plan shape')
    samples = load_samples(data, validate_images=False)
    if len(samples) != plan['physical_plan_length'] or any(s.split != 'train' for s in samples):
        raise ValueError('mixed train sample count or split mismatch')
    if sum(s.mode == 'sequential' for s in samples) + sum(s.mode == 'from_zero' for s in samples) != len(samples):
        raise ValueError('unexpected sample mode')
    checkpoint_steps = sorted(x['step'] for x in plan['checkpoints'].values())
    if checkpoint_steps != [460, 919]:
        raise ValueError('checkpoint schedule differs from frozen plan')
    checkpoint_steps = sorted(set([s for s in checkpoint_steps if s <= a.max_optimizer_steps] + [a.max_optimizer_steps]))
    manifest = {
        'kind': 'v7_full_parameter_mixed_success_failure_sft', 'model_path': plan['model_path'],
        'data_path': str(data), 'data_sha256': plan['mixed_train_sha256'],
        'exposure_plan_path': str(run / 'data/exposure_plan.jsonl'),
        'exposure_plan_sha256': plan['exposure_plan_sha256'],
        'prompt_version': VERSION, 'template_sha256': TEMPLATE_SHA256,
        'prompt_file_sha256': sha(Path(__file__).with_name('prompts.py')),
        'split': 'train', 'sample_count': len(samples), 'modes': dict(Counter(s.mode for s in samples)),
        'logical_success_exposures': plan['success_exposures'],
        'logical_failure_exposures': plan['failure_exposures'],
        'physical_tail_padding': plan['failure_padding_exposures'],
        'image_pixels': 65536, 'max_seq_length': 8192,
        'global_sampling': 'round(i*(N-1)/7), i=0..7, paired main/wrist',
        'image_order': 'GLOBAL16, INITIAL2, optional PREVIOUS2, CURRENT2',
        'output_protocol': '<think><update>...</update><state>...</state><progress>...</progress></think><answer>integer%</answer>',
        'trainable': 'full', 'precision': 'bf16', 'distributed': 'FSDP full_shard auto_wrap',
        'activation_checkpointing': True, 'optimizer': 'PyTorch fused AdamW',
        'learning_rate': 1e-5, 'betas': [0.9, 0.95], 'weight_decay': 0.1,
        'max_grad_norm': 1.0, 'lr_scheduler': 'constant', 'warmup_steps': 0,
        'per_device_batch_size': 1, 'gradient_accumulation_steps': accumulation,
        'world_size': a.world_size, 'effective_batch_size': 16, 'epochs': 1.0,
        'optimizer_steps': a.max_optimizer_steps, 'checkpoint_steps': checkpoint_steps,
        'sampler': 'SequentialSampler over one frozen global mixed plan; Accelerate BatchSamplerShard',
        'resume': 'Trainer FSDP model, optimizer, scheduler, per-rank RNG and global-step data skip',
        'save_total_limit': 2,
    }
    target = run / 'protocol_manifest.json'
    if target.exists():
        if json.loads(target.read_text()) != manifest:
            raise ValueError('existing mixed-run manifest differs; refuse to resume')
    elif a.preflight_only:
        atomic_json(target, manifest)
    else:
        raise FileNotFoundError('node 0 must complete preflight before launch')
    if a.preflight_only:
        if not (Path(plan['model_path']) / 'config.json').is_file():
            raise FileNotFoundError('local base model config missing')
        print(json.dumps({'status': 'preflight_pass', 'samples': len(samples),
                          'steps': a.max_optimizer_steps, 'checkpoint_steps': checkpoint_steps,
                          'data_sha256': plan['mixed_train_sha256']}))
        return
    import torch
    from torch.utils.data import Dataset, SequentialSampler
    from transformers import Qwen3VLForConditionalGeneration, Trainer, TrainerCallback, TrainingArguments
    from processing import V7Collator, encode_sample, load_processor

    world = int(os.environ.get('WORLD_SIZE', '0'))
    if world != a.world_size or not torch.cuda.is_available():
        raise RuntimeError(f'expected {a.world_size} torchrun GPU ranks with CUDA, got {world}')
    processor = load_processor(plan['model_path'], 65536)

    class LazyDataset(Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self, i): return encode_sample(samples[i], processor, 8192)

    class OrderedTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        plan['model_path'], local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa', low_cpu_mem_usage=True)
    model.config.use_cache = False
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != total:
        raise RuntimeError('formal mixed SFT requires every parameter trainable')

    class CheckpointCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in checkpoint_steps or state.global_step == a.stop_after_step:
                control.should_save = True
            if state.global_step == a.stop_after_step:
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
                atomic_json(checkpoint / 'v7_protocol_manifest.json', manifest)

    args = TrainingArguments(
        output_dir=str(run / 'checkpoints'), num_train_epochs=1.0,
        max_steps=a.max_optimizer_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=accumulation,
        learning_rate=1e-5, lr_scheduler_type='constant', warmup_steps=0,
        adam_beta1=0.9, adam_beta2=0.95, weight_decay=0.1,
        max_grad_norm=1.0, optim='adamw_torch_fused', bf16=True,
        gradient_checkpointing=False, fsdp='full_shard auto_wrap',
        fsdp_config={'transformer_layer_cls_to_wrap': ['Qwen3VLTextDecoderLayer', 'Qwen3VLVisionBlock'],
                     'activation_checkpointing': True, 'offload_params': False,
                     'use_orig_params': True, 'limit_all_gathers': True},
        remove_unused_columns=False, dataloader_num_workers=0,
        save_strategy='steps', save_steps=920, save_total_limit=2,
        eval_strategy='no', logging_steps=min(10, a.max_optimizer_steps), report_to=[],
        seed=7, data_seed=7, save_safetensors=True, ignore_data_skip=False)
    trainer = OrderedTrainer(model=model, args=args, train_dataset=LazyDataset(),
        data_collator=V7Collator(processor.tokenizer.pad_token_id),
        processing_class=processor, callbacks=[CheckpointCallback()])
    checkpoints = sorted((run / 'checkpoints').glob('checkpoint-*'),
                         key=lambda p: int(p.name.split('-')[-1]))
    if checkpoints and a.resume == 'never':
        raise ValueError('checkpoints exist but --resume never was selected')
    resume = None
    if checkpoints and a.resume == 'auto':
        c = checkpoints[-1]
        needed = [c / 'trainer_state.json', c / 'scheduler.pt', c / 'v7_protocol_manifest.json']
        if (c / 'pytorch_model_fsdp_0/.metadata').is_file():
            needed += [c / 'optimizer_0/.metadata']
            checkpoint_type = 'SHARDED_STATE_DICT'
        else:
            index = c / 'model.safetensors.index.json'
            needed += [index, c / 'pytorch_model_fsdp.bin', c / 'optimizer.bin']
            if index.is_file():
                shards = set(json.loads(index.read_text())['weight_map'].values())
                if not shards:
                    raise ValueError('empty checkpoint weight index')
                needed += [c / name for name in shards]
            checkpoint_type = 'FULL_STATE_DICT'
        plugin = trainer.accelerator.state.fsdp_plugin
        plugin.state_dict_config = None
        plugin.optim_state_dict_config = None
        plugin.set_state_dict_type(checkpoint_type)
        needed += [c / f'rng_state_{i}.pth' for i in range(world)]
        if any(not p.is_file() for p in needed):
            raise ValueError(f'incomplete checkpoint; refuse model-only resume: {c}')
        if json.loads((c / 'v7_protocol_manifest.json').read_text()) != manifest:
            raise ValueError('checkpoint protocol differs from frozen run')
        completed = json.loads((c / 'trainer_state.json').read_text())['global_step']
        if a.stop_after_step is not None and completed >= a.stop_after_step:
            raise ValueError('stop-after-step must be later than the resumed checkpoint')
        resume = str(c)
    started = time.monotonic()
    result = trainer.train(resume_from_checkpoint=resume)
    expected_step = a.stop_after_step or a.max_optimizer_steps
    if result.global_step != expected_step:
        raise RuntimeError(f'unexpected final optimizer step {result.global_step}')
    if trainer.is_world_process_zero():
        atomic_json(run / 'train_summary.json', {
            'status': 'paused' if a.stop_after_step else 'train_complete', 'optimizer_steps': result.global_step,
            'train_loss': result.training_loss, 'train_runtime_seconds': time.monotonic() - started,
            'world_size': world, 'total_parameters': total, 'trainable_parameters': trainable,
            'effective_batch_size': 16, 'resumed_from': resume,
            'checkpoint_steps': checkpoint_steps})
    if a.stop_after_step:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        return
    plugin = trainer.accelerator.state.fsdp_plugin
    plugin.state_dict_config = None
    plugin.optim_state_dict_config = None
    plugin.set_state_dict_type('FULL_STATE_DICT')
    trainer.save_model(str(run / 'final_model'))
    if trainer.is_world_process_zero():
        processor.save_pretrained(run / 'final_model')
        atomic_json(run / 'final_model/v7_protocol_manifest.json', manifest)
        print(json.dumps({'status': 'completed', 'global_step': result.global_step,
                          'final_model': str(run / 'final_model')}))

    # Keep every rank alive until rank 0 finishes exporting the final model.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == '__main__': main()
