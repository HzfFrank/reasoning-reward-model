"""Local Qwen3-VL model and smoke checkpoint utilities."""
from __future__ import annotations
import json, hashlib, random
from pathlib import Path
import torch
from transformers import Qwen3VLForConditionalGeneration
from prompts import VERSION as PROMPT_VERSION, TEMPLATE_SHA256


def load_model(model_path: str, device: str = 'cuda:0', checkpoint: str | None = None):
    source = str(Path(checkpoint)/'model') if checkpoint and (Path(checkpoint)/'model').is_dir() else model_path
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        source, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa', low_cpu_mem_usage=True)
    model.to(device)
    if checkpoint:
        ckpt = Path(checkpoint)
        metadata = json.loads((ckpt/'metadata.json').read_text())
        if Path(metadata['base_model']).resolve() != Path(model_path).resolve():
            raise ValueError('checkpoint base model differs from requested model')
        if metadata.get('kind') == 'smoke_trainable_parameters_only':
            state = torch.load(ckpt/'trainable_state.pt', map_location='cpu', weights_only=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if unexpected: raise ValueError('smoke checkpoint has unexpected keys')
    return model


def select_trainable(model, mode: str):
    for p in model.parameters(): p.requires_grad_(False)
    if mode == 'last_layer':
        target = model.model.language_model.layers[-1]
        for p in target.parameters(): p.requires_grad_(True)
    elif mode == 'full':
        for p in model.parameters(): p.requires_grad_(True)
    else: raise ValueError('trainable mode must be last_layer or full')
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_smoke_checkpoint(model, model_path: str, output_dir: Path, step: int, trainable_mode: str,
                          processor=None, image_pixels: int = 65536, max_seq_length: int = 8192):
    checkpoint = output_dir/'smoke_checkpoint'
    checkpoint.mkdir(parents=True, exist_ok=False)
    state = {name: p.detach().cpu().contiguous() for name,p in model.named_parameters() if p.requires_grad}
    torch.save(state, checkpoint/'trainable_state.pt')
    if processor is not None: processor.save_pretrained(checkpoint/'processor')
    (checkpoint/'metadata.json').write_text(json.dumps({
        'kind':'smoke_trainable_parameters_only', 'base_model':str(Path(model_path).resolve()),
        'step':step, 'trainable_mode':trainable_mode, 'diagnostic_only':True,
        'protocol':protocol_metadata(model_path,processor,image_pixels,max_seq_length,Path(__file__).with_name('prompts.py')),
        'parameter_count':sum(x.numel() for x in state.values())},indent=2)+'\n')
    return checkpoint

def protocol_metadata(model_path: str, processor, image_pixels: int, max_seq_length: int, prompt_file: Path):
    return {'base_model':str(Path(model_path).resolve()),'prompt_version':PROMPT_VERSION,'template_sha256':TEMPLATE_SHA256,'prompt_file_sha256':hashlib.sha256(prompt_file.read_bytes()).hexdigest(),
            'image_pixels':image_pixels,'max_seq_length':max_seq_length,
            'mode_definitions':{'sequential':['GLOBAL16','INITIAL2','PREVIOUS2','CURRENT2','previous_progress'],'from_zero':['GLOBAL16','INITIAL2','CURRENT2']},
            'storyboard_sampling':'round(i*(N-1)/7), i=0..7; paired main/wrist; chronological',
            'image_order':'GLOBAL main/wrist, INITIAL main/wrist, optional PREVIOUS main/wrist, CURRENT main/wrist',
            'output_protocol':'<think><update>...</update><state>...</state><progress>...</progress></think><answer>integer%</answer>'}

def save_full_checkpoint(model, processor, optimizer, scheduler, output_dir: Path, model_path: str,
                         config: dict, global_step: int, prompt_file: Path):
    ckpt=output_dir/'full_checkpoint'; ckpt.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(ckpt/'model',safe_serialization=True)
    processor.save_pretrained(ckpt/'processor')
    torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict() if scheduler else None,'global_step':global_step,
                'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,'rng_python':random.getstate()},ckpt/'training_state.pt')
    meta={'kind':'full_sft_checkpoint','base_model':str(Path(model_path).resolve()),'global_step':global_step,'config':config,
          'protocol':protocol_metadata(model_path,processor,config['image_pixels'],config['max_seq_length'],prompt_file),
          'diagnostic_only':config.get('diagnostic_only',False),'split_manifest_hash':config.get('split_manifest_hash')}
    (ckpt/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return ckpt
