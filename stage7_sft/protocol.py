"""Input validation, message assembly, and strict XML answer parsing."""
from __future__ import annotations

import hashlib
import os
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from PIL import Image

from prompts import FROM_ZERO_USER, SEQUENTIAL_USER, SYSTEM, VERSION, TEMPLATE_SHA256

DATA_ROOT = Path(os.environ.get("V7_DATA_ROOT", str(Path(__file__).resolve().parents[2]))).expanduser().resolve()
PROJECT_ALIASES = (
    Path('/vepfs-mlp2/c20250301/251503016'),
    Path('/project_c20250301/251503016'),
)
ANSWER_RE = re.compile(r'-?\d+%\Z')
FORBIDDEN = ('teacher json', 'evidence_frames', 'uncertainty', 'v3 completion', 'plan_text', 'completion_criterion')

@lru_cache(maxsize=65536)
def _check_image(path: Path, expected_hash: str | None):
    with Image.open(path) as image:
        image.verify()
    if expected_hash is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f'image hash mismatch: {path}')


def parse_completion(text: str) -> dict:
    result = {'ok': False, 'raw': text, 'progress': None, 'out_of_range': None, 'error': None}
    try:
        root = ET.fromstring('<root>' + text.strip() + '</root>')
        if [x.tag for x in root] != ['think', 'answer'] or (root.text or '').strip():
            raise ValueError('expected think followed by answer')
        think, answer = root
        if [x.tag for x in think] != ['update', 'state', 'progress']:
            raise ValueError('expected update, state, progress')
        if any(list(x) or not (x.text or '').strip() for x in think):
            raise ValueError('empty or nested think section')
        if list(answer) or not ANSWER_RE.fullmatch((answer.text or '').strip()):
            raise ValueError('answer must be an integer percentage')
        if (think.tail or '').strip() or (answer.tail or '').strip():
            raise ValueError('trailing text')
        value = int(answer.text.strip()[:-1])
        result.update(ok=True, progress=value, out_of_range=not -100 <= value <= 100)
    except (ET.ParseError, ValueError) as exc:
        result['error'] = str(exc)
    return result


def check_progress_contract(text: str, mode: str, previous_progress: int | None) -> list[str]:
    """Check the approved numerical sentences without changing generated text."""
    parsed = parse_completion(text)
    if not parsed['ok']:
        return [parsed['error']]
    root = ET.fromstring('<root>' + text.strip() + '</root>')
    progress = root.find('./think/progress').text.strip()
    first = (f'Previous task progress was {previous_progress}%.' if mode == 'sequential'
             else 'Initial task progress was 0%.')
    last = f'Current task progress is estimated at {parsed["progress"]}%.'
    errors = []
    if not progress.startswith(first): errors.append(f'progress must start with {first}')
    if not progress.endswith(last): errors.append(f'progress must end with {last}')
    return errors


@dataclass(frozen=True)
class Sample:
    sample_id: str
    mode: str
    task: str
    split: str
    episode: str
    initial: int
    previous: int | None
    current: int
    previous_progress: int | None
    previous_progress_provenance: str | None
    images: tuple[dict, ...]
    completion: str | None = None
    source_path: str | None = None
    is_initialization: bool = False


def normalize_image_refs(images: list[dict] | tuple[dict, ...]) -> tuple[dict, ...]:
    """Resolve the two known mount aliases without changing source image identity."""
    output = []
    for image in images:
        source = Path(image['path'])
        relative = None
        for alias in PROJECT_ALIASES:
            try:
                relative = source.relative_to(alias)
                break
            except ValueError:
                pass
        if relative is not None:
            resolved = DATA_ROOT / relative
        elif source.is_absolute():
            resolved = source.resolve()
        else:
            resolved = (DATA_ROOT / source).resolve()
        if not resolved.is_relative_to(DATA_ROOT):
            raise ValueError(f'image path outside configured V7_DATA_ROOT: {source}')
        mapped = dict(image)
        mapped['path'] = str(resolved)
        output.append(mapped)
    return tuple(output)


def validate_sample(sample: Sample, verify_hash: bool = True, verify_image: bool = True) -> None:
    if sample.mode not in ('sequential', 'from_zero'):
        raise ValueError('invalid mode')
    expected_roles = (['GLOBAL'] * 16 + ['INITIAL'] * 2 +
                      (['PREVIOUS'] * 2 if sample.mode == 'sequential' else []) + ['CURRENT'] * 2)
    if len(sample.images) != len(expected_roles):
        raise ValueError(f'{sample.mode} needs {len(expected_roles)} images')
    global_frames = []
    for i, (im, role) in enumerate(zip(sample.images, expected_roles)):
        label = im.get('label', '')
        if not label.startswith(role + ' frame=') or f'role={role.lower()}' not in label:
            raise ValueError(f'image {i}: wrong role label')
        view = 'main' if i % 2 == 0 else 'wrist'
        if im['view'] != view or f'view={view}' not in label or f'frame={im["frame"]}' not in label:
            raise ValueError(f'image {i}: order/frame/view mismatch')
        expected_frame = (sample.initial if role == 'INITIAL' else sample.previous if role == 'PREVIOUS' else sample.current if role == 'CURRENT' else im['frame'])
        if im['frame'] != expected_frame:
            raise ValueError(f'image {i}: query frame mismatch')
        path = Path(im['path']).resolve()
        if not path.is_relative_to(DATA_ROOT) or not path.is_file():
            raise ValueError(f'image {i}: path missing or outside data disk')
        if verify_image:
            _check_image(path, im['sha256'] if verify_hash else None)
        if role == 'GLOBAL': global_frames.append(im['frame'])
    if len(set(global_frames)) != 8 or any(global_frames[i] != global_frames[i+1] for i in range(0,16,2)) or global_frames != sorted(global_frames):
        raise ValueError('GLOBAL needs 8 ordered two-view time points')
    if sample.initial != global_frames[0]:
        raise ValueError('INITIAL must match first global frame')
    if sample.mode == 'sequential':
        valid_transition = sample.initial <= sample.previous < sample.current if sample.previous is not None else False
        valid_initialization = (sample.is_initialization and sample.previous == sample.current == sample.initial)
        if sample.previous is None or sample.previous_progress is None or not (valid_transition or valid_initialization):
            raise ValueError('sequential needs previous frame and model progress')
    elif sample.previous is not None or sample.previous_progress is not None:
        raise ValueError('from_zero must omit previous frame and progress')
    if not sample.task or not sample.episode or not sample.split:
        raise ValueError('missing task/episode/split provenance')
    if sample.completion is not None and not parse_completion(sample.completion)['ok']:
        raise ValueError('invalid assistant completion')


def messages(sample: Sample) -> list[dict]:
    validate_sample(sample, verify_hash=False)
    if sample.mode == 'sequential':
        user = SEQUENTIAL_USER.format(task=sample.task, initial_frame_id=sample.initial,
            previous_frame_id=sample.previous, current_frame_id=sample.current,
            previous_progress=sample.previous_progress)
    else:
        user = FROM_ZERO_USER.format(task=sample.task, initial_frame_id=sample.initial,
            current_frame_id=sample.current)
    content = [{'type':'text','text':user}]
    for im in sample.images:
        content.extend([{'type':'text','text':im['label']}, {'type':'image','image':im['path']}])
    result = [{'role':'system','content':[{'type':'text','text':SYSTEM}]},
              {'role':'user','content':content}]
    user_text = json.dumps(result, ensure_ascii=False).lower()
    if any(term in user_text for term in FORBIDDEN):
        raise ValueError('audit data leaked into student messages')
    return result


def load_samples(path: Path, mode: str = 'all', max_samples: int | None = None,
                 validate_images: bool = True) -> list[Sample]:
    files = sorted(path.glob('samples/*.json')) if path.is_dir() else [path]
    selection_path = (path if path.is_dir() else path.parent)/'selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.is_file() else None
    output = []
    for file in files:
        rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()] if file.suffix == '.jsonl' else [json.loads(file.read_text())]
        for row in rows:
            if mode != 'all' and row['mode'] != mode: continue
            if row.get('status') != 'structural_pass': continue
            if 'images' in row and 'episode_id' in row and 'source_jsonl_line' in row:
                if row.get('prompt_version') != VERSION or row.get('template_sha256') != TEMPLATE_SHA256:
                    raise ValueError(f'{file}: student prompt version/hash missing or incompatible; explicit re-export required')
                sample = Sample(row['sample_id'], row['mode'], row['task'], row['split'], row['episode_id'],
                    row['initial_frame'], row['previous_frame'], row['current_frame'], row['previous_progress'],
                    row['previous_progress_provenance'], normalize_image_refs(row['images']), row['completion'], str(file),
                    row.get('is_initialization', False))
                validate_sample(sample, verify_hash=validate_images, verify_image=validate_images)
                if parse_completion(sample.completion)['progress'] != row['answer']:
                    raise ValueError(f'{file}: source answer differs from completion')
                output.append(sample)
                if max_samples and len(output) >= max_samples: return output
                continue
            query = row['query']; is_seq = row['mode'] == 'sequential'
            if selection is None or row['split'] != selection['split']:
                raise ValueError('sample split/episode selection manifest missing or inconsistent')
            episode = selection['episode']
            selected = selection['sources'].get(str(query['current']))
            if selected is None or selected['source_sha256'] != query['source_sha256']:
                raise ValueError('sample query not traceable to source selection')
            sample = Sample(row['sample_id'], row['mode'], row['task'], row['split'], episode,
                0, query['previous'] if is_seq else None, query['current'],
                query['previous_progress'] if is_seq else None,
                'source_previous_progress' if is_seq else None,
                normalize_image_refs(row['images']), row['completion'], str(file))
            validate_sample(sample, verify_hash=validate_images, verify_image=validate_images)
            if parse_completion(sample.completion)['progress'] != query['answer']:
                raise ValueError(f'{file}: source answer differs from completion')
            output.append(sample)
            if max_samples and len(output) >= max_samples: return output
    if not output: raise ValueError('no usable samples')
    return output
