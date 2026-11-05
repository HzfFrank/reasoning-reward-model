#!/usr/bin/env python3
"""Final accounting and integrity audit for the event-aligned local-only snapshot."""
import collections,csv,hashlib,json,os,pathlib,re,sys
ROOT=pathlib.Path('/project_c20250301/251503016/reasoning')
BASE=ROOT/'stage7_artifacts/v7_spatial_failure_production_20261025_r1/balanced_failure_20261026_r1'
OUT=pathlib.Path(os.environ.get('FAILURE_EVENT_ALIGNED_FINAL',str(BASE/'failure_event_aligned_sampling_v1/revision_r3')))
R2=pathlib.Path(os.environ.get('FAILURE_EVENT_ALIGNED_PRIMARY',str(BASE/'failure_event_aligned_sampling_v1/revision_r2')))
OLD=BASE/'short_interval_corrected_20261027_r2'
KINDS=('miss','drop','wrong_place','d1','d2','wrong_object')
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def putl(p,x):p.write_text(''.join(json.dumps(z,ensure_ascii=False)+'\n' for z in x))
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
train=rows(OUT/'train_candidates.jsonl');primary=rows(R2/'train_candidates.jsonl');secondary=rows(OUT/'secondary_selection_audit.jsonl')
old=rows(OLD/'train_candidates.jsonl');audit=rows(R2/'selection_audit.jsonl');quarantine=rows(OUT/'quarantined_windows.jsonl');excluded=rows(OUT/'excluded_windows.jsonl')
assert len(train)==176 and len(primary)==122 and len(old)==537
trainkeys={(x['trajectory_dir'],x['previous_frame'],x['current_frame']) for x in train}
assert len(trainkeys)==len(train)
# A newly admitted second-phase pair cannot remain excluded under its former one-phase decision.
restored=[x for x in rows(R2/'excluded_windows.jsonl') if (x['trajectory_dir'],*x['original_pair']) in trainkeys]
excluded=[x for x in excluded if (x['trajectory_dir'],*x['original_pair']) not in trainkeys]
putl(OUT/'excluded_windows.jsonl',excluded)
dump(OUT/'restored_old_pairs_by_second_phase.json',restored)
combined=[dict(x,phase='primary_event_window') for x in audit]+secondary
putl(OUT/'selection_audit.jsonl',combined)
assert len(combined)==len(audit)+len(secondary)
oldkeys={(x['trajectory_dir'],x['previous_frame'],x['current_frame']) for x in old}
assert len(oldkeys)==len(old)
accounted={((x['trajectory_dir'],*x['original_pair'])) for x in excluded}
accounted|={((x['trajectory_dir'],*x['original_pair'])) for x in quarantine if 'old_sample_id' in x}
accounted|=(trainkeys&oldkeys)
assert accounted==oldkeys,(len(accounted),len(oldkeys))
hold=json.loads((OUT/'holdout_trajectories.json').read_text());groups={x['group_id'] for x in json.loads((OUT/'excluded_training_groups.json').read_text())}
assert len(hold)==13 and not (OUT/'holdout_windows.jsonl').exists()
assert json.loads((BASE/'API_DISABLED.json').read_text())['api_submission_enabled'] is False
assert not any(x['group_id'] in groups for x in train)
scorefiles={};configs=set();hash_images=0
for x in train:
 assert x['split']=='train' and x['mode']=='sequential' and x['assistant_completion'] is None
 assert x['review_status']=='machine_checked_pending_independent_visual_review'
 assert x['previous_progress_provenance']==x['current_progress_provenance']=='geometry_v2'
 assert 2<=x['current_frame']-x['previous_frame']<=8
 assert len(x['images'])==22 and [z['role'] for z in x['images']]==['global']*16+['initial']*2+['previous']*2+['current']*2
 assert [z['view'] for z in x['images']]==['main','wrist']*11
 assert len(x['student_messages'])==2 and [z['role'] for z in x['student_messages']]==['system','user']
 text=' '.join(z['text'] for m in x['student_messages'] for z in m['content'] if z['type']=='text').lower()
 for forbidden in ('failure_type','geometry_v2','event_label','teacher json','simulator'):
  assert forbidden not in text
 assert not re.search(r'current task progress is estimated at\s+\d+%',text)
 sf=x['per_frame_progress_file']
 if sf not in scorefiles:
  assert sha(sf)==x['per_frame_progress_sha256'];scorefiles[sf]={r['frame']:r for r in rows(pathlib.Path(sf))}
 if x['progress_config'] not in configs:
  assert sha(x['progress_config'])==x['progress_config_sha256'];configs.add(x['progress_config'])
 for role in ('previous','current'):
  y=scorefiles[sf][x[role+'_frame']]
  assert y['P']==x[role+'_progress_float'] and int(y['P']+.5)==x[role+'_progress']
  assert y['success_current']==x[role+'_success_current']
 for z in x['images'][-4:]:assert sha(z['path'])==z['sha256'];hash_images+=1
 if x['failure_type'] in ('drop','d1','d2'):assert x['current_progress']<x['previous_progress']
 else:assert x['current_progress']<=x['previous_progress']
# Two selected windows of the same trajectory must not overlap, and the latter must add a distinct stage.
bytraj=collections.defaultdict(list)
for x in train:bytraj[x['trajectory_dir']].append(x)
for xs in bytraj.values():
 assert len(xs)<=2
 if len(xs)==2:
  xs.sort(key=lambda z:z['previous_frame'])
  assert xs[1]['previous_frame']>=xs[0]['current_frame']+2

mainby=collections.Counter(x['failure_type'] for x in train)
primaryby=collections.Counter(x['failure_type'] for x in primary)
secondaryby=collections.Counter(x['failure_type'] for x in train if 'primary_window_sample_id' in x)
with (OUT/'counts_by_type.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['type','old_windows','event_clear_trajectories','physically_selected_decrease','physically_selected_equal','physically_selected_increase','primary_kept','second_distinct_kept','final_main','score_direction_conflict_quarantine','insensitive_label_quarantine','no_event_old_windows','excluded_old_windows'])
 for kind in KINDS:
  chosen=[x for x in audit if x['failure_type']==kind and x['selected_pair'] is not None]
  q=[x for x in quarantine if x['failure_type']==kind]
  w.writerow([kind,sum(x['failure_type']==kind for x in old),len(chosen),sum(x['score']['direction']=='decrease' for x in chosen),sum(x['score']['direction']=='equal' for x in chosen),sum(x['score']['direction']=='increase' for x in chosen),primaryby[kind],secondaryby[kind],mainby[kind],sum(x['reason']=='event_and_frozen_label_direction_conflict' for x in q),sum(x['reason']=='frozen_integer_label_insensitive_to_regression' for x in q),sum('old_sample_id' in x for x in q),sum(x['failure_type']==kind for x in excluded)])
# Historical six CoT strings are never read. Only their source identity/frame records are mapped.
review=ROOT/'stage7_artifacts/v7_short_failure_six_review_20261027_r1/preflight'
by_primary={x['trajectory_dir']:x for x in audit}
six=[]
for kind in KINDS:
 p=json.loads((review/f'{kind}.json').read_text());src=p['source_candidate'];x=by_primary[src['trajectory_dir']]
 matches=[z for z in train if z['trajectory_dir']==src['trajectory_dir']]
 six.append(dict(type=kind,old_cot_sample_id=p['sample_id'],old_cot_pair=[src['previous_frame'],src['current_frame']],event_first_primary_pair=x['selected_pair'],primary_decision=x['decision'],new_main_pairs=[[z['previous_frame'],z['current_frame']] for z in matches],old_cot_text_used_for_selection=False))
dump(OUT/'six_old_cot_reconciliation.json',six)
summary=dict(old_windows=len(old),old_trajectories=len({x['trajectory_dir'] for x in old}),primary_kept=len(primary),second_distinct_kept=len(train)-len(primary),main_kept=len(train),quarantine_records=len(quarantine),excluded_old_windows=len(excluded),old_rows_accounted=len(accounted),old_pairs_recovered_by_second_phase=len(restored),main_by_type=dict(mainby),second_by_type=dict(secondaryby),frozen_progress_files_hashed=len(scorefiles),frozen_config_files_hashed=len(configs),selected_endpoint_images_hashed=hash_images,holdout_trajectories=len(hold),holdout_groups=len(groups),validation_candidates=0,persistent_candidates=0,api_requests=0,rollouts=0,cot_generated=0,training=0,visual_review_status='machine_checked_pending_independent_visual_review')
dump(OUT/'audit_validation.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
