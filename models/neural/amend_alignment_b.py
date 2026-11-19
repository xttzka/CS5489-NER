"""Create an auditable alignment-fix v2 of an existing frozen B protocol.

Only validation-derived planning is recomputed. The selected checkpoints,
tasks and model-selection evidence must match the parent exactly. No test
scores select a model or parameter. The parent manifest is never modified.
"""
import argparse
from pathlib import Path

from models.neural.run_b_evaluation import ROOT, freeze_plan, json_hash, load_frozen, read_json, sha256, write_new_json

ALLOWED = {
    'models/neural/token_alignment.py',
    'models/neural/predict_neural.py',
    'models/neural/predict_bert.py',
    'models/neural/bilstm_representations.py',
    'models/neural/export_bert_entities.py',
    'models/neural/run_b_evaluation.py',
    'models/neural/finish_b.py',
    'models/neural/amend_alignment_b.py',
    'models/neural/compare_evaluation_versions_b.py',
}


def make_amendment(parent, draft, selection, audit_path):
    original = load_frozen(parent)
    fresh = freeze_plan(ROOT, read_json(draft), selection)
    for field in ('tasks', 'policy', 'search_evidence', 'runs', 'bootstrap'):
        if original.get(field) != fresh.get(field):
            raise ValueError(f'cannot amend: previously frozen {field} changed')
    a, b = dict(original['selection']), dict(fresh['selection'])
    a.pop('evidence_sha256', None); b.pop('evidence_sha256', None)
    if a != b:
        raise ValueError('cannot amend: validation model selection changed')
    old, new = original['code_sha256'], fresh['code_sha256']
    changed = {path for path in set(old) | set(new) if old.get(path) != new.get(path)}
    if not changed or changed - ALLOWED:
        raise ValueError(f'unexpected or absent code changes relative to parent: {sorted(changed)}')
    for required in ('models/neural/token_alignment.py', 'models/neural/predict_neural.py'):
        if required not in changed or required not in new:
            raise ValueError(f'required alignment fix missing: {required}')
    audit=read_json(audit_path)
    if audit.get('frozen_manifest_sha256') != original['manifest_sha256'] or audit.get('frozen_task_count') != len(original['tasks']):
        raise ValueError('alignment audit is not bound to the parent manifest and its full task inventory')
    input_paths = {str(Path(task['input'])) for task in original['tasks']}
    audit_inputs = audit.get('inputs', [])
    if len(audit_inputs) != audit.get('unique_input_count') or {item.get('input') for item in audit_inputs} != input_paths:
        raise ValueError('alignment audit does not cover exactly all parent task inputs')
    for item in audit_inputs:
        path = (ROOT/item['input']).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha256(path) != item.get('sha256'):
            raise ValueError(f'alignment audit input changed: {path}')
    if audit.get('summary', {}).get('over_model_limit_sentences') != 0:
        raise ValueError('alignment-only amendment does not handle overlength inputs')
    fresh.pop('manifest_sha256')
    fresh['amendment'] = {
        'kind': 'inference_token_alignment_unknown_fallback_v2',
        'parent_manifest_path': str(parent.resolve()), 'parent_manifest_sha256': original['manifest_sha256'],
        'reason': 'A token consisting of U+200E was dropped by the fast tokenizer, violating word/subword alignment. Invisible tokens with no subwords are represented by the model unknown-token ID; original word positions are preserved.',
        'changed_files': {name: {'parent_sha256':old.get(name), 'new_sha256':new.get(name)} for name in sorted(changed)},
        'input_audit_path': str(audit_path.resolve()), 'input_audit_sha256': sha256(audit_path),
        'input_audit_summary': audit,
        'policy': 'All planned tasks must be rerun in a new result root; old predictions are retained and compared by ID, not used to select a version.',
        'training_history': 'All checkpoints were trained before this inference-only amendment; the current inference code was not used for their training.',
        'limitation': 'Some parent held-out predictions and metrics existed before the alignment failure was discovered. This technical correction was not selected using held-out scores.',
    }
    fresh['manifest_sha256']=json_hash(fresh)
    return fresh


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('parent','plan','selection','audit','output'):
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    try:
        if a.output.exists() or a.output.resolve()==a.parent.resolve():
            raise ValueError('new manifest output must be new and distinct from parent')
        amended=make_amendment(a.parent,a.plan,a.selection,a.audit)
        write_new_json(a.output,amended)
        print(f'Created amended frozen manifest: {a.output}; parent={amended["amendment"]["parent_manifest_sha256"]}; new={amended["manifest_sha256"]}')
    except (OSError,ValueError,KeyError,TypeError) as error:p.exit(2,f'Error: {error}\n')


if __name__=='__main__':main()
