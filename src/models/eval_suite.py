"""Evaluate the initial/final checkpoints and every requested WiSE-FT alpha."""

import copy
import json
from pathlib import Path

from src import tracking
from src.models.interpolation import merge_weights


DEFAULT_EVAL_DATASETS = ['ImageNet', 'ImageNetV2', 'ImageNetR', 'ImageNetA', 'ImageNetSketch']
DEFAULT_ALPHAS = [i / 10 for i in range(11)]


def evaluate_all_settings(image_classifier, args, reference_checkpoint=None, reference_state=None):
    # Imports stay local so the interpolation/report layer can be tested without CLIP.
    from src.models.eval import evaluate
    from src.models.utils import fisher_load

    if reference_state is None:
        if reference_checkpoint is None:
            raise ValueError('Final evaluation requires an initial reference checkpoint')
        from src.models.modeling import ImageClassifier
        reference = ImageClassifier.load(reference_checkpoint)
        reference_state = {k: v.detach().cpu().clone() for k, v in reference.state_dict().items()}
        del reference
    else:
        reference_state = {k: v.detach().cpu().clone() for k, v in reference_state.items()}
    final_state = {k: v.detach().cpu().clone() for k, v in image_classifier.state_dict().items()}
    alphas = getattr(args, 'alpha', DEFAULT_ALPHAS)
    if isinstance(alphas, (int, float)):
        alphas = [alphas]
    alphas = sorted(set([0.0, 1.0, *alphas]))
    if any(not 0 <= alpha <= 1 for alpha in alphas):
        raise ValueError('All evaluation alphas must be in [0, 1]')
    # Validate compatibility before the first (potentially long) dataset pass.
    merge_weights(0, reference_state, final_state)
    eval_args = copy.deepcopy(args)
    eval_args.eval_datasets = list(getattr(args, 'eval_datasets', None) or DEFAULT_EVAL_DATASETS)
    fishers = None
    if getattr(args, 'fisher', None) is not None:
        if len(args.fisher) != 2:
            raise ValueError('Provide exactly two Fisher files')
        fishers = [fisher_load(str(Path(path).expanduser()), device='cpu') for path in args.fisher]
    root = Path(args.save or '.').expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / 'evaluation_summary.json'
    settings = []
    report = dict(datasets=eval_args.eval_datasets, reference_checkpoint=reference_checkpoint,
                  settings=settings, complete=False)
    was_training = image_classifier.training
    try:
        for alpha in alphas:
            eval_args.alpha = alpha
            eval_args.eval_phase = 'final'
            image_classifier.load_state_dict(merge_weights(
                alpha, reference_state, final_state, fishers, getattr(args, 'fisher_floor', 1e-8)))
            if getattr(args, 'save_interpolated', False):
                image_classifier.save(str(root / f'wise_ft_alpha={alpha:.3f}.pt'))
            print(f'Final evaluation: alpha={alpha:.2f}', flush=True)
            info = evaluate(image_classifier, eval_args)
            metrics = {key: value for key, value in info.items() if ':' in key}
            settings.append(dict(alpha=alpha, metrics=metrics))
            temporary = report_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(report, indent=2) + '\n')
            temporary.replace(report_path)
            tracking.log({'wise_ft/alpha': alpha,
                          **{'wise_ft/' + key.replace(':', '/'): value for key, value in metrics.items()}})
        report['complete'] = True
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        tracking.log_final_results(settings, report_path)
    finally:
        image_classifier.load_state_dict(final_state)
        image_classifier.train(was_training)
    print('Final evaluation saved to', report_path, flush=True)
    return report
