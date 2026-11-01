import copy
import os

from src import tracking
from src.args import parse_arguments
from src.models.eval_suite import evaluate_all_settings
from src.models.finetune import finetune
from src.models.interpolation import merge_weights as _merge
from src.models.modeling import ImageEncoder, ImageClassifier
from src.models.zeroshot import get_zeroshot_classifier


def wise_ft(args):
    if args.save is None:
        raise ValueError('Please provide --save for checkpoints and evaluation results')
    os.makedirs(args.save, exist_ok=True)
    with tracking.experiment_run(args, job_type='wise-ft'):
        if args.load is None:
            image_encoder = ImageEncoder(args, keep_lang=True)
            classification_head = get_zeroshot_classifier(args, image_encoder.model)
            delattr(image_encoder.model, 'transformer')
            classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
            zeroshot_checkpoint = os.path.join(args.save, 'zeroshot.pt')
            classifier.save(zeroshot_checkpoint)
            del classifier, image_encoder, classification_head

            train_args = copy.deepcopy(args)
            train_args.load = zeroshot_checkpoint
            train_args.save = os.path.join(args.save, 'finetuned')
            # The outer run owns the final suite; do not evaluate it twice.
            finetuned_checkpoint = finetune(train_args, evaluate_at_end=False)
        else:
            if isinstance(args.load, str) or len(args.load) != 2:
                raise ValueError('--load requires reference.pt,finetuned.pt for WiSE-FT')
            zeroshot_checkpoint, finetuned_checkpoint = args.load

        finetuned = ImageClassifier.load(finetuned_checkpoint)
        return evaluate_all_settings(finetuned, args, reference_checkpoint=zeroshot_checkpoint)


if __name__ == '__main__':
    wise_ft(parse_arguments())
