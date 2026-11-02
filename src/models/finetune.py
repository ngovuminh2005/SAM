import os
import copy
import time
import tqdm

import torch

import clip.clip as clip

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing
from src.models.training import build_optimizer_and_policy, train_batch
from src.models.eval_suite import evaluate_all_settings
from src import tracking
from src.randomness import seed_everything

import src.datasets as datasets


def finetune(args, evaluate_at_end=True):
    with tracking.experiment_run(args):
        seed_everything(getattr(args, 'seed', 42))
        return _finetune(args, evaluate_at_end)


def _finetune(args, evaluate_at_end):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    assert args.train_dataset is not None, "Please provide a training dataset."
    
    
    image_classifier = ImageClassifier.load(args.load)

    if args.freeze_encoder:
        print('Fine-tuning a linear classifier')
        model = image_classifier.classification_head
        input_key = 'features'
        preprocess_fn = image_classifier.val_preprocess
        image_enc = image_classifier.image_encoder
        print_every = 1000
    else:
        print('Fine-tuning end-to-end')
        model = image_classifier
        input_key = 'images'
        preprocess_fn = image_classifier.train_preprocess
        image_enc = None
        image_classifier.process_images = True
        print_every = 100
    
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=getattr(args, 'num_workers', 8),
    )
    num_batches = len(dataset.train_loader)

    model = model.to(args.device)
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    if args.device.startswith('cuda') and len(devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    optimizer, sam_policy, sam_parameter_names = build_optimizer_and_policy(model, args)
    print('Optimizer:', type(optimizer).__name__, 'SAM:', getattr(args, 'sam', 'none'),
          'perturbed tensors:', len(sam_parameter_names))

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    for epoch in range(args.epochs):
        model.train()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=image_enc)

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)

            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].to(args.device)
            labels = batch['labels'].to(args.device)
            data_time = time.time() - start_time

            loss = train_batch(model, inputs, labels, loss_fn, optimizer,
                               policy=sam_policy, grad_clip=getattr(args, 'grad_clip_norm', 1.0))
            batch_time = time.time() - start_time

            if step % getattr(args, 'log_every', 100) == 0:
                tracking.log({'train/step': step + 1,
                              'train/epoch': epoch + (i + 1) / len(data_loader),
                              'train/loss': loss.item(),
                              'train/lr': optimizer.param_groups[0]['lr'],
                              'train/rho': sam_policy.options['rho'] if sam_policy else 0.0,
                              'train/batch_seconds': batch_time})

            if i % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
        
        trained_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        if args.freeze_encoder:
            image_classifier = ImageClassifier(image_classifier.image_encoder, trained_model)
        else:
            image_classifier = trained_model

        # Saving model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch+1}.pt')
            print('Saving model to', model_path)
            image_classifier.save(model_path)
            optim_path = os.path.join(args.save, f'optim_{epoch+1}.pt')
            torch.save(optimizer.state_dict(), optim_path)
            if sam_policy is not None:
                torch.save({'parameter_names': sam_parameter_names,
                            'policy': sam_policy.state_dict()},
                           os.path.join(args.save, f'sam_{epoch+1}.pt'))

        # Evaluate
        args.current_epoch = epoch
        eval_results = evaluate(image_classifier, args)

    if evaluate_at_end:
        evaluate_all_settings(image_classifier, args, reference_checkpoint=args.load)

    if args.save is not None:
        return model_path


if __name__ == '__main__':
    args = parse_arguments()
    finetune(args)
