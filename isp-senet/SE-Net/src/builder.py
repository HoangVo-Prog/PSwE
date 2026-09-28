import sys

sys.path.append('../common')

from common.dataset import process_data
from common.air_data import load_air_scanpaths, normalize_air_scanpaths
from .models import UserEmbeddingNet
from common.utils import adjust_subjects
import json
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader


def build(hparams, dataset_root, device, is_eval=False, split=1):
    dataset_name = hparams.Data.name
    configured_subject_num = hparams.Data.num_subjects

    bbox_annos = np.load(
        join(dataset_root, 'bbox_annos.npy'),
        allow_pickle=True).item() if dataset_name == 'COCO-Search18' else {}

    if dataset_name in ('Air-D', 'AiR', 'Air'):
        human_scanpaths = load_air_scanpaths(
            dataset_root,
            fix_path=hparams.Data.fix_path,
            fixation_file_pattern=getattr(hparams.Data, 'fixation_file_pattern', 'AiR_fixations_{}.json'),
        )
        human_scanpaths = normalize_air_scanpaths(
            human_scanpaths,
            target_width=hparams.Data.im_w,
            target_height=hparams.Data.im_h,
        )
    else:
        with open(join(dataset_root, hparams.Data.fix_path), 'r') as json_file:
            human_scanpaths = json.load(json_file)
    if dataset_name == 'COCO-Search18':
        n_tasks = 18
    elif dataset_name in ('Air-D', 'AiR', 'Air'):
        n_tasks = len({str(record['task']) for record in human_scanpaths})
    else:
        n_tasks = 1

    if dataset_name in ('Air-D', 'AiR', 'Air') and hparams.Data.subject[0] != -1 and hparams.Data.fewshot_subject[0] != -1:
        raise ValueError('Choose subject exclusion or few-shot support, not both')

    # hparams.Data.subject indicating which subjects are unseen subjects
    if hparams.Data.subject[0] != -1:
        print(f"skip subject {hparams.Data.subject} data!")
        human_scanpaths = adjust_subjects(human_scanpaths, hparams.Data.subject)

    if dataset_name in ('Air-D', 'AiR', 'Air'):
        subjects = sorted({record['subject'] for record in human_scanpaths})
        if subjects != list(range(len(subjects))):
            raise ValueError('Air-D subject_idx rows must be contiguous from zero before exporting a table')
        hparams.Data.subject_ids = [next(record['subject_idx'] for record in human_scanpaths
                                        if record['subject'] == row) for row in subjects]
        hparams.Data.num_subjects = len(subjects)
        if hparams.Data.fewshot_subject[0] != -1:
            hparams.Data.subject_ids = list(hparams.Data.fewshot_subject)
            if len(set(hparams.Data.subject_ids)) != len(hparams.Data.subject_ids) or not set(hparams.Data.subject_ids).issubset(subjects):
                raise ValueError('Few-shot subject IDs must be unique and present in the Air-D records')
            hparams.Data.num_subjects = configured_subject_num

    # Filtering training data
    if hparams.Data.TAP == 'TP':
        human_scanpaths = list(
            filter(lambda x: x['condition'] == 'present', human_scanpaths))
        human_scanpaths = list(
            filter(lambda x: x['fixOnTarget'], human_scanpaths))
    elif hparams.Data.TAP == 'FV':
        human_scanpaths = list(
            filter(lambda x: x['condition'] == 'freeview', human_scanpaths))
        n_tasks = 1

    # process fixation data
    dataset = process_data(
        human_scanpaths,
        dataset_root,
        bbox_annos,
        hparams,
        device)

    if dataset_name in ('Air-D', 'AiR', 'Air'):
        dataset['gaze_train'].evaluation_only = is_eval
        dataset['gaze_valid'].evaluation_only = True

    batch_size = hparams.Train.batch_size
    n_workers = hparams.Train.n_workers

    tag = False if is_eval else True
    bs = batch_size // 2 if is_eval else batch_size
    train_HG_loader = DataLoader(dataset['gaze_train'],
                                 batch_size=bs,
                                 shuffle=tag,
                                 num_workers=n_workers,
                                 drop_last=not (is_eval and dataset_name in ('Air-D', 'AiR', 'Air')),
                                 pin_memory=True)
    print('num of training batches =', len(train_HG_loader))

    
    valid_HG_loader = DataLoader(dataset['gaze_valid'],
                                    batch_size=batch_size//2,
                                    shuffle=False,
                                    num_workers=n_workers,
                                    drop_last=False,
                                    pin_memory=True)


    # Create model
    emb_size = hparams.Model.embedding_dim
    n_heads = hparams.Model.n_heads
    hidden_size = hparams.Model.hidden_dim


    model = UserEmbeddingNet(
            hparams.Data,
            num_decoder_layers=hparams.Model.n_dec_layers,
            hidden_dim=emb_size,
            nhead=n_heads,
            ntask=n_tasks,
            num_output_layers=hparams.Model.num_output_layers,
            train_encoder=hparams.Train.train_backbone,
            train_pixel_decoder=hparams.Train.train_pixel_decoder,
            dropout=hparams.Train.dropout,
            dim_feedforward=hidden_size,
            num_encoder_layers=hparams.Model.n_enc_layers)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=hparams.Train.adam_lr,
                                  betas=hparams.Train.adam_betas)

    # Load weights from checkpoint when available
    if len(hparams.Model.checkpoint) > 0:
        print(f"loading weights from {hparams.Model.checkpoint} in {hparams.Train.transfer_learn} setting.")
        ckp = torch.load(join(hparams.Train.log_dir, hparams.Model.checkpoint), map_location=device)

        model.load_state_dict(ckp['model'], strict=False)
        # optimizer.load_state_dict(ckp['optimizer'])
        global_step = ckp['step']
    else:
        global_step = 0

    if hparams.Train.parallel:
        model = torch.nn.DataParallel(model)

    bbox_annos = dataset['bbox_annos']
    human_cdf = dataset['human_cdf']


    is_lasts = [x[5] for x in dataset['gaze_train'].fix_labels]
    term_pos_weight = len(is_lasts) / np.sum(is_lasts) - 1
    print("termination pos weight: {:.3f}".format(term_pos_weight))

    return (model, optimizer, train_HG_loader, valid_HG_loader, term_pos_weight, global_step)
