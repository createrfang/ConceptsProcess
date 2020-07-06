import time

import torch
import numpy as np


def train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                  coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    outs = None

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        print(f'input.shape : {input.shape}')
        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output = model(input)
            output = output.cpu().detach().numpy()
            if outs is None:
                outs = output
            else:
                outs = np.concatenate((outs, output), axis=0)

            n_batches += 1

        del input
        del output

    end = time.time()

    log('\ttime: \t{0}'.format(end - start))
    # p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()

    return outs
