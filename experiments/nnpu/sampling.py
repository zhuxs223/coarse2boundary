"""Visit every tile once; ensure each PU batch has a coarse positive reference."""
import math
import torch


class PositiveBatchSampler:
    def __init__(self, keys, priors, batch_size, seed):
        if batch_size<1:
            raise ValueError('Invalid batch size')
        self.keys, self.priors = list(keys), priors
        self.batch_size, self.seed = batch_size, seed

    def __len__(self):
        return math.ceil(len(self.keys)/self.batch_size)

    def __iter__(self):
        order = torch.randperm(len(self.keys),generator=torch.Generator().manual_seed(self.seed)).tolist()
        batches = [order[i:i+self.batch_size] for i in range(0,len(order),self.batch_size)]
        has_p = lambda i: self.priors[self.keys[i]]['positive']>0
        needs_p = lambda i: self.priors[self.keys[i]]['unlabeled']>0 and self.priors[self.keys[i]]['pi']>0
        missing = [b for b in batches if any(map(needs_p,b)) and not any(map(has_p,b))]
        donors = []
        for b in batches:
            positives = [j for j,i in enumerate(b) if has_p(i)]
            donors.extend((b,j) for j in positives[1:])
        if len(missing)>len(donors):
            raise RuntimeError('Too few positive tiles to form PU batches without repeating data; increase batch size')
        for b,(donor,j) in zip(missing,donors):
            b[0],donor[j] = donor[j],b[0]
        yield from batches
