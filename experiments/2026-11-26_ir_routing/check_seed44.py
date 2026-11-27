"""Gate-2 follow-up: does seed 44 reproduce 0.678 when predicted per fold, as the notebook did?"""
import sys, numpy as np, torch
sys.path.insert(0, __import__('os').path.dirname(__file__))
import oof
from buck.benchmark import architectures as arch
from buck.benchmark.data import decode_images
from buck.benchmark.ensemble import _probabilities
records, ca, labels, groups, dev_idx, folds = oof.split()
dev=[records[i] for i in dev_idx]; y=labels[dev_idx]
M=oof.MODEL; size=arch.input_size(M); mean,std=arch.normalisation(M)
imgs=decode_images(dev,size); d=torch.device('cuda')
saved=np.load(oof.HERE/'outputs'/'baseline.npz')['oof'][2]
per=np.zeros_like(saved)
for k,(_,va) in enumerate(folds):
    m=arch.build_model(M,len(ca),pretrained=False)
    m.load_state_dict(torch.load(oof.ARMS['baseline'][0]/f'{M}_s44/checkpoints/{M}/fold{k+1}.pth',map_location='cpu')['model_state_dict']); m.to(d)
    per[va]=_probabilities(m,imgs[va],y[va],32,d,False,mean,std)
print('per-fold acc', (per.argmax(1)==y).mean(), ' all-at-once acc', (saved.argmax(1)==y).mean())
diff=np.flatnonzero(per.argmax(1)!=saved.argmax(1))
print('max |prob diff|', np.abs(per-saved).max(), ' flipped images', [(dev[i].filename, per[i].round(3).tolist(), saved[i].round(3).tolist()) for i in diff])
