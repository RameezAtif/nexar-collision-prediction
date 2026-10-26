"""Selection-optimism check (E10): train the notebook-05 CNN+LSTM on the FULL training set for a fixed 10 epochs (5 seeds)
and record the validation AP at every epoch, to separate (a) the cost of choosing the epoch on the evaluation videos from
(b) the cost of the 20% training data held out in notebook 06. Run from the project root; writes results/selection_check_fulldata.csv."""
import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.metrics import average_precision_score
P=Path("data/processed"); F=P/"cnn_features"
torch.set_num_threads(4)
class CNNLSTM(nn.Module):
    def __init__(s):
        super().__init__(); s.lstm=nn.LSTM(512,128,batch_first=True); s.fc=nn.Linear(128,1)
    def forward(s,x):
        _,(h,_)=s.lstm(x); return s.fc(h[-1]).squeeze(1)
tc=pd.read_csv(P/"train_clips.csv"); vc=pd.read_csv(P/"val_clips.csv")
rows=[]
for tte in (0.5,1.0,1.5):
    Xtr=torch.tensor(np.load(F/f"train_tte_{tte}.npy")); ytr=torch.tensor(tc[tc.tte==tte].target.to_numpy(),dtype=torch.float32)
    Xva=torch.tensor(np.load(F/f"val_tte_{tte}.npy")); yva=vc[vc.tte==tte].target.to_numpy()
    for seed in (42,43,44,45,46):
        torch.manual_seed(seed); np.random.seed(seed)
        m=CNNLSTM(); opt=torch.optim.Adam(m.parameters(),lr=1e-3); crit=nn.BCEWithLogitsLoss(); aps=[]
        for ep in range(10):
            m.train(); perm=torch.randperm(len(Xtr))
            for i in range(0,len(perm),32):
                b=perm[i:i+32]; opt.zero_grad(); crit(m(Xtr[b]),ytr[b]).backward(); opt.step()
            m.eval()
            with torch.no_grad(): aps.append(average_precision_score(yva,torch.sigmoid(m(Xva)).numpy()))
        rows.append(dict(tte=tte,seed=seed,final_epoch=aps[-1],mean_over_epochs=np.mean(aps),best_epoch=max(aps),best_epoch_idx=int(np.argmax(aps))+1))
df=pd.DataFrame(rows)
g=df.groupby("tte").agg(final_epoch=("final_epoch","mean"),final_std=("final_epoch","std"),mean_over_epochs=("mean_over_epochs","mean"),
    best_epoch_oracle=("best_epoch","mean"),best_epoch_idx=("best_epoch_idx","mean")).round(3)
print(g.to_string())
df.to_csv("results/selection_check_fulldata.csv",index=False)
