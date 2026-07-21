import copy, random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def fit_model(model,X,y,X_val,y_val,cfg,device=None):
    seed_all(cfg.seed); device=device or ("cuda" if torch.cuda.is_available() else "cpu"); model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay); criterion=torch.nn.HuberLoss(); best=float("inf"); best_state=None; wait=0; history=[]
    loader=DataLoader(TensorDataset(torch.as_tensor(X,dtype=torch.float32),torch.as_tensor(y,dtype=torch.float32)),batch_size=cfg.batch_size,shuffle=True)
    xv=torch.as_tensor(X_val,dtype=torch.float32,device=device); yv=torch.as_tensor(y_val,dtype=torch.float32,device=device)
    for epoch in range(cfg.epochs):
        model.train(); train_losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); optimizer.zero_grad(set_to_none=True); loss=criterion(model(xb),yb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.gradient_clip); optimizer.step(); train_losses.append(loss.item())
        model.eval()
        with torch.no_grad(): val_loss=criterion(model(xv),yv).item()
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(train_losses)),"val_loss":val_loss})
        if val_loss<best-1e-6: best=val_loss; best_state=copy.deepcopy(model.state_dict()); wait=0
        else: wait+=1
        if wait>=cfg.patience: break
    if best_state is None: raise RuntimeError("Training produced no valid checkpoint")
    model.load_state_dict(best_state); return model,best,history

def predict_with_embeddings(model,X,batch=512):
    device=next(model.parameters()).device; preds=[]; embeddings=[]; model.eval()
    with torch.no_grad():
        for start in range(0,len(X),batch):
            y,e=model(torch.as_tensor(X[start:start+batch],dtype=torch.float32,device=device),return_embedding=True); preds.append(y.cpu().numpy()); embeddings.append(e.cpu().numpy())
    return np.vstack(preds),np.vstack(embeddings)
