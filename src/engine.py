import torch
import numpy as np
from tqdm import tqdm
from utils import compute_metrics, unpad_torch

def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, model_type):
    model.train()
    total_loss = 0.0
    comp_accumulator = {}
    
    # Robust length check:
    # 1. List (e.g. single batch list) -> len(list)
    # 2. Has dataset (Standard DataLoader) -> len(dataset)
    # 3. Fallback (Custom/Iterable) -> len(loader)
    # if isinstance(loader, list):
    #     dataset_size = len(loader)
    # elif hasattr(loader, 'dataset'):
    #     dataset_size = len(loader.dataset)
    # else:
    #     dataset_size = len(loader)

    for batch in tqdm(loader, leave=False):
        if model_type == "mlp":
            feats, coords, yb = batch
            feats, coords, yb = feats.to(device), coords.to(device), yb.to(device)
            
            with torch.amp.autocast(device_type=device.type):
                pred = model(feats, coords)
                loss, components = loss_fn(pred, yb)
                
        elif model_type == "cnn":
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            
            with torch.amp.autocast(device_type=device.type):
                pred = model(xb)
                loss, components = loss_fn(pred, yb)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        curr_batch_size = yb.size(0)
        # total_loss += loss.item() * curr_batch_size
        total_loss += loss.item()
        
        for k, v in components.items():
            comp_accumulator[k] = comp_accumulator.get(k, 0.0) + (v * curr_batch_size)

    # avg_loss = total_loss / dataset_size
    # avg_components = {k: v / dataset_size for k, v in comp_accumulator.items()}
    avg_loss = total_loss / len(loader)
    avg_components = {k: v / len(loader) for k, v in comp_accumulator.items()}

    return avg_loss, avg_components

@torch.no_grad()
def evaluate(model, feats_mri, ct, device, model_type, pad_vals):
    model.eval()

    if isinstance(ct, np.ndarray):
        ct_tensor = torch.from_numpy(ct).float().to(device)
    else:
        ct_tensor = ct.float().to(device)
    if ct_tensor.ndim == 3:
        ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)

    
    if model_type == "mlp":
        C, D, H, W = feats_mri.shape
        feats_flat = torch.from_numpy(feats_mri).permute(1, 2, 3, 0).reshape(-1, C).float().to(device)
        
        z = torch.linspace(0, 1, D, device=device)
        y = torch.linspace(0, 1, H, device=device)
        x = torch.linspace(0, 1, W, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        
        preds = []
        batch_size = 100_000
        for i in range(0, feats_flat.size(0), batch_size):
            f_batch = feats_flat[i:i+batch_size]
            c_batch = coords[i:i+batch_size]
            preds.append(model(f_batch, c_batch))
        
        pred_flat = torch.cat(preds, dim=0)
        pred_ct_tensor = pred_flat.reshape(1, 1, D, H, W)

    elif model_type == "cnn":
        feats_t = torch.from_numpy(feats_mri).unsqueeze(0).float().to(device)
        pred_ct_tensor = model(feats_t)

    pred_ct_tensor_unpad = unpad_torch(pred_ct_tensor, pad_vals)
    ct_tensor_unpad = unpad_torch(ct_tensor, pad_vals)
    metrics = compute_metrics(pred_ct_tensor_unpad, ct_tensor_unpad)
    # pred_ct_numpy_unpad = pred_ct_tensor_unpad.squeeze().cpu().numpy()
    # return metrics, pred_ct_numpy_unpad
    pred_ct_tensor = pred_ct_tensor.squeeze().cpu().numpy()
    return metrics, pred_ct_tensor