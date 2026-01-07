import torch
import numpy as np
from tqdm import tqdm
from utils import compute_metrics, unpad_torch

def get_mlp_samples(features, target, locations, vol_shapes, num_samples=16384):
    """
    features: [B, C, W, H, D] # [1, 16, 96, 96, 96]
    target: [B, 1, D, H, W] # [1, 1, 96, 96, 96]
    locations: [B, 6] -> (i_ini, j_ini, k_ini, ...) # [ 34,  25,   0, 130, 121,  96]
    vol_shapes: [B, 3] -> (W, H, D) # [155., 122.,  91. -> 96. (updated)]
    """
    B, C, dim_w, dim_h, dim_d = features.shape

    # 1. Permute to [B, W, H, D, C] for easier flattening
    # Moves Channel to last dim. Spatial dims (W,H,D) stay in 1,2,3
    feats_flat = features.permute(0, 2, 3, 4, 1).reshape(-1, C)
    targ_flat = target.permute(0, 2, 3, 4, 1).reshape(-1, 1)
    
    # 2. Generate GLOBAL Coordinates
    # Local grid matches feature dims: (W, H, D)
    w_local = torch.arange(dim_w, device=features.device).float()
    h_local = torch.arange(dim_h, device=features.device).float()
    d_local = torch.arange(dim_d, device=features.device).float()
    
    # 'ij' indexing preserves order: Dim0=W, Dim1=H, Dim2=D
    grid_w, grid_h, grid_d = torch.meshgrid(w_local, h_local, d_local, indexing='ij')
    
    # Stack in (x, y, z) order which corresponds to (W, H, D)
    # Shape: [1, W, H, D, 3]
    coords_local = torch.stack([grid_w, grid_h, grid_d], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1, 1)
    
    # --- 3. Global Normalization ---
    # offsets and shapes are already in (x, y, z) / (W, H, D) order from utils.py
    
    off_w = locations[:, 0].view(B, 1, 1, 1).float().to(features.device)
    off_h = locations[:, 1].view(B, 1, 1, 1).float().to(features.device)
    off_d = locations[:, 2].view(B, 1, 1, 1).float().to(features.device)
    
    max_w = vol_shapes[:, 0].view(B, 1, 1, 1).float().to(features.device)
    max_h = vol_shapes[:, 1].view(B, 1, 1, 1).float().to(features.device)
    max_d = vol_shapes[:, 2].view(B, 1, 1, 1).float().to(features.device)
    
    # Normalize: (Local + Offset) / Max
    # Index 0 is W(x), 1 is H(y), 2 is D(z). Everything aligns now.
    global_x = (coords_local[..., 0] + off_w) / (max_w - 1)
    global_y = (coords_local[..., 1] + off_h) / (max_h - 1)
    global_z = (coords_local[..., 2] + off_d) / (max_d - 1)
    
    # Stack [B, W, H, D, 3]
    coords_global = torch.stack([global_x, global_y, global_z], dim=-1)
    coords_flat = coords_global.reshape(-1, 3)
    
    # 4. Random Subsampling
    # Optional: You can add the "Valid Mask" logic here if you want to skip air
    total_voxels = feats_flat.shape[0]
    actual_samples = min(num_samples, total_voxels)
    indices = torch.randperm(total_voxels)[:actual_samples]

    return feats_flat[indices], coords_flat[indices], targ_flat[indices]

def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, model_type, feat_extractor, args):
    # Anatomix should be in eval mode even during training (unless finetuning it)
    feat_extractor.eval() 
    
    model.train()
    total_loss = 0.0
    comp_accumulator = {}
    num_batches = len(loader)
    
    for mri, ct, seg, location, vol_shape in tqdm(loader, leave=False):
        mri, ct = mri.to(device), ct.to(device)
        
        # --- 1. Just-in-Time Feature Extraction ---
        with torch.no_grad():
            # Run Anatomix on the augmented patch
            # Input: [B, 1, D, H, W] -> Output: [B, 16, D, H, W]
            features = feat_extractor(mri) 
            
            # --- 2. Just-in-Time One-Hot Encoding ---
            if seg is not None:
                seg = seg.to(device) # [B, 1, D, H, W]
                # One-hot encode using PyTorch
                # seg is long/int. F.one_hot expects long.
                seg_long = seg.long().squeeze(1) # [B, D, H, W]
                
                # Safety clamp for classes > seg_classes
                seg_long = torch.clamp(seg_long, 0, args.seg_classes - 1)
                
                # one_hot: [B, D, H, W, C]
                seg_hot = F.one_hot(seg_long, num_classes=args.seg_classes).float()
                # Permute to [B, C, D, H, W]
                seg_hot = seg_hot.permute(0, 4, 1, 2, 3)
                
                features = torch.cat([features, seg_hot], dim=1)

        # --- 3. Model Forward Pass ---
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type=device.type):
            if model_type == "mlp":
                # Flatten and Subsample Points
                f_pts, c_pts, t_pts = get_mlp_samples(features, ct, location, vol_shape, num_samples=args.mlp_batch_size)
                pred = model(f_pts, c_pts)
                loss, components = loss_fn(pred, t_pts)
            
            elif model_type == "cnn":
                # Standard Convolution
                pred = model(features)
                loss, components = loss_fn(pred, ct)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("LOSS CONTAINS NaN!!")
            exit()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        for k, v in components.items():
            comp_accumulator[k] = comp_accumulator.get(k, 0.0) + v

    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in comp_accumulator.items()}

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
        C, dim_w, dim_h, dim_d = feats_mri.shape
        
        # Permute to (W, H, D, C) for flattening
        feats_flat = torch.from_numpy(feats_mri).permute(1, 2, 3, 0).reshape(-1, C).float().to(device)
        
        # Grid must be (W, H, D) -> (x, y, z)
        w_local = torch.arange(dim_w, device=device).float()
        h_local = torch.arange(dim_h, device=device).float()
        d_local = torch.arange(dim_d, device=device).float()
        
        # 'ij' index -> grid_w, grid_h, grid_d
        grid_w, grid_h, grid_d = torch.meshgrid(w_local, h_local, d_local, indexing='ij')
        
        # Normalize 0..1
        norm_w = grid_w / (dim_w - 1)
        norm_h = grid_h / (dim_h - 1)
        norm_d = grid_d / (dim_d - 1)
        
        # Stack [W, H, D, 3] -> (x, y, z)
        coords = torch.stack([norm_w, norm_h, norm_d], dim=-1).reshape(-1, 3)
        
        preds = []
        batch_size = 100000
        for i in range(0, feats_flat.size(0), batch_size):
            f_batch = feats_flat[i:i+batch_size]
            c_batch = coords[i:i+batch_size]
            preds.append(model(f_batch, c_batch))
        
        pred_flat = torch.cat(preds, dim=0)
        # Reshape back to (W, H, D)
        pred_ct_tensor = pred_flat.reshape(1, 1, dim_w, dim_h, dim_d)

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