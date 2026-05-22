import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.utils.tensorboard
from torch.optim import AdamW
import torch.cuda.amp as amp

import itertools

from . import dist_util, logger
from .resample import LossAwareSampler, UniformSampler
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        in_channels,
        image_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        contr,
        save_interval,
        resume_checkpoint,
        resume_step,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        dataset='brats',
        summary_writer=None,
        mode='default',
        loss_level='image',
        use_wandb=False,
        wandb_run=None,
        val_loader=None,
        val_diffusion=None,
        val_interval=20000,
        val_subj_id='',
        val_ddim_steps=50,
        save_last_interval=0,
    ):
        self.summary_writer = summary_writer
        self.mode = mode
        # Optional wandb + in-training validation hook (None by default = unchanged behavior).
        self.use_wandb = use_wandb
        self.wandb_run = wandb_run
        self.val_loader = val_loader
        self.val_diffusion = val_diffusion
        self.val_interval = int(val_interval) if val_interval else 0
        self.val_subj_id = val_subj_id
        self.val_ddim_steps = int(val_ddim_steps)
        self.save_last_interval = int(save_last_interval) if save_last_interval else 0
        self.model = model
        self.diffusion = diffusion
        self.datal = data
        self.dataset = dataset
        self.iterdatal = iter(data)
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.image_size = image_size
        self.contr = contr
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        if self.use_fp16:
            self.grad_scaler = amp.GradScaler()
        else:
            self.grad_scaler = amp.GradScaler(enabled=False)

        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.dwt = DWT_3D('haar')
        self.idwt = IDWT_3D('haar')

        self.loss_level = loss_level

        self.step = 1
        self.resume_step = resume_step
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        # Wall-clock for info/cumulative_time.
        import time as _time
        self._t_start = _time.time()

        self._load_and_sync_parameters()

        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            print("Resume Step: " + str(self.resume_step))
            self._load_optimizer_state()

        if not th.cuda.is_available():
            logger.warn(
                "Training requires CUDA. "
            )

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model ...')
            # For rolling-last checkpoints (synthrad_last.pt), the step number isn't
            # in the filename — it's stored in a `last_step.txt` sidecar.
            if os.path.basename(resume_checkpoint).endswith('_last.pt'):
                sidecar = os.path.join(os.path.dirname(resume_checkpoint), 'last_step.txt')
                if os.path.exists(sidecar):
                    with open(sidecar) as f:
                        self.resume_step = int(f.read().strip())
                else:
                    logger.log(f"WARNING: loading {resume_checkpoint} but no last_step.txt sidecar; step counter will start at 0.")
                    self.resume_step = 0
            else:
                self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}... (step={self.resume_step})")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())


    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        # Rolling-last opt state lives at `optlast.pt`, milestones at `opt<step>.pt`.
        if os.path.basename(main_checkpoint).endswith('_last.pt'):
            opt_checkpoint = bf.join(bf.dirname(main_checkpoint), 'optlast.pt')
        else:
            opt_checkpoint = bf.join(
                bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
            )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
        else:
            print('no optimizer checkpoint exists')

    def run_loop(self):
        import time
        t = time.time()
        while not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
            t_total = time.time() - t
            t = time.time()
            if self.dataset in ['brats', 'synthrad']:
                try:
                    batch = next(self.iterdatal)
                    cond = {}
                except StopIteration:
                    self.iterdatal = iter(self.datal)
                    batch = next(self.iterdatal)
                    cond = {}

            if self.mode == 'i2i':
                if self.dataset == 'synthrad':
                    batch['ct']  = batch['ct'].to(dist_util.dev())
                    batch['mri'] = batch['mri'].to(dist_util.dev())
                else:
                    batch['t1n'] = batch['t1n'].to(dist_util.dev())
                    batch['t1c'] = batch['t1c'].to(dist_util.dev())
                    batch['t2w'] = batch['t2w'].to(dist_util.dev())
                    batch['t2f'] = batch['t2f'].to(dist_util.dev())
            else:
                batch = batch.to(dist_util.dev())

            t_fwd = time.time()
            t_load = t_fwd-t

            lossmse, sample, sample_idwt = self.run_step(batch, cond)

            t_fwd = time.time()-t_fwd

            names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]

            global_step = self.step + self.resume_step

            if self.summary_writer is not None:
                self.summary_writer.add_scalar('time/load', t_load, global_step=global_step)
                self.summary_writer.add_scalar('time/forward', t_fwd, global_step=global_step)
                self.summary_writer.add_scalar('time/total', t_total, global_step=global_step)
                self.summary_writer.add_scalar('loss/MSE', lossmse.item(), global_step=global_step)

            if self.use_wandb:
                import wandb
                payload = {
                    'time/load': t_load,
                    'time/forward': t_fwd,
                    'time/total': t_total,
                    'loss/MSE': lossmse.item(),
                }
                # info/* — mirror amix's per-step training-progress fields.
                samples_seen = (self.step + self.resume_step) * self.global_batch
                payload['info/lr'] = float(self.opt.param_groups[0]['lr'])
                payload['info/global_step'] = float(global_step)
                payload['info/samples_seen'] = float(samples_seen)
                payload['info/cumulative_time'] = float(time.time() - self._t_start)
                if self.lr_anneal_steps:
                    payload['info/train_pct'] = 100.0 * global_step / float(self.lr_anneal_steps)
                wandb.log(payload, step=global_step)

            if self.step % 200 == 0 and self.summary_writer is not None:
                image_size = sample_idwt.size()[2]
                midplane = sample_idwt[0, 0, :, :, image_size // 2]
                self.summary_writer.add_image('sample/x_0', midplane.unsqueeze(0),
                                              global_step=global_step)

                image_size = sample.size()[2]
                for ch in range(8):
                    midplane = sample[0, ch, :, :, image_size // 2]
                    self.summary_writer.add_image('sample/{}'.format(names[ch]), midplane.unsqueeze(0),
                                                  global_step=global_step)

                # BraTS-only source-image previews (uses t1n/t1c/t2w/t2f keys).
                if self.mode == 'i2i' and self.dataset == 'brats':
                    if not self.contr == 't1n':
                        image_size = batch['t1n'].size()[2]
                        midplane = batch['t1n'][0, 0, :, :, image_size // 2]
                        self.summary_writer.add_image('source/t1n', midplane.unsqueeze(0),
                                                      global_step=global_step)
                    if not self.contr == 't1c':
                        image_size = batch['t1c'].size()[2]
                        midplane = batch['t1c'][0, 0, :, :, image_size // 2]
                        self.summary_writer.add_image('source/t1c', midplane.unsqueeze(0),
                                                      global_step=global_step)
                    if not self.contr == 't2w':
                        midplane = batch['t2w'][0, 0, :, :, image_size // 2]
                        self.summary_writer.add_image('source/t2w', midplane.unsqueeze(0),
                                                      global_step=global_step)
                    if not self.contr == 't2f':
                        midplane = batch['t2f'][0, 0, :, :, image_size // 2]
                        self.summary_writer.add_image('source/t2f', midplane.unsqueeze(0),
                                                      global_step=global_step)


            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            # In-training light val sanity check: sample one fixed subject every val_interval steps.
            if (
                self.val_loader is not None
                and self.val_interval > 0
                and self.step % self.val_interval == 0
            ):
                try:
                    self._validate_one()
                except Exception as e:
                    logger.log(f"[val hook] exception: {e}")

                # Free train-side viz: sample_idwt is the in-loop denoised reconstruction
                # of the current training batch (computed during forward_backward at
                # line 205). Logging at val cadence lets us compare to val/* for overfitting.
                if self.use_wandb and self.mode == 'i2i' and self.dataset == 'synthrad':
                    try:
                        self._log_train_sample(batch, sample_idwt, global_step)
                    except Exception as e:
                        logger.log(f"[train viz hook] exception: {e}")

            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            # Rolling last-checkpoint (overwrites each fire) for cheap resume in case of crash/SLURM-kill.
            if self.save_last_interval > 0 and self.step % self.save_last_interval == 0:
                self.save_last()
            self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, label=None, info=dict()):
        lossmse, sample, sample_idwt = self.forward_backward(batch, cond, label)

        if self.use_fp16:
            self.grad_scaler.unscale_(self.opt)  # check self.grad_scaler._per_optimizer_states

        # compute norms
        with torch.no_grad():
            param_max_norm = max([p.abs().max().item() for p in self.model.parameters()])
            grad_max_norm = max([p.grad.abs().max().item() for p in self.model.parameters()])
            info['norm/param_max'] = param_max_norm
            info['norm/grad_max'] = grad_max_norm

        if not torch.isfinite(lossmse): #infinite
            if not torch.isfinite(torch.tensor(param_max_norm)):
                logger.log(f"Model parameters contain non-finite value {param_max_norm}, entering breakpoint", level=logger.ERROR)
                breakpoint()
            else:
                logger.log(f"Model parameters are finite, but loss is not: {lossmse}"
                           "\n -> update will be skipped in grad_scaler.step()", level=logger.WARN)

        if self.use_fp16:
            print("Use fp16 ...")
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()
            info['scale'] = self.grad_scaler.get_scale()
        else:
            self.opt.step()
        self._anneal_lr()
        self.log_step()
        return lossmse, sample, sample_idwt

    def forward_backward(self, batch, cond, label=None):
        for p in self.model.parameters():  # Zero out gradient
            p.grad = None

        if self.mode == 'i2i':
            # Pick any key from the batch to read the leading batch dim (target modality).
            ref_key = 'ct' if self.dataset == 'synthrad' else 't1n'
            t, weights = self.schedule_sampler.sample(batch[ref_key].shape[0], dist_util.dev())
        else:
            t, weights = self.schedule_sampler.sample(batch.shape[0], dist_util.dev())

        compute_losses = functools.partial(self.diffusion.training_losses,
                                           self.model,
                                           x_start=batch,
                                           t=t,
                                           model_kwargs=cond,
                                           labels=label,
                                           mode=self.mode,
                                           contr=self.contr
                                           )
        losses1 = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses1["loss"].detach())

        losses = losses1[0]         # Loss value
        sample = losses1[1]         # Denoised subbands at t=0
        sample_idwt = losses1[2]    # Inverse wavelet transformed denoised subbands at t=0

        # Log wavelet level loss
        global_step = self.step + self.resume_step
        band_names = ['lll', 'llh', 'lhl', 'lhh', 'hll', 'hlh', 'hhl', 'hhh']
        band_vals = {f'loss/mse_wav_{n}': losses["mse_wav"][i].item() for i, n in enumerate(band_names)}
        if self.summary_writer is not None:
            for k, v in band_vals.items():
                self.summary_writer.add_scalar(k, v, global_step=global_step)
        if self.use_wandb:
            import wandb
            wandb.log(band_vals, step=global_step)

        weights = th.ones(len(losses["mse_wav"])).cuda()  # Equally weight all wavelet channel losses

        loss = (losses["mse_wav"] * weights).mean()
        lossmse = loss.detach()

        log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})

        # perform some finiteness checks
        if not torch.isfinite(loss):
            logger.log(f"Encountered non-finite loss {loss}")
        if self.use_fp16:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        return lossmse.detach(), sample, sample_idwt

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, state_dict):
            if dist.get_rank() == 0:
                logger.log("Saving model...")
                if self.dataset == 'brats':
                    filename = f"brats_{(self.step+self.resume_step):06d}.pt"
                elif self.dataset == 'lidc-idri':
                    filename = f"lidc-idri_{(self.step+self.resume_step):06d}.pt"
                elif self.dataset == 'brats_inpainting':
                    filename = f"brats_inpainting_{(self.step + self.resume_step):06d}.pt"
                elif self.dataset == 'synthrad':
                    filename = f"synthrad_{(self.step + self.resume_step):06d}.pt"
                else:
                    raise ValueError(f'dataset {self.dataset} not implemented')

                with bf.BlobFile(bf.join(get_blob_logdir(), 'checkpoints', filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.model.state_dict())

        if dist.get_rank() == 0:
            checkpoint_dir = os.path.join(logger.get_dir(), 'checkpoints')
            with bf.BlobFile(
                bf.join(checkpoint_dir, f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

    def save_last(self):
        """Overwrite a rolling 'last' checkpoint (model + opt + step sidecar) for cheap resume.

        Writes 3 files under checkpoints/ that get overwritten each call:
          - <dataset>_last.pt       : model state_dict
          - optlast.pt              : optimizer state
          - last_step.txt           : current step number (sidecar, read on resume)
        """
        if dist.get_rank() != 0:
            return
        checkpoint_dir = os.path.join(logger.get_dir(), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        global_step = self.step + self.resume_step
        # Model
        with bf.BlobFile(bf.join(checkpoint_dir, f"{self.dataset}_last.pt"), "wb") as f:
            th.save(self.model.state_dict(), f)
        # Optimizer
        with bf.BlobFile(bf.join(checkpoint_dir, "optlast.pt"), "wb") as f:
            th.save(self.opt.state_dict(), f)
        # Step sidecar
        with open(os.path.join(checkpoint_dir, "last_step.txt"), "w") as f:
            f.write(str(global_step))

    # ------------------------------------------------------------------ #
    # In-training light validation: sample one fixed subject and log it. #
    # ------------------------------------------------------------------ #
    def _log_train_sample(self, batch, sample_idwt, global_step):
        """Free train-side viz: mid-slice (MRI / GT CT / Pred / |diff|) on the
        current training batch. sample_idwt is the t=0 reconstruction already
        computed during forward_backward — no extra forward pass.
        """
        import wandb
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        mri = batch['mri'].detach()
        ct  = batch['ct'].detach()
        pred = sample_idwt.detach().clamp(0., 1.)

        gt_mri_np = mri[0, 0].float().cpu().numpy()
        gt_ct_np  = ct [0, 0].float().cpu().numpy()
        pred_np   = pred[0, 0].float().cpu().numpy()
        diff_np   = np.abs(pred_np - gt_ct_np)
        mid_z     = pred_np.shape[-1] // 2

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, img, title, cmap, vmin, vmax in [
            (axes[0], gt_mri_np[..., mid_z], "MRI",       "gray",  0, 1),
            (axes[1], gt_ct_np [..., mid_z], "GT CT",     "gray",  0, 1),
            (axes[2], pred_np  [..., mid_z], "Pred",      "gray",  0, 1),
            (axes[3], diff_np  [..., mid_z], "|Pred-GT|", "magma", 0, 0.5),
        ]:
            ax.imshow(np.rot90(img), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([])
        subj = batch.get('subj_id', ['?'])
        subj = subj[0] if isinstance(subj, (list, tuple)) else subj
        fig.suptitle(f"[train] {subj} | step {global_step}")
        plt.tight_layout()
        wandb.log({"train/sample_image": wandb.Image(fig)}, step=global_step)
        plt.close(fig)

    def _validate_one(self):
        """Sample one fixed val subject with reduced-step DDPM and log to wandb.

        Uses `self.val_diffusion` (a SpacedDiffusion with timestep_respacing="ddimN")
        if provided; otherwise falls back to `self.diffusion`. The val diffusion shares
        the same trained weights (SpacedDiffusion just maps timesteps).
        """
        if self.val_loader is None:
            return
        try:
            batch = next(iter(self.val_loader))
        except StopIteration:
            return
        if 'mri' not in batch or 'ct' not in batch:
            return

        global_step = self.step + self.resume_step
        device = dist_util.dev()

        mri = batch['mri'].to(device).float()
        ct_gt = batch['ct'].to(device).float()
        mask = batch.get('body_mask', None)
        if mask is not None:
            mask = mask.to(device).float()
        orig_shape = batch.get('original_shape', None)
        if orig_shape is not None and th.is_tensor(orig_shape):
            orig_shape = orig_shape[0].tolist() if orig_shape.ndim > 1 else orig_shape.tolist()
        subj_id = batch.get('subj_id', ['val'])
        if isinstance(subj_id, (list, tuple)):
            subj_id = subj_id[0]

        self.model.eval()
        try:
            with th.no_grad():
                # 1 conditioning modality (MRI) -> 8-ch DWT
                LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.dwt(mri)
                cond_dwt = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

                B, _, D, H, W = mri.shape
                noise_shape = (B, 8, D // 2, H // 2, W // 2)
                noise = th.randn(*noise_shape, device=device)

                diffusion = self.val_diffusion if self.val_diffusion is not None else self.diffusion
                x0_wav = diffusion.p_sample_loop(
                    model=self.model,
                    shape=noise.shape,
                    noise=noise,
                    cond=cond_dwt,
                    clip_denoised=True,
                    model_kwargs={},
                    progress=False,
                )

                # 8-ch wavelet -> single-channel volume via IDWT (LLL is scaled by *3)
                Bo, _, Do, Ho, Wo = x0_wav.size()
                pred = self.idwt(
                    x0_wav[:, 0:1] * 3.,
                    x0_wav[:, 1:2], x0_wav[:, 2:3], x0_wav[:, 3:4],
                    x0_wav[:, 4:5], x0_wav[:, 5:6], x0_wav[:, 6:7], x0_wav[:, 7:8],
                ).clamp(0., 1.)

                # Unpad back to original spatial shape (recorded before res_mult pad).
                if orig_shape is not None and len(orig_shape) == 3:
                    w_o, h_o, d_o = int(orig_shape[0]), int(orig_shape[1]), int(orig_shape[2])
                    pred = pred[..., :w_o, :h_o, :d_o]
                    ct_gt_u = ct_gt[..., :w_o, :h_o, :d_o]
                    mri_u = mri[..., :w_o, :h_o, :d_o]
                    if mask is not None:
                        mask = mask[..., :w_o, :h_o, :d_o]
                else:
                    ct_gt_u = ct_gt
                    mri_u = mri

                if mask is not None:
                    pred = pred * mask

                # Metrics: import lazily so non-synthrad runs don't pay the cost.
                try:
                    from src.common.utils import compute_metrics  # noqa: E402
                    m = compute_metrics(pred.float(), ct_gt_u.float(), hu_range=2048)
                except Exception as e:
                    logger.log(f"[val hook] compute_metrics failed: {e}")
                    m = {}

                if self.use_wandb:
                    import wandb
                    log_payload = {f"val/{k}": v for k, v in m.items()}
                    log_payload["val/subj_id"] = subj_id
                    log_payload["val/ddim_steps"] = self.val_ddim_steps

                    # Mid-slice grid via matplotlib (matches src/common/utils.visualize_lite layout).
                    try:
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        import numpy as np

                        gt_ct_np  = ct_gt_u.detach().cpu().numpy()[0, 0]
                        gt_mri_np = mri_u.detach().cpu().numpy()[0, 0]
                        pred_np   = pred.detach().cpu().numpy()[0, 0]
                        D_dim = gt_ct_np.shape[-1]
                        zs = np.linspace(0.2 * D_dim, 0.8 * D_dim, 3, dtype=int)
                        items = [
                            (gt_mri_np, 'GT MRI', 'gray', (0, 1)),
                            (gt_ct_np,  'GT CT',  'gray', (0, 1)),
                            (pred_np,   'Pred CT','gray', (0, 1)),
                            (pred_np - gt_ct_np, 'Residual', 'seismic', (-0.5, 0.5)),
                        ]
                        fig, axes = plt.subplots(len(zs), len(items), figsize=(3 * len(items), 3 * len(zs)))
                        if len(zs) == 1:
                            axes = axes.reshape(1, -1)
                        for i, z in enumerate(zs):
                            for j, (data, title, cmap, clim) in enumerate(items):
                                ax = axes[i, j]
                                ax.imshow(data[:, :, z], cmap=cmap, vmin=clim[0], vmax=clim[1])
                                if i == 0:
                                    ax.set_title(title)
                                ax.axis('off')
                        cap = f"subj={subj_id} step={global_step}"
                        if m:
                            cap += " | " + " ".join(f"{k}={v:.4f}" for k, v in m.items())
                        fig.suptitle(cap, fontsize=10)
                        log_payload["val/sample_image"] = wandb.Image(fig)
                        plt.close(fig)
                    except Exception as e:
                        logger.log(f"[val hook] image build failed: {e}")

                    wandb.log(log_payload, step=global_step)

                logger.log(
                    f"[val hook] step={global_step} subj={subj_id} "
                    + " ".join(f"{k}={v:.4f}" for k, v in m.items())
                )
        finally:
            self.model.train()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """

    split = os.path.basename(filename)
    split = split.split(".")[-2]  # remove extension
    split = split.split("_")[-1]  # remove possible underscores, keep only last word
    # extract trailing number
    reversed_split = []
    for c in reversed(split):
        if not c.isdigit():
            break
        reversed_split.append(c)
    split = ''.join(reversed(reversed_split))
    split = ''.join(c for c in split if c.isdigit())  # remove non-digits
    try:
        return int(split)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
