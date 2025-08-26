import logging
import os

import omegaconf
import torch
import tqdm
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader

import argparse
import wandb
from sdib.data import DiTDataset, PromptImageDataset
from sdib.hooks import CrossAttentionExtractionHook, FeedForwardHooker, NormHooker
from sdib.utils import (
    calculate_mask_sparsity,
    calculate_reg_loss,
    get_precision,
    load_config,
    load_pipeline,
    save_image_binarize_seed,
    save_image_seed,
)


# Clean up the code, move it into src package
def pruning_loss(
    reconstruction_loss_func,
    image_pt,
    image,
    cross_attn_hooker,
    ff_hooker,
    device,
    torch_dtype,
    cfg,
    logger=None,
    norm_hooker=None,
):
    loss_reconstruct = reconstruction_loss_func(image_pt, image["images"])
    attn_loss_reg = torch.tensor(0.0, device=device, dtype=torch_dtype)
    if cfg.loss.use_attn_reg:
        attn_loss_reg = calculate_reg_loss(
            attn_loss_reg,
            cross_attn_hooker.lambs,
            cfg.loss.reg,
            mean=cfg.loss.mean,
            reg=cfg.loss.lambda_reg,
            reg_alpha=cfg.loss.reg_alpha,
            reg_beta=cfg.loss.reg_beta,
        )
    ff_loss_reg = torch.tensor(0.0, device=device, dtype=torch_dtype)
    if cfg.loss.use_ffn_reg:
        ff_loss_reg = calculate_reg_loss(
            ff_loss_reg,
            ff_hooker.lambs,
            cfg.loss.reg,
            mean=cfg.loss.mean,
            reg=cfg.loss.lambda_reg,
            reg_alpha=cfg.loss.reg_alpha,
            reg_beta=cfg.loss.reg_beta,
        )
    loss_reg = attn_loss_reg + ff_loss_reg
    if norm_hooker:
        norm_loss_reg = calculate_reg_loss(
            ff_loss_reg,
            norm_hooker.lambs,
            cfg.loss.reg,
            mean=cfg.loss.mean,
            reg=cfg.loss.lambda_reg,
            reg_alpha=cfg.loss.reg_alpha,
            reg_beta=cfg.loss.reg_beta,
        )
        ff_loss_reg += norm_loss_reg
    loss = loss_reconstruct + cfg.trainer.beta * loss_reg
    if logger:
        log_output = f"ff_loss_reg: {ff_loss_reg.item()}" + f" attn_loss_reg: {attn_loss_reg.item()}"
        if norm_hooker:
            log_output += f" norm_loss_reg: {norm_loss_reg.item()}"
        logger.info(log_output)
    return loss, loss_reconstruct, loss_reg


def main(args):
    # inital config
    cfg = load_config(args.cfg)
    device = torch.device(cfg.trainer.device)
    with open(args.validation_prompts_path, "r") as f:
        validation_prompts = yaml.safe_load(f)
    if cfg.trainer.model == "dit":
        validation_prompts = [1]  # for dit we need index number as prompt
    else:
        validation_prompts = validation_prompts

    # inital wandb
    if cfg.logger.type == "wandb":
        config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        import time

        timestr = time.strftime("%Y%m%d-%H%M%S")
        name = f"{cfg.logger.notes}_{timestr}"
        run = wandb.init(
            project=cfg.logger.project, notes=cfg.logger.notes, tags=cfg.logger.tags, config=config, name=name
        )

    # inital logging
    logger = logging.getLogger(__name__)
    filename = f"{cfg.logger.output_dir}/{cfg.logger.project}/{cfg.logger.notes}/report.log"
    os.makedirs(f"{cfg.logger.output_dir}/{cfg.logger.project}/{cfg.logger.notes}", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(filename), logging.StreamHandler()],
    )
    logger.info(f"Validation prompts: {validation_prompts}")

    # setup for accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.logger.output_dir, logging_dir=cfg.logger.output_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator_log_with = "all" if cfg.logger.type == "csv" else cfg.logger.type
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.accelerator.gradient_accumulation_steps,
        log_with=accelerator_log_with,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.islaunch:
        device = accelerator.device

    # setup seed
    seed = cfg.trainer.seed
    set_seed(seed)  # use accelerate set_seed
    g_cpu = torch.Generator(device.type).manual_seed(seed)

    # set the precision, only support bf16, fp16, fp32
    torch_dtype = get_precision(cfg.trainer.precision)

    # initialize pipeline
    pipe = load_pipeline(cfg.trainer.model, torch_dtype, cfg.trainer.disable_progress_bar)
    pipe.to(device)

    # set required_grad to False for all parameters
    # unet, vae, transformer (for sd3)
    pipe.vae.requires_grad_(False)
    if cfg.trainer.model in ["sd3", "dit", "flux", "flux_dev"]:
        pipe.transformer.requires_grad_(False)
    else:
        pipe.unet.requires_grad_(False)

    # prepare for the datasets and dataloader
    if cfg.trainer.model == "dit":
        train_dataset = DiTDataset(
            pipe=pipe,
            save_dir=cfg.data.save_dir,
            device=device,
            size=cfg.data.size,
            num_inference_steps=cfg.trainer.num_intervention_steps,
            seed=seed,
        )
    else:
        train_dataset = PromptImageDataset(
            metadata=cfg.data.metadata,
            pipe=pipe,
            num_inference_steps=cfg.trainer.num_intervention_steps,
            save_dir=cfg.data.save_dir,
            device=device,
            seed=seed,
            size=cfg.data.size,
        )
        try:
            batch_size = cfg.data.batch_size
            logger.info(f"Batch size: {batch_size}")
        except Exception as e:
            logger.info(f"Error: {e}, setting batch size to 1")
            batch_size = 1
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # save the original image
    if cfg.logger.type == "wandb":
        img = save_image_seed(pipe, validation_prompts, cfg.trainer.num_intervention_steps, device, seed, save_dir=None)
        wandb.log({"image": [wandb.Image(i) for i in img]})
    else:
        path = os.path.join(args.save_dir, cfg.logger.project, cfg.logger.notes, "images")
        val_path = os.path.join(path, "validation", "initial_image")
        save_image_seed(pipe, validation_prompts, cfg.trainer.num_intervention_steps, device, seed, save_dir=val_path)
        train_path = os.path.join(path, "train", "initial_image")
        save_image_seed(
            pipe, train_dataset[0]["prompt"], cfg.trainer.num_intervention_steps, device, seed, save_dir=train_path
        )

    # define loss for reconstruction
    if cfg.loss.reconstruct == 1:
        reconstruction_loss_func = torch.nn.L1Loss(reduction="mean")
    elif cfg.loss.reconstruct == 2:
        reconstruction_loss_func = torch.nn.MSELoss()
    else:
        raise ValueError(f"Reconstruction loss {cfg.loss.reconstruct} not supported")

    # initialize cross attention hooks
    cross_attn_hooker = CrossAttentionExtractionHook(
        pipe,
        regex=cfg.trainer.regex,
        dtype=torch_dtype,
        head_num_filter=cfg.trainer.head_num_filter,
        masking=cfg.trainer.masking,
        dst=cfg.logger.save_lambda_path.attn,
        epsilon=cfg.trainer.epsilon,
        model_name=cfg.trainer.model,
        attn_name=cfg.trainer.attn_name,
        use_log=cfg.trainer.use_log,
        eps=cfg.trainer.masking_eps,
    )
    cross_attn_hooker.add_hooks(init_value=cfg.trainer.init_lambda)
    lamda_block_names = cross_attn_hooker.get_lambda_block_names

    # initialize feedforward hooks
    ff_hooker = FeedForwardHooker(
        pipe,
        regex=cfg.trainer.regex,
        dtype=torch_dtype,
        masking=cfg.trainer.masking,
        dst=cfg.logger.save_lambda_path.ffn,
        epsilon=cfg.trainer.epsilon,
        eps=cfg.trainer.masking_eps,
        use_log=cfg.trainer.use_log,
    )
    ff_hooker.add_hooks(init_value=cfg.trainer.init_lambda)
    ff_lambda_block_names = ff_hooker.get_lambda_block_names

    # initialize norm hooks if lr is not 0
    if cfg.trainer.n_lr != 0:
        norm_hooker = NormHooker(
            pipe,
            regex=cfg.trainer.regex,
            dtype=torch_dtype,
            masking=cfg.trainer.masking,
            dst=cfg.logger.save_lambda_path.norm,
            epsilon=cfg.trainer.epsilon,
            eps=cfg.trainer.masking_eps,
            use_log=cfg.trainer.use_log,
        )
        norm_hooker.add_hooks(init_value=cfg.trainer.init_lambda)
        norm_lambda_block_names = norm_hooker.get_lambda_block_names

    # dummy generation to initialize the lambda
    logger.info(f"Initializing lambda to be {cfg.trainer.init_lambda}")
    _ = pipe(validation_prompts, generator=g_cpu, num_inference_steps=1)
    if args.load_lambda:
        cross_attn_hooker.binary = False
        ff_hooker.binary = False
        cross_attn_hooker.load(device=device)
        ff_hooker.load(device=device)
        for i, lambs in enumerate(cross_attn_hooker.lambs):
            lambs = lambs.detach().clone().requires_grad_(True)
            lambs.to(device)
            cross_attn_hooker.lambs[i] = lambs
        for i, lambs in enumerate(ff_hooker.lambs):
            lambs = lambs.detach().clone().requires_grad_(True)
            lambs.to(device)
            ff_hooker.lambs[i] = lambs
        # add norm_hooker

    # optimizer and scheduler
    params = [
        {"params": cross_attn_hooker.lambs, "lr": cfg.trainer.attn_lr},
        {"params": ff_hooker.lambs, "lr": cfg.trainer.ff_lr},
    ]
    if cfg.trainer.n_lr != 0:
        params += ({"params": norm_hooker.lambs, "lr": cfg.trainer.n_lr},)

    optimizer = torch.optim.AdamW(params, lr=cfg.trainer.lr)  # redundant lr param
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler.type,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_scheduler.warmup_steps,
        num_training_steps=cfg.trainer.epochs * cfg.data.size,
        num_cycles=cfg.lr_scheduler.num_cycles,
        power=cfg.lr_scheduler.power,
    )

    # prepare with accelerator
    pipe, optimizer, lr_scheduler = accelerator.prepare(pipe, optimizer, lr_scheduler)
    logger.info("Start Training ...")

    torch.cuda.empty_cache()

    optimizer.zero_grad()
    # loss_min = 1e3  # for saving the best lambda
    total_step = cfg.trainer.epochs * cfg.data.size
    with tqdm.tqdm(total=total_step) as pbar:
        for i in range(cfg.trainer.epochs):
            for idx, data in enumerate(dataloader):
                image_pt = data["image"]
                prompt = data["prompt"]
                if cfg.trainer.grad_checkpointing:
                    # use grad checkpointing to save memory
                    g_cpu = torch.Generator(device.type).manual_seed(seed)
                    with torch.no_grad():
                        preparation_phase_output = pipe.inference_preparation_phase(
                            prompt,
                            generator=g_cpu,
                            num_inference_steps=cfg.trainer.num_intervention_steps,
                            output_type="latent",
                        )
                        intermediate_latents = [preparation_phase_output.latents]
                        timesteps = preparation_phase_output.timesteps
                        for timesteps_idx, t in enumerate(timesteps):
                            latents = pipe.inference_with_grad_denoising_step(
                                timesteps_idx, t, preparation_phase_output
                            )
                            # update latents in output class
                            preparation_phase_output.latents = latents
                            intermediate_latents.append(latents)
                        # pop the last latents
                        intermediate_latents.pop()
                        latents.requires_grad = True

                    # backprop from loss to the last latents
                    with torch.set_grad_enabled(True):
                        prompt_embeds = preparation_phase_output.prompt_embeds
                        image = pipe.inference_with_grad_aft_denoising(
                            latents, prompt_embeds, g_cpu, "latent", True, device
                        )
                        # calculate loss
                        norm_hooker = None if cfg.trainer.n_lr == 0 else norm_hooker
                        loss, loss_reconstruct, loss_reg = pruning_loss(
                            reconstruction_loss_func,
                            image_pt,
                            image,
                            cross_attn_hooker,
                            ff_hooker,
                            device,
                            torch_dtype,
                            cfg,
                            logger=logger,
                            norm_hooker=norm_hooker,
                        )
                        accelerator.backward(loss)
                        grad = latents.grad.detach()

                    # backprop from the last latents to the first latents
                    timesteps = preparation_phase_output.timesteps

                    for timesteps_idx, t in enumerate(reversed(timesteps)):
                        current_latents = intermediate_latents[-(timesteps_idx + 1)].detach()
                        current_latents.requires_grad = True
                        timesteps_idx = len(timesteps) - timesteps_idx - 1
                        with torch.set_grad_enabled(True):
                            preparation_phase_output.latents = current_latents
                            latents = pipe.inference_with_grad_denoising_step(
                                timesteps_idx,
                                t,
                                preparation_phase_output,
                                step_index=timesteps_idx,
                            )
                            # calculate grad w.r.t. lambda
                            if cfg.trainer.n_lr == 0:
                                trainable_lambs = cross_attn_hooker.lambs + ff_hooker.lambs
                            else:
                                trainable_lambs = cross_attn_hooker.lambs + ff_hooker.lambs + norm_hooker.lambs
                            lamb_grads = torch.autograd.grad(
                                latents,
                                trainable_lambs,
                                grad_outputs=grad,
                                retain_graph=True,
                            )

                            for lamb, lamb_grad in zip(trainable_lambs, lamb_grads):
                                if lamb.grad is None:
                                    lamb.grad = lamb_grad
                                else:
                                    lamb.grad += lamb_grad
                            # calculate grad w.r.t. unet input
                            grad = torch.autograd.grad(latents, current_latents, grad_outputs=grad)
                else:
                    # reset seed at each step to make sure the generated image is identical under same randomness
                    g_cpu = torch.Generator(device.type).manual_seed(seed)
                    image = pipe.inference_with_grad(
                        prompt,
                        generator=g_cpu,
                        num_inference_steps=cfg.trainer.num_intervention_steps,
                        output_type="latent",
                        attn_res=(16, 16),  # might need to change this based on different models
                    )
                    # calculate loss
                    loss, loss_reconstruct, loss_reg = pruning_loss(
                        reconstruction_loss_func,
                        image_pt,
                        image,
                        cross_attn_hooker,
                        ff_hooker,
                        device,
                        torch_dtype,
                        cfg,
                    )
                    accelerator.backward(loss)

                if (idx * batch_size) % cfg.trainer.accumulate_grad_batches == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # START LOGGING
                if (idx * batch_size) % cfg.logger.plot_interval == 0:
                    if cfg.logger.type == "wandb":
                        wandb.log(
                            {
                                "loss_reconstruct": loss_reconstruct,
                                "loss_reg": loss_reg,
                                "loss": loss,
                                "lr": lr_scheduler.get_last_lr()[0],
                                "vram": torch.cuda.max_memory_allocated(device) / 1024**3,
                            },
                            commit=False,
                        )
                        # log learned lambdas
                        for index, lamb in enumerate(cross_attn_hooker.lambs):
                            heads = [f"head_{j}" for j in range(lamb.shape[0])]
                            # convert to fp32 to avoid numpy error(numpy does not support bf16 yet)
                            # data = [[h, l] for h, l in zip(heads, lamb.detach().float().cpu().numpy().clip(min=0))]
                            data = [[h, l] for h, l in zip(heads, lamb.detach().float().cpu().numpy())]
                            table = wandb.Table(data=data, columns=["head", "lambda"])
                            wandb.log(
                                {
                                    f"lambda_{lamda_block_names[index]}": wandb.plot.bar(
                                        table, "head", "lambda", title=f"lambda_{lamda_block_names[index]}"
                                    )
                                },
                                commit=False,
                            )
                        img_continues_mask = save_image_seed(
                            pipe, validation_prompts, cfg.trainer.num_intervention_steps, device, seed, save_dir=None
                        )
                        img_discrete_mask = save_image_binarize_seed(
                            pipe,
                            [cross_attn_hooker, ff_hooker],
                            validation_prompts,
                            cfg.trainer.num_intervention_steps,
                            device,
                            seed,
                            save_dir=None,
                        )
                        wandb.log(
                            {
                                "image with continuous mask": [wandb.Image(i) for i in img_continues_mask],
                                "image with discrete mask": [wandb.Image(i) for i in img_discrete_mask],
                            }
                        )
                    else:
                        path = os.path.join(args.save_dir, cfg.logger.project, cfg.logger.notes, "images")
                        val_path = os.path.join(path, "validation", f"epoch_{i}_step_{idx}")
                        train_path = os.path.join(path, "train", f"epoch_{i}_step_{idx}")
                        for path, prompts in zip(
                            [val_path, train_path], [validation_prompts, train_dataset[0]["prompt"]]
                        ):
                            save_image_seed(
                                pipe,
                                prompts,
                                cfg.trainer.num_intervention_steps,
                                device,
                                seed,
                                save_dir=os.path.join(path, "continues mask"),
                            )
                            torch.cuda.empty_cache()
                            hookers = [cross_attn_hooker, ff_hooker]
                            if cfg.trainer.n_lr != 0:
                                hookers.append(norm_hooker)
                            save_image_binarize_seed(
                                pipe,
                                hookers,
                                prompts,
                                cfg.trainer.num_intervention_steps,
                                device,
                                seed,
                                save_dir=os.path.join(path, "discrete mask"),
                            )
                            torch.cuda.empty_cache()

                    # log attn head sparsity
                    for n, lamb in zip(lamda_block_names, cross_attn_hooker.lambs):
                        logger.info(f"lambda in {n}: {lamb.clamp(min=0).tolist()}")

                    # log ffn sparsity
                    for n, lamb in zip(ff_lambda_block_names, ff_hooker.lambs):
                        logger.info(
                            f"lambda {n}: max {lamb.max().item()}, min {lamb.min().item()}, mean {lamb.mean().item()}"
                        )
                    # log norm sparsity
                    if cfg.trainer.n_lr != 0:
                        for n, lamb in zip(norm_lambda_block_names, norm_hooker.lambs):
                            logger.info(
                                f"lambda {n}: max {lamb.max().item()}, "
                                + f"min {lamb.min().item()}, mean {lamb.mean().item()}"
                            )

                    masking_threshold = 0
                    remain_head, total_head, sparsity = calculate_mask_sparsity(cross_attn_hooker, masking_threshold)
                    ff_remain_head, ff_total_head, ff_sparsity = calculate_mask_sparsity(ff_hooker, masking_threshold)

                    if cfg.trainer.n_lr != 0:
                        norm_remain_head, norm_total_head, norm_sparsity = calculate_mask_sparsity(
                            norm_hooker, masking_threshold
                        )
                    else:
                        norm_remain_head, norm_total_head, norm_sparsity = 0, 0, 0

                    logger.info(
                        f"mask sparsity for threshold {masking_threshold}: "
                        f"{remain_head}/{total_head}, {sparsity:.2%} \n"
                        f"ff_mask sparsity for threshold {masking_threshold}: "
                        f"{ff_remain_head}/{ff_total_head}, {ff_sparsity:.2%} \n"
                        f"norm_mask sparsity for threshold {masking_threshold}: "
                        f"{norm_remain_head}/{norm_total_head}, {norm_sparsity:.2%} \n"
                    )
                    logger.info(f"loss_reconstruct: {loss_reconstruct}, loss_reg: {loss_reg}, total_loss: {loss}")

                    cross_attn_hooker.save(os.path.join("lambda", f"epoch_{i}_step_{idx}_attn.pt"))
                    ff_hooker.save(os.path.join("lambda", f"epoch_{i}_step_{idx}_ff.pt"))

                    if cfg.trainer.n_lr != 0:
                        norm_hooker.save(os.path.join("lambda", f"epoch_{i}_step_{idx}_norm.pt"))
                    logger.info(f"epoch: {i}, step: {idx}: saving lambda")
                pbar.update()

        logger.info(f"epoch {i+1}/{cfg.trainer.epochs}")

        hookers = [cross_attn_hooker, ff_hooker]
        if cfg.trainer.n_lr != 0:
            hookers.append(norm_hooker)

        if cfg.logger.type == "wandb":
            logger.info("Saving final image to wandb ...")
            img = save_image_binarize_seed(
                pipe,
                hookers,
                validation_prompts,
                cfg.trainer.num_intervention_steps,
                device,
                seed,
                save_dir=None,
            )
            wandb.log({"image": [wandb.Image(i) for i in img]})
            run.finish()
            wandb.finish()
            logger.info("Done")
        else:
            path = os.path.join(args.save_dir, cfg.logger.project, cfg.logger.notes, "images")
            train_path = os.path.join(path, "train", "final_image")
            val_path = os.path.join(path, "validation", "final_image")
            prompts = [validation_prompts, train_dataset[0]["prompt"]]
            for path, prompt in zip([val_path, train_path], prompts):
                save_image_binarize_seed(
                    pipe,
                    hookers,
                    prompt,
                    cfg.trainer.num_intervention_steps,
                    device,
                    seed,
                    save_dir=path,
                )
        # END LOGGING

    logger.info(f"Training finished with cfg:{args.cfg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train SDIB model, supporting SD1 SD2 SDXL SD3")
    parser.add_argument("--cfg", type=str, default="configs/sdxl.yaml", help="config file to load all parameters")
    parser.add_argument(
        "--validation_prompts_path",
        "-v",
        type=str,
        default="configs/validation_prompts_small.yaml",  # by default only plot one image to accelerate the process
        help="Path to validation prompts",
    )
    parser.add_argument("--save_dir", "-s", type=str, default="./results", help="Directory to save images")
    parser.add_argument("--notes", type=str, default="inital_run", help="Notes for wandb")
    parser.add_argument("--islaunch", action="store_true", help="Launch accelerator")
    parser.add_argument("--task", "-t", type=str, default="general", help="Task to perform, might need to change")
    parser.add_argument("--load_lambda", "-l", action="store_true", help="Load lambda from checkpoint")
    args = parser.parse_args()
    main(args)
