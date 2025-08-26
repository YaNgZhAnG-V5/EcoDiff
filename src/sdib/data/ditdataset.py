import os

import torch
import tqdm
from torch.utils.data import Dataset


class DiTDataset(Dataset):
    """
    Dataset for Diffusion Transformers (DiT) with class-conditional generation.
    
    This dataset generates images using DiT models with ImageNet class labels
    and stores both the generated images and intermediate tensor representations.
    
    Args:
        pipe: DiT pipeline for image generation
        save_dir (str): Directory to save generated images and tensors
        device (str): Device for computation (e.g., 'cuda:0')
        size (int): Number of samples to generate
        num_inference_steps (int, optional): Number of denoising steps. Defaults to 50.
        seed (int, optional): Random seed for reproducibility. Defaults to 44.
        grad_checkpointing (bool, optional): Enable gradient checkpointing. Defaults to True.
    """
    
    def __init__(
        self,
        pipe,
        save_dir,
        device,
        size,
        num_inference_steps=50,
        seed=44,
        grad_checkpointing=True,
    ):
        self.save_dir = save_dir
        self.pipe = pipe
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.df = None
        self.grad_checkpointing = grad_checkpointing
        self.seed = seed
        self.size = size

        # list placeholder
        self.prompt_name = []
        self.prompt_id = []
        self.images_path = []
        self.pt_path = []

        if not os.path.exists(save_dir):
            print(f"save_dir {save_dir} does not exist, creating the directory")
            os.makedirs(save_dir)
        self.prepare_data()

    @torch.no_grad()
    def image_generation(self, prompt, image_path, pt_path):
        g_cpu = torch.Generator(self.device).manual_seed(self.seed)
        preparation_phase_output = self.pipe.inference_preparation_phase(
            prompt,
            generator=g_cpu,
            num_inference_steps=self.num_inference_steps,
            output_type="latent",
        )
        intermediate_latents = [preparation_phase_output.latents]
        timesteps = preparation_phase_output.timesteps
        for timesteps_idx, time in enumerate(timesteps):
            latents = self.pipe.inference_with_grad_denoising_step(timesteps_idx, time, preparation_phase_output)
            # update latents in output class
            preparation_phase_output.latents = latents
            intermediate_latents.append(latents)
        # use the last latents for generating images
        g_cpu = torch.Generator(self.device).manual_seed(self.seed)
        image_tensor = self.pipe.inference_with_grad_aft_denoising(latents, None, g_cpu, "latent", True, self.device)
        tmp_image_tensor = image_tensor["images"][0]
        g_cpu = torch.Generator(self.device).manual_seed(self.seed)
        img = self.pipe.inference_with_grad_aft_denoising(latents, None, g_cpu, "pil", True, self.device)
        torch.save(tmp_image_tensor, pt_path)
        img["images"][0].save(image_path)

    def prepare_data(self):
        label = self.pipe.labels  # as dict with name: id
        with tqdm.tqdm(total=self.size) as pbar:
            for idx, (name, id) in enumerate(label.items()):
                if idx >= self.size:
                    break
                self.prompt_name.append(name)
                self.prompt_id.append([id])
                tmp_image_path = os.path.join(self.save_dir, f"{id}.png")
                tmp_pt_path = os.path.join(self.save_dir, f"{id}.pt")
                self.images_path.append(os.path.join(tmp_image_path))
                self.pt_path.append(os.path.join(tmp_pt_path))
                self.image_generation([id], tmp_image_path, tmp_pt_path)
                pbar.update()

    def __len__(self):
        return len(self.prompt_id)

    def __getitem__(self, index):
        example = {}
        example["image"] = torch.load(self.pt_path[index])
        example["prompt"] = self.prompt_id[index]
        return example


if __name__ == "__main__":
    from diffusers import DPMSolverMultistepScheduler

    from sdib.models import DiTIBPipeline

    pipe = DiTIBPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    pipe = pipe.to("cuda:0")

    ds = DiTDataset(
        pipe=pipe,
        save_dir="./datasets/dit",
        device="cuda:0",
        size=10,
        num_inference_steps=25,
        seed=33,
        grad_checkpointing=True,
    )

    ds.prepare_data()
