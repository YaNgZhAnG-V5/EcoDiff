import os

import torch
import tqdm
from torch.utils.data import Dataset

from sdib.data.prompt_template import DECONCEPT_DATASET_TEMPLATES


class DeConceptDataset(Dataset):
    def __init__(
        self,
        pipe,
        save_dir,
        device,
        prompts=DECONCEPT_DATASET_TEMPLATES,
        concept: str = "dog",
        num_inference_steps=50,
        seed=44,
        grad_checkpointing=True,
    ):
        self.concept = concept
        self.save_dir = save_dir
        self.pipe = pipe
        self.device = device
        self.prompts = prompts
        self.num_inference_steps = num_inference_steps
        self.df = None
        self.grad_checkpointing = grad_checkpointing
        self.seed = seed

        # list placeholder
        self.with_concept_ptpaths = []
        self.no_concept_ptpaths = []
        self.with_concept_imgpaths = []
        self.no_concept_imgpaths = []
        self.prompts_with_concept = []
        self.prompts_without_concept = []

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
        prompt_embeds = preparation_phase_output.prompt_embeds
        g_cpu = torch.Generator(self.device).manual_seed(self.seed)
        image_tensor = self.pipe.inference_with_grad_aft_denoising(
            latents, prompt_embeds, g_cpu, "latent", True, self.device
        )
        tmp_image_tensor = image_tensor["images"][0]
        g_cpu = torch.Generator(self.device).manual_seed(self.seed)
        img = self.pipe.inference_with_grad_aft_denoising(latents, prompt_embeds, g_cpu, "pil", True, self.device)
        torch.save(tmp_image_tensor, pt_path)
        img["images"][0].save(image_path)

    def prepare_data(self):
        "use the same concepts with different prompts to generate images with same seed"
        for prompt in self.prompts:
            prompt_with_concept = prompt(self.concept)
            prompt_without_concept = prompt("")
            self.prompts_with_concept.append(prompt_with_concept)
            self.prompts_without_concept.append(prompt_without_concept)
            self.with_concept_ptpaths.append(os.path.join(self.save_dir, f"{prompt_with_concept}_{self.seed}.pt"))
            self.no_concept_ptpaths.append(os.path.join(self.save_dir, f"{prompt_without_concept}_{self.seed}.pt"))
            self.with_concept_imgpaths.append(os.path.join(self.save_dir, f"{prompt_with_concept}_{self.seed}.png"))
            self.no_concept_imgpaths.append(os.path.join(self.save_dir, f"{prompt_without_concept}_{self.seed}.png"))

        total_length = len(self.prompts_with_concept) * 2
        with tqdm.tqdm(total=total_length) as pbar:
            for prompt, pt_path, img_path in zip(
                self.prompts_with_concept, self.with_concept_ptpaths, self.with_concept_imgpaths
            ):
                self.image_generation(prompt, img_path, pt_path)
                pbar.update()

            for prompt, pt_path, img_path in zip(
                self.prompts_without_concept, self.no_concept_ptpaths, self.no_concept_imgpaths
            ):
                self.image_generation(prompt, img_path, pt_path)
                pbar.update()

    def __len__(self):
        return len(self.prompts_with_concept)

    def __getitem__(self, index):
        """
        for deconcept dataset,
        need to return
        image w/o concept
        prompt w concept

        aim to generate image w/o concept from prompt w concept
        """
        example = {}
        example["image"] = torch.load(self.no_concept_ptpaths[index])
        example["prompt"] = self.prompts_with_concept[index]
        return example


class ConceptDataset(Dataset):
    """
    Dataset for generating prompted images from different concepts
    using different seeds for generating different concepts images.

    params:
        prompt: str, the concepts for generating images, e.g. dog, cat, etc.
        seed: int, the starting seed for generating images, 44, 45, 46, etc.
    """

    def __init__(self, prompt, pipe, num_inference_steps, save_dir, device, seed=44, size=20, grad_checkpointing=True):
        self.prompt = prompt
        self.save_dir = save_dir
        self.size = size
        self.pipe = pipe
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.df = None
        self.grad_checkpointing = grad_checkpointing
        self.seed = seed

        if not os.path.exists(save_dir):
            print(f"save_dir {save_dir} does not exist, creating the directory")
            os.makedirs(save_dir)

        self.prepare_data()

    def prepare_data(self):
        "use the same prompts with different seeds to generate images"
        ptpaths, imgpaths, seed_list = [], [], []
        for i in range(self.size):
            cur_seed = self.seed + i
            ptpath = os.path.join(self.save_dir, f"{self.prompt}_{cur_seed}.pt")
            imgpath = os.path.join(self.save_dir, f"{self.prompt}_{cur_seed}.png")
            ptpaths.append(ptpath)
            imgpaths.append(imgpath)
            seed_list.append(cur_seed)

        # save latent tensor / images
        with tqdm.tqdm(total=self.size) as pbar:
            for seed, pp, ip in zip(seed_list, ptpaths, imgpaths):
                with torch.no_grad():
                    g_cpu = torch.Generator(self.device).manual_seed(seed)
                    preparation_phase_output = self.pipe.inference_preparation_phase(
                        f"a photo of a {self.prompt}",
                        generator=g_cpu,
                        num_inference_steps=self.num_inference_steps,
                        output_type="latent",
                    )
                    intermediate_latents = [preparation_phase_output.latents]
                    timesteps = preparation_phase_output.timesteps
                    for timesteps_idx, time in enumerate(timesteps):
                        latents = self.pipe.inference_with_grad_denoising_step(
                            timesteps_idx, time, preparation_phase_output
                        )
                        # update latents in output class
                        preparation_phase_output.latents = latents
                        intermediate_latents.append(latents)
                    # use the last latents for generating images
                    prompt_embeds = preparation_phase_output.prompt_embeds
                    g_cpu = torch.Generator(self.device).manual_seed(self.seed)
                    image_tensor = self.pipe.inference_with_grad_aft_denoising(
                        latents, prompt_embeds, g_cpu, "latent", True, self.device
                    )
                    tmp_image_tensor = image_tensor["images"][0]
                    g_cpu = torch.Generator(self.device).manual_seed(self.seed)
                    img = self.pipe.inference_with_grad_aft_denoising(
                        latents, prompt_embeds, g_cpu, "pil", True, self.device
                    )
                torch.save(tmp_image_tensor, pp)
                img["images"][0].save(ip)
                pbar.update()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        example = {}
        example["image"] = torch.load(os.path.join(self.save_dir, f"{self.prompt}_{self.seed + index}.pt"))
        example["prompt"] = f"a photo of {self.prompt}"
        example["seed"] = self.seed + index
        return example


if __name__ == "__main__":
    from sdib.models import SDXLDiffusionPipeline

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = SDXLDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda:0")

    concept = False
    if concept:
        ds = ConceptDataset(
            prompt="dog",
            pipe=pipe,
            num_inference_steps=50,
            save_dir="./datasets/concepts_dog",
            seed=44,
            device="cuda:0",
            size=40,
        )
        ds.prepare_data()
    else:
        ds = DeConceptDataset(
            pipe=pipe,
            save_dir="./datasets/deconcepts_dog",
            device="cuda:0",
            prompts=DECONCEPT_DATASET_TEMPLATES,
            concept="and, a dog",
            num_inference_steps=50,
            seed=44,
            grad_checkpointing=True,
        )
