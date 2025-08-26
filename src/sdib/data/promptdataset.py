import os

import gdown
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset

METADICT = {"gcc": "https://drive.google.com/file/d/1VCWJ9YeLwqbT_TyvdV_aZWp0qkpHdEkz/view?usp=sharing"}


class PromptImageDataset(Dataset):
    """
    A dataset for generating prompted images from different metadata, such as gcc3m, gcc12m, yfcc, laion400m, etc.
    store the image in the tmp folder and return the path of the image.
    """

    def __init__(self, metadata, pipe, num_inference_steps, save_dir, seed, device, size=45, grad_checkpointing=True):
        self.metadata = metadata
        self.save_dir = save_dir
        self.size = size
        self.pipe = pipe
        self.seed = seed
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.df = None
        self.grad_checkpointing = grad_checkpointing
        self.validity_check()
        self.prepare_metadata()

    def validity_check(self):
        if not os.path.exists(self.save_dir):
            print(f"save_dir {self.save_dir} does not exist, creating the directory")
            os.makedirs(self.save_dir)

        if not os.path.exists(self.metadata):
            base_dir = os.path.dirname(self.metadata)
            if not os.path.exists(base_dir):
                os.mkdir(base_dir)
            print(f"save_dir {self.metadata} does not exist, downloading the meta data ...")
            if "gcc" in self.metadata:
                url = METADICT["gcc"]
            gdown.download(url, self.metadata, fuzzy=True)

    def prepare_metadata(self):
        # read tsv file
        df = pd.read_csv(self.metadata, sep="\t", header=None)
        self.df = df.iloc[:, 0][: self.size]

        # always overwrite the save_dir, need to find a way to avoid this
        ptpaths, imgpaths, idxlist = [], [], []
        for i in range(self.size):
            ptpaths.append(os.path.join(self.save_dir, f"{i}.pt"))
            imgpaths.append(os.path.join(self.save_dir, f"{i}.png"))
            idxlist.append(i)

        # save latent tensor
        print("Generating latent tensors and images ...")
        with tqdm.tqdm(total=len(self.df)) as pbar:
            for p, i, idx in zip(ptpaths, imgpaths, idxlist):
                if os.path.exists(p) and os.path.exists(i):
                    pbar.update()
                    continue
                
                if self.grad_checkpointing:
                    with torch.no_grad():
                        g_cpu = torch.Generator(self.device).manual_seed(self.seed)
                        preparation_phase_output = self.pipe.inference_preparation_phase(
                            self.df[int(idx)],
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
                else:
                    g_cpu = torch.Generator(self.device).manual_seed(self.seed)
                    image_tensor = self.pipe(
                        self.df[int(idx)],
                        generator=g_cpu,
                        num_inference_steps=self.num_inference_steps,
                        output_type="latent",
                        attn_res=(16, 16),
                    )
                    g_cpu = torch.Generator(self.device).manual_seed(self.seed)
                    tmp_image_tensor = image_tensor["images"][0]
                    img = self.pipe(
                        self.df[int(idx)],
                        generator=g_cpu,
                        num_inference_steps=self.num_inference_steps,
                        attn_res=(16, 16),
                    )
                torch.save(tmp_image_tensor, p)
                img["images"][0].save(i)

                pbar.update()

    def __len__(self):
        if self.df is None:
            return 0
        return len(self.df)

    def __getitem__(self, idx):
        if self.df is None:
            raise ValueError("metadata is not prepared")
        example = {}
        example["image"] = torch.load(os.path.join(self.save_dir, f"{idx}.pt"), weights_only=False)
        example["prompt"] = self.df[idx]
        return example


if __name__ == "__main__":
    metadata = "./datasets/gcc3m/Validation_GCC-1.1.0-Validation.tsv"
    save_dir = "./datasets/gcc3m"

    from diffusers import EulerDiscreteScheduler

    from sdib.models.sdib_pipeline import SDIBDiffusionPipeline

    # make pipe more automatic, put it in src
    model_id = "stabilityai/stable-diffusion-2-base"
    # Use the Euler scheduler here instead
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = SDIBDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda:0")
    g_cpu = torch.Generator("cuda:0").manual_seed(44)
    ds = PromptImageDataset(
        metadata, pipe=pipe, num_inference_steps=10, save_dir=save_dir, device="cuda:0", seed=44, size=8
    )
    for idx in range(len(ds)):
        print(ds[idx]["prompt"])
