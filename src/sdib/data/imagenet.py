import glob
import os

import torch
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from sdib.data.imagenet_labels import IMAGENET_LABEL


class ImageNetDataset(Dataset):
    """
    a dataset for evaluation the semantic similarity of the masked generated image
    and the original image from sdxl/sd3 model
    """

    def __init__(
        self,
        save_dir,
        device,
        pipe,
        num_inference_steps,
        dataset_name="imagenet",
        task="",
        seed: list = [44],
        resize: int = 224,
        cropsize: int = 224,
        is_transform: bool = True,
        batch_size: int = 1,
    ):
        self.save_dir = os.path.join(save_dir, dataset_name)
        if task:
            self.save_dir = os.path.join(self.save_dir, task)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.imagenet_labels = IMAGENET_LABEL
        self.device = device
        self.seed = seed
        self.pipe = pipe
        self.dataset_name = dataset_name
        self.num_inference_steps = num_inference_steps
        self.is_transform = is_transform
        self.batch_size = batch_size
        self.image_path = []
        self.labels = []
        self.labels_text = []

        # define transform
        self.transform = T.Compose(
            [
                T.Resize(resize, InterpolationMode.BICUBIC),
                T.CenterCrop(cropsize),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]
        )

        self.base_transform = T.Compose(
            [T.Resize(resize, InterpolationMode.BICUBIC), T.CenterCrop(cropsize), T.ToTensor()]
        )

    def load_data(self):
        print("Loading synthetic images ...")
        if not os.path.exists(self.save_dir):
            raise ValueError(f"Directory {self.save_dir} does not exist")

        # load image path and labels
        self.image_path = glob.glob(os.path.join(self.save_dir, "*.png"))
        self.image_path.sort()
        # path format <seed>_<label>.png
        self.labels = [int(os.path.basename(p).split("_")[-1][:-4]) for p in self.image_path]

    def prepare_data(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        for seed in self.seed:
            for idx, label in self.imagenet_labels.items():
                path = os.path.join(self.save_dir, f"{seed}_{idx}.png")
                self.image_path.append(path)
                self.labels_text.append(label)

        print("Generating synthetic images ...")
        with tqdm.tqdm(total=len(self.image_path) // self.batch_size) as pbar:
            for p, l in zip(self.image_path, self.labels_text):
                seed = int(os.path.basename(p).split("_")[0])
                with torch.no_grad():
                    g_cpu = torch.Generator(self.device).manual_seed(seed)
                    preparation_phase_output = self.pipe.inference_preparation_phase(
                        l,
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
                    img = self.pipe.inference_with_grad_aft_denoising(
                        latents, prompt_embeds, g_cpu, "pil", True, self.device
                    )
                img["images"][0].save(p)
                pbar.update()
        self.load_data()

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img = Image.open(self.image_path[idx])
        if self.is_transform:
            img = self.transform(img)
        else:
            img = self.base_transform(img)
        return img, self.labels[idx]


if __name__ == "__main__":
    from sdib.models import SDXLDiffusionPipeline
    from sdib.utils import create_pipeline

    device = "cuda:0"
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = SDXLDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)

    save_dir = "datasets"
    ds = ImageNetDataset(
        save_dir=save_dir, device=device, pipe=pipe, num_inference_steps=50, seed=[44, 45, 46, 47], batch_size=16
    )
    ds.prepare_data()

    save_pth = "./results/prune_results/latest_lambda.pt"
    pipe = create_pipeline("sdxl", device, torch.bfloat16, save_pt=save_pth)
    masked_ds = ImageNetDataset(
        save_dir=save_dir, device=device, dataset_name="masked_imagenet", pipe=pipe, num_inference_steps=50
    )
