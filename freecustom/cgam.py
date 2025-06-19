import torch
import open_clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import to_pil_image

class CGAMController:
    def __init__(self, ref_images, vae, mrsa_layers, device, k=8, tau=0.07):
        self.ref_images = ref_images
        self.vae = vae
        self.device = device
        self.k = k
        self.tau = tau
        self.step = 0
        self.mrsa_layers = mrsa_layers

        # Load CLIP
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        self.clip_model = self.clip_model.to(device).eval()

        # Pre-encode reference image embeddings
        with torch.no_grad():
            self.ref_embs = [
                self.clip_model.encode_image(
                    self.preprocess(to_pil_image((img * 0.5 + 0.5).cpu().squeeze(0)))
                    .unsqueeze(0).to(device)
                )
                for img in self.ref_images
            ]

    def update(self, z_t):
        if self.step % self.k != 0:
            self.step += 1
            return

        with torch.no_grad():
            # decode and convert to PIL
            img = self.vae.decode(z_t / 0.18215).sample.clamp(0, 1)
            img_pil = to_pil_image(img[0].cpu())

            # embed and compare with reference
            pred_emb = self.clip_model.encode_image(
                self.preprocess(img_pil).unsqueeze(0).to(self.device)
            )
            sims = torch.stack([
                torch.nn.functional.cosine_similarity(pred_emb, ref_emb, dim=-1)
                for ref_emb in self.ref_embs
            ])
            sims = sims / sims.max()
            weights = torch.softmax(sims / self.tau, dim=0) * len(self.ref_images)
            float_weights = weights.detach().cpu().tolist()

            for m in self.mrsa_layers:
                m.mask_weights = float_weights  # ✅ list[float] not Tensor

            print(f"[CGAM] step {self.step}: weights = {float_weights}")

        self.step += 1