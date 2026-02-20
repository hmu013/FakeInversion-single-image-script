import torch
import albumentations as A
import base64
import io
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models

CHECKPOINT_PATH = "FakeInversion-single-image-script/weights/resnet_wResiduals_epoch_4.pt"

USE_RESIDUALS = True

#set for better performance
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16

#to test images as base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
        base64_bytes = base64.b64encode(binary_data)
        base64_string = base64_bytes.decode('utf-8')
    return base64_string

#---------------------Helpers-------------------------------------

def img_to_latents(images: torch.Tensor, vae: AutoencoderKL):
    images = 2 * images - 1       # to go from [0,1] to [-1,1]
    posterior = vae.encode(images).latent_dist
    latents = posterior.mean * vae.config.scaling_factor
    return latents

image_transform = A.Compose([
    A.SmallestMaxSize(512), 
    A.CenterCrop(512, 512),
])  # same methode as in fakeinversion rezise and keeps aspectratio

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) #resize input for classifier and normalize

def build_input_single_image(org, inv, rec):
    if USE_RESIDUALS:
        residual = torch.abs(org - rec)
        return torch.cat([org, inv, rec, residual], dim=1)
    else:
        return torch.cat([org, inv, rec], dim=1)

def base64_to_tensor(b64_str):
    img_data = base64.b64decode(b64_str)
    image = Image.open(io.BytesIO(img_data)).convert("RGB")
    
    image_np = np.array(image) #make np array and apply albumentations transform
    transformed = image_transform(image=image_np)["image"]

    tensor = torch.from_numpy(transformed).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)



#-------------------- blip2 aptioning model ------------------------------------

blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

blip2_model.to(device)

#------------------- sd 1.5 feature extraction pipe ---------------------------

inverse_scheduler = DDIMInverseScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='scheduler')
forward_scheduler = DDIMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder= 'scheduler')

pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                scheduler=inverse_scheduler, #start with inverse sheduler, gets swaped in loop
                                                safety_checker=None,
                                                torch_dtype=dtype)
pipe.to(device)

#channels_last should help speed by using a more efficent format (only heps when processing batches but left here anyway)
pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)

#sett pipe to eval mode infecrence only
pipe.vae.eval()
pipe.text_encoder.eval()
vae = pipe.vae

#----------------------------- Resnet detector ---------------------------------------------
# We modify resnet model to take 3 or 4 input images depending on if you include residual 
# i hav added residuals to the as a fourth feature to the FakeInversion mode - slightly better performace SynRIS (dataet introduced in FakeInversion paper)

NUM_IMAGES = 4 if USE_RESIDUALS else 3
IN_CHANNELS = 3 * NUM_IMAGES

detector = models.resnet50(weights=None, norm_layer=nn.InstanceNorm2d)
detector.conv1 = nn.Conv2d(IN_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
detector.fc = nn.Linear(2048, 1)
detector.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
detector.to(device)
detector.eval()

#------------------------------ feature extraction + classification -----------------------------------------------

image = image_to_base64("FakeInversion-single-image-script/test_images/fake_01.png")

image_tensor = base64_to_tensor(image)

with torch.inference_mode():
    
    #-----------------------captioning ----------------------------------------
    blip_input = blip_processor(image_tensor, return_tensors="pt").to(device, torch.float16)
    image_caption_embedding = blip2_model.generate(**blip_input, max_new_tokens=20)
    caption = blip_processor.batch_decode(image_caption_embedding, skip_special_tokens=True)[0].strip()

    #-------------------------- Feature extraxtion---------------------------
    gpu_image_tensor = image_tensor.to(device=device, dtype=vae.dtype, memory_format=torch.channels_last)

    #make batch of images into batch of latents
    latents = img_to_latents(gpu_image_tensor, vae=vae)

    #calculate the inverted latents
    inv_latents, _ = pipe(prompt = caption, 
                        negative_prompt="", 
                        guidance_scale=1.,
                        num_inference_steps=50, 
                        return_dict=False,
                        latents=latents,
                        output_type='latent')
    
    #set pipe to use forward scheduler
    pipe.scheduler = forward_scheduler
    
    #reconstruct img from inv latent 
    reconstructed_tensor = pipe(prompt=caption, 
                                negative_prompt="", 
                                guidance_scale=1.,
                                num_inference_steps=50,
                                latents=inv_latents,
                                output_type="pt").images

    #reset to inverse shedueler again
    pipe.scheduler = inverse_scheduler

    #decode inverted latent to tensor for classification
    inv_img_tensor = vae.decode(inv_latents / vae.config.scaling_factor).sample
    inv_img_tensor = (inv_img_tensor / 2 + 0.5).clamp(0, 1) # Range [0, 1]

    #-------------------------classification----------------------------------
    org_proc = resnet_transform(gpu_image_tensor.to(dtype=torch.float32))
    inv_proc = resnet_transform(inv_img_tensor.to(device, dtype=torch.float32))
    rec_proc = resnet_transform(reconstructed_tensor.to(device, dtype=torch.float32))
    
    x = build_input_single_image(org_proc,inv_proc,rec_proc)

    logit = detector(x).squeeze(1)
    prob = torch.sigmoid(logit).item()

predicted_class = 1 if prob >= 0.5 else 0

print(f"Prediction: {predicted_class} | Probability: {prob:.4f}")