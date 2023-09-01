import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import streamlit as st
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.caption import CaptionTask
from models.ofa import OFAModel

# Register caption task
tasks.register_task('caption', CaptionTask)

# Turn on CUDA if GPU is available
use_cuda = torch.cuda.is_available()
# Use fp16 only when GPU is available
use_fp16 = False

# Download the pretrained checkpoint
checkpoint_url = "https://ofa-silicon.oss-us-west-1.aliyuncs.com/checkpoints/caption_large_best_clean.pt"
checkpoint_path = "checkpoints/caption.pt"

if not os.path.isfile(checkpoint_path):
    st.warning("Downloading the checkpoint...")
    os.system(f"wget {checkpoint_url}")
    os.system(f"mkdir -p checkpoints")
    os.system(f"mv caption_large_best_clean.pt {checkpoint_path}")
    st.success("Checkpoint downloaded successfully!")

# Load pretrained checkpoint & config
overrides = {"bpe_dir": "utils/BPE", "eval_cider": False, "beam": 5,
             "max_len_b": 16, "no_repeat_ngram_size": 3, "seed": 7}
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    utils.split_paths(checkpoint_path),
    arg_overrides=overrides
)

# Move models to GPU
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

# Image preprocessing
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()


def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s


# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


# Function for image captioning
def image_caption(image):
    st.image(image, caption="Uploaded Image", use_column_width=True)
    sample = construct_sample(image)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
    with torch.no_grad():
        result, scores = eval_step(task, generator, models, sample)
    return result[0]['caption']


def construct_sample(image):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(" what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id": np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask
        }
    }
    return sample


st.title("OFA-Image_Caption Streamlit App")
st.write("Upload an image, and the app will generate a caption for it.")
uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

if uploaded_image:
    caption = image_caption(uploaded_image)
    st.write("Generated Caption:")
    st.write(caption)
