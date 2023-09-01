import os
os.system('cd fairseq;'
          'pip install ./; cd ..')
os.system('ls -l')
os.system('wget https://ofa-silicon.oss-us-west-1.aliyuncs.com/checkpoints/caption_large_best_clean.pt; '
          'mkdir -p checkpoints; mv caption_large_best_clean.pt checkpoints/caption.pt')
import torch
import numpy as np
import sqlite3
import gradio as gr
from PIL import Image
from torchvision import transforms
from googletrans import Translator
from fairseq import utils, tasks, checkpoint_utils
from fairseq.tasks.mm_tasks.caption import CaptionTask
from models.ofa import OFAModel
from utils.eval_utils import eval_step


# Supported languages for translation
supported_languages = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    # Add more languages as needed
}

# Register caption task
tasks.register_task('caption', CaptionTask)

# Check for GPU availability
use_cuda = torch.cuda.is_available()

# Use FP16 only when GPU is available
use_fp16 = False

# Constants
SIGNUP_SUCCESS_MSG = "Signup successful!"
SIGNUP_ERROR_EXISTING_USER = "Username already exists. Please choose a different one."
LOGIN_SUCCESS_MSG = "Login successful!"
LOGIN_ERROR_INVALID_CREDENTIALS = "Invalid username or password."

# Function to create a table for user authentication
def create_table():
    with sqlite3.connect("login.db") as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                email TEXT NOT NULL,
                role TEXT NOT NULL
            )
        ''')

# Function for user signup
def signup_interface(new_username, new_password, new_email):
    response = {}
    if not new_username or not new_password or not new_email:
        response["message"] = "All fields are required for signup."
        response["success"] = False
    else:
        role = "user"
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)",
                           (new_username, new_password, new_email, role))
            conn.commit()
            response["message"] = SIGNUP_SUCCESS_MSG
            response["success"] = True
        except sqlite3.IntegrityError:
            response["message"] = SIGNUP_ERROR_EXISTING_USER
            response["success"] = False
    return response

# Create a Gradio interface for user signup
signup_inputs = [
    gr.inputs.Textbox(label="New Username", type="text"),
    gr.inputs.Textbox(label="New Password", type="password"),
    gr.inputs.Textbox(label="Email", type="text"),
]
signup_output = gr.outputs.JSON()
signup_app = gr.Interface(
    fn=signup_interface,
    inputs=signup_inputs,
    outputs=signup_output,
    title="Signup",
    description="Enter your new username, password, and email to sign up.",
    allow_screenshot=False,
    allow_flagging=False,
)

# Function for user login
def login_interface(username, password):
    response = {}
    if not username or not password:
        response["message"] = "Username and password are required for login."
        response["success"] = False
    else:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()

            if user and user[2] == password:
                response["message"] = LOGIN_SUCCESS_MSG
                response["success"] = True
                response["username"] = user[1]
            else:
                response["message"] = LOGIN_ERROR_INVALID_CREDENTIALS
                response["success"] = False
        except sqlite3.OperationalError as e:
            response["message"] = f"An error occurred while trying to log in: {e}"
            response["success"] = False
    return response

# Create a Gradio interface for user login
login_inputs = [
    gr.inputs.Textbox(label="Username", type="text"),
    gr.inputs.Textbox(label="Password", type="password"),
]
login_output = gr.outputs.JSON()
login_app = gr.Interface(
    fn=login_interface,
    inputs=login_inputs,
    outputs=login_output,
    title="Login",
    description="Enter your username and password to log in.",
    allow_screenshot=False,
    allow_flagging=False,
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

# Constants for image preprocessing
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocessing
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

# Construct input for the caption task
def construct_sample(image: Image):
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

# Function to convert FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

# Function for image captioning
def image_caption(Image):
    sample = construct_sample(Image)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
    with torch.no_grad():
        result, scores = eval_step(task, generator, models, sample)
    generated_caption = result[0]['caption']

    # Translate the generated caption to the user-selected target language
    translated_caption = translate_caption(generated_caption, target_language)
    return translated_caption

# Function to translate a caption
def translate_caption(caption, target_language="en"):
    translated = translator.translate(caption, dest=target_language)
    return translated.text

# Gradio interface input component for language selection
language_selector = gr.inputs.Dropdown(choices=list(supported_languages.keys()), label="Target Language")
title = "OFA-Image_Caption"
description = "Image Caption. Upload your own image or click any one of the examples, and click " \
              "\"Submit\" and then wait for the generated caption.  "

io = gr.Interface(fn=image_caption, inputs=[gr.inputs.Image(type='pil'), language_selector],
                  outputs=gr.outputs.Textbox(label="Translated Caption"),
                  title=title, description=description,
                  allow_flagging=False, allow_screenshot=False)

tabs_to_apps = {
    "Login": login_app,
    "Signup": signup_app,
    "Generate Caption": io,
}

# Define a variable to track login status
is_login_successful = False

# Function to set login status when login is successful
def set_login_successful():
    global is_login_successful  # Declare that we're using the global variable
    is_login_successful = True

# Implement a function to check login success and call set_login_successful() when successful
def check_login(username, password):
    # Add your login logic here to check if the provided username and password are correct
    if username == "correct_username" and password == "correct_password":
        set_login_successful()
        print("Login successful")
    else:
        print("Login failed")

# Check login success by calling check_login with username and password
check_login("correct_username", "correct_password")

# Function to launch the selected app
def launch_selected_app(selected_tab):
    global is_login_successful  # Declare that we're using the global variable

    if selected_tab == "Login":
        login_app.launch()
    elif selected_tab == "Signup":
        signup_app.launch()
    elif selected_tab == "Generate Caption":
        if is_login_successful:  # Check if login is successful using the variable
            io.launch()
        else:
            print("Login is required for caption generation.")

# Define the main Gradio interface with navigation tabs
def main():
    gr.Interface(
        fn=launch_selected_app,
        inputs=gr.inputs.Textbox(label="selected_tab", type="text"),
        outputs="auto",
        title="Main App",
        description="Select a tab: 'Login', 'Signup', or 'Generate Caption'",
    ).launch()

if __name__ == "__main__":
    main()
