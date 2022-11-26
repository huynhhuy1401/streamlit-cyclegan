import streamlit as st

import torch
import torchvision.transforms as transforms

from models import define_G
from utils import save_image, tensor2im

import numpy as np

from PIL import Image

st.header("Demo CycleGAN")
st.write("Choose any image and choose correspond mode to get the transformed image")

mode = st.radio(
    "Choose mode:",
    ('Not Blond To Blond', 'Blond to Not blond'))


uploaded_file = st.file_uploader("Choose an image with black hair")
device =  torch.device("cpu")

gB = define_G(3, 3, 64, "resnet_9blocks", norm="instance")
model_file_path = "latest_net_G_B.pth"
state_dict = torch.load(model_file_path, map_location=device)
gB.load_state_dict(state_dict)


gA = define_G(3, 3, 64, "resnet_9blocks", norm="instance")
model_file_path = "latest_net_G_A.pth"
state_dict = torch.load(model_file_path, map_location=device)
gA.load_state_dict(state_dict)


if uploaded_file is not None:

     #src_image = load_image(uploaded_file)
     image = Image.open(uploaded_file)	

     trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
     if mode == 'Not Blond To Blond':
          transformed = gB(trans(image).unsqueeze(0))
     else:
          transformed = gA(trans(image).unsqueeze(0))
     out_img = tensor2im(transformed)
     save_image(out_img, "out.jpg")


     col1, col2 = st.columns(2)
     with col1:

          st.image(image, caption='Original', use_column_width='always')
     with col2:
          st.image(out_img, caption='Transformed', use_column_width='always')