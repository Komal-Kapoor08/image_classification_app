import streamlit as st
import requests
import torch
from PIL import Image   
import torchvision.transforms as transforms
import torchvision.models as models




st.title("Image Classification App")

model = models.resnet18(pretrained=True)
model.eval()

LABELS_URL ="https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
label = [line.strip() for line in requests.get(LABELS_URL).text.splitlines()]

def imagepro(image):
    transform = transforms.Compose([
        transforms.Resize((224,244)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

upload = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if upload is not None:
    image = Image.open(upload).convert('RGB')
    st.image(image, caption="Uploaded Image")

    with st.spinner("Processing..."):
        i_tensor = imagepro(image)
        with torch.no_grad():
            outputs = model(i_tensor)
            a, predicted = torch.max(outputs, 1)
            label = label[predicted.item()]

    st.success(f"Predicted Items is: {label}")

    

