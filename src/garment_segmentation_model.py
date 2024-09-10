from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import requests
from io import BytesIO
import base64

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
model_path = "segformer/"  # Replace with the actual local path
processor = SegformerImageProcessor.from_pretrained(model_path)
model = AutoModelForSemanticSegmentation.from_pretrained(model_path).to(device)
print("using device :", device)

def resize_image(image, max_size=1024):
    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image
    else:
        return image

def segment_image(image_url, padding_pixels=100):
    # Load image from local path
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")  # Convert image to RGB format
        image = resize_image(image)  # Resize image if necessary
    except IOError as e:
        print(f"Error opening image: {e}")
        return None
    
    # Process image and predict
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    # Upsample output logits to match the size of the input image
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False
    )

    # Get the predicted segmentation
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

    # Define labels and groups
    labels = ["Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress",
              "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm",
              "Bag", "Scarf"]
    gray_labels = {"Hair", "Left-arm", "Right-arm", "Face", "Left-leg", "Right-leg"}
    content_labels = {"Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress", "Bag", "Scarf", "Belt", "Left-shoe", "Right-shoe"}

    # Create an output image initialized to white
    output_image = np.full((pred_seg.shape[0], pred_seg.shape[1], 3), fill_value=255, dtype=np.uint8)

    # Process each label
    for i, label in enumerate(labels):
        mask = (pred_seg == i)
        if label in gray_labels:
            output_image[mask] = [128, 128, 128]
        elif label in content_labels:
            output_image[mask] = np.array(image)[mask]

    # Find the bounding box of the content region
    content_mask = np.isin(pred_seg, [labels.index(label) for label in content_labels])
    rows, cols = np.where(content_mask)
    if len(rows) > 0 and len(cols) > 0:
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)

        # Crop the image based on the content region bounding box
        cropped_image = output_image[min_row:max_row+1, min_col:max_col+1]

        # Fill the gray mask region inside the bounding box with the actual content
        gray_mask = np.isin(pred_seg[min_row:max_row+1, min_col:max_col+1], [labels.index(label) for label in gray_labels])
        cropped_image[gray_mask] = np.array(image)[min_row:max_row+1, min_col:max_col+1][gray_mask]

        # Create a larger white canvas for padding
        total_width = cropped_image.shape[1] + 2 * padding_pixels
        total_height = cropped_image.shape[0] + 2 * padding_pixels
        final_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))

        # Calculate the position to paste the cropped image on the canvas
        paste_position = (padding_pixels, padding_pixels)
        final_image.paste(Image.fromarray(cropped_image), paste_position)
    else:
        final_image = Image.fromarray(output_image)

    
    # Save the final image to the specified output path
    # Convert the final image to base64
    buffered = BytesIO()
    final_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

if __name__ == "__main__":
    image_url = "https://voilastudio.in/old_website_assets/images/VMM11/2520.webp"
    final_image = segment_image(image_url)
    print(final_image)