import cv2
from PIL import Image
import numpy as np
''''''
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# One example: Qwen
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained(model_name)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

image_path = "./img/to.jpg"
text_prompt = "What is within the image? Does it have any hidden texts or image content?"

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": f"{image_path}",
#             },
#             {"type": "text", "text": f"{text_prompt}"},
#         ],
#     }
# ]

# Preparation for inference
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# inputs = inputs.to("cuda")

# # Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )

''''''
''''''

def dynamic_zoom(image):  # , base_size=512
    # """
    # Dynamically adjust the scaling ratio based on image size:
    # - High resolution (>= 2048x2048): downscale to 1/8
    # - Medium resolution (1024x1024 ~ 2047x2047): downscale to 1/4
    # - Low resolution (< 1024x1024): no scaling or downscale to 1/2
    # """
    # factor = 32
    # width, height = image.size
    # if width >= 2048 and height >= 2048:
    #     scale = 64 // factor
    # elif 1024 <= width < 2048 or 1024 <= height < 2048:
    #     scale = 32 // factor
    # elif 512 <= width < 1024 or 512 <= height < 1024:
    #     scale = 16 // factor
    # else:
    #     scale = 1
    """
    We find Rou_{model} is a constant in the specific model.
    Scale should be width/Rou
    """
    rou = 40
    width, height = image.size
    t_ht = int(float(height) / width * rou)
    if t_ht <= 0: t_ht = 1
    return image.resize((rou, t_ht), resample=Image.BILINEAR)
    # return image.resize((width // scale, height // scale), resample=Image.BILINEAR)

def enhance_features(image):
    """
    Perform the following enhancement operations on the scaled image:
    1. Grayscale conversion + Canny edge detection: highlight line structures
    2. HSV color segmentation: isolate specific color regions (e.g., red/blue watermarks)
    3. Histogram equalization: enhance contrast
    """
    # Convert to OpenCV format
    img_cv = np.array(image.convert('RGB'))[:, :, ::-1].copy()  # RGB转BGR
    
    # 1. Edge detection
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    
    # 2. Color segmentation (example: extract red regions)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # 3. Histogram equalization
    eq_hist = cv2.equalizeHist(gray)
    
    # Return the enhanced image (optional: edges/mask/eq_hist as supplementary inputs)
    enhanced_image = Image.fromarray(eq_hist)
    return enhanced_image

def zoom_out_and_infer(image_path, text_prompt):
    # 1. Load and scale the image
    image = Image.open(image_path)
    
    # original_size = image.size
    # # zoom_out_ratio = 32  # Zoom out by 1/zoom_out_ratio  # bingo!
    # zoom_out_ratio = 32  # 
    # thumbnail_size = (original_size[0] // zoom_out_ratio, original_size[1] // zoom_out_ratio)
    # zoomed_image = image.resize(thumbnail_size, resample=Image.BILINEAR)
    
    zoomed_image = dynamic_zoom(image)  # Automatically adapt the scaling ratio
    enhanced_image = enhance_features(image)  # Input the scaled image
    edited_image = zoomed_image
    # edited_image = image
    # edited_image = enhanced_image
    # edited_image = enhanced_image.resize(thumbnail_size, resample=Image.BILINEAR)

    # 1.5. New messages
    new_msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    # "image": f"{image_path}",
                    "image": edited_image,
                },
                {"type": "text", "text": f"{text_prompt}"},
            ],
        }
    ]
    
    # 2. Process multimodal input
    text = processor.apply_chat_template(
        new_msgs, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(new_msgs)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # 3. Model inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # 4. Construct enhanced CoT description
    cot_description = (
        f"The original image size is {image.size}."
        f"The image has been resized to {edited_image.size}."
        f"After scaling, the following observations were made: {output_text}"
    )
    
    return cot_description
''''''
''''''

# print(output_text)
print(f'Picture: {image_path}\nPrompt: {text_prompt}')
result = zoom_out_and_infer(image_path, text_prompt)
print(result)
