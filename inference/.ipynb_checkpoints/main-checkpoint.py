import cv2
from PIL import Image
import numpy as np
''''''
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

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
text_prompt = "图中有什么？是否存在隐藏的文字或图案？"

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
    # 根据图片尺寸动态调整缩放比例：
    # - 高分辨率图片 (>= 2048x2048): 缩小至1/8
    # - 中等分辨率 (1024x1024~2047x2047): 缩小至1/4
    # - 低分辨率 (<1024x1024): 不缩放或缩小至1/2
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
    对缩放后的图片进行以下增强操作：
    1. 灰度化 + Canny边缘检测：突出线条结构
    2. HSV颜色分割：分离特定颜色区域（如红色/蓝色水印）
    3. 直方图均衡化：提升对比度
    """
    # 转换为OpenCV格式
    img_cv = np.array(image.convert('RGB'))[:, :, ::-1].copy()  # RGB转BGR
    
    # 1. 边缘检测
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    
    # 2. 颜色分割（示例：提取红色区域）
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # 3. 直方图均衡化
    eq_hist = cv2.equalizeHist(gray)
    
    # 返回增强后的图像（可选edges/mask/eq_hist作为补充输入）
    enhanced_image = Image.fromarray(eq_hist)
    return enhanced_image

def zoom_out_and_infer(image_path, text_prompt):
    # 1. 加载并缩放图片
    image = Image.open(image_path)
    
    # original_size = image.size
    # # zoom_out_ratio = 32  # Zoom out by 1/zoom_out_ratio  # bingo!
    # zoom_out_ratio = 32  # 
    # thumbnail_size = (original_size[0] // zoom_out_ratio, original_size[1] // zoom_out_ratio)
    # zoomed_image = image.resize(thumbnail_size, resample=Image.BILINEAR)
    
    zoomed_image = dynamic_zoom(image)  # 自动适配缩放比例
    enhanced_image = enhance_features(image)  # 输入缩放后的图片
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
    
    # 2. 处理多模态输入
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
    
    # 3. 模型推理
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # 4. 构建增强的CoT描述
    cot_description = (
        f"图片原尺寸为{image.size}。"
        f"已将图片缩放至{edited_image.size}尺寸。"
        f"缩放后观察到：{output_text}"
    )
    
    return cot_description
''''''
''''''

# print(output_text)
print(f'Picture: {image_path}\nPrompt: {text_prompt}')
result = zoom_out_and_infer(image_path, text_prompt)
print(result)
