# Copyright 2024 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");

# 安装必要的库

# 导入库
import io
import json
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, Image
from datasets import load_dataset
from tqdm import tqdm
import torch
from PIL import Image as PILImage
import argparse

# 工具函数
def pil_image_to_bytes(img, format='PNG'):
  if not img:
    return None
  img_byte_arr = io.BytesIO()
  img.save(img_byte_arr, format=format)
  return img_byte_arr.getvalue()


def get_multi_image_question_parts(record, question_appendix, max_num_images=6):
  """Converts the question into multiple parts corresponding to texts/images.

  Args:
    record: This is a dictionary with a key "question" containing the text of
            the question, and multiple keys named "image_i" corresponding to
            the i-th image.
    question_appendix: A text to be appended at the end of the question.
    max_num_images: Maximum number of images in a problem (6 for ReMI).

  Returns:
    A list of vertexai parts where each part is either a piece of text or an
    image.
  """

  parts, images = [], []
  for i in range(max_num_images):
    img_bytes = pil_image_to_bytes(record[f'image_{i+1}'])
    if img_bytes:
      images.append(Part.from_data(img_bytes, mime_type='image/png'))

  question = 'Question: ' + record['question'] + question_appendix
  for j, img in enumerate(images):
    question_parts = question.split(f'<image{j+1}>')
    parts += [question_parts[0], img] if question_parts[0] else [img]
    question = question_parts[1]
  return parts + [question] if question else []

# 评估函数
def strip_json(s):
  """Takes a text possibly containing a json, and extrcts the json part."""
  return s[s.index('{') if '{' in s else 0: s.rindex('}') + 1 if '}' in s else 0]


def is_float(x):
  try: float(x); return True
  except (ValueError, TypeError): return False


def exact_match(label, pred, eps=0.01):
  return (
      str(label) == str(pred) or
      (is_float(label) and is_float(pred) and float(label) == float(pred)) or
      ((pred == 'f(x)' and label == 'f') or (pred == 'g(x)' and label == 'g') or (pred == 'h(x)' and label == 'h')) or
      (f'({pred})' == label or f'({label})' == pred) or
      (str(pred).replace(' ', '') == str(label) or str(label).replace(' ', '') == str(pred)) or
      relaxed_accuracy(label, pred, eps=eps))

def relaxed_accuracy(label, pred, eps=0.03):
  if not is_float(label) or not is_float(pred): return False
  return (1-eps) * float(label) <= float(pred) <= (1+eps) * float(label)

def accuracy_with_tolerance(label, pred, tolerance=10):
  if not is_float(label) or not is_float(pred): return False
  return float(label) - tolerance <= float(pred) <= float(label) + tolerance

def get_pred(model_response):
  model_response_json = strip_json(model_response).replace('\\"', '').replace('\\', '')
  try:
    pred = str(json.loads(model_response_json)['answer']).lower()
    return pred.split('%')[0].strip()
  except (KeyError, json.JSONDecodeError):
    return 'BAD_JSON'

def prep_label(label):
  return label.lower().replace('\\', '').split('%')[0].strip()

def evaluate(task, labels, model_responses):
  correct = 0
  for orig_label, model_response in zip(labels, model_responses):
    pred, label = get_pred(model_response), prep_label(orig_label)
    if task == 'RefCoco':
      correct += 1 if str(pred) in label.split(',') else 0  # whether pred is in label
    elif task in ['GeomShape', 'GeomCost']:
      correct += 1 if relaxed_accuracy(label, pred, eps=0.03) else 0
    elif task == 'Clocks':
      correct += 1 if accuracy_with_tolerance(label, pred, tolerance=10) else 0
    else:
      correct += 1 if exact_match(label, pred) else 0
  return correct / len(labels)

# 准备问题附加信息
QUESTION_APPENDICES = {
    'Collisions': ' Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the numeric value corresponding to your final answer. If it is a yes or no question, the answer field must be 0 for no and 1 for yes.',
    'Clocks': ' Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the numeric value corresponding to your final answer.',
    'Schedule': ' Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains a string corresponding to your final answer.',
    'EmojiAlgebra': ' Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the numeric value corresponding to your final answer.',
    'Charts': ' Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains a string or numerical value corresponding to your final answer.',
    'CodeEdit': ' Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the line of code corresponding to your final answer.',
    'GeomShape': ' Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the numeric value corresponding to your final answer.',
    'GeomCost': ' Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the numeric value corresponding to your final answer.',
    'FuncRead': ' Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains a string or numeric value corresponding to your final answer.',
    'RefCoco': ' Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the numeric value corresponding to your final answer.',
    'IQ': ' Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains the a string corresponding to your final choice.',
    'Isomorphism': ' Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field must be 1 if the two graphs are isomorphic and 0 otherwise.',
    'Maps': ' Output only a valid JSON string with two fields: "explanation" and "answer". Do not output anything else. The explanation field contains your reasoning. The answer field contains a string corresponding to your final answer.'}

# 准备数据
def prepare_data():
    prompts, labels = {}, {}
    for example in tqdm(load_dataset("mehrankazemi/ReMI")['test']):
        task = example['task']
        prompts[task] = prompts.get(task, []) + [get_multi_image_question_parts(example, QUESTION_APPENDICES[task])]
        labels[task] = labels.get(task, []) + [example['label']]
    return prompts, labels

# 初始化Qwen模型
def init_qwen_model(model_path=None):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch
    
    model_id = model_path if model_path else "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, max_pixels=2048*28*28)
    print(f"加载模型：{model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    return model, processor

# Qwen模型推理函数
def qwen_multimodal_call(prompt_parts, qwen_model, qwen_processor):
    """对Qwen模型进行调用"""
    # 解析prompt_parts为Qwen接受的格式
    messages = [{"role": "user", "content": []}]
    
    for part in prompt_parts:
        if isinstance(part, str):
            messages[0]["content"].append({"type": "text", "text": part})
        else:  # 这是一个图像Part
            img_bytes = part.inline_data.data
            img = PILImage.open(io.BytesIO(img_bytes))
            messages[0]["content"].append({"type": "image", "image": img})
    
    # 构建Qwen模型的输入
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs = [content["image"] for content in messages[0]["content"] 
                   if content.get("type") == "image"]
    
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    )
    
    inputs = inputs.to(qwen_model.device)
    
    # 生成回复
    with torch.no_grad():  # 禁用梯度计算以节省内存
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
    
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    torch.cuda.empty_cache()  # 清理缓存
    return output_text

# 主函数
def main():
    parser = argparse.ArgumentParser(description="Qwen模型推理")
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="模型路径，可以根据需要修改")
    args = parser.parse_args()

    print("开始准备数据...")
    prompts, labels = prepare_data()
    
    print("初始化Qwen模型...")
    model_path = args.model_path  # 使用传入的模型路径
    qwen_model, qwen_processor = init_qwen_model(model_path)
    
    print("开始模型推理...")
    model_responses = {}
    for task in prompts:
        model_responses[task] = []
        for prompt in tqdm(prompts[task], desc=f"处理任务: {task}"):
            model_response = qwen_multimodal_call(prompt, qwen_model, qwen_processor)
            model_responses[task].append(model_response)
    
    print("开始评估...")
    task_scores = {}
    for task in model_responses:
        score = evaluate(task, labels[task], model_responses[task])
        task_scores[task] = score
        print(f"{task}: {score:.2f}")

    # 计算总体平均分
    average_score = sum(task_scores.values()) / len(task_scores)
    print("\n" + "="*50)
    print(f"ReMI 总分: {average_score:.2f}")
    print("="*50)

    # 以表格形式展示所有分数
    print("\n详细任务分数:")
    for task, score in task_scores.items():
        print(f"{task:<15}: {score:.2f}")

if __name__ == "__main__":
    main()