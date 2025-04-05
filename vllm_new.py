# Copyright 2024 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");

# 导入库
import io
import json
import os
import requests
from base64 import b64encode
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image as PILImage
import argparse
import math
import openai
from tqdm.asyncio import tqdm_asyncio
from loguru import logger
import asyncio
import sys

# 工具函数
def pil_image_to_bytes(img, format='PNG'):
	if not img:
		return None
	img_byte_arr = io.BytesIO()
	img.save(img_byte_arr, format=format)
	return img_byte_arr.getvalue()

def get_multi_image_question_parts(record, question_appendix, max_num_images=6):
	"""处理包含多个图像的问题"""
	parts, images = [], []
	for i in range(max_num_images):
		img_key = f'image_{i+1}'
		if img_key in record and record[img_key] is not None:
			images.append(record[img_key])

	question = 'Question: ' + record['question'] + question_appendix
	for j, img in enumerate(images):
		question_parts = question.split(f'<image{j+1}>')
		parts += [question_parts[0], img] if question_parts[0] else [img]
		question = question_parts[1]
	return parts + [question] if question else []

# 评估函数
def strip_json(s):
	"""提取文本中的JSON部分"""
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
		logger.warning(f'json parse error: {model_response}')
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
	"""加载ReMI数据集并准备评估数据"""
	logger.info("开始加载ReMI数据集...")
	prompts, labels = {}, {}
	for example in tqdm(load_dataset("mehrankazemi/ReMI")['test']):
			task = example['task']
			prompts[task] = prompts.get(task, []) + [get_multi_image_question_parts(example, QUESTION_APPENDICES[task])]
			labels[task] = labels.get(task, []) + [example['label']]
	return prompts, labels

# 图像预处理：确保不超过最大像素限制
def preprocess_image(img, max_pixels=2048*28*28):
	"""预处理图像，确保不超过最大像素限制"""
	width, height = img.size
	pixels = width * height
	
	if pixels <= max_pixels:
		return img  # 不需要调整尺寸
	
	# 计算需要的缩放比例
	scale_factor = math.sqrt(max_pixels / pixels)
	new_width = int(width * scale_factor)
	new_height = int(height * scale_factor)
	
	# 调整图像大小
	resized_img = img.resize((new_width, new_height), PILImage.LANCZOS)
	logger.info(f"调整图像尺寸: {width}x{height} -> {new_width}x{new_height}")
	
	return resized_img

# 保存并编码图像
def process_images_for_api(image_parts, image_dir="./temp_images", max_pixels=2048*28*28):
	"""将图像保存到临时文件并编码为Base64格式"""
	os.makedirs(image_dir, exist_ok=True)
	
	image_data = []
	image_paths = []
	
	for i, img in enumerate(image_parts):
		# 预处理图像
		preprocessed_img = preprocess_image(img, max_pixels)
		
		# 保存图像到临时目录
		image_path = f"{image_dir}/image_{i}.png"
		preprocessed_img.save(image_path)
		image_paths.append(image_path)
		
		# 将图像编码为Base64
		with open(image_path, 'rb') as img_file:
			encoded_image = b64encode(img_file.read()).decode('utf-8')
			image_data.append({
				"data": encoded_image,
				"id": f"image_{i}"
			})

	return image_data, image_paths
import base64
from io import BytesIO
from PIL import Image
def read_img_as_base64(img):
	if isinstance(img, str):
		pil_img = Image.open(img)
	else:
		pil_img = img
	buffered = BytesIO()
	format = "PNG" if pil_img.mode in ("RGBA", "LA") else "JPEG"
	pil_img.save(buffered, format=format)
	return f"data:image/{format.lower()};base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging
logger_std = logging.getLogger(__name__)
@retry(
	stop=stop_after_attempt(10),  # 最大重试次数
	wait=wait_exponential(multiplier=1, min=1, max=128),  # 指数退避 (1, 2, 4, 8, ..., 128)
	retry=(retry_if_exception_type(Exception)),  # 对所有异常重试
	before_sleep=before_sleep_log(logger_std, logging.WARNING),  # 重试前记录日志
	reraise=True  # 重试结束后重新抛出异常
)
async def get_model_response(prompt_parts, client, model):
	text_question = ""
	messages = [{"role": "user", 'content': []}]  
	for part in prompt_parts:
		if isinstance(part, str):
			messages[0]['content'].append({"type": "text", "text": part})
			text_question += part
		else:
			messages[0]['content'].append({"type": "image_url", "image_url": {"url": read_img_as_base64(part)}})
			text_question += '<image>'
	completion = await client.chat.completions.create(
			model=model,
			messages=messages,
			temperature=0.0,
			max_tokens=512,
			stream=False,
		)
	logger.debug(f"Question: {text_question}\n Answer: {completion.choices[0].message.content}")
	return completion.choices[0].message.content

# 主函数
async def main():
	logger.remove()
	logger.add(sys.stdout, level="INFO")
	logger.add("log/{time}.log", level="DEBUG")
	parser = argparse.ArgumentParser(description="通过VLLM服务评估Qwen模型在ReMI数据集上的表现")
	parser.add_argument('--api_url', type=str, default="http://localhost:8000/v1",
						help="VLLM服务的API URL")
	parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
						help="模型名称，用于API请求")
	parser.add_argument('--max_pixels', type=int, default=2048*28*28,
						help="图像最大像素数量")
	parser.add_argument('--task', type=str, default=None,
						help="指定要评估的单个任务，默认评估所有任务")
	parser.add_argument('--save_dir',default=None,
						help="保存结果的目录")
	args = parser.parse_args()

	logger.info(f"VLLM服务地址: {args.api_url}")
	logger.info(f"模型名称: {args.model_name}")
	logger.info(f"图像最大像素数: {args.max_pixels}")
	openai_api_key = "EMPTY"
	openai_api_base = args.api_url
	client = openai.AsyncOpenAI(
		api_key=openai_api_key,
		base_url=openai_api_base,
	)
	model = args.model_name
		
	# 准备数据
	prompts, labels = prepare_data()

	# 如果指定了特定任务，则只评估该任务
	if args.task:
		if args.task not in prompts:
			logger.error(f"错误：找不到任务 '{args.task}'。可用任务: {list(prompts.keys())}")
			return
		selected_tasks = [args.task]
	else:
		selected_tasks = list(prompts.keys())

	logger.info("开始模型推理...")
	model_responses = {}
	for task in selected_tasks:
		logger.info(f"评估任务: {task}")
		model_responses[task] = []
		async_tasks = []
		for prompt in tqdm(prompts[task], desc=f"处理任务: {task}"):
			async_tasks.append(asyncio.create_task(get_model_response(prompt, client, model)))
		results = await tqdm_asyncio.gather(*async_tasks)
		model_responses[task].extend(results)
	
	if args.save_dir:
		os.makedirs(args.save_dir, exist_ok=True)
		with open(os.path.join(args.save_dir, f"model_responses_{args.model_name}.json"), "w") as f:
			json.dump(model_responses, f, indent=4)

	logger.info("\n开始评估...")
	task_scores = {}
	for task in model_responses:
		if not model_responses[task]:
			logger.warning(f"警告: 任务 {task} 没有响应")
			continue
			
		score = evaluate(task, labels[task], model_responses[task])
		task_scores[task] = score
		logger.info(f"{task}: {score:.4f}")

	# 计算总体平均分
	if task_scores:
		average_score = sum(task_scores.values()) / len(task_scores)
		logger.info("\n" + "="*50)
		logger.info(f"ReMI 总分: {average_score:.4f}")
		logger.info("="*50)
		logger.info(f"模型名称: {args.model_name}")
		logger.info(f"VLLM服务地址: {args.api_url}")
		
		# 以表格形式展示所有分数
		logger.info("\n详细任务分数:")
		for task, score in sorted(task_scores.items(), key=lambda x: x[0]):
			logger.info(f"{task:<15}: {score:.4f}")
	else:
		logger.info("没有可用的评估结果")

if __name__ == "__main__":
	asyncio.run(main())
