FROM continuumio/miniconda3

# 复制环境配置文件
COPY environment.yml .

# 创建 conda 环境
RUN conda env create -f environment.yml

# 激活环境时用的 SHELL 设置
SHELL ["conda", "run", "-n", "docker", "/bin/bash", "-c"]

# 设置默认工作目录（可选）
WORKDIR /workspace

# 复制项目代码（可选）
# COPY . /workspace

# 默认启动命令（可选）
CMD ["python"]