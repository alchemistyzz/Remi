FROM continuumio/miniconda3

# 拷贝环境配置
COPY environment.yml .

# 创建 conda 环境
RUN conda env create -f environment.yml

# 激活环境（通过 SHELL 设置）
SHELL ["conda", "run", "-n", "your_env_name", "/bin/bash", "-c"]

# 设置默认命令（可选）
CMD ["python"]