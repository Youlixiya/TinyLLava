python -m llava.serve.controller --host 127.0.0.1 --port 10000
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
python -m llava.serve.model_worker --host 127.0.0.1 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ./checkpoints/tinytape-v1.0-1.1b
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.model_worker --host 127.0.0.1 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ./checkpoints/tinytape-llava-llama-2-v1.0-1.1b --load-4bit

python -m llava.serve.cli \
    --model-path ./checkpoints/tinytape-llava-llama-2-v1.0-1.1b \
    --image-file "serve_images/2024-01-31/b939abf2c4553ce07e642170aee3a3d7.jpg" \
    --load-4bit