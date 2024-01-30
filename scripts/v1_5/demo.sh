python -m llava.serve.controller --host 127.0.0.1 --port 10000
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
python -m llava.serve.model_worker --host 127.0.0.1 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ./checkpoints/tinyllava-v1.0-1.1b-rec
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.model_worker --host 127.0.0.1 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ./checkpoints/tinyllava-v1.0-1.1b-rec --load-4bit