# backend/main.py
from fastapi import FastAPI
import uvicorn
import gradio as gr

# 각각의 Gradio 페이지에서 만든 "demo_kr", "demo_jp", "demo_en" import
from gradio_korean import demo_kr
from gradio_japanese import demo_jp
from gradio_english import demo_en

app = FastAPI()

# Gradio 앱들을 각기 다른 path로 mount
app = gr.mount_gradio_app(app, demo_kr, path="/korean")
app = gr.mount_gradio_app(app, demo_jp, path="/japanese")
app = gr.mount_gradio_app(app, demo_en, path="/english")

@app.get("/")
def read_root():
    return {"message": "Hello from Multi-Language Gradio!"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Azure가 지정하는 포트 사용
    uvicorn.run(app, host="0.0.0.0", port=port)



