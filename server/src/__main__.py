from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import uvicorn
import zhconv
import os
import tempfile
import traceback

model_id = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

print("加载模型中...")
asr_pipeline = pipeline(task=Tasks.auto_speech_recognition, model=model_id)
print("模型加载完成")

app = FastAPI()


@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        print(f"收到文件：{file.filename}, content_type={file.content_type}")

        audio_data = await file.read()
        print(f"文件大小：{len(audio_data)} bytes")

        # 用 tempfile 自动生成临时文件，跨平台
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
            temp_file_path = tmp.name
            tmp.write(audio_data)

        print(f"临时文件路径：{temp_file_path}")

        # 调用模型
        result = asr_pipeline(temp_file_path)
        print(f"ASR 原始输出：{result}")

        # 有些模型返回字典，有些返回 list
        text = ""
        if isinstance(result, list):
            if len(result) > 0 and isinstance(result[0], dict):
                text = result[0].get("text", "")
        elif isinstance(result, dict):
            text = result.get("text", "")

        print(f"识别结果：{text}")

        os.remove(temp_file_path)
        print("临时文件已删除")

        return JSONResponse(content={"text": zhconv.convert(text, "zh-cn")})
    except Exception as e:
        print("发生错误：", str(e))
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
