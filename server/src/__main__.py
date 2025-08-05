from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import uvicorn
import zhconv
import os

model_id = "models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

asr_pipeline = pipeline(task=Tasks.auto_speech_recognition, model=model_id)

app = FastAPI()


@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        audio_data = await file.read()

        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(audio_data)

        result = asr_pipeline(temp_file_path)
        text = result[0].get("text", "")
        os.remove(temp_file_path)

        return JSONResponse(content={"text": zhconv.convert(text, "zh-cn")})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
