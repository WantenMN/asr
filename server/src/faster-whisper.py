from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import torch
import uvicorn
import zhconv

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = WhisperModel("models/faster-whisper-large-v3-turbo-ct2")

app = FastAPI()


@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        audio_data = await file.read()

        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(audio_data)

        segments, info = model.transcribe(
            temp_file_path, language="zh", initial_prompt="中文"
        )
        text = ""
        for segment in segments:
            text += segment.text

        return JSONResponse(content={"text": zhconv.convert(text, "zh-cn")})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
