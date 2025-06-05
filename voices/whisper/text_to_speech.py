from TTS.api import TTS

# Init TTS
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# Run TTS
tts.tts_to_file(text="Hello, how are you today? I'm testing a new function.", file_path="/Users/phuongtxq/Desktop/test/voices/whisper/ouput/output2.wav")
