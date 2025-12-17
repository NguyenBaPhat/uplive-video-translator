## uplive-video-translator

A **Python-based AI-powered video translation tool** that takes an **English video (15–30s)** and produces:

- **Translated video** with a synthetic voice track in the target language
- **Original SRT subtitles** (English, timed)
- **Translated SRT subtitles** (timed, target language)

The core translation step uses **Google Gemini models via the official Python SDK** (`google-generativeai`) as described in the Gemini API docs `[https://ai.google.dev/gemini-api/docs]`.

### 1. Project structure

- **`main.py`**: CLI entrypoint for running the pipeline end-to-end
- **`src/config.py`**: Environment-based config (Gemini API key, model, target language)
- **`src/gemini_client.py`**: Thin wrapper around the Gemini model for subtitle translation
- **`src/transcription.py`**: Whisper-based transcription and SRT generation
- **`src/tts.py`**: gTTS-based text-to-speech for translated subtitles
- **`src/pipeline.py`**: Orchestrates the full workflow (audio extraction → transcription → Gemini translation → TTS → mux)
- **`data/input/`**: Sample input video(s)
- **`data/output/`**: Generated outputs (translated video, audio, SRT files)
- **`requirements.txt`**: Python dependencies

### 2. Setup

```bash
git clone <repository-url>
cd uplive-video-translator
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

System dependencies:

- `ffmpeg` available on PATH (required by `moviepy`, `pydub`, and Whisper)

Install `ffmpeg`:

- **Ubuntu / Debian (Linux)**:

  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```

- **macOS (Homebrew)**:

  ```bash
  brew install ffmpeg
  ```

- **Windows (choco or winget)**:

  With Chocolatey:

  ```powershell
  choco install ffmpeg
  ```

  Or with Winget:

  ```powershell
  winget install Gyan.FFmpeg
  ```

After installation, verify:

```bash
ffmpeg -version
```

### 3. Configure Gemini

Create a `.env` file in the project root with your **Gemini API key**:

```bash
echo "GEMINI_API_KEY=YOUR_KEY_HERE" > .env
```

Optional overrides:

- `GEMINI_MODEL` (default: `gemini-2.5-flash`)
- `TARGET_LANGUAGE` (default: `vi` – Vietnamese)

### 4. Sample input video

A sample English video has been downloaded to:

- `data/input/sample_en_video.mp4`

You can replace it with any 15–30 second English video if desired.

### 5. Running the translation

Basic usage (uses `TARGET_LANGUAGE` from `.env` or default Vietnamese):

```bash
python main.py --input data/input/sample_en_video.mp4
```

Override the target language (e.g. Spanish):

```bash
python main.py --input data/input/sample_en_video.mp4 --target-lang es
```

Multi-language batch (bonus): translate into several languages in one run:

```bash
python main.py --input data/input/sample_en_video.mp4 --target-langs vi,es,fr
```

Outputs in `data/output/`:

- `*_translated.mp4` – video with translated audio track and hard-coded translated subtitles
- `*_translated_audio.mp3` – synthetic translated audio
- `*_original.srt` – original English subtitles
- `*_translated.srt` – translated subtitles

### 6. Notes and limitations

- Transcription uses **Whisper** for accurate timestamps; **Gemini** focuses on **natural, context-aware translation** of subtitle lines.
- TTS uses **gTTS** for simplicity; in production you might swap in a higher-quality neural TTS.
- Subtitle timing is inherited from Whisper; for very noisy audio the timestamps may be slightly off.


