# Looper

**Looper** is a desktop application for detecting seamless loop points in MP3 files.

It analyzes the frequency characteristics of an audio track, searches for sections with similar frequency patterns, and identifies a potential loop point. The detected loop can then be previewed and exported as a new MP3 file.

## Features

* Analyze a selected MP3 file to automatically detect a loop
*  View the detected loop start and end points
*  See the loop duration
*  See the detected match percentage
*  Preview the detected loop
*  Export the detected loop as an MP3 file
*  Track analysis progress


## How It Works

Looper uses the following process:

1. The MP3 file is decoded using **mpg123**.
2. Audio frames are analyzed using **NumPy**.
3. Frequency information is extracted using Fast Fourier Transform (FFT).
4. The application searches for sections with similar frequency patterns.
5. Correlation is used to determine the strongest potential loop point.
6. The detected section can be previewed.
7. The loop can be exported as a separate MP3 file using **FFmpeg**.

## Tech Stack

* **Python**
* **Tkinter** 
* **NumPy** 
* **mpg123** 
* **pygame** 
* **FFmpeg**

## Requirements

* Python 3.10+
* Windows
* FFmpeg

Python dependencies are listed in `requirements.txt`.

## Installation

Clone the repository:

```bash
git clone https://github.com/itstxti/Looper.git
cd Looper
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

### FFmpeg

FFmpeg must also be installed separately and available on the system.

You can verify the installation with:

```bash
ffmpeg -version
```

If you don't have it install, the easiest option is using WinGet:

```bash
winget install Gyan.FFmpeg
```

## Usage

Run the application with:

```bash
python interface.py
```

Then:

1. Click **Choose MP3**.
2. Select an MP3 file.
3. Click **Analyze track**.
4. Wait for the analysis to complete.
5. Review the detected loop information.
6. Click **Play loop** to preview it.
7. Click **Export MP3** to save the MP3 loop.

## Notes

Looper currently supports **MP3 files**.

Loop detection is based on frequency-domain analysis and correlation, so the detected point is an automatically estimated loop rather than a guarantee of musical perfection for every track.

Analysis time depends on the length and complexity of the audio file.
