# Looper

**Looper** is a desktop application for detecting, previewing, and exporting music loops from MP3 files.

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

Looper detects suitable loop points in MP3 files by combining spectral analysis with waveform-level refinement.

1. The MP3 file is decoded into PCM audio using **mpg123**.
2. Stereo audio is converted to mono for analysis.
3. The audio is divided into overlapping windows and analyzed using **Fast Fourier Transform (FFT)**.
4. Spectral and RMS features are extracted from each window.
5. Looper searches for sections with similar spectral patterns over time.
6. The strongest candidates are refined by comparing the transitions between the end and beginning of the loop.
7. The candidate points are further refined at the **sample level** using the original waveform.
8. Candidates are scored based on spectral similarity and transition quality.
9. The detected loop can be previewed directly in the application and exported as a separate MP3 file using **FFmpeg**.

## Tech Stack

* **Python**
* **Tkinter** 
* **NumPy** 
* **mpg123** 
* **pygame** 
* **FFmpeg**

## Requirements

* Python 3.12
* Windows
* FFmpeg

Python dependencies are listed in `requirements.txt`.

## Installation

Clone the repository:

```bash
git clone https://github.com/itstxti/Looper.git
cd Looper
```

### Python environment

Create a virtual environment:

```bash
py -3.12 -m venv .venv
```

Activate it:

```bash
.\.venv\Scripts\activate
```

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

### FFmpeg

FFmpeg must also be installed separately and available on the system.

You can verify the installation with:

```bash
ffmpeg -version
```

If you don't have it installed, the easiest option on Windows is using WinGet:

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

## License

Looper is licensed under the MIT License.

Looper also includes mpg123, which is licensed under the GNU Lesser General
Public License (LGPL) version 2.1.
