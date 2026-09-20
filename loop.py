import os
import sys
import subprocess
import shutil

# =========================================================
# MPG123 DLL SETUP
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MPG123_DIR = os.path.join(
    BASE_DIR,
    "mpg123"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

if sys.platform == "win32":

    if not os.path.isdir(MPG123_DIR):
        raise FileNotFoundError(
            f"MPG123 directory not found: {MPG123_DIR}"
        )

    os.add_dll_directory(
        MPG123_DIR
    )

    os.environ["PATH"] = (
        MPG123_DIR
        + os.pathsep
        + os.environ.get("PATH", "")
    )

    os.environ["MPG123_MODDIR"] = os.path.join(
        MPG123_DIR,
        "plugins"
    )

    os.environ["OUT123_MODULE"] = "win32"


import numpy as np
from mpg123 import Mpg123, Out123


# =========================================================
# MUSIC FILE
# =========================================================

class MusicFile:

    def __init__(self, filename):

        if not os.path.exists(filename):
            raise FileNotFoundError(
                "Specified file not found."
            )

        if not os.path.isfile(filename):
            raise FileNotFoundError(
                "Specified path is not a file."
            )

        if not filename.lower().endswith(".mp3"):
            raise TypeError(
                "This script can currently only handle MP3 files."
            )

        self.filename = filename

        mp3 = Mpg123(
            filename,
            library_path=os.path.join(
                MPG123_DIR,
                "libmpg123-0.dll"
            )
        )

        print("Decoding MP3...")

        self.frames = list(
            mp3.iter_frames()
        )

        print(
            f"Decoded {len(self.frames):,} frames."
        )

        self.rate, self.channels, self.encoding = (
            mp3.get_format()
        )

        print(
            f"Format: {self.rate} Hz, "
            f"{self.channels} channel(s)"
        )

    # =====================================================
    # FREQUENCY ANALYSIS
    # =====================================================

    def calculate_max_frequencies(
        self,
        progress_callback=None
    ):

        print()
        print("Analyzing frequencies...")

        if progress_callback:
            progress_callback(
                10,
                "Analyzing frequencies..."
            )

        frame_ffts = []

        start_frame = 1
        end_frame = len(self.frames) - 2

        total_frames = (
            end_frame - start_frame
        )

        if total_frames <= 0:
            raise ValueError(
                "The MP3 file does not contain enough frames."
            )

        for index, i in enumerate(
            range(
                start_frame,
                end_frame
            ),
            start=1
        ):

            arr = np.frombuffer(
                self.frames[i],
                dtype=np.int16
            )

            arr = arr[::self.channels]

            frame_fft = np.abs(
                np.fft.rfft(arr)
            )

            frame_ffts.append(
                frame_fft
            )

            progress_step = max(
                1,
                total_frames // 20
            )

            if (
                index % progress_step == 0
                or index == total_frames
            ):

                progress = (
                    index / total_frames
                ) * 100

                print(
                    f"\rAnalyzing frequencies: "
                    f"{progress:6.1f}%",
                    end="",
                    flush=True
                )

                if progress_callback:

                    gui_progress = (
                        10
                        + progress * 0.40
                    )

                    progress_callback(
                        gui_progress,
                        "Analyzing frequencies..."
                    )

        print()

        fft_2d = np.stack(
            frame_ffts
        )

        frame_freq = np.fft.rfftfreq(
            len(arr)
        )

        clip_start = 1
        clip_end = 25

        frame_freq_sub = frame_freq[
            clip_start:clip_end
        ]

        fft_2d_sub = fft_2d[
            :,
            clip_start:clip_end
        ]

        fft_2d_denoise = np.ma.masked_where(
            (
                fft_2d_sub.T
                <
                fft_2d_sub.max() * 0.25
            ),
            fft_2d_sub.T,
            0
        )

        max_freq = frame_freq_sub[
            np.argmax(
                fft_2d_denoise,
                axis=0
            )
        ]

        self.max_freq = np.ma.masked_where(
            max_freq == frame_freq_sub[0],
            max_freq
        )

        print(
            f"Frequency analysis complete: "
            f"{len(self.max_freq):,} samples."
        )

    # =====================================================
    # CORRELATION
    # =====================================================

    def sig_corr(
        self,
        s1,
        s2,
        comp_length
    ):

        return np.corrcoef(
            self.max_freq[
                s1:s1 + comp_length
            ],
            self.max_freq[
                s2:s2 + comp_length
            ]
        )[1, 0]

    # =====================================================
    # PERCENTAGE MATCH
    # =====================================================

    def pct_match(
        self,
        s1,
        s2,
        comp_length
    ):

        matches = (
            self.max_freq[
                s1:s1 + comp_length
            ]
            ==
            self.max_freq[
                s2:s2 + comp_length
            ]
        )

        count = np.ma.count(
            matches
        )

        if count == 0:
            return 0

        return (
            np.ma.sum(matches)
            / count
        )

    # =====================================================
    # FIND LOOP POINT
    # =====================================================

    def find_loop_point(
        self,
        start_offset=200,
        test_len=500,
        progress_callback=None
    ):

        print()
        print("Searching for loop point...")

        if progress_callback:
            progress_callback(
                55,
                "Searching for loop point..."
            )

        max_corr = 0

        best_start = None
        best_end = None

        start_step = max(
            1,
            int(
                len(self.max_freq) / 10
            )
        )

        starts = list(
            range(
                start_offset,
                len(self.max_freq)
                - test_len,
                start_step
            )
        )

        total_starts = len(starts)

        if total_starts == 0:
            return (
                None,
                None,
                0
            )

        for start_index, start in enumerate(
            starts,
            start=1
        ):

            for end in range(
                start + 500,
                len(self.max_freq)
                - test_len
            ):

                sc = self.sig_corr(
                    start,
                    end,
                    test_len
                )

                if (
                    np.isfinite(sc)
                    and sc > max_corr
                ):

                    best_start = start
                    best_end = end
                    max_corr = sc

            progress = (
                start_index
                / total_starts
            ) * 100

            print(
                f"\rSearching loop point: "
                f"{progress:6.1f}%",
                end="",
                flush=True
            )

            if progress_callback:

                gui_progress = (
                    55
                    + progress * 0.25
                )

                progress_callback(
                    gui_progress,
                    "Searching for loop point..."
                )

        print()

        return (
            best_start,
            best_end,
            max_corr
        )

    # =====================================================
    # FRAME TO SECONDS
    # =====================================================

    def frame_to_seconds(
        self,
        frame
    ):

        samples_per_sec = (
            self.rate
            * self.channels
        )

        samples_per_frame = (
            len(self.frames[1])
            / 2
        )

        frames_per_sec = (
            samples_per_sec
            / samples_per_frame
        )

        return (
            frame
            / frames_per_sec
        )

    # =====================================================
    # FRAME TO TIME STRING
    # =====================================================

    def time_of_frame(
        self,
        frame
    ):

        time_sec = self.frame_to_seconds(
            frame
        )

        return "{:02.0f}:{:06.3f}".format(
            time_sec // 60,
            time_sec % 60
        )

    # =====================================================
    # PLAY LOOP
    # =====================================================

    def play_looping(
        self,
        start_offset,
        loop_offset,
        stop_event=None
    ):

        out = Out123(
            library_path=os.path.join(
                MPG123_DIR,
                "libout123-0.dll"
            )
        )

        out.start(
            self.rate,
            self.channels,
            self.encoding
        )

        i = 0

        try:

            while True:

                if (
                    stop_event
                    and stop_event.is_set()
                ):
                    break

                out.play(
                    self.frames[i]
                )

                i += 1

                if i == loop_offset:
                    i = start_offset

        finally:

            try:
                out.close()
            except Exception:
                pass

    # =====================================================
    # FIND FFMPEG
    # =====================================================

    def find_ffmpeg(self):

        ffmpeg_path = os.path.join(
            os.environ.get(
                "LOCALAPPDATA",
                ""
            ),
            "Microsoft",
            "WinGet",
            "Links",
            "ffmpeg.exe"
        )

        if os.path.isfile(
            ffmpeg_path
        ):
            return ffmpeg_path

        ffmpeg_path = "ffmpeg"

        try:

            subprocess.run(
                [
                    ffmpeg_path,
                    "-version"
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )

            return ffmpeg_path

        except (
            FileNotFoundError,
            subprocess.CalledProcessError
        ):

            raise RuntimeError(
                "FFmpeg was not found."
            )

    # =====================================================
    # EXPORT LOOP
    # =====================================================

    def export_loop(
        self,
        start_offset,
        loop_offset,
        output_file
    ):

        start_time = self.frame_to_seconds(
            start_offset
        )

        loop_time = self.frame_to_seconds(
            loop_offset
        )

        loop_duration = (
            loop_time
            - start_time
        )

        if loop_duration <= 0:
            raise ValueError(
                "Invalid loop duration."
            )

        output_directory = os.path.dirname(
            output_file
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True
            )

        ffmpeg_path = self.find_ffmpeg()

        subprocess.run(
            [
                ffmpeg_path,

                "-y",

                "-ss",
                f"{start_time:.6f}",

                "-i",
                self.filename,

                "-t",
                f"{loop_duration:.6f}",

                "-vn",

                "-c:a",
                "libmp3lame",

                "-q:a",
                "2",

                output_file
            ],
            check=True
        )

        return output_file


# =========================================================
# ANALYZE TRACK
# =========================================================

def analyze_track(
    filename,
    progress_callback=None
):

    print()
    print(
        f"Loading {filename}..."
    )

    if progress_callback:
        progress_callback(
            0,
            "Loading MP3..."
        )

    track = MusicFile(
        filename
    )

    if progress_callback:
        progress_callback(
            10,
            "MP3 loaded."
        )

    track.calculate_max_frequencies(
        progress_callback
    )

    (
        start_offset,
        best_offset,
        best_corr
    ) = track.find_loop_point(
        progress_callback=progress_callback
    )

    if (
        start_offset is None
        or best_offset is None
    ):
        raise RuntimeError(
            "Could not find a suitable loop point."
        )

    start_time = track.frame_to_seconds(
        start_offset
    )

    end_time = track.frame_to_seconds(
        best_offset
    )

    duration = (
        end_time
        - start_time
    )

    if progress_callback:
        progress_callback(
            100,
            "Loop found."
        )

    return {
        "track": track,
        "start_offset": start_offset,
        "end_offset": best_offset,
        "start": start_time,
        "end": end_time,
        "duration": duration,
        "match": best_corr
    }


# =========================================================
# LOOP TRACK
# =========================================================

def loop_track(
    filename,
    progress_callback=None
):

    result = analyze_track(
        filename,
        progress_callback
    )

    result["track"].play_looping(
        result["start_offset"],
        result["end_offset"]
    )

    return result


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print(
        "MP3 Loop Detector"
    )

    print(
        "-----------------"
    )

    filename = input(
        "Enter the full path to the MP3 file: "
    ).strip()

    filename = filename.strip(
        '"'
    ).strip(
        "'"
    )

    if not filename:

        print(
            "Error: No file specified."
        )

        sys.exit(1)

    result = analyze_track(
        filename
    )

    print()
    print(
        "Loop found:"
    )

    print(
        "Loop starts at: {}".format(
            result["track"].time_of_frame(
                result["start_offset"]
            )
        )
    )

    print(
        "Loop returns at: {}".format(
            result["track"].time_of_frame(
                result["end_offset"]
            )
        )
    )

    print(
        "Match: {:.0f}%".format(
            result["match"] * 100
        )
    )

    print()
    print(
        "Playing loop..."
    )

    result["track"].play_looping(
        result["start_offset"],
        result["end_offset"]
    )