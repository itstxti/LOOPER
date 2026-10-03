import os
import sys
import subprocess

import numpy as np


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


from mpg123 import Mpg123, Out123


# =========================================================
# MUSIC FILE
# =========================================================

class MusicFile:

    def __init__(
        self,
        filename
    ):

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

        print(
            "Decoding MP3..."
        )

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

        # Filled by the audio analysis stage.
        self.audio_samples = None
        self.spectral_features = None
        self.rms = None
        self.feature_times = None

        self.analysis_window_size = None
        self.analysis_hop_size = None

        self.frame_start_samples = None


    # =====================================================
    # DECODE TO MONO
    # =====================================================

    def _decode_mono(
        self
    ):
        """
        Convert the decoded PCM frames into a mono
        float32 signal.

        The original MP3 channel structure is preserved
        during decoding and then converted to mono.
        """

        if not self.frames:
            raise ValueError(
                "The MP3 file contains no audio frames."
            )

        pcm = np.frombuffer(
            b"".join(
                self.frames
            ),
            dtype=np.int16
        )

        if pcm.size == 0:
            raise ValueError(
                "The MP3 file contains no PCM samples."
            )

        if self.channels > 1:

            usable_samples = (
                pcm.size
                - (
                    pcm.size
                    % self.channels
                )
            )

            pcm = pcm[
                :usable_samples
            ]

            pcm = pcm.reshape(
                -1,
                self.channels
            )

            pcm = pcm.mean(
                axis=1
            )

        pcm = (
            pcm.astype(
                np.float32
            )
            / 32768.0
        )

        return pcm


    # =====================================================
    # BUILD FRAME SAMPLE INDEX
    # =====================================================

    def _build_frame_sample_index(
        self
    ):
        """
        Build a mapping between mpg123 frame indices
        and decoded PCM sample positions.
        """

        starts = np.empty(
            len(self.frames),
            dtype=np.int64
        )

        current_sample = 0

        for index, frame in enumerate(
            self.frames
        ):

            starts[index] = (
                current_sample
            )

            # Frame data contains int16 samples.
            total_values = (
                len(frame)
                // 2
            )

            if self.channels > 1:

                frame_samples = (
                    total_values
                    // self.channels
                )

            else:

                frame_samples = (
                    total_values
                )

            current_sample += (
                frame_samples
            )

        self.frame_start_samples = (
            starts
        )


    # =====================================================
    # AUDIO ANALYSIS
    # =====================================================

    def calculate_audio_features(
        self,
        progress_callback=None
    ):
        """
        Analyse the track using STFT spectral features
        and RMS energy.
        """

        print()
        print(
            "Analyzing audio..."
        )

        if progress_callback:

            progress_callback(
                10,
                "Analyzing audio..."
            )

        # -------------------------------------------------
        # Decode PCM
        # -------------------------------------------------

        self.audio_samples = (
            self._decode_mono()
        )

        if len(
            self.audio_samples
        ) < 4096:

            raise ValueError(
                "The MP3 file does not contain enough audio data."
            )

        # -------------------------------------------------
        # Frame/sample mapping
        # -------------------------------------------------

        self._build_frame_sample_index()

        # -------------------------------------------------
        # STFT configuration
        # -------------------------------------------------

        window_size = 2048
        hop_size = 512

        self.analysis_window_size = (
            window_size
        )

        self.analysis_hop_size = (
            hop_size
        )

        # -------------------------------------------------
        # Hann window
        # -------------------------------------------------

        window = np.hanning(
            window_size
        ).astype(
            np.float32
        )

        # -------------------------------------------------
        # Number of analysis frames
        # -------------------------------------------------

        total_frames = (
            1
            +
            (
                len(self.audio_samples)
                - window_size
            )
            // hop_size
        )

        if total_frames <= 0:

            raise ValueError(
                "The MP3 file is too short for audio analysis."
            )

        spectral_features = []
        rms_values = []

        # -------------------------------------------------
        # Frequency bins
        # -------------------------------------------------

        frequencies = np.fft.rfftfreq(
            window_size,
            d=1.0 / self.rate
        )

        frequency_mask = (
            frequencies >= 40.0
        )

        progress_step = max(
            1,
            total_frames // 20
        )

        # -------------------------------------------------
        # STFT
        # -------------------------------------------------

        for frame_index in range(
            total_frames
        ):

            start = (
                frame_index
                * hop_size
            )

            end = (
                start
                + window_size
            )

            samples = (
                self.audio_samples[
                    start:end
                ]
            )

            windowed = (
                samples
                * window
            )

            # ---------------------------------------------
            # RMS
            # ---------------------------------------------

            rms = np.sqrt(
                np.mean(
                    windowed ** 2
                )
                + 1e-12
            )

            rms_values.append(
                rms
            )

            # ---------------------------------------------
            # FFT
            # ---------------------------------------------

            spectrum = np.abs(
                np.fft.rfft(
                    windowed
                )
            )

            spectrum = spectrum[
                frequency_mask
            ]

            # ---------------------------------------------
            # Log compression
            # ---------------------------------------------

            spectrum = np.log1p(
                spectrum
            )

            # ---------------------------------------------
            # Normalize spectral shape
            # ---------------------------------------------

            norm = np.linalg.norm(
                spectrum
            )

            if norm > 0:

                spectrum = (
                    spectrum
                    / norm
                )

            spectral_features.append(
                spectrum
            )

            # ---------------------------------------------
            # Progress
            # ---------------------------------------------

            if (
                frame_index % progress_step == 0
                or frame_index == total_frames - 1
            ):

                progress = (
                    frame_index
                    /
                    max(
                        1,
                        total_frames - 1
                    )
                ) * 100

                print(
                    f"\rAnalyzing audio: "
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
                        "Analyzing audio..."
                    )

        print()

        # -------------------------------------------------
        # Store analysis
        # -------------------------------------------------

        self.spectral_features = np.asarray(
            spectral_features,
            dtype=np.float32
        )

        self.rms = np.asarray(
            rms_values,
            dtype=np.float32
        )

        self.feature_times = (
            np.arange(
                total_frames
            )
            * hop_size
            / self.rate
        )

        print(
            "Audio analysis complete: "
            f"{len(self.spectral_features):,} samples."
        )


    # =====================================================
    # SEGMENT SIMILARITY
    # =====================================================

    def segment_similarity(
        self,
        start_a,
        start_b,
        length
    ):
        """
        Compare two audio segments using their spectral
        features and RMS energy.

        Higher score means greater similarity.
        """

        end_a = (
            start_a
            + length
        )

        end_b = (
            start_b
            + length
        )

        if (
            end_a
            > len(self.spectral_features)
            or
            end_b
            > len(self.spectral_features)
        ):

            return -1.0

        features_a = (
            self.spectral_features[
                start_a:end_a
            ]
        )

        features_b = (
            self.spectral_features[
                start_b:end_b
            ]
        )

        # -------------------------------------------------
        # Spectral similarity
        # -------------------------------------------------

        spectral_similarity = np.mean(
            np.sum(
                features_a
                * features_b,
                axis=1
            )
        )

        # -------------------------------------------------
        # RMS similarity
        # -------------------------------------------------

        rms_a = (
            self.rms[
                start_a:end_a
            ]
        )

        rms_b = (
            self.rms[
                start_b:end_b
            ]
        )

        rms_a_mean = (
            np.mean(
                rms_a
            )
            + 1e-8
        )

        rms_b_mean = (
            np.mean(
                rms_b
            )
            + 1e-8
        )

        rms_ratio = (
            min(
                rms_a_mean,
                rms_b_mean
            )
            /
            max(
                rms_a_mean,
                rms_b_mean
            )
        )

        # -------------------------------------------------
        # Combined score
        # -------------------------------------------------

        score = (
            spectral_similarity * 0.8
            +
            rms_ratio * 0.2
        )

        return float(
            score
        )


    # =====================================================
    # FIND LOOP CANDIDATES
    # =====================================================

    def find_loop_candidates(
        self,
        comparison_seconds=2.0,
        minimum_loop_seconds=5.0,
        maximum_candidates=10,
        progress_callback=None
    ):
        """
        Find the strongest repeated sections in the track.

        Returns:

            [
                (
                    similarity,
                    start_feature,
                    end_feature
                ),
                ...
            ]
        """

        print()
        print(
            "Searching for loop candidates..."
        )

        if progress_callback:

            progress_callback(
                55,
                "Searching for loop candidates..."
            )

        feature_rate = (
            self.rate
            /
            self.analysis_hop_size
        )

        comparison_length = max(
            1,
            int(
                comparison_seconds
                * feature_rate
            )
        )

        minimum_loop_length = max(
            comparison_length,
            int(
                minimum_loop_seconds
                * feature_rate
            )
        )

        total_features = len(
            self.spectral_features
        )

        if total_features <= (
            minimum_loop_length
            + comparison_length
        ):

            return []

        # -------------------------------------------------
        # Candidate start points
        # -------------------------------------------------

        start_step = max(
            1,
            int(
                total_features
                / 100
            )
        )

        starts = list(
            range(
                0,
                total_features
                - minimum_loop_length
                - comparison_length,
                start_step
            )
        )

        total_starts = len(
            starts
        )

        candidates = []

        # Compare end positions approximately every second.
        end_step = max(
            1,
            int(
                feature_rate
            )
        )

        # -------------------------------------------------
        # Search
        # -------------------------------------------------

        for start_index, start in enumerate(
            starts,
            start=1
        ):

            end_min = (
                start
                + minimum_loop_length
            )

            for end in range(
                end_min,
                total_features
                - comparison_length,
                end_step
            ):

                score = self.segment_similarity(
                    start,
                    end,
                    comparison_length
                )

                if score <= 0:
                    continue

                candidates.append(
                    (
                        score,
                        start,
                        end
                    )
                )

            # ---------------------------------------------
            # Progress
            # ---------------------------------------------

            progress = (
                start_index
                /
                max(
                    1,
                    total_starts
                )
            ) * 100

            print(
                f"\rSearching candidates: "
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
                    "Searching for loop candidates..."
                )

        print()

        # -------------------------------------------------
        # Sort
        # -------------------------------------------------

        candidates.sort(
            key=lambda candidate: candidate[0],
            reverse=True
        )

        # -------------------------------------------------
        # Remove near-duplicate candidates
        # -------------------------------------------------

        filtered = []

        minimum_distance = int(
            feature_rate
            * 2.0
        )

        for candidate in candidates:

            score, start, end = (
                candidate
            )

            too_close = False

            for (
                _,
                existing_start,
                existing_end
            ) in filtered:

                if (
                    abs(
                        start
                        - existing_start
                    )
                    < minimum_distance
                    and
                    abs(
                        end
                        - existing_end
                    )
                    < minimum_distance
                ):

                    too_close = True
                    break

            if not too_close:

                filtered.append(
                    candidate
                )

            if len(
                filtered
            ) >= maximum_candidates:

                break

        print(
            f"Found {len(filtered)} "
            f"loop candidates."
        )

        return filtered


    # =====================================================
    # TRANSITION SCORE
    # =====================================================

    def transition_score(
        self,
        start,
        end,
        window_seconds=0.25
    ):
        """
        Measure how smoothly the end of a loop connects
        back to its beginning.

        Lower score means a better transition.
        """

        feature_rate = (
            self.rate
            /
            self.analysis_hop_size
        )

        window = max(
            1,
            int(
                window_seconds
                * feature_rate
            )
        )

        if (
            start < window
            or end < window
            or start + window
                >= len(self.spectral_features)
            or end >= len(self.spectral_features)
        ):

            return float("inf")

        # -------------------------------------------------
        # Spectral transition
        # -------------------------------------------------

        start_features = (
            self.spectral_features[
                start:start + window
            ]
        )

        end_features = (
            self.spectral_features[
                end - window:end
            ]
        )

        spectral_difference = np.mean(
            np.abs(
                start_features
                - end_features
            )
        )

        # -------------------------------------------------
        # RMS transition
        # -------------------------------------------------

        start_rms = (
            self.rms[
                start:start + window
            ]
        )

        end_rms = (
            self.rms[
                end - window:end
            ]
        )

        rms_difference = np.mean(
            np.abs(
                start_rms
                - end_rms
            )
        )

        rms_scale = (
            np.mean(
                self.rms
            )
            + 1e-8
        )

        rms_difference /= (
            rms_scale
        )

        # -------------------------------------------------
        # Combined score
        # -------------------------------------------------

        return float(
            spectral_difference * 0.8
            +
            rms_difference * 0.2
        )


    # =====================================================
    # REFINE LOOP CANDIDATE
    # =====================================================

    def refine_loop_candidate(
        self,
        start,
        end,
        search_seconds=1.0,
        step_seconds=0.05
    ):
        """
        Refine an approximate loop candidate using
        spectral transition similarity.
        """

        feature_rate = (
            self.rate
            /
            self.analysis_hop_size
        )

        search_radius = max(
            1,
            int(
                search_seconds
                * feature_rate
            )
        )

        step = max(
            1,
            int(
                step_seconds
                * feature_rate
            )
        )

        best_start = start
        best_end = end

        best_score = float(
            "inf"
        )

        # -------------------------------------------------
        # Search local neighbourhood
        # -------------------------------------------------

        for start_offset in range(
            -search_radius,
            search_radius + 1,
            step
        ):

            refined_start = (
                start
                + start_offset
            )

            if refined_start < 1:
                continue

            for end_offset in range(
                -search_radius,
                search_radius + 1,
                step
            ):

                refined_end = (
                    end
                    + end_offset
                )

                if (
                    refined_end
                    <= refined_start
                ):
                    continue

                score = (
                    self.transition_score(
                        refined_start,
                        refined_end
                    )
                )

                if score < best_score:

                    best_score = score

                    best_start = (
                        refined_start
                    )

                    best_end = (
                        refined_end
                    )

        return (
            best_start,
            best_end,
            best_score
        )


    # =====================================================
    # SAMPLE-LEVEL TRANSITION SCORE
    # =====================================================

    def sample_transition_score(
        self,
        start_sample,
        end_sample,
        window_seconds=0.08
    ):
        """
        Compare the end of the loop directly against
        its beginning using the original PCM waveform.

        Lower score means a smoother transition.
        """

        window_samples = max(
            1,
            int(
                window_seconds
                * self.rate
            )
        )

        if (
            start_sample < window_samples
            or end_sample < window_samples
            or
            start_sample
            + window_samples
            > len(self.audio_samples)
            or
            end_sample
            > len(self.audio_samples)
        ):

            return float("inf")

        start_audio = (
            self.audio_samples[
                start_sample:
                start_sample
                + window_samples
            ]
        )

        end_audio = (
            self.audio_samples[
                end_sample
                - window_samples:
                end_sample
            ]
        )

        # -------------------------------------------------
        # Normalize local amplitude
        # -------------------------------------------------

        start_rms = np.sqrt(
            np.mean(
                start_audio ** 2
            )
            + 1e-12
        )

        end_rms = np.sqrt(
            np.mean(
                end_audio ** 2
            )
            + 1e-12
        )

        if (
            start_rms > 1e-8
            and end_rms > 1e-8
        ):

            normalized_start = (
                start_audio
                / start_rms
            )

            normalized_end = (
                end_audio
                / end_rms
            )

        else:

            normalized_start = (
                start_audio
            )

            normalized_end = (
                end_audio
            )

        # -------------------------------------------------
        # Waveform correlation
        # -------------------------------------------------

        correlation = np.corrcoef(
            normalized_start,
            normalized_end
        )[0, 1]

        if not np.isfinite(
            correlation
        ):

            correlation = -1.0

        correlation_score = (
            1.0
            - correlation
        ) / 2.0

        # -------------------------------------------------
        # Amplitude difference
        # -------------------------------------------------

        amplitude_difference = abs(
            start_rms
            - end_rms
        )

        global_rms = (
            np.mean(
                np.abs(
                    self.audio_samples
                )
            )
            + 1e-8
        )

        amplitude_difference /= (
            global_rms
        )

        # -------------------------------------------------
        # Combined score
        # -------------------------------------------------

        return float(
            correlation_score * 0.75
            +
            amplitude_difference * 0.25
        )


    # =====================================================
    # SAMPLE-LEVEL REFINEMENT
    # =====================================================

    def refine_loop_samples(
        self,
        start_feature,
        end_feature,
        search_ms=50,
        step_ms=2
    ):
        """
        Refine the loop points using the original PCM
        waveform.
        """

        hop_size = (
            self.analysis_hop_size
        )

        approximate_start = (
            start_feature
            * hop_size
        )

        approximate_end = (
            end_feature
            * hop_size
        )

        search_samples = max(
            1,
            int(
                self.rate
                * search_ms
                / 1000
            )
        )

        step_samples = max(
            1,
            int(
                self.rate
                * step_ms
                / 1000
            )
        )

        best_start = (
            approximate_start
        )

        best_end = (
            approximate_end
        )

        best_score = float(
            "inf"
        )

        # -------------------------------------------------
        # Search local neighbourhood
        # -------------------------------------------------

        for start_offset in range(
            -search_samples,
            search_samples + 1,
            step_samples
        ):

            candidate_start = (
                approximate_start
                + start_offset
            )

            if candidate_start < 1:
                continue

            for end_offset in range(
                -search_samples,
                search_samples + 1,
                step_samples
            ):

                candidate_end = (
                    approximate_end
                    + end_offset
                )

                if (
                    candidate_end
                    <= candidate_start
                ):

                    continue

                score = (
                    self.sample_transition_score(
                        candidate_start,
                        candidate_end
                    )
                )

                if score < best_score:

                    best_score = score

                    best_start = (
                        candidate_start
                    )

                    best_end = (
                        candidate_end
                    )

        return (
            best_start,
            best_end,
            best_score
        )


    # =====================================================
    # SAMPLE TO SECONDS
    # =====================================================

    def sample_to_seconds(
        self,
        sample
    ):

        return (
            sample
            / self.rate
        )


    # =====================================================
    # SAMPLE TO FRAME
    # =====================================================

    def sample_to_frame(
        self,
        sample
    ):
        """
        Convert a PCM sample position into the corresponding
        mpg123 frame index.
        """

        if (
            self.frame_start_samples is None
            or len(
                self.frame_start_samples
            ) == 0
        ):

            raise RuntimeError(
                "Frame sample index has not been built."
            )

        frame = np.searchsorted(
            self.frame_start_samples,
            sample,
            side="right"
        ) - 1

        frame = max(
            0,
            min(
                frame,
                len(self.frames) - 1
            )
        )

        return int(
            frame
        )


    # =====================================================
    # FRAME TO SAMPLES
    # =====================================================

    def frame_to_samples(
        self,
        frame
    ):

        if (
            self.frame_start_samples is None
        ):

            self._build_frame_sample_index()

        frame = max(
            0,
            min(
                frame,
                len(self.frames) - 1
            )
        )

        return int(
            self.frame_start_samples[
                frame
            ]
        )


    # =====================================================
    # FRAME TO SECONDS
    # =====================================================

    def frame_to_seconds(
        self,
        frame
    ):

        sample = (
            self.frame_to_samples(
                frame
            )
        )

        return (
            sample
            / self.rate
        )


    # =====================================================
    # FRAME TO TIME STRING
    # =====================================================

    def time_of_frame(
        self,
        frame
    ):

        time_sec = (
            self.frame_to_seconds(
                frame
            )
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

    def find_ffmpeg(
        self
    ):

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

        start_time = (
            self.frame_to_seconds(
                start_offset
            )
        )

        loop_time = (
            self.frame_to_seconds(
                loop_offset
            )
        )

        loop_duration = (
            loop_time
            - start_time
        )

        if loop_duration <= 0:

            raise ValueError(
                "Invalid loop duration."
            )

        output_directory = (
            os.path.dirname(
                output_file
            )
        )

        if output_directory:

            os.makedirs(
                output_directory,
                exist_ok=True
            )

        ffmpeg_path = (
            self.find_ffmpeg()
        )

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

    # -----------------------------------------------------
    # STEP 1
    # STFT + RMS analysis
    # -----------------------------------------------------

    track.calculate_audio_features(
        progress_callback
    )

    # -----------------------------------------------------
    # STEP 2
    # Candidate search
    # -----------------------------------------------------

    candidates = (
        track.find_loop_candidates(
            comparison_seconds=2.0,
            minimum_loop_seconds=5.0,
            maximum_candidates=10,
            progress_callback=progress_callback
        )
    )

    if not candidates:

        raise RuntimeError(
            "Could not find suitable loop candidates."
        )

    # -----------------------------------------------------
    # STEP 3
    # Spectral transition refinement
    # -----------------------------------------------------

    print()
    print(
        "Refining loop candidates..."
    )

    refined_candidates = []

    total_candidates = len(
        candidates
    )

    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        (
            similarity,
            start_feature,
            end_feature
        ) = candidate

        (
            refined_start,
            refined_end,
            transition_score
        ) = track.refine_loop_candidate(
            start_feature,
            end_feature
        )

        refined_candidates.append(
            (
                transition_score,
                similarity,
                refined_start,
                refined_end
            )
        )

        progress = (
            index
            / total_candidates
        ) * 100

        print(
            f"\rRefining candidates: "
            f"{progress:6.1f}%",
            end="",
            flush=True
        )

    print()

    # -----------------------------------------------------
    # Sort by transition quality
    # -----------------------------------------------------

    refined_candidates.sort(
        key=lambda candidate: (
            candidate[0]
        )
    )

    # -----------------------------------------------------
    # STEP 4
    # Sample-level refinement
    # -----------------------------------------------------

    print()
    print(
        "Performing sample-level refinement..."
    )

    best_result = None

    best_final_score = float(
        "inf"
    )

    total_candidates = len(
        refined_candidates
    )

    for index, candidate in enumerate(
        refined_candidates,
        start=1
    ):

        (
            transition_score,
            similarity,
            start_feature,
            end_feature
        ) = candidate

        (
            start_sample,
            end_sample,
            sample_score
        ) = track.refine_loop_samples(
            start_feature,
            end_feature,
            search_ms=50,
            step_ms=2
        )

        # -------------------------------------------------
        # Final score
        #
        # Similarity is rewarded.
        # Transition smoothness is prioritized.
        # -------------------------------------------------

        final_score = (
            sample_score
            - (
                similarity
                * 0.10
            )
        )

        if final_score < best_final_score:

            best_final_score = (
                final_score
            )

            best_result = {
                "start_sample": start_sample,
                "end_sample": end_sample,
                "sample_score": sample_score,
                "similarity": similarity,
                "transition_score": transition_score
            }

        progress = (
            index
            / max(
                1,
                total_candidates
            )
        ) * 100

        print(
            f"\rSample refinement: "
            f"{progress:6.1f}%",
            end="",
            flush=True
        )

    print()

    if best_result is None:

        raise RuntimeError(
            "Could not refine a suitable loop."
        )

    # -----------------------------------------------------
    # Convert sample positions to mpg123 frames
    # -----------------------------------------------------

    start_sample = (
        best_result[
            "start_sample"
        ]
    )

    end_sample = (
        best_result[
            "end_sample"
        ]
    )

    start_offset = (
        track.sample_to_frame(
            start_sample
        )
    )

    end_offset = (
        track.sample_to_frame(
            end_sample
        )
    )

    if end_offset <= start_offset:

        raise RuntimeError(
            "The detected loop points are invalid."
        )

    # -----------------------------------------------------
    # Exact times from PCM
    # -----------------------------------------------------

    start_time = (
        track.sample_to_seconds(
            start_sample
        )
    )

    end_time = (
        track.sample_to_seconds(
            end_sample
        )
    )

    duration = (
        end_time
        - start_time
    )

    if duration <= 0:

        raise RuntimeError(
            "The detected loop duration is invalid."
        )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    if progress_callback:

        progress_callback(
            100,
            "Loop found."
        )

    print()
    print(
        "Loop analysis complete."
    )

    print(
        f"Start: {start_time:.3f}s"
    )

    print(
        f"End: {end_time:.3f}s"
    )

    print(
        f"Duration: {duration:.3f}s"
    )

    print(
        f"Similarity: "
        f"{best_result['similarity'] * 100:.1f}%"
    )

    print(
        f"Transition score: "
        f"{best_result['sample_score']:.6f}"
    )

    return {
        "track": track,

        # mpg123 frame positions
        "start_offset": start_offset,
        "end_offset": end_offset,

        # Exact PCM positions
        "start_sample": start_sample,
        "end_sample": end_sample,

        # Exact times
        "start": start_time,
        "end": end_time,
        "duration": duration,

        # Quality information
        "match": best_result[
            "similarity"
        ],

        "transition_score": best_result[
            "sample_score"
        ]
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

    try:

        result = analyze_track(
            filename
        )

    except Exception as error:

        print()
        print(
            f"Error: {error}"
        )

        sys.exit(1)

    print()
    print(
        "Loop found:"
    )

    print(
        "Loop starts at: {}".format(
            result["start"]
        )
    )

    print(
        "Loop returns at: {}".format(
            result["end"]
        )
    )

    print(
        "Duration: {:.3f}s".format(
            result["duration"]
        )
    )

    print(
        "Similarity: {:.1f}%".format(
            result["match"] * 100
        )
    )

    print(
        "Transition score: {:.6f}".format(
            result["transition_score"]
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

