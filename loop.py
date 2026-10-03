import os
import sys
import subprocess
import threading

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
# OPTIONAL PCM PLAYBACK
# =========================================================

try:

    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True

except ImportError:

    sd = None

    SOUNDDEVICE_AVAILABLE = False


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

        # -------------------------------------------------
        # Analysis data
        # -------------------------------------------------

        self.audio_samples = None

        self.spectral_features = None

        self.phase_features = None

        self.rms = None

        self.feature_times = None

        self.analysis_window_size = None

        self.analysis_hop_size = None

        self.frame_start_samples = None

        # -------------------------------------------------
        # Playback state
        # -------------------------------------------------

        self.stop_playback_event = threading.Event()

        self.playback_stream = None


    # =====================================================
    # DECODE TO MONO
    # =====================================================

    def _decode_mono(
        self
    ):
        """
        Convert the decoded PCM frames into a mono
        float32 signal.

        This signal is used for analysis and sample-level
        loop detection.
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
        Analyse the track using:

        - STFT spectral features
        - RMS energy
        - phase information
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

        phase_features = []

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

            fft_result = np.fft.rfft(
                windowed
            )

            magnitude = np.abs(
                fft_result
            )

            phase = np.angle(
                fft_result
            )

            magnitude = magnitude[
                frequency_mask
            ]

            phase = phase[
                frequency_mask
            ]

            # ---------------------------------------------
            # Log compression
            # ---------------------------------------------

            magnitude = np.log1p(
                magnitude
            )

            # ---------------------------------------------
            # Normalize spectral shape
            # ---------------------------------------------

            norm = np.linalg.norm(
                magnitude
            )

            if norm > 0:

                magnitude = (
                    magnitude
                    / norm
                )

            spectral_features.append(
                magnitude
            )

            phase_features.append(
                phase
            )

            # ---------------------------------------------
            # Progress
            # ---------------------------------------------

            if (
                frame_index % progress_step == 0
                or
                frame_index == total_frames - 1
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

        self.phase_features = np.asarray(
            phase_features,
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
        Compare two audio segments using spectral
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

        spectral_similarity = np.mean(
            np.sum(
                features_a
                * features_b,
                axis=1
            )
        )

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

        score = (
            spectral_similarity * 0.8
            +
            rms_ratio * 0.2
        )

        return float(
            score
        )


    # =====================================================
    # DTW SIMILARITY
    # =====================================================

    def dtw_similarity(
        self,
        start_a,
        start_b,
        length,
        max_warp=8
    ):
        """
        Compare two spectral segments using Dynamic
        Time Warping.

        Higher score means greater similarity.
        """

        features = (
            self.spectral_features
        )

        if features is None:

            return 0.0

        end_a = (
            start_a
            + length
        )

        end_b = (
            start_b
            + length
        )

        if (
            end_a > len(features)
            or
            end_b > len(features)
        ):

            return 0.0

        a = features[
            start_a:end_a
        ]

        b = features[
            start_b:end_b
        ]

        n = len(a)

        m = len(b)

        if n == 0 or m == 0:

            return 0.0

        infinity = float(
            "inf"
        )

        cost = np.full(
            (
                n + 1,
                m + 1
            ),
            infinity,
            dtype=np.float32
        )

        cost[
            0,
            0
        ] = 0.0

        for i in range(
            1,
            n + 1
        ):

            j_start = max(
                1,
                i - max_warp
            )

            j_end = min(
                m,
                i + max_warp
            )

            current = a[
                i - 1
            ]

            for j in range(
                j_start,
                j_end + 1
            ):

                other = b[
                    j - 1
                ]

                similarity = float(
                    np.dot(
                        current,
                        other
                    )
                )

                similarity = max(
                    -1.0,
                    min(
                        1.0,
                        similarity
                    )
                )

                distance = (
                    1.0
                    - similarity
                )

                cost[
                    i,
                    j
                ] = (
                    distance
                    +
                    min(
                        cost[
                            i - 1,
                            j
                        ],
                        cost[
                            i,
                            j - 1
                        ],
                        cost[
                            i - 1,
                            j - 1
                        ]
                    )
                )

        final_cost = cost[
            n,
            m
        ]

        if not np.isfinite(
            final_cost
        ):

            return 0.0

        normalized_cost = (
            final_cost
            /
            max(
                n,
                m
            )
        )

        similarity = (
            1.0
            - normalized_cost
        )

        return float(
            max(
                0.0,
                min(
                    1.0,
                    similarity
                )
            )
        )


    # =====================================================
    # ADVANCED SEGMENT SIMILARITY
    # =====================================================

    def advanced_segment_similarity(
        self,
        start_a,
        start_b,
        length
    ):
        """
        Combine spectral similarity with DTW.
        """

        spectral = (
            self.segment_similarity(
                start_a,
                start_b,
                length
            )
        )

        if spectral <= 0:

            return 0.0

        dtw = (
            self.dtw_similarity(
                start_a,
                start_b,
                length
            )
        )

        score = (
            spectral * 0.55
            +
            dtw * 0.45
        )

        return float(
            score
        )


    # =====================================================
    # PHASE SIMILARITY
    # =====================================================

    def phase_similarity(
        self,
        start_a,
        start_b,
        length
    ):
        """
        Compare phase structure between two segments.

        Phase is treated as a secondary signal.
        """

        phases = (
            self.phase_features
        )

        if phases is None:

            return 0.0

        end_a = (
            start_a
            + length
        )

        end_b = (
            start_b
            + length
        )

        if (
            end_a > len(phases)
            or
            end_b > len(phases)
        ):

            return 0.0

        phase_a = phases[
            start_a:end_a
        ]

        phase_b = phases[
            start_b:end_b
        ]

        difference = (
            phase_a
            - phase_b
        )

        difference = np.angle(
            np.exp(
                1j * difference
            )
        )

        similarity = np.cos(
            difference
        )

        similarity = np.mean(
            similarity
        )

        return float(
            max(
                0.0,
                min(
                    1.0,
                    (
                        similarity
                        + 1.0
                    )
                    * 0.5
                )
            )
        )


    # =====================================================
    # ULTIMATE SEGMENT SIMILARITY
    # =====================================================

    def ultimate_segment_similarity(
        self,
        start_a,
        start_b,
        length
    ):
        """
        Combine spectral similarity, DTW and phase.
        """

        spectral = (
            self.segment_similarity(
                start_a,
                start_b,
                length
            )
        )

        if spectral <= 0:

            return 0.0

        dtw = (
            self.dtw_similarity(
                start_a,
                start_b,
                length
            )
        )

        phase = (
            self.phase_similarity(
                start_a,
                start_b,
                length
            )
        )

        score = (
            spectral * 0.40
            +
            dtw * 0.40
            +
            phase * 0.20
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
        Fast first-stage candidate search.

        This stage intentionally does NOT use DTW.
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

        end_step = max(
            1,
            int(
                feature_rate
            )
        )

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

                score = (
                    self.segment_similarity(
                        start,
                        end,
                        comparison_length
                    )
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

        candidates.sort(
            key=lambda candidate: candidate[0],
            reverse=True
        )

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
            or
            end < window
            or
            start + window
            >= len(self.spectral_features)
            or
            end >= len(self.spectral_features)
        ):

            return float(
                "inf"
            )

        start_features = (
            self.spectral_features[
                start:
                start + window
            ]
        )

        end_features = (
            self.spectral_features[
                end - window:
                end
            ]
        )

        spectral_difference = np.mean(
            np.abs(
                start_features
                - end_features
            )
        )

        start_rms = (
            self.rms[
                start:
                start + window
            ]
        )

        end_rms = (
            self.rms[
                end - window:
                end
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
        Refine an approximate loop candidate.
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
    # WAVEFORM SIMILARITY
    # =====================================================

    def waveform_similarity(
        self,
        sample_a,
        sample_b,
        window_seconds=0.2
    ):
        """
        Compare two waveform regions directly.

        Higher score means greater similarity.
        """

        if self.audio_samples is None:

            return 0.0

        window_samples = int(
            self.rate
            * window_seconds
        )

        if window_samples <= 0:

            return 0.0

        half = (
            window_samples
            // 2
        )

        start_a = max(
            0,
            sample_a - half
        )

        end_a = min(
            len(self.audio_samples),
            sample_a + half
        )

        start_b = max(
            0,
            sample_b - half
        )

        end_b = min(
            len(self.audio_samples),
            sample_b + half
        )

        segment_a = (
            self.audio_samples[
                start_a:end_a
            ]
        )

        segment_b = (
            self.audio_samples[
                start_b:end_b
            ]
        )

        length = min(
            len(segment_a),
            len(segment_b)
        )

        if length < 32:

            return 0.0

        segment_a = (
            segment_a[
                :length
            ]
            .astype(
                np.float64
            )
        )

        segment_b = (
            segment_b[
                :length
            ]
            .astype(
                np.float64
            )
        )

        segment_a -= np.mean(
            segment_a
        )

        segment_b -= np.mean(
            segment_b
        )

        norm_a = np.linalg.norm(
            segment_a
        )

        norm_b = np.linalg.norm(
            segment_b
        )

        if (
            norm_a < 1e-10
            or
            norm_b < 1e-10
        ):

            return 0.0

        segment_a /= norm_a

        segment_b /= norm_b

        correlation = float(
            np.dot(
                segment_a,
                segment_b
            )
        )

        return max(
            0.0,
            min(
                1.0,
                (
                    correlation
                    + 1.0
                )
                * 0.5
            )
        )


    # =====================================================
    # ZERO CROSSING SCORE
    # =====================================================

    def zero_crossing_score(
        self,
        sample_a,
        sample_b,
        window_samples=1024
    ):
        """
        Measure how close the candidate boundaries are
        to waveform zero crossings.

        Lower score is better.
        """

        if self.audio_samples is None:

            return float(
                "inf"
            )

        audio = self.audio_samples

        start_a = max(
            0,
            sample_a
            - window_samples // 2
        )

        end_a = min(
            len(audio),
            start_a
            + window_samples
        )

        start_b = max(
            0,
            sample_b
            - window_samples // 2
        )

        end_b = min(
            len(audio),
            start_b
            + window_samples
        )

        segment_a = (
            audio[
                start_a:end_a
            ]
        )

        segment_b = (
            audio[
                start_b:end_b
            ]
        )

        length = min(
            len(segment_a),
            len(segment_b)
        )

        if length < 16:

            return float(
                "inf"
            )

        segment_a = (
            segment_a[
                :length
            ]
        )

        segment_b = (
            segment_b[
                :length
            ]
        )

        boundary_a = np.argmin(
            np.abs(
                segment_a
            )
        )

        boundary_b = np.argmin(
            np.abs(
                segment_b
            )
        )

        zero_a = abs(
            float(
                segment_a[
                    boundary_a
                ]
            )
        )

        zero_b = abs(
            float(
                segment_b[
                    boundary_b
                ]
            )
        )

        return (
            zero_a
            +
            zero_b
        ) * 0.5


    # =====================================================
    # PERIODICITY SCORE
    # =====================================================

    def periodicity_score(
        self,
        sample,
        window_seconds=0.5
    ):
        """
        Estimate local waveform periodicity.

        Higher score means stronger periodic structure.
        """

        if self.audio_samples is None:

            return 0.0

        window_samples = int(
            self.rate
            * window_seconds
        )

        if window_samples < 256:

            return 0.0

        start = max(
            0,
            sample
            - window_samples // 2
        )

        end = min(
            len(self.audio_samples),
            start
            + window_samples
        )

        segment = (
            self.audio_samples[
                start:end
            ]
            .astype(
                np.float64
            )
        )

        if len(segment) < 256:

            return 0.0

        segment -= np.mean(
            segment
        )

        energy = np.sqrt(
            np.mean(
                segment ** 2
            )
        )

        if energy < 1e-8:

            return 0.0

        segment /= energy

        min_lag = max(
            1,
            int(
                self.rate
                * 0.01
            )
        )

        max_lag = min(
            len(segment) // 2,
            int(
                self.rate
                * 0.5
            )
        )

        if max_lag <= min_lag:

            return 0.0

        best = -1.0

        step = max(
            1,
            int(
                self.rate
                * 0.002
            )
        )

        for lag in range(
            min_lag,
            max_lag,
            step
        ):

            a = segment[
                :-lag
            ]

            b = segment[
                lag:
            ]

            if len(a) < 128:

                continue

            correlation = float(
                np.dot(
                    a,
                    b
                )
                /
                len(a)
            )

            if correlation > best:

                best = correlation

        return max(
            0.0,
            min(
                1.0,
                best
            )
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
        Compare the loop boundary using waveform,
        amplitude, zero crossing and periodicity.

        Lower score means a better transition.
        """

        if self.audio_samples is None:

            return float(
                "inf"
            )

        window_samples = max(
            1,
            int(
                window_seconds
                * self.rate
            )
        )

        if (
            start_sample < window_samples
            or
            end_sample < window_samples
            or
            start_sample
            + window_samples
            > len(self.audio_samples)
            or
            end_sample
            > len(self.audio_samples)
        ):

            return float(
                "inf"
            )

        similarity = (
            self.waveform_similarity(
                start_sample,
                end_sample,
                window_seconds
            )
        )

        waveform_error = (
            1.0
            - similarity
        )

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

        max_rms = max(
            start_rms,
            end_rms,
            1e-8
        )

        amplitude_difference = (
            abs(
                start_rms
                - end_rms
            )
            /
            max_rms
        )

        zero_score = (
            self.zero_crossing_score(
                start_sample,
                end_sample,
                window_samples
            )
        )

        zero_score = min(
            1.0,
            zero_score * 10.0
        )

        periodicity_start = (
            self.periodicity_score(
                start_sample
            )
        )

        periodicity_end = (
            self.periodicity_score(
                end_sample
            )
        )

        periodicity = (
            periodicity_start
            +
            periodicity_end
        ) * 0.5

        periodicity_error = (
            1.0
            - periodicity
        )

        score = (
            waveform_error * 0.50
            +
            amplitude_difference * 0.20
            +
            zero_score * 0.15
            +
            periodicity_error * 0.15
        )

        return float(
            score
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
        Refine loop points using the original PCM waveform.
        """

        if self.audio_samples is None:

            raise RuntimeError(
                "Audio samples have not been calculated."
            )

        hop_size = (
            self.analysis_hop_size
        )

        approximate_start = int(
            start_feature
            * hop_size
        )

        approximate_end = int(
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
            or
            len(
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
    # STOP PLAYBACK
    # =====================================================

    def stop_playback(
        self
    ):
        """
        Stop current PCM playback.
        """

        self.stop_playback_event.set()

        if (
            self.playback_stream is not None
        ):

            try:

                self.playback_stream.stop()

            except Exception:

                pass

            try:

                self.playback_stream.close()

            except Exception:

                pass

            self.playback_stream = None


    # =====================================================
    # PLAY LOOP — SAMPLE ACCURATE
    # =====================================================

    def play_looping_samples(
        self,
        start_sample,
        end_sample,
        stop_event=None
    ):
        """
        Play the detected loop directly from the decoded PCM
        samples.

        This uses the exact detected sample positions instead
        of mpg123 frame boundaries.
        """

        if self.audio_samples is None:

            raise RuntimeError(
                "Audio samples have not been calculated."
            )

        if not SOUNDDEVICE_AVAILABLE:

            raise RuntimeError(
                "sounddevice is required for sample-accurate "
                "PCM playback. Install it with: "
                "pip install sounddevice"
            )

        start_sample = int(
            start_sample
        )

        end_sample = int(
            end_sample
        )

        if (
            start_sample < 0
            or
            end_sample > len(
                self.audio_samples
            )
            or
            end_sample <= start_sample
        ):

            raise ValueError(
                "Invalid sample loop boundaries."
            )

        loop_audio = (
            self.audio_samples[
                start_sample:end_sample
            ]
        )

        if len(loop_audio) == 0:

            raise ValueError(
                "The detected loop contains no audio."
            )

        self.stop_playback_event.clear()

        if stop_event is None:

            stop_event = (
                self.stop_playback_event
            )

        print()

        print(
            "Playing sample-accurate loop..."
        )

        print(
            f"Start sample: {start_sample:,}"
        )

        print(
            f"End sample:   {end_sample:,}"
        )

        print(
            f"Duration: "
            f"{len(loop_audio) / self.rate:.3f}s"
        )

        try:

            while not stop_event.is_set():

                sd.play(
                    loop_audio,
                    samplerate=self.rate,
                    blocking=False
                )

                while (
                    not stop_event.is_set()
                    and
                    sd.get_stream()
                    is not None
                ):

                    try:

                        if not sd.get_stream().active:

                            break

                    except Exception:

                        break

                    sd.sleep(
                        20
                    )

                sd.stop()

        finally:

            try:

                sd.stop()

            except Exception:

                pass


    # =====================================================
    # PLAY LOOP — FRAME FALLBACK
    # =====================================================

    def play_looping(
        self,
        start_offset,
        loop_offset,
        stop_event=None
    ):
        """
        Legacy mpg123-frame playback.

        Kept as a fallback for compatibility.
        """

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
        """
        Export the detected loop using FFmpeg.
        """

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
    # STFT + RMS + PHASE
    # -----------------------------------------------------

    track.calculate_audio_features(
        progress_callback
    )

    # -----------------------------------------------------
    # STEP 2
    # FAST CANDIDATE SEARCH
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
    # DTW RE-RANKING
    # -----------------------------------------------------

    print()

    print(
        "Running advanced DTW analysis..."
    )

    dtw_candidates = []

    comparison_length = max(
        1,
        int(
            2.0
            * track.rate
            / track.analysis_hop_size
        )
    )

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

        advanced_similarity = (
            track.advanced_segment_similarity(
                start_feature,
                end_feature,
                comparison_length
            )
        )

        dtw_candidates.append(
            (
                advanced_similarity,
                similarity,
                start_feature,
                end_feature
            )
        )

        progress = (
            index
            /
            max(
                1,
                total_candidates
            )
        ) * 100

        print(
            f"\rDTW analysis: "
            f"{progress:6.1f}%",
            end="",
            flush=True
        )

    print()

    dtw_candidates.sort(
        key=lambda candidate: candidate[0],
        reverse=True
    )

    dtw_candidates = (
        dtw_candidates[:5]
    )

    print(
        f"Selected {len(dtw_candidates)} "
        f"candidates after DTW."
    )

    # -----------------------------------------------------
    # STEP 4
    # SPECTRAL TRANSITION REFINEMENT
    # -----------------------------------------------------

    print()

    print(
        "Refining loop candidates..."
    )

    refined_candidates = []

    total_candidates = len(
        dtw_candidates
    )

    for index, candidate in enumerate(
        dtw_candidates,
        start=1
    ):

        (
            advanced_similarity,
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
                advanced_similarity,
                similarity,
                refined_start,
                refined_end
            )
        )

        progress = (
            index
            /
            max(
                1,
                total_candidates
            )
        ) * 100

        print(
            f"\rRefining candidates: "
            f"{progress:6.1f}%",
            end="",
            flush=True
        )

    print()

    refined_candidates.sort(
        key=lambda candidate: candidate[0]
    )

    # -----------------------------------------------------
    # STEP 5
    # SAMPLE-LEVEL REFINEMENT
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
            advanced_similarity,
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
        # -------------------------------------------------

        final_score = (
            sample_score
            -
            (
                advanced_similarity
                * 0.15
            )
        )

        if (
            final_score
            <
            best_final_score
        ):

            best_final_score = (
                final_score
            )

            best_result = {

                "start_sample":
                    start_sample,

                "end_sample":
                    end_sample,

                "sample_score":
                    sample_score,

                "similarity":
                    similarity,

                "advanced_similarity":
                    advanced_similarity,

                "transition_score":
                    transition_score
            }

        progress = (
            index
            /
            max(
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
    # Exact samples
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

    # -----------------------------------------------------
    # Convert sample positions to frames
    # -----------------------------------------------------

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
    # Exact times
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
        f"Start: {start_time:.6f}s"
    )

    print(
        f"End: {end_time:.6f}s"
    )

    print(
        f"Duration: {duration:.6f}s"
    )

    print(
        f"Similarity: "
        f"{best_result['similarity'] * 100:.1f}%"
    )

    print(
        f"Advanced similarity: "
        f"{best_result['advanced_similarity'] * 100:.1f}%"
    )

    print(
        f"Transition score: "
        f"{best_result['sample_score']:.6f}"
    )

    return {

        "track":
            track,

        # mpg123 frame positions
        "start_offset":
            start_offset,

        "end_offset":
            end_offset,

        # Exact PCM positions
        "start_sample":
            start_sample,

        "end_sample":
            end_sample,

        # Exact times
        "start":
            start_time,

        "end":
            end_time,

        "duration":
            duration,

        # Quality information
        "match":
            best_result[
                "similarity"
            ],

        "advanced_match":
            best_result[
                "advanced_similarity"
            ],

        "transition_score":
            best_result[
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

    track = result[
        "track"
    ]

    # -----------------------------------------------------
    # Prefer sample-accurate playback.
    # -----------------------------------------------------

    if SOUNDDEVICE_AVAILABLE:

        track.play_looping_samples(
            result[
                "start_sample"
            ],
            result[
                "end_sample"
            ]
        )

    else:

        print()

        print(
            "sounddevice is not installed."
        )

        print(
            "Falling back to mpg123 frame playback."
        )

        track.play_looping(
            result[
                "start_offset"
            ],
            result[
                "end_offset"
            ]
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
            result[
                "start"
            ]
        )
    )

    print(
        "Loop returns at: {}".format(
            result[
                "end"
            ]
        )
    )

    print(
        "Duration: {:.6f}s".format(
            result[
                "duration"
            ]
        )
    )

    print(
        "Similarity: {:.1f}%".format(
            result[
                "match"
            ]
            * 100
        )
    )

    print(
        "Advanced similarity: {:.1f}%".format(
            result[
                "advanced_match"
            ]
            * 100
        )
    )

    print(
        "Transition score: {:.6f}".format(
            result[
                "transition_score"
            ]
        )
    )

    print()

    if SOUNDDEVICE_AVAILABLE:

        print(
            "Playing sample-accurate loop..."
        )

        try:

            result[
                "track"
            ].play_looping_samples(
                result[
                    "start_sample"
                ],
                result[
                    "end_sample"
                ]
            )

        except KeyboardInterrupt:

            print()

            print(
                "Playback stopped."
            )

        except Exception as error:

            print()

            print(
                f"Playback error: {error}"
            )

    else:

        print(
            "Playing loop using mpg123..."
        )

        print(
            "Install sounddevice for "
            "sample-accurate playback."
        )

        try:

            result[
                "track"
            ].play_looping(
                result[
                    "start_offset"
                ],
                result[
                    "end_offset"
                ]
            )

        except KeyboardInterrupt:

            print()

            print(
                "Playback stopped."
            )

