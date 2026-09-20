import os
import threading
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox

import pygame

from loop import analyze_track


# =========================================================
# COLORS
# =========================================================

BG = "#0D0F12"
CARD = "#15181D"
CARD_LIGHT = "#191D23"
BORDER = "#252A31"

TEXT = "#F4F6F8"
TEXT_SECONDARY = "#9CA3AF"
TEXT_MUTED = "#6B7280"

ACCENT = "#7C5CFC"
ACCENT_HOVER = "#8B6DFF"

SUCCESS = "#35C98A"

BUTTON = "#7C5CFC"
BUTTON_HOVER = "#8B6DFF"

PROGRESS_BG = "#252A31"


# =========================================================
# APPLICATION
# =========================================================

class LooperApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Looper")
        self.root.configure(bg=BG)
        self.root.minsize(1100, 720)

        try:
            self.root.state("zoomed")
        except Exception:
            pass

        # -------------------------------------------------
        # STATE
        # -------------------------------------------------

        self.filename = None
        self.track = None

        self.start_offset = None
        self.end_offset = None

        self.start_time = None
        self.end_time = None
        self.duration = None
        self.match = None

        self.stop_event = threading.Event()
        self.preview_thread = None
        self.preview_file = None

        # -------------------------------------------------
        # PYGAME
        # -------------------------------------------------

        pygame.mixer.init()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.on_close
        )

        # -------------------------------------------------
        # FONTS
        # -------------------------------------------------

        self.FONT = "Segoe UI"

        self.font_title = (
            self.FONT,
            28,
            "bold"
        )

        self.font_subtitle = (
            self.FONT,
            11
        )

        self.font_section = (
            self.FONT,
            11,
            "bold"
        )

        self.font_body = (
            self.FONT,
            10
        )

        self.font_small = (
            self.FONT,
            9
        )

        self.font_result = (
            self.FONT,
            20,
            "bold"
        )

        # -------------------------------------------------
        # MAIN
        # -------------------------------------------------

        self.main = tk.Frame(
            self.root,
            bg=BG
        )

        self.main.pack(
            fill="both",
            expand=True,
            padx=42,
            pady=30
        )

        # =================================================
        # HEADER
        # =================================================

        header = tk.Frame(
            self.main,
            bg=BG
        )

        header.pack(
            fill="x",
            pady=(0, 28)
        )

        title_frame = tk.Frame(
            header,
            bg=BG
        )

        title_frame.pack(
            side="left"
        )

        tk.Label(
            title_frame,
            text="Looper",
            font=self.font_title,
            fg=TEXT,
            bg=BG
        ).pack(
            anchor="w"
        )

        tk.Label(
            title_frame,
            text="Detect and preview seamless loops in MP3 files.",
            font=self.font_subtitle,
            fg=TEXT_SECONDARY,
            bg=BG
        ).pack(
            anchor="w",
            pady=(3, 0)
        )

        # =================================================
        # CONTENT
        # =================================================

        content = tk.Frame(
            self.main,
            bg=BG
        )

        content.pack(
            fill="both",
            expand=True
        )

        content.grid_columnconfigure(
            0,
            weight=1
        )

        content.grid_columnconfigure(
            1,
            weight=1
        )

        content.grid_rowconfigure(
            0,
            weight=1
        )

        # =================================================
        # LEFT COLUMN
        # =================================================

        left = tk.Frame(
            content,
            bg=BG
        )

        left.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 10)
        )

        # -------------------------------------------------
        # SOURCE CARD
        # -------------------------------------------------

        source_card = self.create_card(
            left
        )

        source_card.pack(
            fill="x",
            pady=(0, 12)
        )

        tk.Label(
            source_card,
            text="SOURCE FILE",
            font=self.font_section,
            fg=TEXT,
            bg=CARD
        ).pack(
            anchor="w",
            padx=20,
            pady=(18, 10)
        )

        self.file_label = tk.Label(
            source_card,
            text="No MP3 file selected",
            font=self.font_body,
            fg=TEXT_SECONDARY,
            bg=CARD,
            anchor="w"
        )

        self.file_label.pack(
            fill="x",
            padx=20,
            pady=(0, 14)
        )

        self.choose_button = self.create_button(
            source_card,
            "Choose MP3",
            self.choose_file
        )

        self.choose_button.pack(
            fill="x",
            padx=20,
            pady=(0, 18)
        )

        # -------------------------------------------------
        # ANALYSIS CARD
        # -------------------------------------------------

        analysis_card = self.create_card(
            left
        )

        analysis_card.pack(
            fill="x",
            pady=(0, 12)
        )

        tk.Label(
            analysis_card,
            text="ANALYSIS",
            font=self.font_section,
            fg=TEXT,
            bg=CARD
        ).pack(
            anchor="w",
            padx=20,
            pady=(18, 10)
        )

        self.analysis_status = tk.Label(
            analysis_card,
            text="Ready to analyze",
            font=self.font_body,
            fg=TEXT_SECONDARY,
            bg=CARD
        )

        self.analysis_status.pack(
            anchor="w",
            padx=20
        )

        # -------------------------------------------------
        # PROGRESS
        # -------------------------------------------------

        self.progress_canvas = tk.Canvas(
            analysis_card,
            height=6,
            bg=PROGRESS_BG,
            highlightthickness=0,
            bd=0
        )

        self.progress_canvas.pack(
            fill="x",
            padx=20,
            pady=(12, 5)
        )

        self.progress_fill = self.progress_canvas.create_rectangle(
            0,
            0,
            0,
            6,
            fill=ACCENT,
            outline=""
        )

        self.progress_percent = tk.Label(
            analysis_card,
            text="0%",
            font=self.font_small,
            fg=TEXT_MUTED,
            bg=CARD
        )

        self.progress_percent.pack(
            anchor="e",
            padx=20
        )

        self.analyze_button = self.create_button(
            analysis_card,
            "Analyze track",
            self.analyze
        )

        self.analyze_button.pack(
            fill="x",
            padx=20,
            pady=(10, 18)
        )

        # -------------------------------------------------
        # HOW IT WORKS CARD
        # -------------------------------------------------

        how_card = self.create_card(
            left
        )

        how_card.pack(
            fill="both",
            expand=True
        )

        tk.Label(
            how_card,
            text="HOW IT WORKS",
            font=self.font_section,
            fg=TEXT,
            bg=CARD
        ).pack(
            anchor="w",
            padx=20,
            pady=(18, 15)
        )

        steps = [
            (
                "01",
                "Load your MP3",
                "Select the track you want to analyze."
            ),
            (
                "02",
                "Analyze the audio",
                "Looper compares frequency patterns throughout the track."
            ),
            (
                "03",
                "Preview and export",
                "Listen to the detected loop and export it as MP3."
            )
        ]

        for number, title, description in steps:

            row = tk.Frame(
                how_card,
                bg=CARD
            )

            row.pack(
                fill="x",
                padx=20,
                pady=6
            )

            tk.Label(
                row,
                text=number,
                font=(
                    self.FONT,
                    9,
                    "bold"
                ),
                fg=ACCENT,
                bg=CARD,
                width=3
            ).pack(
                side="left",
                anchor="n"
            )

            text_frame = tk.Frame(
                row,
                bg=CARD
            )

            text_frame.pack(
                side="left",
                fill="x",
                expand=True
            )

            tk.Label(
                text_frame,
                text=title,
                font=(
                    self.FONT,
                    10,
                    "bold"
                ),
                fg=TEXT,
                bg=CARD
            ).pack(
                anchor="w"
            )

            tk.Label(
                text_frame,
                text=description,
                font=self.font_small,
                fg=TEXT_SECONDARY,
                bg=CARD,
                wraplength=400,
                justify="left"
            ).pack(
                anchor="w",
                pady=(2, 0)
            )

        # =================================================
        # RIGHT COLUMN
        # =================================================

        right = tk.Frame(
            content,
            bg=BG
        )

        right.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(10, 0)
        )

        # -------------------------------------------------
        # DETECTION RESULT CARD
        # -------------------------------------------------

        result_card = self.create_card(
            right
        )

        result_card.pack(
            fill="x",
            pady=(0, 12)
        )

        tk.Label(
            result_card,
            text="DETECTION RESULT",
            font=self.font_section,
            fg=TEXT,
            bg=CARD
        ).pack(
            anchor="w",
            padx=20,
            pady=(18, 14)
        )

        result_grid = tk.Frame(
            result_card,
            bg=CARD
        )

        result_grid.pack(
            fill="x",
            padx=20,
            pady=(0, 14)
        )

        result_grid.grid_columnconfigure(
            0,
            weight=1
        )

        result_grid.grid_columnconfigure(
            1,
            weight=1
        )

        # IMPORTANT:
        # create_result returns the FRAME and the VALUE LABEL.
        # The frame uses grid; the label inside uses pack.

        self.start_card, self.start_value = self.create_result(
            result_grid,
            "LOOP START",
            "—"
        )

        self.start_card.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 6),
            pady=(0, 6)
        )

        self.end_card, self.end_value = self.create_result(
            result_grid,
            "LOOP END",
            "—"
        )

        self.end_card.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(6, 0),
            pady=(0, 6)
        )

        self.duration_card, self.duration_value = self.create_result(
            result_grid,
            "DURATION",
            "—"
        )

        self.duration_card.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=(0, 6),
            pady=(6, 0)
        )

        self.match_card, self.match_value = self.create_result(
            result_grid,
            "MATCH",
            "—"
        )

        self.match_card.grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(6, 0),
            pady=(6, 0)
        )

        # -------------------------------------------------
        # LOOP PREVIEW CARD
        # -------------------------------------------------

        preview_card = self.create_card(
            right
        )

        preview_card.pack(
            fill="both",
            expand=True
        )

        tk.Label(
            preview_card,
            text="LOOP PREVIEW",
            font=self.font_section,
            fg=TEXT,
            bg=CARD
        ).pack(
            anchor="w",
            padx=20,
            pady=(18, 14)
        )

        # -------------------------------------------------
        # VISUAL
        # -------------------------------------------------

        visual = tk.Frame(
            preview_card,
            bg=CARD_LIGHT,
            height=145
        )

        visual.pack(
            fill="x",
            padx=20
        )

        visual.pack_propagate(
            False
        )

        self.music_symbol = tk.Label(
            visual,
            text="♪",
            font=(
                self.FONT,
                52
            ),
            fg=ACCENT,
            bg=CARD_LIGHT
        )

        self.music_symbol.place(
            relx=0.5,
            rely=0.42,
            anchor="center"
        )

        self.preview_status = tk.Label(
            visual,
            text="No loop available",
            font=(
                self.FONT,
                10,
                "bold"
            ),
            fg=TEXT,
            bg=CARD_LIGHT
        )

        self.preview_status.place(
            relx=0.5,
            rely=0.73,
            anchor="center"
        )

        # -------------------------------------------------
        # PREVIEW CONTROLS
        # -------------------------------------------------

        controls = tk.Frame(
            preview_card,
            bg=CARD
        )

        controls.pack(
            fill="x",
            padx=20,
            pady=(12, 0)
        )

        self.play_button = self.create_button(
            controls,
            "▶  Play loop",
            self.play_preview
        )

        self.play_button.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 5)
        )

        self.stop_button = self.create_button(
            controls,
            "■  Stop",
            self.stop_preview
        )

        self.stop_button.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(5, 0)
        )

        self.play_button.config(
            state="disabled"
        )

        self.stop_button.config(
            state="disabled"
        )

        # -------------------------------------------------
        # EXPORT
        # -------------------------------------------------

        self.export_button = self.create_button(
            preview_card,
            "Export MP3",
            self.export_loop
        )

        self.export_button.pack(
            fill="x",
            padx=20,
            pady=(8, 6)
        )

        self.export_button.config(
            state="disabled"
        )

        self.preview_hint = tk.Label(
            preview_card,
            text="Analyze a track to enable loop preview.",
            font=self.font_small,
            fg=TEXT_MUTED,
            bg=CARD
        )

        self.preview_hint.pack(
            anchor="w",
            padx=20,
            pady=(0, 12)
        )

    # =====================================================
    # CARD
    # =====================================================

    def create_card(
        self,
        parent
    ):

        return tk.Frame(
            parent,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
            bd=0
        )

    # =====================================================
    # BUTTON
    # =====================================================

    def create_button(
        self,
        parent,
        text,
        command
    ):

        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=(
                self.FONT,
                10,
                "bold"
            ),
            fg="#FFFFFF",
            bg=BUTTON,
            activeforeground="#FFFFFF",
            activebackground=BUTTON_HOVER,
            disabledforeground="#6B7280",
            relief="flat",
            bd=0,
            padx=16,
            pady=10,
            cursor="hand2"
        )

        button.bind(
            "<Enter>",
            lambda event: self.button_enter(
                button
            )
        )

        button.bind(
            "<Leave>",
            lambda event: self.button_leave(
                button
            )
        )

        return button

    # =====================================================
    # BUTTON HOVER
    # =====================================================

    def button_enter(
        self,
        button
    ):

        if button["state"] != "disabled":

            button.configure(
                bg=BUTTON_HOVER
            )

    def button_leave(
        self,
        button
    ):

        if button["state"] != "disabled":

            button.configure(
                bg=BUTTON
            )

    # =====================================================
    # RESULT
    # =====================================================

    def create_result(
        self,
        parent,
        label,
        value
    ):

        frame = tk.Frame(
            parent,
            bg=CARD_LIGHT,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        tk.Label(
            frame,
            text=label,
            font=(
                self.FONT,
                8,
                "bold"
            ),
            fg=TEXT_MUTED,
            bg=CARD_LIGHT
        ).pack(
            anchor="w",
            padx=14,
            pady=(9, 1)
        )

        value_label = tk.Label(
            frame,
            text=value,
            font=self.font_result,
            fg=TEXT,
            bg=CARD_LIGHT
        )

        value_label.pack(
            anchor="w",
            padx=14,
            pady=(0, 8)
        )

        # Return BOTH:
        # 1. frame -> managed with grid()
        # 2. label -> updated later with config()

        return frame, value_label

    # =====================================================
    # CHOOSE FILE
    # =====================================================

    def choose_file(self):

        filename = filedialog.askopenfilename(
            title="Select MP3 file",
            filetypes=[
                (
                    "MP3 files",
                    "*.mp3"
                )
            ]
        )

        if not filename:
            return

        self.filename = filename

        self.file_label.config(
            text=os.path.basename(filename),
            fg=TEXT
        )

        self.analysis_status.config(
            text="Ready to analyze",
            fg=TEXT_SECONDARY
        )

        self.reset_results()

    # =====================================================
    # RESET RESULTS
    # =====================================================

    def reset_results(self):

        self.stop_preview()

        self.track = None

        self.start_offset = None
        self.end_offset = None

        self.start_time = None
        self.end_time = None
        self.duration = None
        self.match = None

        self.start_value.config(
            text="—"
        )

        self.end_value.config(
            text="—"
        )

        self.duration_value.config(
            text="—"
        )

        self.match_value.config(
            text="—"
        )

        self.preview_status.config(
            text="No loop available"
        )

        self.preview_hint.config(
            text="Analyze a track to enable loop preview."
        )

        self.export_button.config(
            state="disabled"
        )

        self.play_button.config(
            state="disabled"
        )

        self.stop_button.config(
            state="disabled"
        )

        self.update_progress(
            0
        )

    # =====================================================
    # ANALYZE
    # =====================================================

    def analyze(self):

        if not self.filename:

            messagebox.showwarning(
                "No file selected",
                "Please select an MP3 file first."
            )

            return

        self.stop_preview()

        self.analyze_button.config(
            state="disabled"
        )

        self.choose_button.config(
            state="disabled"
        )

        self.play_button.config(
            state="disabled"
        )

        self.export_button.config(
            state="disabled"
        )

        self.analysis_status.config(
            text="Analyzing...",
            fg=TEXT
        )

        self.update_progress(
            0
        )

        thread = threading.Thread(
            target=self.analyze_thread,
            daemon=True
        )

        thread.start()

    # =====================================================
    # ANALYSIS THREAD
    # =====================================================

    def analyze_thread(self):

        try:

            result = analyze_track(
                self.filename,
                progress_callback=self.update_progress
            )

            self.root.after(
                0,
                lambda: self.analysis_complete(
                    result
                )
            )

        except Exception as e:

            self.root.after(
                0,
                lambda: self.analysis_failed(
                    e
                )
            )

    # =====================================================
    # ANALYSIS COMPLETE
    # =====================================================

    def analysis_complete(
        self,
        result
    ):

        self.track = result["track"]

        self.start_offset = result[
            "start_offset"
        ]

        self.end_offset = result[
            "end_offset"
        ]

        self.start_time = result[
            "start"
        ]

        self.end_time = result[
            "end"
        ]

        self.duration = result[
            "duration"
        ]

        self.match = result[
            "match"
        ]

        self.start_value.config(
            text=self.format_time(
                self.start_time
            )
        )

        self.end_value.config(
            text=self.format_time(
                self.end_time
            )
        )

        self.duration_value.config(
            text=self.format_duration(
                self.duration
            )
        )

        self.match_value.config(
            text="{:.0f}%".format(
                self.match * 100
            )
        )

        self.analysis_status.config(
            text="Loop detected successfully",
            fg=SUCCESS
        )

        self.preview_status.config(
            text="Loop ready to preview"
        )

        self.preview_hint.config(
            text="Play the detected loop or export it as MP3."
        )

        self.analyze_button.config(
            state="normal"
        )

        self.choose_button.config(
            state="normal"
        )

        self.play_button.config(
            state="normal"
        )

        self.export_button.config(
            state="normal"
        )

        self.stop_button.config(
            state="disabled"
        )

        self.update_progress(
            100,
            "Loop found."
        )

    # =====================================================
    # ANALYSIS FAILED
    # =====================================================

    def analysis_failed(
        self,
        error
    ):

        self.analyze_button.config(
            state="normal"
        )

        self.choose_button.config(
            state="normal"
        )

        self.analysis_status.config(
            text="Analysis failed",
            fg="#EF6B73"
        )

        self.update_progress(
            0
        )

        messagebox.showerror(
            "Analysis error",
            str(error)
        )

    # =====================================================
    # PROGRESS
    # =====================================================

    def update_progress(
        self,
        value,
        message=None
    ):

        value = max(
            0,
            min(
                100,
                value
            )
        )

        def update():

            width = self.progress_canvas.winfo_width()

            if width <= 1:
                width = 400

            fill_width = (
                width * value / 100
            )

            self.progress_canvas.coords(
                self.progress_fill,
                0,
                0,
                fill_width,
                6
            )

            self.progress_percent.config(
                text=f"{value:.0f}%"
            )

            if message:

                self.analysis_status.config(
                    text=message
                )

        self.root.after(
            0,
            update
        )

    # =====================================================
    # PLAY PREVIEW
    # =====================================================

    def play_preview(self):

        if (
            self.track is None
            or self.start_offset is None
            or self.end_offset is None
        ):
            return

        self.stop_event.set()

        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass

        self.stop_event.clear()

        self.play_button.config(
            state="disabled"
        )

        self.stop_button.config(
            state="normal"
        )

        self.preview_status.config(
            text="Preparing loop..."
        )

        self.preview_thread = threading.Thread(
            target=self.play_loop_thread,
            daemon=True
        )

        self.preview_thread.start()

    # =====================================================
    # PLAY LOOP THREAD
    # =====================================================

    def play_loop_thread(self):

        try:

            pygame.mixer.music.stop()

            try:
                pygame.mixer.music.unload()
            except Exception:
                pass

            # -------------------------------------------------
            # UNIQUE TEMPORARY FILE
            # -------------------------------------------------

            fd, self.preview_file = tempfile.mkstemp(
                suffix=".mp3",
                prefix="looper_preview_"
            )

            os.close(
                fd
            )

            # -------------------------------------------------
            # EXPORT LOOP
            # -------------------------------------------------

            self.track.export_loop(
                self.start_offset,
                self.end_offset,
                self.preview_file
            )

            if self.stop_event.is_set():
                return

            # -------------------------------------------------
            # PLAY
            # -------------------------------------------------

            pygame.mixer.music.load(
                self.preview_file
            )

            pygame.mixer.music.play(
                loops=-1
            )

            self.root.after(
                0,
                lambda: self.preview_status.config(
                    text="Playing detected loop"
                )
            )

            while not self.stop_event.is_set():

                if not pygame.mixer.music.get_busy():
                    break

                self.stop_event.wait(
                    0.05
                )

        except Exception as e:

            print(
                f"Preview error: {e}"
            )

            self.root.after(
                0,
                lambda: self.preview_status.config(
                    text="Preview failed"
                )
            )

        finally:

            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

            try:
                pygame.mixer.music.unload()
            except Exception:
                pass

            if not self.stop_event.is_set():

                self.root.after(
                    0,
                    self.preview_finished
                )

    # =====================================================
    # PREVIEW FINISHED
    # =====================================================

    def preview_finished(self):

        self.play_button.config(
            state="normal"
        )

        self.stop_button.config(
            state="disabled"
        )

        self.preview_status.config(
            text="Loop ready to preview"
        )

    # =====================================================
    # STOP PREVIEW
    # =====================================================

    def stop_preview(self):

        self.stop_event.set()

        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

        try:
            pygame.mixer.music.unload()
        except Exception:
            pass

        self.play_button.config(
            state=(
                "normal"
                if self.track is not None
                else "disabled"
            )
        )

        self.stop_button.config(
            state="disabled"
        )

        if self.track is not None:

            self.preview_status.config(
                text="Loop ready to preview"
            )

    # =====================================================
    # EXPORT
    # =====================================================

    def export_loop(self):

        if (
            self.track is None
            or self.start_offset is None
            or self.end_offset is None
        ):
            return

        filename = filedialog.asksaveasfilename(
            title="Export loop",
            defaultextension=".mp3",
            filetypes=[
                (
                    "MP3 files",
                    "*.mp3"
                )
            ]
        )

        if not filename:
            return

        try:

            self.export_button.config(
                state="disabled"
            )

            self.analysis_status.config(
                text="Exporting loop...",
                fg=TEXT
            )

            self.root.update_idletasks()

            self.track.export_loop(
                self.start_offset,
                self.end_offset,
                filename
            )

            self.analysis_status.config(
                text="Loop exported successfully",
                fg=SUCCESS
            )

            messagebox.showinfo(
                "Export complete",
                "The loop was exported successfully."
            )

        except Exception as e:

            self.analysis_status.config(
                text="Export failed",
                fg="#EF6B73"
            )

            messagebox.showerror(
                "Export error",
                str(e)
            )

        finally:

            self.export_button.config(
                state="normal"
            )

    # =====================================================
    # TIME FORMATTING
    # =====================================================

    def format_time(
        self,
        seconds
    ):

        minutes = int(
            seconds // 60
        )

        remaining = seconds % 60

        return "{:02d}:{:06.3f}".format(
            minutes,
            remaining
        )

    def format_duration(
        self,
        seconds
    ):

        if seconds < 60:

            return "{:.2f}s".format(
                seconds
            )

        minutes = int(
            seconds // 60
        )

        remaining = seconds % 60

        return "{}:{:05.2f}".format(
            minutes,
            remaining
        )

    # =====================================================
    # CLOSE
    # =====================================================

    def on_close(self):

        self.stop_event.set()

        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

        try:
            pygame.mixer.music.unload()
        except Exception:
            pass

        try:
            pygame.mixer.quit()
        except Exception:
            pass

        self.root.destroy()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = LooperApp(
        root
    )

    root.mainloop()