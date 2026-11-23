from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .engine import VideoEngine
from .states import Visibility


class VanishCamApp:
    def __init__(self, root: tk.Tk, model_dir: Path):
        self.root = root
        self.engine = VideoEngine(model_dir)
        root.title("VanishCam Lite")
        root.resizable(False, False)
        self.preview = ttk.Label(root)
        self.preview.pack(padx=12, pady=(12, 6))
        self.status = tk.StringVar(value="Iniciando webcam…")
        ttk.Label(root, textvariable=self.status).pack(pady=4)
        self.virtual = tk.BooleanVar(value=False)
        ttk.Checkbutton(root, text="Enviar a cámara virtual (OBS)", variable=self.virtual).pack(pady=4)
        row = ttk.Frame(root)
        row.pack(pady=(4, 12))
        self.capture_button = ttk.Button(row, text="Capturar fondo vacío", command=self.capture_background)
        self.capture_button.pack(side="left", padx=4)
        self.start_button = ttk.Button(row, text="Iniciar", command=self.start_processing, state="disabled")
        self.start_button.pack(side="left", padx=4)
        self.reappear_button = ttk.Button(row, text="Reaparecer (F8)", command=self.reappear, state="disabled")
        self.reappear_button.pack(side="left", padx=4)
        self.root.bind_all("<F8>", lambda _event: self.reappear())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        try:
            self.engine.start_worker()
        except Exception as exc:
            messagebox.showerror("VanishCam Lite", str(exc))
            root.after(0, self.close)
            return
        self.root.after(30, self.refresh)

    def capture_background(self):
        if self.engine.capture_background():
            self.status.set("Fondo capturado. Asegúrate de estar fuera del encuadre y pulsa Iniciar.")
            self.start_button.configure(state="normal")

    def start_processing(self):
        try:
            self.engine.start_processing(self.virtual.get())
        except Exception as exc:
            messagebox.showerror("No se pudo iniciar", str(exc))
            return
        self.capture_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.reappear_button.configure(state="normal")
        self.status.set("VISIBLE — mantén 🤫 o cruza los brazos en X durante 600 ms.")

    def reappear(self):
        self.engine.reappear()

    def refresh(self):
        if self.engine.error:
            self.status.set("Error: " + self.engine.error)
        latest = self.engine.latest()
        if latest:
            frame, state = latest
            rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            image = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.preview.configure(image=image)
            self.preview.image = image
            if self.engine.started:
                text = {
                    Visibility.VISIBLE: "VISIBLE — 🤫 o brazos en X durante 600 ms para desaparecer.",
                    Visibility.DISAPPEARING: "DESAPARECIENDO…",
                    Visibility.INVISIBLE: "INVISIBLE — F8 o Reaparecer para volver.",
                    Visibility.APPEARING: "REAPARECIENDO…",
                }[state]
                self.status.set(text)
        self.root.after(30, self.refresh)

    def close(self):
        self.engine.close()
        self.root.destroy()


def run(model_dir: Path):
    root = tk.Tk()
    VanishCamApp(root, model_dir)
    root.mainloop()
