import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from audio_recorder import grabar_audio, guardar_audio
from audio_processor import obtener_formantes
from analyzer import comparar_formantes, obtener_score_formantes
from feedback_generator import generar_feedback
import sounddevice as sd
import soundfile as sf
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

record_path = os.path.join("data", "records")
referencia_path_carpeta = os.path.join("data", "formantes_referencia")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Entrenador de Pronunciación")
        self.root.geometry("600x700")  # Aumentamos un poco más la altura
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", font=("Segoe UI", 11))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("TCombobox", padding=5)

        self.sonidos_objetivo = {
            "æ": ("cat", "I have a cat.", "The black cat sat on the mat."),
            "ɪ": ("bit", "A little bit.", "Can you give me a little bit of sugar?"),
            "ʌ": ("cut", "A sharp cut.", "He made a sharp cut with the knife."),
        }
        self.sonido_seleccionado = tk.StringVar(root)
        self.sonido_seleccionado.set("æ")
        self.grabacion_audio = None
        self.frecuencia_muestreo = None
        self.ultimo_archivo_grabado = None

        # Variables para el score
        self.intentos = {sonido: 0 for sonido in self.sonidos_objetivo}
        self.correctos = {sonido: 0 for sonido in self.sonidos_objetivo}
        self.datos_referencia = self.cargar_datos_referencia()

        # Main visual frame
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill="both", expand=True)

        # Widgets
        ttk.Label(main_frame, text="🎧 Selecciona el sonido a practicar:").pack(pady=(0, 10))

        self.combo = ttk.Combobox(
            main_frame, textvariable=self.sonido_seleccionado,
            values=list(self.sonidos_objetivo.keys()), state="readonly"
        )
        self.combo.pack(pady=(0, 10))
        self.combo.bind("<<ComboboxSelected>>", self.actualizar_sugerencia)

        self.sugerencia_label = ttk.Label(main_frame, text="", wraplength=360, justify="center", font=("Segoe UI", 10, "italic"))
        self.sugerencia_label.pack(pady=(0, 15))
        self.actualizar_sugerencia()

        self.grabar_button = ttk.Button(main_frame, text="🎤 Grabar Pronunciación", command=self.iniciar_grabacion)
        self.grabar_button.pack(pady=(0, 10))

        self.analizar_button = ttk.Button(main_frame, text="🚀 Enviar y Analizar", command=self.enviar_y_analizar, state=tk.DISABLED)
        self.analizar_button.pack(pady=(0, 10))

        self.escuchar_button = ttk.Button(main_frame, text="👂 Escuchar Grabación", command=self.reproducir_grabacion, state=tk.DISABLED)
        self.escuchar_button.pack(pady=(0, 10))

        self.retroalimentacion_label = ttk.Label(main_frame, text="Retroalimentación:", justify="left", font=("Segoe UI", 11))
        self.retroalimentacion_label.pack(pady=(10, 0), anchor="w")
        self.retroalimentacion_text = scrolledtext.ScrolledText(main_frame, height=10, wrap=tk.WORD)
        self.retroalimentacion_text.pack(pady=(0, 10), fill="x", expand=True)
        self.retroalimentacion_text.config(state=tk.DISABLED)

        self.score_label = ttk.Label(main_frame, text="Score:", justify="center", font=("Segoe UI", 12, "bold"))
        self.score_label.pack(pady=10)
        self.actualizar_score_display()

    def cargar_datos_referencia(self):
        """Carga los datos de referencia de formantes desde archivos .npy."""
        datos_referencia = {}
        ruta_referencia_carpeta = os.path.join("data", "formantes_referencia")
        try:
            for sonido, info in self.sonidos_objetivo.items():
                clave = ""
                if sonido == "æ":
                    clave = "ae"
                elif sonido == "ɪ":
                    clave = "ih"
                elif sonido == "ʌ":
                    clave = "uh"

                if clave:
                    ruta_archivo = os.path.join(ruta_referencia_carpeta, f"referencia_{clave}.npy")
                    if os.path.exists(ruta_archivo):
                        datos_referencia[sonido] = np.load(ruta_archivo)
                    else:
                        messagebox.showerror("Error", f"No se encontró el archivo de referencia: {ruta_archivo}")
                        return {}
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar los datos de referencia: {e}")
            return {}
        return datos_referencia

    def actualizar_sugerencia(self, event=None):
        sonido = self.sonido_seleccionado.get()
        sugerencias = self.sonidos_objetivo[sonido]
        self.sugerencia_label.config(text=f"Puedes intentar decir la palabra: '{sugerencias[0]}', o la frase: '{sugerencias[1]}'.")

    def iniciar_grabacion(self):
        print("Iniciando grabación...")
        self.grabacion_audio, self.frecuencia_muestreo = grabar_audio(duracion=3)
        sonido = self.sonido_seleccionado.get()
        self.ultimo_archivo_grabado = os.path.join(record_path, f"grabacion_{sonido}_temp_escucha.wav")
        guardar_audio(self.grabacion_audio, self.frecuencia_muestreo, self.ultimo_archivo_grabado)
        self.grabar_button.config(state=tk.DISABLED, text="Grabando...")
        self.analizar_button.config(state=tk.NORMAL)
        self.escuchar_button.config(state=tk.NORMAL) # Habilitar el botón de escuchar
        print("Grabación finalizada y guardada para escuchar.")
        self.grabar_button.config(state=tk.NORMAL, text="🎤 Grabar Pronunciación")

    def visualizar_audio(self, archivo_wav):
        if archivo_wav and os.path.exists(archivo_wav):
            data, samplerate = sf.read(archivo_wav)
            time = np.linspace(0., len(data)/samplerate, len(data))

            fig, ax = plt.subplots()
            ax.plot(time, data)
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.set_title("Forma de Onda de la Grabación")

            ventana_top = tk.Toplevel(self.root)
            ventana_top.title("Visualización de Audio")
            canvas = FigureCanvasTkAgg(fig, master=ventana_top)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            canvas.draw()
        else:
            messagebox.showinfo("Información", "No hay grabación para visualizar.")
    

    def enviar_y_analizar(self):
        nombre_archivo_wav = os.path.join(record_path, f"grabacion_{sonido}.wav")
        guardar_audio(self.grabacion_audio, self.frecuencia_muestreo, nombre_archivo_wav)
        self.visualizar_audio(nombre_archivo_wav) # Llama a la función de visualización
        self.analizar_pronunciacion(sonido, nombre_archivo_wav)
        if self.grabacion_audio is not None and self.frecuencia_muestreo is not None:
            sonido = self.sonido_seleccionado.get()
            nombre_archivo_wav = os.path.join(record_path, f"grabacion_{sonido}.wav")
            guardar_audio(self.grabacion_audio, self.frecuencia_muestreo, nombre_archivo_wav)
            self.analizar_pronunciacion(sonido, nombre_archivo_wav)
            self.intentos[sonido] += 1
            self.ultimo_archivo_grabado = nombre_archivo_wav # Actualizar la ruta del último archivo analizado
        else:
            messagebox.showerror("Error", "No se ha grabado ningún audio.")
    def analizar_pronunciacion(self, sonido, archivo_wav):
        formantes_usuario = obtener_formantes(archivo_wav)
        datos_referencia = self.datos_referencia.get(sonido)
        es_correcto = False
        retroalimentacion = ""
        score = 0.0  # Inicializamos el score

        if formantes_usuario is not None and datos_referencia is not None and datos_referencia.size > 0:
            # Convertir datos de referencia a array de NumPy si no lo es
            datos_referencia_np = np.array(datos_referencia)

            # Calcular la distancia entre los formantes
            distancia = comparar_formantes(formantes_usuario, datos_referencia_np)

            # Obtener el score basado en la distancia
            score = obtener_score_formantes(distancia)

            umbral_correcto = 70  # Ajusta este umbral según consideres
            es_correcto = score >= umbral_correcto
            f1_usuario, f2_usuario = map(int, formantes_usuario)

            input_word = self.sonidos_objetivo[sonido][0]
            phoneme = sonido

            prompt = (
                f"El usuario intentó pronunciar la vocal /{phoneme}/.\n"
                f"Según el análisis, la primera F1 (apertura de la boca) fue de {f1_usuario} Hz, "
                f"mientras F2 (posición de la lengua) fue de {f2_usuario} Hz.\n"
                f"Para la vocal /{phoneme}/, las frecuencias usuales suelen estar alrededor de {int(np.mean(datos_referencia_np[:, 0]))} Hz para el primer formante F1"
                f"y {int(np.mean(datos_referencia_np[:, 1]))} Hz para el formante F2.\n"
                f"Considerando esta información, brinda un consejo breve y amigable para ayudar al usuario a mejorar su pronunciación de esta vocal. "
                f"El consejo debe ser fácil de entender y enfocado en cómo podría sentir o mover su boca y lengua. Evita usar términos técnicos como 'Formantes', 'F1' o 'F2'."
            )
            try:
                respuesta = generar_feedback(input_word, phoneme, f1_usuario, f2_usuario, datos_referencia)
                retroalimentacion_con_score = f"Score: {score:.2f}/100\n{respuesta}"
                self.actualizar_retroalimentacion_texto(retroalimentacion_con_score)
            except Exception as e:
                self.actualizar_retroalimentacion_texto(f"Score: {score:.2f}/100\nError al generar retroalimentación: {e}")
        else:
            self.actualizar_retroalimentacion_texto("⚠️ No se pudieron analizar los formantes o no hay datos de referencia para esta vocal.")

        if es_correcto:
            self.correctos[sonido] += 1
        self.actualizar_score_display()
        self.analizar_button.config(state=tk.DISABLED)

    def reproducir_grabacion(self):
        if self.ultimo_archivo_grabado and os.path.exists(self.ultimo_archivo_grabado):
            try:
                data, fs = sf.read(self.ultimo_archivo_grabado)
                sd.play(data, fs)
                sd.wait()
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo reproducir la grabación: {e}")
        else:
            messagebox.showinfo("Información", "No hay ninguna grabación para reproducir.")

    def actualizar_retroalimentacion_texto(self, texto):
        self.retroalimentacion_text.config(state=tk.NORMAL)
        self.retroalimentacion_text.delete(1.0, tk.END)
        self.retroalimentacion_text.insert(tk.END, texto)
        self.retroalimentacion_text.config(state=tk.DISABLED)

    def actualizar_score_display(self):
        score_text = "Score:\n"
        for sonido, intentos in self.intentos.items():
            if intentos > 0:
                precision = (self.correctos[sonido] / intentos) * 100
                score_text += f"{sonido} ({self.sonidos_objetivo[sonido][0]}): {precision:.2f}% ({intentos} intentos)\n"
            else:
                score_text += f"{sonido} ({self.sonidos_objetivo[sonido][0]}): 0% (0 intentos)\n"
        self.score_label.config(text=score_text)

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()