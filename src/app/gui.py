import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Label, Style, Frame
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from src.data.dataset import preprocess_image
from src.config import MODEL_PATH

def select_image():
    return filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])

def display_image(file_path, img_label):
    img = Image.open(file_path)
    max_size = 500
    original_width, original_height = img.size
    if original_width > max_size or original_height > max_size:
        if original_width > original_height:
            new_width = max_size
            new_height = int((max_size / original_width) * original_height)
        else:
            new_height = max_size
            new_width = int((max_size / original_height) * original_width)
        img = img.resize((new_width, new_height), Image.LANCZOS)

    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

def update_result_label(predicted_class, result_label):
    result_text = "Damage Detected" if predicted_class == 1 else "No Damage"
    result_label.config(text=result_text, foreground="green" if predicted_class == 1 else "blue")

def process_and_predict(model, file_path, result_label):
    try:
        processed_image = preprocess_image(file_path)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        update_result_label(predicted_class, result_label)
    except Exception as e:
        result_label.config(text=f"Error: {e}", foreground="red")

def classify_image(model, img_label, result_label):
    file_path = select_image()
    if not file_path:
        return

    try:
        display_image(file_path, img_label)
        process_and_predict(model, file_path, result_label)
    except Exception as e:
        result_label.config(text=f"Error loading image: {e}", foreground="red")

def setup_ui(root, model):
    style = Style()
    style.configure("TFrame", background="#f7f7f7")
    style.configure("TLabel", background="#f7f7f7", foreground="#333333", font=("Arial", 12))

    frame = Frame(root)
    frame.pack(pady=20)

    title_label = Label(frame, text="Seed Damage Classifier", font=("Arial", 20, "bold"), foreground="#333333")
    title_label.pack(pady=10)

    desc_label = Label(frame, text="Load an image to classify as 'Damage' or 'No Damage'.", font=("Arial", 14), foreground="#555555")
    desc_label.pack(pady=10)

    img_label = Label(frame)
    img_label.pack(pady=20)

    result_label = Label(frame, text="", font=("Arial", 16, "bold"), foreground="#333333")
    result_label.pack(pady=20)

    select_button = tk.Button(
        frame,
        text="Select Image",
        command=lambda: classify_image(model, img_label, result_label),
        font=("Arial", 14),
        bg="#4CAF50",
        fg="white",
        bd=0,
        relief="flat"
    )
    select_button.pack(pady=20)

def run_gui():
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    root = tk.Tk()
    root.title("Seed Damage Classifier")
    root.geometry("800x900")
    root.configure(bg="#f7f7f7")

    setup_ui(root, model)

    root.mainloop()
