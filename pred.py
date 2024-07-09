import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

img_width=224
img_height=224

# Load the saved model
model = load_model('oil.h5')



# Test on a single image
def test_on_img(model, img_path, classes):
    image = Image.open(img_path)
    image = image.resize((img_width, img_height))
    image_array = np.array(image) / 255.0
    X_test = np.expand_dims(image_array, axis=0)
    Y_pred_probs = model.predict(X_test)[0]
    Y_pred_label = np.argmax(Y_pred_probs)
    
    plt.imshow(image)
    plt.title('Predicted Label: ' + classes[Y_pred_label])
    plt.show()

# Create the Tkinter GUI window
root = tk.Tk()
root.title("Melanoma Cancer Image Classification")
root.geometry("400x400")

# Function to handle image upload and prediction
def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        test_on_img(model, file_path, cm_plot_labels)
        # Show the uploaded image on the GUI
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img  # Keep a reference to avoid garbage collection

# Button for uploading an image
upload_btn = tk.Button(root, text="Upload Image", command=upload_and_predict)
upload_btn.pack(pady=10)

# Label for displaying the uploaded image
image_label = tk.Label(root)
image_label.pack()

# Example usage
cm_plot_labels = ["benign melanoma lesion", "malignant melanoma lesion"]

# Run the Tkinter event loop
root.mainloop()