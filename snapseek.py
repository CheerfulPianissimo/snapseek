import os
import tkinter as tk
import pyautogui
from PIL import Image, ImageTk
from search import find_closest

def show_images():
    # Create the main window

    inputfile = pyautogui.prompt("INPUT FILE PATH")
    print(inputfile)

    root = tk.Tk()
    root.title("Image Viewer")

    # Create a frame to contain the images and filenames
    frame = tk.Frame(root)
    frame.pack()

    # Get all file names in the folder
    file_names = find_closest(inputfile)
    image = Image.open(inputfile)
    image.thumbnail((300, 300))  # Resize the image to fit within a 300x300 box

    # Convert image to Tkinter-compatible format
    tk_image = ImageTk.PhotoImage(image)

    # Create a label to display the image and its filename
    image_label = tk.Label(frame, image=tk_image, text="Input File", compound=tk.TOP, wraplength=100)
    image_label.image = tk_image  # Keep a reference to prevent garbage collection
    image_label.pack(side=tk.LEFT, padx=10, pady=10)

    # Create a label for each image and its filename
    for file_name in file_names:
        # Open the image file
        image = Image.open(file_name)
        image.thumbnail((300, 300))  # Resize the image to fit within a 300x300 box

        # Convert image to Tkinter-compatible format
        tk_image = ImageTk.PhotoImage(image)

        # Create a label to display the image and its filename
        image_label = tk.Label(frame, image=tk_image, text=file_name, compound=tk.TOP, wraplength=100)
        image_label.image = tk_image  # Keep a reference to prevent garbage collection
        image_label.pack(side=tk.LEFT, padx=10, pady=10)

    # Run the Tkinter event loop
    root.mainloop()

# Example usage:
show_images()
