import os
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import cv2
from tf_utils import suppress_tf_warnings

# Class names for prediction
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_DESCRIPTIONS = {
    "akiec": "Actinic Keratosis & Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion"
}
RISK_LEVELS = {
    "akiec": "Moderate to High",
    "bcc": "Moderate",
    "bkl": "Low",
    "df": "Low",
    "mel": "High",
    "nv": "Low",
    "vasc": "Low to Moderate"
}

class DermatologyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dermatological Diagnosis System")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        # Suppress TensorFlow warnings
        suppress_tf_warnings()
        
        # Create frames
        self.header_frame = Frame(root, bg="#4e6a85", padx=10, pady=10)
        self.header_frame.pack(fill=tk.X)
        
        self.content_frame = Frame(root, bg="#f0f0f0", padx=20, pady=20)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_frame = Frame(self.content_frame, bg="#f0f0f0", width=400, height=400)
        self.image_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.result_frame = Frame(self.content_frame, bg="#f0f0f0", width=400, height=400)
        self.result_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Header
        self.title_label = Label(
            self.header_frame, 
            text="Dermatological Diagnosis System", 
            font=("Arial", 20, "bold"),
            bg="#4e6a85",
            fg="white"
        )
        self.title_label.pack(pady=10)
        
        # Buttons
        self.btn_frame = Frame(self.header_frame, bg="#4e6a85")
        self.btn_frame.pack(pady=10)
        
        self.load_btn = Button(
            self.btn_frame, 
            text="Load Image", 
            command=self.load_image,
            font=("Arial", 12),
            bg="#204060",
            fg="white",
            padx=10,
            pady=5
        )
        self.load_btn.grid(row=0, column=0, padx=10)
        
        self.predict_btn = Button(
            self.btn_frame, 
            text="Diagnose", 
            command=self.predict,
            font=("Arial", 12),
            bg="#204060",
            fg="white",
            padx=10,
            pady=5,
            state=tk.DISABLED
        )
        self.predict_btn.grid(row=0, column=1, padx=10)
        
        self.model_selection_btn = Button(
            self.btn_frame,
            text="Select Model",
            command=self.select_model,
            font=("Arial", 12),
            bg="#204060",
            fg="white",
            padx=10,
            pady=5
        )
        self.model_selection_btn.grid(row=0, column=2, padx=10)
        
        # Image display
        self.image_label = Label(
            self.image_frame, 
            text="No image loaded",
            font=("Arial", 14),
            bg="#f0f0f0",
            width=40,
            height=20
        )
        self.image_label.pack(pady=10)
        
        # Results display
        self.result_title = Label(
            self.result_frame,
            text="Diagnosis Results",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0"
        )
        self.result_title.pack(pady=10)
        
        self.result_label = Label(
            self.result_frame,
            text="Upload an image and click Diagnose",
            font=("Arial", 12),
            bg="#f0f0f0",
            justify=tk.LEFT,
            wraplength=350
        )
        self.result_label.pack(pady=10)
        
        # Initialize variables
        self.image_path = None
        self.image_array = None
        self.model = None
        self.model_path = None
        self.current_prediction = None
        
        # Status bar
        self.status_bar = Label(
            root, 
            text="Ready. Please load an image and select a model.",
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            bg="#e0e0e0"
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Check for available models
        self.find_available_models()
    
    def find_available_models(self):
        """Find available models in the current directory"""
        self.available_models = [f for f in os.listdir('.') if f.endswith(('.h5', '.keras', '.tflite'))]
        if self.available_models:
            self.status_bar.config(text=f"Found {len(self.available_models)} models. Please select one.")
        else:
            self.status_bar.config(text="Warning: No models found. Please check your model files.")
    
    def select_model(self):
        """Open a dialog to select a model file"""
        if not self.available_models:
            self.status_bar.config(text="No models available to select.")
            return
            
        model_window = tk.Toplevel(self.root)
        model_window.title("Select Model")
        model_window.geometry("400x300")
        model_window.configure(bg="#f0f0f0")
        
        Label(
            model_window,
            text="Select a Model",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0"
        ).pack(pady=10)
        
        models_frame = Frame(model_window, bg="#f0f0f0")
        models_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        for i, model_name in enumerate(self.available_models):
            model_btn = Button(
                models_frame,
                text=model_name,
                font=("Arial", 12),
                bg="#e0e0e0",
                command=lambda m=model_name: self.load_model(m, model_window),
                width=30,
                anchor="w",
                padx=10,
                pady=5
            )
            model_btn.pack(pady=5)
    
    def load_model(self, model_name, window):
        """Load the selected model"""
        self.model_path = model_name
        window.destroy()
        
        self.status_bar.config(text=f"Loading model: {model_name}...")
        
        try:
            if model_name.endswith('.tflite'):
                # Handle TFLite model
                self.interpreter = tf.lite.Interpreter(model_path=model_name)
                self.interpreter.allocate_tensors()
                self.model = None  # We'll use the interpreter directly
                self.model_type = 'tflite'
            else:
                # Load regular Keras/TF model
                custom_objects = {
                    'feature_mask_tf': tf.constant(np.ones((2048,), dtype=np.float32))
                }
                self.model = tf.keras.models.load_model(
                    model_name, 
                    custom_objects=custom_objects,
                    compile=False
                )
                self.model_type = 'keras'
            
            self.status_bar.config(text=f"Model loaded: {model_name}")
            
            # Enable predict button if image is also loaded
            if self.image_path:
                self.predict_btn.config(state=tk.NORMAL)
        
        except Exception as e:
            self.status_bar.config(text=f"Error loading model: {e}")
    
    def load_image(self):
        """Open a file dialog to load an image"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
        
        self.image_path = file_path
        self.status_bar.config(text=f"Image loaded: {os.path.basename(file_path)}")
        
        try:
            # Load and display the image
            img = Image.open(file_path)
            img = img.resize((350, 350), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=img_tk, text="")
            self.image_label.image = img_tk  # Keep a reference
            
            # Preprocess image for the model
            img_cv = cv2.imread(file_path)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_cv = cv2.resize(img_cv, (299, 299))
            self.image_array = img_cv / 255.0
            
            # Enable predict button if model is also loaded
            if self.model_path:
                self.predict_btn.config(state=tk.NORMAL)
        
        except Exception as e:
            self.status_bar.config(text=f"Error loading image: {e}")
    
    def predict(self):
        """Run prediction on the loaded image using the loaded model"""
        if self.image_array is None or (self.model is None and self.model_type != 'tflite'):
            self.status_bar.config(text="Please load both an image and a model first")
            return
        
        try:
            img_batch = np.expand_dims(self.image_array, axis=0).astype(np.float32)
            
            if self.model_type == 'tflite':
                # Use TFLite interpreter
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()
                
                self.interpreter.set_tensor(input_details[0]['index'], img_batch)
                self.interpreter.invoke()
                predictions = self.interpreter.get_tensor(output_details[0]['index'])[0]
            else:
                # Use Keras model
                predictions = self.model.predict(img_batch)[0]
            
            # Get top prediction
            pred_class_idx = np.argmax(predictions)
            pred_confidence = predictions[pred_class_idx] * 100
            pred_class = CLASS_NAMES[pred_class_idx]
            
            # Store current prediction
            self.current_prediction = {
                'class_idx': pred_class_idx,
                'class_name': pred_class,
                'confidence': pred_confidence,
                'predictions': predictions
            }
            
            # Format and display results
            result_text = f"Diagnosis: {CLASS_DESCRIPTIONS[pred_class]}\n\n"
            result_text += f"Confidence: {pred_confidence:.2f}%\n\n"
            result_text += f"Risk Level: {RISK_LEVELS[pred_class]}\n\n"
            result_text += "Other possibilities:\n"
            
            # Show top 3 predictions
            top_indices = np.argsort(predictions)[::-1][:3]
            for i, idx in enumerate(top_indices):
                if i == 0:  # Skip the top prediction as it's already shown
                    continue
                result_text += f"- {CLASS_DESCRIPTIONS[CLASS_NAMES[idx]]}: {predictions[idx]*100:.2f}%\n"
            
            # Update result display
            self.result_label.config(text=result_text)
            
            # Update status
            self.status_bar.config(text=f"Prediction complete: {CLASS_DESCRIPTIONS[pred_class]} ({pred_confidence:.2f}%)")
            
            # Color code the result based on risk
            if pred_class in ['mel', 'akiec']:
                self.result_label.config(fg="#b22222")  # Red for high risk
            elif pred_class in ['bcc']:
                self.result_label.config(fg="#ff8c00")  # Orange for moderate risk
            else:
                self.result_label.config(fg="#006400")  # Green for low risk
        
        except Exception as e:
            self.status_bar.config(text=f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = DermatologyApp(root)
    root.mainloop()
