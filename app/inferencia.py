import subprocess
import numpy as np
import tensorflow as tf

def run_cpp(img, out):
    subprocess.run(["./cpp/preprocesamiento", img, out], check=True)

def load_txt(path):
    with open(path) as f:
        values = [float(x) for x in f.read().split(",")]

    arr = np.array(values).reshape(224, 224, 3)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(model, img_path, txt_path):
    run_cpp(img_path, txt_path)
    x = load_txt(txt_path)

    pred = model.predict(x)
    return int(np.argmax(pred))
