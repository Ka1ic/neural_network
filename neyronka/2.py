import numpy as np
import pandas as pd
from tkinter import *


train = pd.read_csv('train.csv')

Y_train = train['label'].values
X_train = train.drop(labels=['label'], axis= 1)
X_train = X_train / 255.0
X_train = X_train.values

inp = X_train[0]
weights = np.zeros((10, 784))
epochs = 1
alpha = 0.001

n = len(X_train)
true_pred = [0] * 10
for epoch in range(epochs):
    error_all = 0
    for i in range(n):
        inp = X_train[i]
        pred = np.dot(weights, inp)
        true_pred[Y_train[i]] = 1
        # error = (pred0 - true_pred0) ** 2

        for j in range(784):
            for k in range(10):
                weights[k][j] -= (pred[k] - true_pred[k]) * inp[j] * alpha

        true_pred[Y_train[i]] = 0
        if i % 1000 == 0:
            print(f'{round(i * 100 / n)}%')
        # error_all += error

    # print(f'Error: {error_all/n}')


# paint part
ind_num = 0
def paint_number():
    clear_canvas()
    global ind_num, lable_current_number
    num = X_train[ind_num]
    lable_current_number['text'] = f"Current number: {Y_train[ind_num]}"
    ind_num += 1
    for i in range(784):
        if num[i] != 0:
            x, y = i * 20 % 560, i * 20 // 560 * 20
            set_pixel(x, y)

def draw(event):
    x, y = event.x // 20 * 20, event.y // 20 * 20
    if x > 540 or y > 540 or x < 0 or y < 0:
        return
    set_pixel(x, y, 40)

def set_pixel(x, y, size = 20):
    global lable_pred_number, canvas_vector
    canvas.create_rectangle(x, y, x + size, y + size, fill=color, width=0)
    canvas_vector[x // 20 + y // 20 * 28] = 1
    res = np.dot(weights, canvas_vector)

    lable_pred_number["text"] = f"Prediction: {res.argmax()}"


def clear_canvas():
    canvas.delete('all')
    canvas['bg'] = 'black'
    for i in range(748):
        canvas_vector[i] = 0

x, y = 0, 0
brush_size = 10
color = "white"
canvas_vector = np.array([0] * 784)


root = Tk()
root.title("Digit recognaizer")
root.geometry("700x700")
root.resizable(0,0)

root.columnconfigure(6, weight = 1)
root.rowconfigure(2, weight = 1)

canvas = Canvas(root, bg="black", width=555, height=575)
canvas.grid(row=2, column=0, columnspan=7, padx=5, pady=5)

canvas.bind("<B1-Motion>", draw)

Button(root, text="Clear", width=10, command=clear_canvas).grid(row=1, column=2)
Button(root, text="next num", width=10, command=paint_number).grid(row=1, column=3)
lable_current_number = Label(root, text=f"Current number: {Y_train[ind_num]}")
lable_current_number.grid(row=1, column=4)
lable_pred_number = Label(root, text=f"Current number: {-1}")
lable_pred_number.grid(row=1, column=5)


root.mainloop()