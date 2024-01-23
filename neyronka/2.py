import copy
import tkinter.ttk

import numpy as np
import pandas as pd
from tkinter import *

# activation function and her derivitev
relu = lambda x: (x >= 0) * x
relu_deriv = lambda x: x >= 0

# load data
train = pd.read_csv('train.csv')

# prepeart data to work
Y = train['label'].values
X = train.drop(labels=['label'], axis=1)
X = X / 255.0
X = X.values

# separation data to test and train
test_size = 7200
X_train = X[:-test_size]
Y_train = Y[:-test_size]

X_test = X[-test_size:]
Y_test = Y[-test_size:]

# a convenient type of response
Y_acum = np.zeros((len(X_train), 10))
for i, l in enumerate(Y_train):
    Y_acum[i][l] = 1
Y_train = copy.deepcopy(Y_acum)

Y_acum = np.zeros((len(X_test), 10))
for i, l in enumerate(Y_test):
    Y_acum[i][l] = 1
Y_test = copy.deepcopy(Y_acum)

# inicialization weights and other variables
layer_0 = X_train[0]
hidden_size = 40
epochs = 10
alpha = 0.005
batch_size = 1
weights_0_1 = np.random.random((784, hidden_size)) * 0.2 - 0.1
weights_1_2 = np.random.random((hidden_size, 10)) * 0.2 - 0.1


def get_pred(inp):
    layer_hidden = relu(np.dot(inp, weights_0_1))
    pred = np.dot(layer_hidden, weights_1_2)
    return pred


# lerning and testing part
for epoch in range(epochs):
    error, correct_cnt = 0.0, 0
    for i in range(len(X_train) - batch_size + 1):
        layer_0 = X_train[i:i + batch_size]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        error += np.sum((Y_train[i:i + batch_size] - layer_2) ** 2)
        correct_cnt += int(np.argmax(layer_2) == np.argmax(Y_train[i:i + 1]))

        layer_2_delta = (Y_train[i:i + batch_size] - layer_2)
        layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * relu_deriv(np.dot(layer_0, weights_0_1))

        weights_1_2 += np.dot(layer_1.T, layer_2_delta) * alpha
        weights_0_1 += np.dot(layer_0.T, layer_1_delta) * alpha

        if i % 10000 == 0:
            print(f'{round(i * 100 / len(X_train))}%')

    test_error, test_correct_cnt = 0.0, 0
    for i in range(len(X_test)):
        res = get_pred(X_test[i:i + 1])
        test_error += np.sum((Y_test[i:i + 1] - res) ** 2)
        test_correct_cnt += int(np.argmax(res) == np.argmax(Y_test[i:i + 1]))

    print(f"\nTrain error: {round(error / len(X_train), 3)} "
          f"Train correct %: {round(correct_cnt / len(X_train) * 100, 3)} "
          f"Test error: {round(test_error / len(X_test), 3)} "
          f"Test correct %: {round(test_correct_cnt / len(X_test) * 100, 3)} "
          f"epoch: {epoch + 1} ")

# paint part
ind_num = 0


def paint_number():
    clear_canvas()
    global ind_num, lable_current_number
    num = X[ind_num]
    lable_current_number['text'] = f"Current number: {Y[ind_num]}"
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


def set_pixel(x, y, size=20):
    global lable_pred_number, canvas_matrix
    canvas.create_rectangle(x, y, x + size, y + size, fill=color, width=0)
    canvas_matrix[0, x // 20 + y // 20 * 28] = 1

    res = get_pred(canvas_matrix)

    lable_pred_number["text"] = f"Prediction: {res.argmax()}"
    for i in range(10):
        bars_digits[i]["value"] = res[0, i] * 100


def clear_canvas():
    global canvas_matrix
    canvas.delete('all')
    canvas['bg'] = 'black'
    canvas_matrix = np.zeros((1, 784))
    text_digits = ""
    for i in range(10):
        bars_digits[i]["value"] = 0


x, y = 0, 0
brush_size = 10
color = "white"
canvas_matrix = np.zeros((1, 784))

root = Tk()
root.title("Digit recognaizer")
root.geometry("800x720")
root.resizable(0, 0)

root.columnconfigure(12, weight=1)
root.rowconfigure(2, weight=1)

canvas = Canvas(root, bg="black", width=555, height=575)
canvas.grid(row=3, column=0, columnspan=6, rowspan=10, padx=5, pady=5)

canvas.bind("<B1-Motion>", draw)

Button(root, text="Clear", width=12, command=clear_canvas).grid(row=2, column=2)
Button(root, text="next num", width=12, command=paint_number).grid(row=2, column=3)
lable_current_number = Label(root, text=f"Current number: {Y[ind_num]}", font="20")
lable_current_number.grid(row=2, column=4)
lable_pred_number = Label(root, text=f"Current number: {-1}", font="20")
lable_pred_number.grid(row=2, column=5)
# digits lables
bars_digits = []
for i in range(10):
    bars_digits.append(tkinter.ttk.Progressbar(orient="horizontal", length=100, value=0, mode="determinate"))
    Label(root, text=f"{i}:", font="20").grid(row=i + 3, column=6)
    bars_digits[i].grid(row=i + 3, column=7)


root.mainloop()
