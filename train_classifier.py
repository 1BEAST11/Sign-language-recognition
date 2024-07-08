import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Определение функции активации и ее производной
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Определение класса нейронной сети
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Инициализация весов и смещений в диапазоне от -0.5 до 0.5
        self.weights_input_hidden = np.random.uniform(-0.5, 0.5, (self.input_size, self.hidden_size))
        self.bias_input_hidden = np.random.uniform(-0.5, 0.5,(1, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(-0.5, 0.5, (self.hidden_size, self.output_size))
        self.bias_hidden_output = np.random.uniform(-0.5, 0.5, (1, self.output_size))

    def forward(self, x):
        # Прямое распространение
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        self.out_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        self.output = sigmoid(self.out_input)
        return self.output

    def backward(self, x, y, output, learning_rate):
        # Обратное распространение
        error_out = 2 * (y - output)
        output_delta = error_out * sigmoid_derivative(output)

        error_hidden = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = error_hidden * sigmoid_derivative(self.hidden_output)

        # Обновление весов между скрытым и выходным слоями
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.bias_hidden_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.weights_input_hidden += learning_rate * np.dot(x.T, hidden_delta)
        self.bias_input_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(x)
            self.backward(x, y, output, learning_rate)

    def save_weights(self, filename):
        weights = {
            'weights_input_hidden': self.weights_input_hidden,
            'bias_input_hidden': self.bias_input_hidden,
            'weights_hidden_output': self.weights_hidden_output,
            'bias_hidden_output': self.bias_hidden_output
        }
        with open(filename, 'wb') as f:
            pickle.dump(weights, f)
        print("Weights saved successfully.")
        f.close()

    def load_weights(self, filename):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        print("Weights loaded successfully.")
        self.weights_input_hidden = weights['weights_input_hidden']
        self.bias_input_hidden = weights['bias_input_hidden']
        self.weights_hidden_output = weights['weights_hidden_output']
        self.bias_hidden_output = weights['bias_hidden_output']
        f.close()

# Загрузка данных из файла pickle
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)
f.close()


# print(data_dict.keys())
# print(data_dict)

data = np.asarray(data_dict['data'], dtype=float)
labels = np.asarray(data_dict['labels'], dtype=float)

# Нормализация данных
data = (data - np.min(data)) / (np.max(data) - np.min(data))

# Разделение данных на обучающий и тестовый наборы
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Определение параметров нейронной сети
input_size = x_train.shape[1]
hidden_size = 24  # Увеличение количества нейронов в скрытом слое
output_size = 9

# Инициализация нейронной сети
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Инициализация и обучение нейронной сети
# model = NeuralNetwork(input_size, hidden_size, output_size)
model.train(x_train, y_train, epochs=1500, learning_rate=0.01)  # Увеличение количества эпох и уменьшение скорости обучения

# Сохранение весов модели
# model.save_weights('weights.pickle')

# Предсказание на тестовых данных перед загрузкой весов
print("Prediction before loading weights:")
y_predict = model.forward(x_test)
for i in range(len(y_predict)):
    print("Input:", np.array2string(y_test[i], formatter={'float_kind':lambda x: "%.3f" % x}))
    print("Output:", np.array2string(y_predict[i], formatter={'float_kind':lambda x: "%.3f" % x}))

# Загрузка весов модели
# model.load_weights('weights.pickle')

# Предсказание на тестовых данных после загрузки весов
print("Prediction after loading weights:")
y_predict = model.forward(x_test)
for i in range(len(y_predict)):
    print("Input:", np.array2string(y_test[i], formatter={'float_kind':lambda x: "%.3f" % x}))
    print("Output:", np.array2string(y_predict[i], formatter={'float_kind':lambda x: "%.3f" % x}))

# Оценка точности модели
accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
print(f"Accuracy: {accuracy:.4f}")

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
