import numpy
import matplotlib.pyplot
import scipy.special


class neuralNetwork:
    # инициализировать нейронную сеть
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задать количество узлов во входном, скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # коэффициент обучения
        self.lr = learningrate
        # Матрицы весовых коэффициентов связей wih (между входным и скрытым слоями) и who (между скрытым и выходным слоями).
        # Весовые коэффициенты связей между узлом i и узлом j следующего слоя обозначены как w_i_j: wll w21 wl2 w22 и т.д.
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # тренировка нейронной сети
    def train(self, inputs_list, targets_list):

        # преобразовать список входных значений в двухмерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # ошибка = целевое значение - фактическое значение
        output_errors = targets - final_outputs
        # ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # обновить весовые коэффициенты связей между скрытым и выходным слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # обновить весовые коэффициенты связеq между входным и скрытым слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    # опрос нейронной сети
    def query(self, inputs_list):
        # преобразовать список входных значений в двухмерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return(final_outputs)
# Тут мы формируем список идеальных выходных значений сети из CSV-файла набора MNIST

training_data_file = open("mnist_dataset//mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Количество входных, скрытых и выходных узлов
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# Коэффициент обучения
learning_rate = 0.1

# Количество эпох (тренировка на одном и том же датасете
epochs = 5

# Экзмемпляр нейрнонной сети
NN = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Тренировка нейронной сети
for epoch in range(epochs):
    # Преобразовать все записи в тренировочном наборе данных
    for record in training_data_list:
        # Получить список значений, используя символы (',')  в качестве разделитилей
        all_values = record.split(',')
        # Данные представлены в диапазоне 0-255, для корректного использования сигмоиды нужен диапазон 0.01-1.0
        # Поэтому данные мы корректируем значения в списке
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # Создать целевые выходные значения (все равны 0,01, за исключением желаемого маркерного значения, равного 0,99
        targets = numpy.zeros(output_nodes) + 0.01

        # all_values[0] - целевое маркерное значение для данной записи
        targets[int(all_values[0])] = 0.99
        NN.train(inputs, targets)
    print("Trained on ", epoch + 1, "epoch")

# Тестирование нейронной сети

test_data_file = open("mnist_dataset//mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# Журнал оценок работы сети
scorecard = []

# Преобразовать все записи в тестовом наборе данных
for record in training_data_list:
    # Полечить значения из записи, используя символы (",") в качестве разделителей
    all_values = record.split(',')
    # Правильный ответ - первое значение
    correct_label = int(all_values[0])
    print(correct_label, " - истинный маркер")
    # Масштабировать и сместить входные значения
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # Опрос сети
    outputs = NN.query(inputs)
    # Индекс наибольшего значения является маркерным значением
    label = numpy.argmax(outputs)
    # Присоединить оценку ответа сети к концу списка
    if (label == correct_label):
        # В случае правильного ответа сети присоеденить к списку 1
        scorecard.append(1)
        print(label, " - ответ сети правельный!")
    else:
        # В ином случае присоеденить к списку 0
        scorecard.append(0)
        print(label, " - ответ сети не правельный (-_-)")
scorecard_array = numpy.asarray(scorecard)
print("Эффективность сети - ", (scorecard_array.sum() / scorecard_array.size) * 100, "%")
image_array = numpy.asfarray(all_values[1:]).reshape(28, 28)
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')

print(NN.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01))

print("Every thing is ok!")