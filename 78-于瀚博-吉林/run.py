import numpy
import matplotlib.pyplot as plt
import scipy.special
class NeuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.lr=learningrate
        self.wih=numpy.random.rand(self.hnodes,self.inodes)-0.5
        self.who=numpy.random.rand(self.onodes,self.hnodes)-0.5
        self.activation_function=lambda x:scipy.special.expit(x)
    def train(self,inputs_list,targets_list):
        inputs=numpy.array(inputs_list,ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors=targets-final_outputs
        hidden_errors=numpy.dot(self.who.T,output_errors*final_outputs*(1-final_outputs))
        self.who+=self.lr*numpy.dot((output_errors*final_outputs*(1-final_outputs)),numpy.transpose(hidden_outputs))
        self.wih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),numpy.transpose(inputs))

    def query(self,inputs):
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs
data_file = open("dataset/mnist_test.csv")
data_list = data_file.readlines()
data_file.close()
print(len(data_list))
print(data_list[0])
all_values=data_list[0].split(',')
image_array=numpy.asfarray(all_values[1:]).reshape((28,28))
onodes=10
targets=numpy.zeros(onodes)+0.01
targets[int(all_values[0])]=0.99
print(targets)
input_nodes=784
hidden_nodes=100
output_nodes=10
learning_rate=0.3
n=NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
training_data_file=open('dataset/mnist_train.csv')
training_data_list=training_data_file.readlines()
training_data_file.close()
epochs=5
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores=[]
for record in test_data_list:
    all_values=record.split(',')
    correct_number=int(all_values[0])
    print(correct_number)
    inputs=(numpy.asfarray(all_values[1:]))//255.0*0.99+0.01
    outputs=n.query(inputs)
    label=numpy.argmax(outputs)
    print(label)
    if label==correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)
scores_array=numpy.asfarray(scores)
print('perference=',scores_array.sum()/scores_array.size)
