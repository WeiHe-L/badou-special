import numpy
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
input_nodes=3
hidden_nodes=3
out_nodes=3
learningrate=0.3
n=NeuralNetwork(input_nodes,hidden_nodes,out_nodes,learningrate)
n.query([1.0,0.5,-1.5])
