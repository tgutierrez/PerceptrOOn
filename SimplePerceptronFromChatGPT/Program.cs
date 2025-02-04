using System;

namespace NeuralNetworkExample
{
    /// <summary>
    /// A simple neural network with one hidden layer.
    /// </summary>
    class NeuralNetwork
    {
        // Number of neurons in each layer
        private int inputNodes;
        private int hiddenNodes;
        private int outputNodes;

        // Weight matrices for connections: input->hidden and hidden->output
        private double[,] weightsInputHidden;
         private double[,] weightsHiddenOutput;

        // Biases for the hidden and output layers
        private double[] biasHidden;
        private double[] biasOutput;

        // Learning rate for gradient descent
        private double _learningRate = 0.1;

        // Random number generator for initializing weights
        private Random rand = new Random();

        public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate)
        {
            this.inputNodes = inputNodes;
            this.hiddenNodes = hiddenNodes;
            this.outputNodes = outputNodes;
            this._learningRate = learningRate;
            // Initialize weight matrices
            weightsInputHidden = new double[inputNodes, hiddenNodes];
            weightsHiddenOutput = new double[hiddenNodes, outputNodes];

            // Initialize biases
            biasHidden = new double[hiddenNodes];
            biasOutput = new double[outputNodes];

            // Randomly initialize weights and biases
            InitializeWeights(weightsInputHidden);
            InitializeWeights(weightsHiddenOutput);

            for (int i = 0; i < hiddenNodes; i++)
                biasHidden[i] = RandomWeight();

            for (int i = 0; i < outputNodes; i++)
                biasOutput[i] = RandomWeight();
        }

        /// <summary>
        /// Fills a weight matrix with random values between -1 and 1.
        /// </summary>
        private void InitializeWeights(double[,] weights)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    weights[i, j] = RandomWeight();
        }

        /// <summary>
        /// Returns a random weight between -1 and 1.
        /// </summary>
        private double RandomWeight()
        {
            return rand.NextDouble() * 2 - 1;
        }

        /// <summary>
        /// The sigmoid activation function.
        /// </summary>
        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        /// <summary>
        /// Derivative of the sigmoid function.
        /// Note: We assume that x has already been passed through Sigmoid.
        /// </summary>
        private double SigmoidDerivative(double x)
        {
            return x * (1 - x);
        }

        /// <summary>
        /// Runs a forward pass through the network and returns the output.
        /// </summary>
        public double[] Predict(double[] input)
        {
            // Compute activations for the hidden layer
            double[] hidden = new double[hiddenNodes];
            for (int i = 0; i < hiddenNodes; i++)
            {
                double sum = 0;
                for (int j = 0; j < inputNodes; j++)
                    sum += input[j] * weightsInputHidden[j, i];
                sum += biasHidden[i];
                hidden[i] = Sigmoid(sum);
            }

            // Compute activations for the output layer
            double[] output = new double[outputNodes];
            for (int i = 0; i < outputNodes; i++)
            {
                double sum = 0;
                for (int j = 0; j < hiddenNodes; j++)
                    sum += hidden[j] * weightsHiddenOutput[j, i];
                sum += biasOutput[i];
                output[i] = Sigmoid(sum);
            }

            return output;
        }

        /// <summary>
        /// Trains the network using one training example (input and target output) via backpropagation.
        /// </summary>
        public void Train(double[] input, double[] target)
        {
            // === Forward pass ===

            // Compute hidden layer activations
            double[] hidden = new double[hiddenNodes];
            for (int i = 0; i < hiddenNodes; i++)
            {
                double sum = 0;
                for (int j = 0; j < inputNodes; j++)
                    sum += input[j] * weightsInputHidden[j, i];
                sum += biasHidden[i];
                hidden[i] = Sigmoid(sum);
            }

            // Compute output layer activations
            double[] output = new double[outputNodes];
            for (int i = 0; i < outputNodes; i++)
            {
                double sum = 0;
                for (int j = 0; j < hiddenNodes; j++)
                    sum += hidden[j] * weightsHiddenOutput[j, i];
                sum += biasOutput[i];
                output[i] = Sigmoid(sum);
            }

            // === Backpropagation ===

            // Calculate output errors (difference between target and actual output)
            double[] outputErrors = new double[outputNodes];
            for (int i = 0; i < outputNodes; i++)
                outputErrors[i] = target[i] - output[i];

            // Calculate gradient for output layer
            double[] gradients = new double[outputNodes];
            for (int i = 0; i < outputNodes; i++)
            {
                // Gradient = error * derivative of activation
                gradients[i] = outputErrors[i] * SigmoidDerivative(output[i]);
                // Scale by learning rate
                gradients[i] *= _learningRate;
            }

            // Adjust weights for the hidden-to-output connections
            for (int i = 0; i < hiddenNodes; i++)
            {
                for (int j = 0; j < outputNodes; j++)
                {
                    double delta = gradients[j] * hidden[i];
                    weightsHiddenOutput[i, j] += delta;
                }
            }
            // Adjust output biases
            for (int i = 0; i < outputNodes; i++)
                biasOutput[i] += gradients[i];

            // Calculate errors for the hidden layer by "backpropagating" the output errors
            double[] hiddenErrors = new double[hiddenNodes];
            for (int i = 0; i < hiddenNodes; i++)
            {
                double error = 0;
                for (int j = 0; j < outputNodes; j++)
                    error += gradients[j] * weightsHiddenOutput[i, j];
                hiddenErrors[i] = error;
            }

            // Calculate gradient for hidden layer
            double[] hiddenGradients = new double[hiddenNodes];
            for (int i = 0; i < hiddenNodes; i++)
            {
                hiddenGradients[i] = hiddenErrors[i] * SigmoidDerivative(hidden[i]);
                hiddenGradients[i] *= _learningRate;
            }

            // Adjust weights for the input-to-hidden connections
            for (int i = 0; i < inputNodes; i++)
            {
                for (int j = 0; j < hiddenNodes; j++)
                {
                    double delta = hiddenGradients[j] * input[i];
                    weightsInputHidden[i, j] += delta;
                }
            }
            // Adjust hidden biases
            for (int i = 0; i < hiddenNodes; i++)
                biasHidden[i] += hiddenGradients[i];
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            // Create a neural network for the XOR problem:
            // 2 input nodes, 2 hidden nodes, and 1 output node.
            NeuralNetwork nn = new NeuralNetwork(2, 2, 1, 1);



            // Define the XOR training data
            double[][] trainingInputs = new double[][]
            {
                new double[] { 0, 0 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 1, 1 }
            };

            double[][] trainingOutputs = new double[][]
            {
                new double[] { 0 },
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 0 }
            };

            // Train the network for 10,000 epochs.
            // Here we cycle through the 4 training examples.
            for (int epoch = 0; epoch < 6100; epoch++)
            {
                int index = epoch % trainingInputs.Length;
                nn.Train(trainingInputs[index], trainingOutputs[index]);
            }

            // Test the network after training.
            Console.WriteLine("Testing the neural network on XOR problem:");
            foreach (var input in trainingInputs)
            {
                double[] output = nn.Predict(input);
                Console.WriteLine($"Input: {input[0]}, {input[1]} -> Output: {output[0]:F4}");
            }
        }
    }
}
