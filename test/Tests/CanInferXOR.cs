﻿namespace Tests
{
    public class CanInferXOR
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void Test1()
        {
            var xorNetwork = new NeuralNetwork(new NetworkDefinition(
                   InputNodes: 2,
                   HiddenLayerNodeDescription: [2, 2],
                   OutputNodes: 1,
                   ActivationStrategy: new SigmoidActivationStrategy(5)
                ));

            var trainingParameters = new TrainingParameters(
                    TrainingDataSet: [
                        new TrainingData([0d, 0d], [0d]),
                        new TrainingData([0d, 1d], [1d]),
                        new TrainingData([1d, 0d], [1d]),
                        new TrainingData([1d, 1d], [0d]),
                    ],
                    Epochs: 10000,
                    TrainingRate: 1
                );

            xorNetwork.Train(trainingParameters);

            var input = new double[] { 1, 0 };

            var output = xorNetwork.Predict(input);

            Array.ForEach(output, Console.WriteLine);

            // TODO: Asserts
        }
    }
}
