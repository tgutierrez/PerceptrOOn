using PerceptrOOn;

namespace Tests
{
    public class CanInferXOR
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public async Task Test1()
        {
            var xorNetwork = new NeuralNetwork(new NetworkDefinition(
                   InputNodes: 2,
                   HiddenLayerNodeDescription: [2, 2],
                   OutputNodes: 1,
                   Strategies: new Strategies(new SigmoidActivationStrategy(5), new DefaultComputeStrategy())
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

            await xorNetwork.Train(trainingParameters);

            var input = new double[] { 1, 0 };

            var output = await xorNetwork.Predict(input);

            // TODO: Asserts
            Assert.That(output, Is.EqualTo(new double[] { 0.98701988175004707d }));
        }
    }
}
