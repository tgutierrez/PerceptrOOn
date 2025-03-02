using PerceptrOOn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tests
{
    [TestFixture]
    public class CanInferWithReLuAndSoftMaxOutput
    {
        [Test]
        public async Task DoTest() {
            Globals.DefaultParallelOptions = new ParallelOptions() { MaxDegreeOfParallelism = 1 };
            var xorNetwork = new NeuralNetwork(new NetworkDefinition(
               InputNodes: 3,
               HiddenLayerNodeDescription: [128, 64],
               OutputNodes: 2,
               Strategies: new Strategies(new ReLuActivationStrategy(0, x => 0.5, x => 0), new DefaultComputeStrategy()),
               UseSoftMaxOutput: true
            ));

            var trainingParameters = new TrainingParameters(
                    TrainingDataSet: [
                        new TrainingData([0d, 0d, 1d], [1d, 0d]),
                        new TrainingData([1d, 1d, 1d], [0d, 1d]),
                        new TrainingData([1d, 0d, 1d], [0d, 1d]),
                        new TrainingData([0d, 1d, 0d], [1d, 0d]),
                    ],
                    Epochs: 10000,
                    TrainingRate: 0.01
                );

            await xorNetwork.Train(trainingParameters);

            
            var output = await xorNetwork.Predict([1d, 1d, 1d]);

            Assert.That(output[0], Is.LessThan(0.1));
            Assert.That(output[1], Is.GreaterThan(0.9));

            output = await xorNetwork.Predict([1d, 0d, 1d]);

            Assert.That(output[0], Is.LessThan(0.1));
            Assert.That(output[1], Is.GreaterThan(0.9));

            output = await xorNetwork.Predict([0d, 1d, 0d]);

            Assert.That(output[0], Is.GreaterThan(0.9));
            Assert.That(output[1], Is.LessThan(0.1));
            

            // TODO: Asserts
            //Assert.That(output, Is.EqualTo(new double[] { 0.98701988175004707d }));

        }
    }
}
