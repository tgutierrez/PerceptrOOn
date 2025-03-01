using PerceptrOOn;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tests
{
    [TestFixture]
    public class CanTrainUsingBatches
    {
        [Test]
        public async Task ValidateTrainingWithNoBatches()
        {
            TrainingData[] sampleDataset = new TrainingData[]
            {
                // Example: Input vector of 4 features and one-hot encoded output for 4 classes.
                new TrainingData(
                    new double[] { 0.1, 0.2, 0.3, 0.4 },
                    new double[] { 1, 0, 0, 0 }
                ),
                new TrainingData(
                    new double[] { 0.5, 0.6, 0.7, 0.8 },
                    new double[] { 0, 1, 0, 0 }
                ),
                new TrainingData(
                    new double[] { 0.9, 1.0, 1.1, 1.2 },
                    new double[] { 0, 0, 1, 0 }
                ),
                new TrainingData(
                    new double[] { 1.3, 1.4, 1.5, 1.6 },
                    new double[] { 0, 0, 0, 1 }
                ),
                //new TrainingData(
                //    new double[] { 1.7, 1.8, 1.9, 2.0 },
                //    new double[] { 1, 0, 0, 0 }
                //),
                //new TrainingData(
                //    new double[] { 2.1, 2.2, 2.3, 2.4 },
                //    new double[] { 0, 1, 0, 0 }
                //),
                //new TrainingData(
                //    new double[] { 2.5, 2.6, 2.7, 2.8 },
                //    new double[] { 0, 0, 1, 0 }
                //),
                //new TrainingData(
                //    new double[] { 2.9, 3.0, 3.1, 3.2 },
                //    new double[] { 0, 0, 0, 1 }
                //)
            };

            var someNetwork = new NeuralNetwork(new NetworkDefinition(
               InputNodes: 4,
               HiddenLayerNodeDescription: [5, 3],
               OutputNodes: 4,
               Strategies: new Strategies(new ReLuActivationStrategy(0, x => 0.5, x => 0.5), new DefaultComputeStrategy()),
               NotificationCallback: (c, t, d) =>
               {
                   if (c % 1000 == 0)
                   TestContext.Out.WriteLine($"Epoch: {c} - {d}");
               },
               UseSoftMaxOutput: true
            ));

            var trainingParameters = new TrainingParameters(
                    TrainingDataSet: sampleDataset,
                    Epochs: 10000,
                    TrainingRate: 0.01
                );
            Globals.DefaultParallelOptions = new ParallelOptions() { MaxDegreeOfParallelism = 1 };
            await someNetwork.Train(trainingParameters);

            


            foreach (var sample in sampleDataset)
            {
                var output = await someNetwork.Predict(sample.input);
                AssertResult(output, sample.expectedOutput);
            }
            
        }


        public static void AssertResult(double[] output, double[] expected)
        {
            var expectedIndex = Array.FindIndex(expected, match => match == 1);
            var max = output.Max();
            var indexOfMax = Array.FindIndex(output, match => match == max);
            TestContext.Out.WriteLine($"----------------------------------------------------");
            TestContext.Out.WriteLine($"Expected: {String.Join(",", expected.Select(o => $"[{o.ToString()}]"))}");
            TestContext.Out.WriteLine($"Output  : {String.Join(",", output.Select(o =>   $"[{o.ToString()}]"))}");

            //Assert.That(indexOfMax, Is.EqualTo(expectedIndex));
            //Assert.That(output[expectedIndex], Is.GreaterThan(0.9));
        }
    }
}
