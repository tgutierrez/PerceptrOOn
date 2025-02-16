//using PerceptrOOn;
//using PerceptrOOn.Exporters;
//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;

//namespace Tests
//{
//    [TestFixture]
//    public class CanExport
//    {
//        [Test]
//        public async Task SerializeToJSON() {

//            var xorNetwork = new NeuralNetwork(new NetworkDefinition(
//               InputNodes: 2,
//               HiddenLayerNodeDescription: [2, 2],
//               OutputNodes: 1,
//               Strategies: new Strategies(new SigmoidActivationStrategy(5), new DefaultComputeStrategy())
//            ));

//                    var trainingParameters = new TrainingParameters(
//                            TrainingDataSet: [
//                                new TrainingData([0d, 0d], [0d]),
//                                new TrainingData([0d, 1d], [1d]),
//                                new TrainingData([1d, 0d], [1d]),
//                                new TrainingData([1d, 1d], [0d]),
//                            ],
//                            Epochs: 10000,
//                            TrainingRate: 1
//                );

//            await xorNetwork.Train(trainingParameters);

//            var exporter = new JSONExporter();

//            var result = xorNetwork.Export(exporter);


//            Assert.That( result, Is.Not.Null ); // too lazy for an actual assert. 
//        }
//    }
//}
