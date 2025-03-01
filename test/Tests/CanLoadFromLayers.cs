using PerceptrOOn;
using PerceptrOOn.Exporters;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tests
{
    [TestFixture]
    public class CanLoadFromLayers
    {
        [Test]
        public async Task LoadFromJsonNoSoftMax() {
            var json = "{\"ActivationStrategy\": \"Sigmoid\", \"ComputeStrategy\": \"Default\", \"Layers\": [{\"LayerId\": 0, \"LayerType\": \"InputLayer\", \"Nodes\": [{\"Bias\": null, \"LayerId\": 0, \"NodeId\": 0, \"Weights\": []}, {\"Bias\": null, \"LayerId\": 0, \"NodeId\": 1, \"Weights\": []}]}, {\"LayerId\": 1, \"LayerType\": \"HiddenLayer\", \"Nodes\": [{\"Bias\": 1.57802623477755, \"LayerId\": 1, \"NodeId\": 0, \"Weights\": [{\"FromNodeId\": 0, \"ToNodeId\": 0, \"Value\": -3.72415677290321}, {\"FromNodeId\": 1, \"ToNodeId\": 0, \"Value\": 3.91305593458224}]}, {\"Bias\": 0.771812339562402, \"LayerId\": 1, \"NodeId\": 1, \"Weights\": [{\"FromNodeId\": 0, \"ToNodeId\": 1, \"Value\": 5.5738090005915}, {\"FromNodeId\": 1, \"ToNodeId\": 1, \"Value\": -5.43265471665853}]}]}, {\"LayerId\": 2, \"LayerType\": \"HiddenLayer\", \"Nodes\": [{\"Bias\": -0.726964453190482, \"LayerId\": 2, \"NodeId\": 0, \"Weights\": [{\"FromNodeId\": 0, \"ToNodeId\": 0, \"Value\": -6.68677170057281}, {\"FromNodeId\": 1, \"ToNodeId\": 0, \"Value\": 4.19493425566212}]}, {\"Bias\": -0.925982405175506, \"LayerId\": 2, \"NodeId\": 1, \"Weights\": [{\"FromNodeId\": 0, \"ToNodeId\": 1, \"Value\": -2.46340445959828}, {\"FromNodeId\": 1, \"ToNodeId\": 1, \"Value\": 10.07095175943}]}]}, {\"LayerId\": 3, \"LayerType\": \"OutputLayer\", \"Nodes\": [{\"Bias\": 4.53610776818702, \"LayerId\": 3, \"NodeId\": 0, \"Weights\": [{\"FromNodeId\": 0, \"ToNodeId\": 0, \"Value\": 9.74010029778418}, {\"FromNodeId\": 1, \"ToNodeId\": 0, \"Value\": -9.36678700196174}]}]}], \"UseSoftMaxOutput\": false}";

            var importer = new JSONImporter();
            var importedLayers = importer.Import(json, null);

            var xorNetwork = new NeuralNetwork(importedLayers, new Strategies(new SigmoidActivationStrategy(), new DefaultComputeStrategy()));

            // Test validity:
            var input = new double[] { 1, 0 };

            var predicted = await xorNetwork.Predict(input);
            var output = Math.Round((predicted)[0], 2);


            // TODO: Asserts
            Assert.That(output, Is.EqualTo(Math.Round(0.98701988175004707d, 2)));
        }

        [Test]
        public async Task LoadFromJsonSoftMax()
        {
            var json = "{\"ActivationStrategy\": \"Sigmoid\", \"ComputeStrategy\": \"Default\", \"Layers\": [{\"LayerId\": 0, \"LayerType\": \"InputLayer\", \"Nodes\": [{\"Bias\": null, \"LayerId\": 0, \"NodeId\": 0, \"Weights\": []}, {\"Bias\": null, \"LayerId\": 0, \"NodeId\": 1, \"Weights\": []}]}, {\"LayerId\": 1, \"LayerType\": \"HiddenLayer\", \"Nodes\": [{\"Bias\": 1.57802623477755, \"LayerId\": 1, \"NodeId\": 0, \"Weights\": [{\"FromNodeId\": 0, \"ToNodeId\": 0, \"Value\": -3.72415677290321}, {\"FromNodeId\": 1, \"ToNodeId\": 0, \"Value\": 3.91305593458224}]}, {\"Bias\": 0.771812339562402, \"LayerId\": 1, \"NodeId\": 1, \"Weights\": [{\"FromNodeId\": 0, \"ToNodeId\": 1, \"Value\": 5.5738090005915}, {\"FromNodeId\": 1, \"ToNodeId\": 1, \"Value\": -5.43265471665853}]}]}, {\"LayerId\": 2, \"LayerType\": \"HiddenLayer\", \"Nodes\": [{\"Bias\": -0.726964453190482, \"LayerId\": 2, \"NodeId\": 0, \"Weights\": [{\"FromNodeId\": 0, \"ToNodeId\": 0, \"Value\": -6.68677170057281}, {\"FromNodeId\": 1, \"ToNodeId\": 0, \"Value\": 4.19493425566212}]}, {\"Bias\": -0.925982405175506, \"LayerId\": 2, \"NodeId\": 1, \"Weights\": [{\"FromNodeId\": 0, \"ToNodeId\": 1, \"Value\": -2.46340445959828}, {\"FromNodeId\": 1, \"ToNodeId\": 1, \"Value\": 10.07095175943}]}]}, {\"LayerId\": 3, \"LayerType\": \"OutputLayer\", \"Nodes\": [{\"Bias\": 4.53610776818702, \"LayerId\": 3, \"NodeId\": 0, \"Weights\": [{\"FromNodeId\": 0, \"ToNodeId\": 0, \"Value\": 9.74010029778418}, {\"FromNodeId\": 1, \"ToNodeId\": 0, \"Value\": -9.36678700196174}]}]}], \"UseSoftMaxOutput\": true}";

            var importer = new JSONImporter();
            var importedLayers = importer.Import(json, null);

            var xorNetwork = new NeuralNetwork(importedLayers, new Strategies(new SigmoidActivationStrategy(), new DefaultComputeStrategy()));

            // Test validity:
            var input = new double[] { 1, 0 };

            var predicted = await xorNetwork.Predict(input);
            var output = Math.Round((predicted)[0], 2);


            // TODO: Asserts
            Assert.That(output, Is.EqualTo(1));
        }
    }
}
