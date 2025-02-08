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
        public void LoadFromJson() {
            var json = "{\"Layers\":[{\"Nodes\":[{\"Weights\":[],\"LayerId\":0,\"NodeId\":0,\"Bias\":null},{\"Weights\":[],\"LayerId\":0,\"NodeId\":1,\"Bias\":null}],\"LayerId\":0},{\"Nodes\":[{\"Weights\":[{\"Value\":-3.7241567729032054,\"FromNodeId\":0,\"ToNodeId\":0},{\"Value\":3.9130559345822379,\"FromNodeId\":1,\"ToNodeId\":0}],\"LayerId\":1,\"NodeId\":0,\"Bias\":1.57802623477755},{\"Weights\":[{\"Value\":5.5738090005914959,\"FromNodeId\":0,\"ToNodeId\":1},{\"Value\":-5.432654716658529,\"FromNodeId\":1,\"ToNodeId\":1}],\"LayerId\":1,\"NodeId\":1,\"Bias\":0.7718123395624025}],\"LayerId\":1},{\"Nodes\":[{\"Weights\":[{\"Value\":-6.686771700572806,\"FromNodeId\":0,\"ToNodeId\":0},{\"Value\":4.194934255662118,\"FromNodeId\":1,\"ToNodeId\":0}],\"LayerId\":2,\"NodeId\":0,\"Bias\":-0.7269644531904816},{\"Weights\":[{\"Value\":-2.463404459598283,\"FromNodeId\":0,\"ToNodeId\":1},{\"Value\":10.070951759429974,\"FromNodeId\":1,\"ToNodeId\":1}],\"LayerId\":2,\"NodeId\":1,\"Bias\":-0.9259824051755065}],\"LayerId\":2},{\"Nodes\":[{\"Weights\":[{\"Value\":9.740100297784176,\"FromNodeId\":0,\"ToNodeId\":0},{\"Value\":-9.366787001961745,\"FromNodeId\":1,\"ToNodeId\":0}],\"LayerId\":3,\"NodeId\":0,\"Bias\":4.536107768187019}],\"LayerId\":3}],\"ActivationStrategy\":\"Sigmoid\"}";

            var importer = new JSONImporter();
            var importedLayers = importer.Import(json, null);

            var xorNetwork = new NeuralNetwork(importedLayers, new SigmoidActivationStrategy());

            // Test validity:
            var input = new double[] { 1, 0 };

            var output = xorNetwork.Predict(input);

            // TODO: Asserts
            Assert.That(output, Is.EqualTo(new double[] { 0.98701988175004707d }));
        }
    }
}
