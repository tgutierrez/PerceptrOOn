using System;
using System.Collections.Generic;
using System.ComponentModel.Design;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace PerceptrOOn.Exporters
{
    public class JSONImporter : INetworkImporter<string>
    {
        public ILayer[] Import(string networkData, Func<Strategies>? strategyProvider = default)
        {
            var exportableNetwork = JsonSerializer.Deserialize(networkData, SourceGenerationContext.Default.ExportableNetwork) ?? throw new InvalidOperationException("Network cannot be serialized");

            var strategies = strategyProvider?.Invoke() ?? new Strategies(ActivationStrategyFactory.Create(exportableNetwork.ActivationStrategy), 
                                                                          ComputeStrategyFactory.Create(exportableNetwork.ComputeStrategy)
                                                                          );

            return Import(exportableNetwork, strategies);
        }


        private ILayer[] Import(ExportableNetwork exportableNetwork, Strategies strategies)
        {
            var layers = new List<ILayer>();


            ILayer? previousLayer = null;
            int lastLayer = exportableNetwork.Layers.Length - 1;
            // extract output and input layer. everything in between will be considered a hidden layer.
            for (int i = 0; i < exportableNetwork.Layers.Length; i++)
            {

                var deserializedLayer = exportableNetwork.Layers[i];

                // check integrity by ensuring that order = id
                if (deserializedLayer.LayerId != i) throw new InvalidOperationException($"Imported Layer Index does not matches the expected order. Found {i}, Expected {deserializedLayer.LayerId}");

                ILayer? layer = null;
                if (i == 0)
                {
                    layer = CreateInputLayer(deserializedLayer);
                } else if (i == lastLayer)
                {
                    layer = CreateOutputLayer(deserializedLayer, previousLayer!, strategies);
                }
                else {
                    layer = CreateHiddenLayer(deserializedLayer, previousLayer!, strategies)!;
                }

                layers.Add(layer);
                previousLayer = layer;

            }

            if (exportableNetwork.UseSoftMaxOutput)
            {
                layers.Add(new SoftMaxOutputLayer(previousLayer!, previousLayer!.Size, strategies));
            }

            return layers.ToArray();
        }

        private ILayer? CreateHiddenLayer(ExportableLayer deserializedLayer, ILayer previousLayer, Strategies strategies)
            => new HiddenLayer(GetNeurons(deserializedLayer.Nodes, previousLayer, strategies), strategies, previousLayer, deserializedLayer.LayerId);

        private ILayer CreateOutputLayer(ExportableLayer deserializedLayer, ILayer previousLayer, Strategies strategies)
            => new OutputLayer(GetNeurons(deserializedLayer.Nodes, previousLayer, strategies), strategies, previousLayer, deserializedLayer.LayerId);

        private ILayer CreateInputLayer(ExportableLayer deserializedLayer)
            => new InputLayer(GetInputs(deserializedLayer.Nodes));

        private InputNode[] GetInputs(ExportableNode[] nodes)
            => nodes.Select(p => new InputNode(p.NodeId)).ToArray();

        private Neuron[] GetNeurons(ExportableNode[] nodes, ILayer previousLayer, Strategies activationStrategy)
            => nodes.Select(n => CreateNeuron(previousLayer, activationStrategy, n)).ToArray();

        private static Neuron CreateNeuron(ILayer previousLayer, Strategies strategies, ExportableNode exportableNode)
        {
            var inputWeights = new MutableArray<Weight>();
            var outputWeights = new MutableArray<Weight>();
            var current = new Neuron(strategies,
                                                        inputWeights,  //n.Weights.Select(p => CreateWeight(n, p, previousLayer)).ToList(),
                                                        outputWeights,
                                                        exportableNode.Bias ?? 0d,
                                                        previousLayer,
                                                        exportableNode.NodeId
                                                        );
            // wire up current weights
           exportableNode.Weights.Select(w => { 
                var weight = new Weight(previousLayer.Content[w.FromNodeId], current, w.Value); 
                return weight;
            }).ToList().ForEach(inputWeights.Add);
            
            return current;
        }

        public TrainingData[] ImportTrainingData(string json) => JsonSerializer.Deserialize(json, SourceGenerationContext.Default.TrainingDataArray) ?? throw new InvalidOperationException("Unable to Deserialize Layers Structure");
    }

}
