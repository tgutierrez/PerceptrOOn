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
            foreach (var deserializedLayer in exportableNetwork.Layers.OrderBy(b => b.LayerId))
            {
                ILayer? layer = null;
                layer = deserializedLayer.LayerType switch
                {
                    "InputLayer" => CreateInputLayer(deserializedLayer),
                    "OutputLayer" => CreateOutputLayer(deserializedLayer, previousLayer!, strategies),
                    "HiddenLayer" => CreateHiddenLayer(deserializedLayer, previousLayer!, strategies),
                    "SoftMaxOutputLayer" => new SoftMaxOutputLayer(previousLayer!, previousLayer!.Size, strategies),
                    _ => throw new InvalidOperationException($"Unknown Layer Type {deserializedLayer.LayerType}")
                };

                layers.Add(layer!);
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
            var inputWeights = new List<Weight>();
            var outputWeights = new List<Weight>();
            var current = new Neuron(strategies,
                                                        inputWeights,  
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
