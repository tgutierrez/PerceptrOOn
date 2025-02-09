using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Text.Json;
using System.Threading.Tasks;

namespace PerceptrOOn.Exporters
{
    #region JSON Exporter

    /// <summary>
    /// Serializes / Imports a network into a JSON with the weights and biases but no values.
    /// </summary>
    /// <remarks>
    /// Will export/import using the following assumtions: 
    ///         - All Layers are contigous (duh!)
    ///         - First layer will always be the input layer
    ///         - Anything in between will always be a hidden layer
    ///         - Last layer will always be the output layer

    /// </remarks>
    public class JSONExporter : INetworkExporter<string>
    {
        public string Export(ILayer[] layers, IActivationStrategy activationStrategy)
        {
            var exportableLayer = new List<ExportableLayer>();

            ExportLayers(layers, exportableLayer);

            var exportableNetwork = new ExportableNetwork(exportableLayer.ToArray(), activationStrategy.Name);

            return JsonSerializer.Serialize(exportableNetwork, SourceGenerationContext.Default.ExportableNetwork);
        }

        private static void ExportLayers(ILayer[] layers, List<ExportableLayer> exportableLayer)
        {
            foreach (var layer in layers)
            {
                var exportedNodes = new List<ExportableNode>();

                ExportNodes(layer, exportedNodes);

                exportableLayer.Add(new ExportableLayer(exportedNodes.ToArray(), layer.Id));
            }
        }

        private static void ExportNodes(ILayer layer, List<ExportableNode> exportedNodes)
        {
            foreach (var node in layer.Content)
            {

                var exportedWeights = new List<ExportableWeight>();
                ExportWeights(node, exportedWeights);

                exportedNodes.Add(new ExportableNode(exportedWeights.ToArray(), layer.Id, node.Id, (node as Neuron)?.Bias));
            }
        }

        private static void ExportWeights(INode node, List<ExportableWeight> exportedWeights)
            => node.InputWeights.Apply(weight => exportedWeights.Add(new ExportableWeight(weight.Value, weight.LinkedFrom.Id, weight.LinksTo.Id)));

        public ILayer[] LoadData(string data)
        {
            throw new NotImplementedException();
        }

        public string ExportTrainingData(TrainingData[] trainingData) => JsonSerializer.Serialize(trainingData, SourceGenerationContext.Default.TrainingDataArray);
    }


    internal record ExportableNetwork(ExportableLayer[] Layers, string ActivationStrategy);

    internal record ExportableLayer(ExportableNode[] Nodes, int LayerId);

    internal record ExportableNode(ExportableWeight[] Weights, int LayerId, int NodeId, double? Bias);

    internal record ExportableWeight(double Value, int FromNodeId, int ToNodeId);

    [JsonSourceGenerationOptions(WriteIndented = true)]
    [JsonSerializable(typeof(ExportableNetwork))]
    [JsonSerializable(typeof(ExportableLayer))]
    [JsonSerializable(typeof(ExportableNode))]
    [JsonSerializable(typeof(ExportableWeight))]
    [JsonSerializable(typeof(TrainingParameters))]
    [JsonSerializable(typeof(TrainingData))]
    internal partial class SourceGenerationContext : JsonSerializerContext
    {
    }

    #endregion
}
