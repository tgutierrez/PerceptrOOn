using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptrOOn
{
    public class DefaultComputeStrategy : IComputeStrategies
    {
        public string Name => "Default";

        public Task ComputeLayer<T>(IActivationStrategy strategy, T layer) where T : ILayer
        {
            if (layer is Layer layerToCompute) { 
                ComputeLayer(strategy, layerToCompute);
            }

            return Task.CompletedTask; // No awaitable code here
        }

        public Task<double> ComputeNode<T>(IActivationStrategy strategy, T node) where T : INode
        {
            double value = 0;
            if (node is Neuron neuron) {
                value = ComputeNeuron(strategy, neuron);
            }

            return Task.FromResult(value);
        }

        private void ComputeLayer(IActivationStrategy strategy, Layer layer)
        {
            Parallel.ForEach(layer.Neurons, async (neuron) => { await neuron.ComputeValue(); });
        }

        private double ComputeNeuron(IActivationStrategy strategy, Neuron neuron) {
            return strategy.ComputeActivation(neuron.InputWeights.Select(x => x.Compute()).Fast_Sum() + neuron.Bias);
        }

    }
}
