using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptrOOn
{
    public class DefaultComputeStrategy: IComputeStrategies
    {
        public readonly Func<IEnumerable<double>, double> GetSum;

        public DefaultComputeStrategy(bool useFastSum = false) {
            GetSum = useFastSum?
                                (s) => s.Fast_Sum():
                                (s) => s.Sum();
        }

        public string Name => "Default";

        public Task ComputeLayer<T>(IActivationStrategy strategy, T layer) where T : ILayer
        {
            if (layer is Layer layerToCompute) { 
                ComputeLayer(strategy, layerToCompute);
            }

            return Task.CompletedTask; // No awaitable code here
        }

        public Task<ComputeNodeOutput> ComputeNode<T>(IActivationStrategy strategy, T node) where T : INode
        {
            if (node is Neuron neuron) {
                return Task.FromResult(ComputeNeuron(strategy, neuron));
            }

            return Task.FromResult(new ComputeNodeOutput(0,0));
        }

        private void ComputeLayer(IActivationStrategy strategy, Layer layer)
        {
            Parallel.ForEach(layer.Neurons, async (neuron) => { await neuron.ComputeValue(); });
        }

        private ComputeNodeOutput ComputeNeuron(IActivationStrategy strategy, Neuron neuron) {

            var sum = 0d;  

            for (int i = 0; i < neuron.InputWeights.Count; i++)
            {
                sum += neuron.InputWeights[i].Compute();
            }

            var logit = sum + neuron.Bias;
            var activated = strategy.ComputeActivation(logit);
            return new ComputeNodeOutput(activated, logit);
        }

    }
}
