using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptrOOn
{
    public class DefaultComputeStrategy: IComputeStrategies
    {
        public readonly Func<double[], double> GetSum;

        private const double MaxValue = 100.0; // Prevent exponential overflow
        private const double MinValue = -100.0;

        public DefaultComputeStrategy(bool useFastSum = true) {
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

            var sum = GetSum(neuron.InputWeights.Select(x =>
                                    x.Compute()
                                            .CoalesceInfinite()
                                            .Clamp(MinValue, MaxValue)
            ));

            var biasedSum = (sum + neuron.Bias).Clamp(MinValue, MaxValue);

            var activatedValue = strategy.ComputeActivation(biasedSum);

            return activatedValue.CoalesceInfinite();
        }

    }
}
