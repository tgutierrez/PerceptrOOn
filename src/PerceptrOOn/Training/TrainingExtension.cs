using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptrOOn
{
    public class TrainingParameters
    {
        public required TrainingData[] TrainingDataSet { get; init; }
        public required TrainingData[] ValidationSet { get; init; }
        public int Epochs { get; init; }
        public double InitialLearningRate { get; init; }
        public int BatchSize { get; init; }
        public required ILossFunction LossFunction { get; init; }
        public required LearningRateScheduler LearningRateScheduler { get; init; }
        public double EarlyStoppingPatience { get; init; }
        public double ValidationSplit { get; init; }
    }

    public interface ILossFunction
    {
        string Name { get; }
        double ComputeLoss(double[] predicted, double[] expected);
        double[] ComputeGradient(double[] predicted, double[] expected);
    }

    public class LearningRateScheduler
    {
        private readonly double initialRate;
        private readonly double decayRate;
        private readonly int decaySteps;

        public LearningRateScheduler(double initialRate, double decayRate, int decaySteps)
        {
            this.initialRate = initialRate;
            this.decayRate = decayRate;
            this.decaySteps = decaySteps;
        }

        public double GetLearningRate(int epoch)
        {
            // Exponential decay
            return initialRate * Math.Pow(decayRate, epoch / decaySteps);
        }
    }

    public class TrainingMetrics
    {
        public double TrainingLoss { get; set; }
        public double ValidationLoss { get; set; }
        public double LearningRate { get; set; }
        public int Epoch { get; set; }
        public TimeSpan EpochDuration { get; set; }
    }

    public partial class NeuralNetwork
    {
        private readonly List<TrainingMetrics> trainingHistory = new();

        public async Task Train(TrainingParameters parameters)
        {
            var bestValidationLoss = double.MaxValue;
            var epochsWithoutImprovement = 0;
            var random = new Random(42); // For shuffling

            for (var epoch = 0; epoch < parameters.Epochs; epoch++)
            {
                var epochStart = DateTime.Now;
                var currentLearningRate = parameters.LearningRateScheduler.GetLearningRate(epoch);

                // Shuffle training data
                var shuffledData = parameters.TrainingDataSet
                    .OrderBy(x => random.Next())
                    .ToArray();

                // Mini-batch training
                for (int i = 0; i < shuffledData.Length; i += parameters.BatchSize)
                {
                    var batch = shuffledData
                        .Skip(i)
                        .Take(parameters.BatchSize)
                        .ToArray();

                    await TrainBatch(batch, currentLearningRate, parameters.LossFunction);
                }

                // Compute metrics
                var metrics = await ComputeMetrics(parameters, epoch, currentLearningRate, epochStart);
                trainingHistory.Add(metrics);

                // Early stopping check
                if (metrics.ValidationLoss < bestValidationLoss)
                {
                    bestValidationLoss = metrics.ValidationLoss;
                    epochsWithoutImprovement = 0;
                    SaveCheckpoint("best_model");
                }
                else
                {
                    epochsWithoutImprovement++;
                    if (epochsWithoutImprovement >= parameters.EarlyStoppingPatience)
                    {
                        Definition.NotificationCallback?.Invoke(
                        epoch,
                        parameters.Epochs,
                        $"Early stopping triggered at epoch {epoch}");
                        
                        break;
                    }
                }

                // Notify progress
                Definition.NotificationCallback?.Invoke(
                    epoch,
                    parameters.Epochs,
                    $"Epoch {epoch + 1}/{parameters.Epochs}, Loss: {metrics.TrainingLoss:F4}, Val Loss: {metrics.ValidationLoss:F4}, LR: {currentLearningRate:F6}");
            }
        }

        private async Task TrainBatch(TrainingData[] batch, double learningRate, ILossFunction lossFunction)
        {
            // Accumulate gradients across the batch
            var accumulatedGradients = new double[OutputLayer.Size];

            foreach (var item in batch)
            {
                var predicted = await Predict(item.input);
                var gradients = lossFunction.ComputeGradient(predicted, item.expectedOutput);

                for (int i = 0; i < gradients.Length; i++)
                    accumulatedGradients[i] += gradients[i];
            }

            // Average the gradients
            for (int i = 0; i < accumulatedGradients.Length; i++)
                accumulatedGradients[i] /= batch.Length;

            // Create error adjustments using averaged gradients
            var outputErrors = OutputLayer.Neurons
                .Index()
                .Select(e => new OutputErrorAdjustment(e.Item, accumulatedGradients[e.Index]))
                .ToArray();

            // Backpropagate
            await OutputLayer.BackPropagate(outputErrors, learningRate);
        }

        private async Task<TrainingMetrics> ComputeMetrics(
            TrainingParameters parameters,
            int epoch,
            double currentLearningRate,
            DateTime epochStart)
        {
            double trainingLoss = 0;
            double validationLoss = 0;

            // Compute training loss
            foreach (var item in parameters.TrainingDataSet)
            {
                var predicted = await Predict(item.input);
                trainingLoss += parameters.LossFunction.ComputeLoss(predicted, item.expectedOutput);
            }
            trainingLoss /= parameters.TrainingDataSet.Length;

            // Compute validation loss
            if (parameters.ValidationSet != null)
            {
                foreach (var item in parameters.ValidationSet)
                {
                    var predicted = await Predict(item.input);
                    validationLoss += parameters.LossFunction.ComputeLoss(predicted, item.expectedOutput);
                }
                validationLoss /= parameters.ValidationSet.Length;
            }

            return new TrainingMetrics
            {
                Epoch = epoch,
                TrainingLoss = trainingLoss,
                ValidationLoss = validationLoss,
                LearningRate = currentLearningRate,
                EpochDuration = DateTime.Now - epochStart
            };
        }

        public void SaveCheckpoint(string path)
        {
            // Implementation for saving model state
        }

        public IReadOnlyList<TrainingMetrics> GetTrainingHistory() => trainingHistory.AsReadOnly();
    }

    /// <summary>
    /// Mean Squared Error (MSE) loss function.
    /// Commonly used for regression problems.
    /// Loss = (1/n) * Σ(y_pred - y_true)²
    /// </summary>
    public class MeanSquaredError : ILossFunction
    {
        public string Name => "MSE";

        public double ComputeLoss(double[] predicted, double[] expected)
        {
            if (predicted.Length != expected.Length)
                throw new ArgumentException("Predicted and expected arrays must be the same length");

            double sumSquaredError = 0;
            for (int i = 0; i < predicted.Length; i++)
            {
                double error = predicted[i] - expected[i];
                sumSquaredError += error * error;
            }

            return sumSquaredError / predicted.Length;
        }

        public double[] ComputeGradient(double[] predicted, double[] expected)
        {
            // The gradient of MSE is: 2(y_pred - y_true)/n
            var gradient = new double[predicted.Length];
            for (int i = 0; i < predicted.Length; i++)
            {
                gradient[i] = 2 * (predicted[i] - expected[i]) / predicted.Length;
            }
            return gradient;
        }
    }

    /// <summary>
    /// Binary Cross Entropy loss function.
    /// Used for binary classification problems.
    /// Loss = -Σ(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    /// </summary>
    public class BinaryCrossEntropy : ILossFunction
    {
        private const double Epsilon = 1e-7; // Small constant to prevent log(0)

        public string Name => "BinaryCrossEntropy";

        public double ComputeLoss(double[] predicted, double[] expected)
        {
            if (predicted.Length != expected.Length)
                throw new ArgumentException("Predicted and expected arrays must be the same length");

            double loss = 0;
            for (int i = 0; i < predicted.Length; i++)
            {
                // Clip predictions to prevent log(0)
                double clippedPred = Math.Max(Epsilon, Math.Min(1 - Epsilon, predicted[i]));
                loss += -(expected[i] * Math.Log(clippedPred) + (1 - expected[i]) * Math.Log(1 - clippedPred));
            }

            return loss / predicted.Length;
        }

        public double[] ComputeGradient(double[] predicted, double[] expected)
        {
            // The gradient of binary cross entropy is: (y_pred - y_true)/(y_pred * (1 - y_pred))
            var gradient = new double[predicted.Length];
            for (int i = 0; i < predicted.Length; i++)
            {
                double clippedPred = Math.Max(Epsilon, Math.Min(1 - Epsilon, predicted[i]));
                gradient[i] = (clippedPred - expected[i]) / (clippedPred * (1 - clippedPred));
            }
            return gradient;
        }
    }

    public class CategoricalCrossEntropy : ILossFunction
    {
        private const double Epsilon = 1e-7;

        public string Name => "CategoricalCrossEntropy";

        public double ComputeLoss(double[] predicted, double[] expected)
        {
            if (predicted.Length != expected.Length)
                throw new ArgumentException("Predicted and expected arrays must be the same length");

            double loss = 0;
            for (int i = 0; i < predicted.Length; i++)
            {
                // Clip predictions to prevent numerical instability
                double clippedPred = Math.Max(Epsilon, Math.Min(1 - Epsilon, predicted[i]));

                // Only calculate for non-zero true values to avoid log(0)
                if (expected[i] > 0)
                {
                    // Use max to prevent -Infinity
                    loss += -expected[i] * Math.Max(-100, Math.Log(clippedPred));
                }
            }

            // Ensure loss is finite
            return double.IsFinite(loss) ? loss : 100.0;
        }

        public double[] ComputeGradient(double[] predicted, double[] expected)
        {
            var gradient = new double[predicted.Length];
            for (int i = 0; i < predicted.Length; i++)
            {
                double clippedPred = Math.Max(Epsilon, Math.Min(1 - Epsilon, predicted[i]));
                gradient[i] = clippedPred - expected[i];

                // Ensure gradient is finite
                if (!double.IsFinite(gradient[i]))
                {
                    gradient[i] = 0.0;
                }
            }
            return gradient;
        }
    }

    /// <summary>
    /// Huber Loss function.
    /// Combines the best properties of MSE and MAE (Mean Absolute Error).
    /// Less sensitive to outliers than MSE, but provides more granular gradients than MAE.
    /// </summary>
    public class HuberLoss : ILossFunction
    {
        private readonly double delta;

        public HuberLoss(double delta = 1.0)
        {
            this.delta = delta;
        }

        public string Name => "Huber";

        public double ComputeLoss(double[] predicted, double[] expected)
        {
            if (predicted.Length != expected.Length)
                throw new ArgumentException("Predicted and expected arrays must be the same length");

            double loss = 0;
            for (int i = 0; i < predicted.Length; i++)
            {
                double error = Math.Abs(predicted[i] - expected[i]);
                loss += error <= delta ?
                    0.5 * error * error :                    // MSE for small errors
                    delta * error - 0.5 * delta * delta;     // MAE for large errors
            }

            return loss / predicted.Length;
        }

        public double[] ComputeGradient(double[] predicted, double[] expected)
        {
            var gradient = new double[predicted.Length];
            for (int i = 0; i < predicted.Length; i++)
            {
                double error = predicted[i] - expected[i];
                gradient[i] = Math.Abs(error) <= delta ?
                    error :                                  // MSE gradient for small errors
                    delta * Math.Sign(error);               // MAE gradient for large errors
            }
            return gradient;
        }
    }
}
