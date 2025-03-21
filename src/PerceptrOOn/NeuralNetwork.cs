﻿#region Base Definitions

using PerceptrOOn;
using System.Collections.Concurrent;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography.X509Certificates;

public static class Globals {
        public static ParallelOptions DefaultParallelOptions = new ParallelOptions { MaxDegreeOfParallelism = -1 };
}
public interface ILayer {
    
    int Id { get; }
    int Size { get; }
    double Loss { get; }

    INode[] Content {  get; }

    string GetLayerName();

    Task BackPropagate(IBackPropagationInput[] backPropagationInput, double rate);
}


public interface IGradientDescentEnabledLayer
{
    public void BackpropagateGradients(IPreviousLayerGradientDescentContainer accumulator);
    public void PerformGradientDescent(IGradientDescentAccumulator accumulator);
}

public interface IOutputLayer : ILayer
{
    public Task Compute();
    public double[] CollectOutput();

    public Task ComputeLoss(double[] expectedOutput);

    public Task<IBackPropagationInput[]> CreateBackPropagationInputFromExpected(TrainingData trainingData);

    public Task<IGradientDescentAccumulator> GetGradientsFromBackPropagation(TrainingData trainingData, double rate);
}

public interface INode : IComparable
{
    int Id { get; }
    public double Value { get; }

    public Task ComputeValue();

    public ILayer NodeLayer { get; }

    public List<Weight> OutputWeights { get; }

    public List<Weight> InputWeights { get; }

}

public interface IGradientDescentAccumulator
{
    Dictionary<int, IEnumerable<IGradientDescentInput>> Gradients { get; }
    double LeariningRate { get; }

    void Add(ILayer layer, IEnumerable<IGradientDescentInput> gradientDescentInputs);

    Task PerformGradientDescent();
}

public interface IPreviousLayerGradientDescentContainer : IGradientDescentAccumulator
{
    List<IGradientDescentInput> PreviousLayerGradientDescent { get; }

    void AddCurrentAsPreviousLayer(IGradientDescentInput[] gradientDescentInputs);
    void AddCurrentAsPreviousLayer(List<IGradientDescentInput> gradientDescentInputs);
}


public class Weight
{
    private readonly INode linkedFrom;
    private readonly INode linksTo;

    public Weight(INode linkedFrom, INode linksTo, double initialValue)
    {
        this.linkedFrom = linkedFrom;
        this.linksTo = linksTo;
        Value = initialValue;
        // add to previous node
        linkedFrom.OutputWeights.Add(this);
    }

    public INode LinkedFrom  => linkedFrom;
    public INode LinksTo => linksTo;

    public double Value { get; private set; }
    public double Gradient { get; private set; }
    public double Compute()
    {
        return Value * LinkedFrom.Value;
    }

    public void SetWeightTo(double value)
    {
        Value = value;
    }
    public void SetGradient(double gradient)
    {
        Gradient = gradient;
    }
};

public interface IActivationStrategy
{
    string Name { get; }
    double ComputeActivation(double input);
    double ComputeActivationDerivative (double input);
    double GetRandomWeight(int inputs);
    double GetInitialBias();
}

public interface IComputeStrategies 
{
    string Name { get; }
    Task ComputeLayer<T>(IActivationStrategy strategy, T layer) where T : ILayer;
    Task<ComputeNodeOutput> ComputeNode<T>(IActivationStrategy strategy, T node) where T : INode;
}

public interface INetworkExporter<T>
{
    public T Export(ILayer[] layer, Strategies strategies);
}

public interface INetworkImporter<T> 
{
    public ILayer[] Import(T networkData, Func<Strategies>? strategyProvider);

}

public delegate void Notify(int current, int total, string description);

public class Strategies(IActivationStrategy activationStrategy, IComputeStrategies computeStrategies) { 
    public IActivationStrategy ActivationStrategy => activationStrategy;
    public IComputeStrategies ComputeStrategies => computeStrategies;
};

public record NetworkDefinition(int InputNodes, int[] HiddenLayerNodeDescription, int OutputNodes, Strategies Strategies, Notify? NotificationCallback = null, bool UseSoftMaxOutput = false);
public record TrainingData(double[] input, double[] expectedOutput);
public record TrainingDataPreservingOriginal<T>(T Original, double[] input, double[] expectedOutput) : TrainingData(input, expectedOutput);
public record TrainingParameters(TrainingData[] TrainingDataSet, int Epochs, double TrainingRate);

public record struct ComputeNodeOutput(double Activated, double Logit);

#endregion

public interface IBackPropagationInput {

    double Value { get; }

    INode Node { get; }

    void ComputeWeights();

    IBackPropagationInput CreateBackPropagationInput(Strategies strategies, double rate);

}

public interface IGradientDescentInput : IBackPropagationInput, IComparable // "Marker" interface for proper method assignment
{
    public Weight[] WeightGradients { get; }
    double BiasGradient { get; }
}

#region Activation strategies

public class SigmoidActivationStrategy : IActivationStrategy
{
    private Random rand; 
    private Func<Random, double> BiasInit;
    public SigmoidActivationStrategy(int? seed = default, Func<Random, double>? biasInitialization = null) {
        rand = seed.HasValue ? new Random(seed.Value) : new Random();
        BiasInit = biasInitialization ?? new Func<Random, double>((r) => rand.NextDouble() * 2 - 1);
    }
    public string Name => "Sigmoid";

    public double ComputeActivation(double x) => 1.0 / (1.0 + Math.Exp(-x));

    public double ComputeActivationDerivative (double x) => x * (1 - x);

    public double GetRandomWeight(int inputs) => rand.NextDouble() *2 -1;

    public double GetInitialBias() => BiasInit(rand);
}

public class ReLuActivationStrategy : IActivationStrategy
{
    public ReLuActivationStrategy(int? seed = default, Func<Random, double>? biasInitializationExpression = null, Func<Random, double>? weightInitializationExpression = null) {
        rand = seed.HasValue? new Random(seed.Value) : new Random();
        this.getBiasInitExpression  = biasInitializationExpression ?? new Func<Random, double>((r) => 0);
        this.weightInitExpression = weightInitializationExpression ?? new Func<Random, double>((r) => r.NextDouble() - 0.5);
    }

    private readonly Func<Random, double> getBiasInitExpression;
    private readonly Func<Random, double> weightInitExpression;

    private readonly Random rand;

    public string Name => "ReLu";

    public double ComputeActivation(double input) => Math.Max(0, input);

    public double ComputeActivationDerivative(double input) => input switch {
        > 0 => 1,
        _ => 0,
    };

    public double GetRandomWeight(int inputs) => weightInitExpression(rand);
    public double GetInitialBias() => getBiasInitExpression(rand);
}

public static class ActivationStrategyFactory {

    public static IActivationStrategy Create(string name, int? seed = default)
        => name switch
        {
            "Sigmoid" => new SigmoidActivationStrategy(seed),
            "ReLu" => new ReLuActivationStrategy(), // ReLu with random weights -1 ... 1
            _ => throw new NotImplementedException()
        };
}

public static class ComputeStrategyFactory {
    public static IComputeStrategies Create(string name, int? seed = default)
    => name switch
    {
        _ => new DefaultComputeStrategy(),
    };
}

#endregion

/// <summary>
/// A neural network consisting of an input layer, N number of hidden layers and an output.
/// It encapsulates the logic that will train / infer / initialize the network
/// </summary>
/// <param name="input">Input data for the netwrokk</param>
/// <param name="numberOfLayers"></param>
/// <param name="actionFunctions"></param>
public class NeuralNetwork
{
    private readonly ILayer[] layers;
    private readonly Strategies strategies;
    
    private InputLayer InputLayer => (layers[0] as InputLayer)!;
    private IOutputLayer OutputLayer => (layers[^1] as IOutputLayer)!;

    private Layer[] HiddenLayer => layers[1..^1].Cast<Layer>().ToArray()!;

    public NetworkDefinition Definition { get; internal set; }

    public double CumulativeLoss {  get; internal set; }


    public List<INode> Nodes { get; internal set; } = new();
    public List<Weight> Weights { get; internal set; } = new();

    /// <summary>
    /// Builds the network using definitions
    /// </summary>
    /// <param name="definition"></param>
    public NeuralNetwork(NetworkDefinition definition) { 
        this.strategies = definition.Strategies;
        layers = BuildLayers(definition);
        this.Definition = definition;
        CollectAllNodes();
    }

    /// <summary>
    /// Builds the network using imported layers
    /// </summary>
    /// <param name="layers"></param>
    /// <param name="activationStrategy"></param>
    /// <exception cref="NotImplementedException"></exception>
    public NeuralNetwork(ILayer[] layers, Strategies strategies) { 
        this.layers = layers;
        this.strategies = strategies;
        var hiddenDefinitions = HiddenLayer.ToList().Select(h => h.Neurons.Length).ToArray();
        bool hasSoftMaxOutput = (layers[^1] is SoftMaxOutputLayer);
        this.Definition = new NetworkDefinition(InputLayer.Size, hiddenDefinitions, this.OutputLayer.Size, strategies,null, hasSoftMaxOutput); // infer definition based on the layer structure
        CollectAllNodes();
    }

    private ILayer[] BuildLayers(NetworkDefinition definition)
    {
        var layerList = new List<ILayer>();

        ILayer previousLayer = new InputLayer(definition.InputNodes);
        layerList.Add(previousLayer);

        int layerIndex = 1;
        foreach (var description in definition.HiddenLayerNodeDescription) {
            var newHiddenLayer = new HiddenLayer(previousLayer, description, this.strategies);
            layerList.Add(newHiddenLayer);
            previousLayer = newHiddenLayer;
            layerIndex++;
        }

        var outputLayer = new OutputLayer(previousLayer, definition.OutputNodes, strategies);
        layerList.Add(outputLayer);

        if (definition.UseSoftMaxOutput)
        {
            layerList.Add(new SoftMaxOutputLayer(outputLayer, definition.OutputNodes, strategies));
        }

        return layerList.ToArray();
    }

    public async Task<double[]> Predict(double[] input)
    {
        // Collect inputs
        InputLayer.CollectInput(input);

        // Compute Hidden Layers
        foreach (var hiddenLayer in this.HiddenLayer)
        {
            await hiddenLayer.Compute();
        }

        // Compute Output
        await OutputLayer.Compute()!;
        
        // Collect Result and  return
        return OutputLayer.CollectOutput();
    }

    /// <summary>
    /// Train the network using backpropagation
    /// </summary>
    /// <param name="trainingParameters">Parameters</param>
    public async Task Train(TrainingParameters trainingParameters)
    {
        foreach (var epoch in Enumerable.Range(0, trainingParameters.Epochs))
        {
            CumulativeLoss = 0;
            await TrainingEpoch(trainingParameters, epoch, trainingParameters.TrainingRate);
            Definition.NotificationCallback?.Invoke(epoch, trainingParameters.TrainingDataSet.Length, $"Finished Training Epoch {epoch + 1}/{trainingParameters.Epochs} - Loss: {(CumulativeLoss/(double)trainingParameters.TrainingDataSet.Length)}");
        }      
    }

    private async Task TrainingEpoch(TrainingParameters trainingParameters, int epoch, double rate) {
        foreach (var trainingSet in trainingParameters.TrainingDataSet)
        {
            await TrainingCycle(trainingSet, epoch, rate);
            CumulativeLoss += OutputLayer.Loss;
        }
    }

    private async Task TrainingCycle(TrainingData trainingSet, int epoch, double rate)
    {
        // 1 - Compute
        await this.Predict(trainingSet.input);
        // 2 - Compute Loss after training
        await this.OutputLayer.ComputeLoss(trainingSet.expectedOutput);
        // 3 - Start the process
        var gradients = await this.OutputLayer.GetGradientsFromBackPropagation(trainingSet, rate);
        // 4 - Compute 
        await gradients.PerformGradientDescent();
        
        
    }

    public T ExportWith<E, T>() where E : INetworkExporter<T>, new()
        => new E().Export(layers, strategies);


    public T Export<T>(INetworkExporter<T> exporter) => exporter.Export(layers, strategies);


    public void CollectAllNodes()
    {
        foreach (var layer in layers)
        {
            foreach (var node in layer.Content)
            {
                Nodes.Add(node);
                foreach (var weight in node.OutputWeights)
                {
                    Weights.Add(weight);
                }
            }
        }
    }
}

/// <summary>
/// an input layer that holds constant values used as parameters for a network
/// </summary>
public class InputLayer : ILayer, IGradientDescentEnabledLayer
{
    private readonly InputNode[] inputNodes;
    private readonly int numberOfParameters;
    public INode[] Content => inputNodes;
    public int Size => numberOfParameters;

    public int Id => 0; // Input Layer Id will always be 0.
    public double Loss => 0; // No loss on Input layer

    /// <param name="input"></param>
    public InputLayer(int numberOfParameters)
    {
        this.numberOfParameters = numberOfParameters;
        inputNodes = new InputNode[numberOfParameters];
        foreach (var index in Enumerable.Range(0, numberOfParameters))
        {
            inputNodes[index] = new InputNode(index, this);
        }
    }

    public InputLayer(Func<InputLayer, InputNode[]> getInputNode) 
    {
        var inputNodes = getInputNode(this);
        this.numberOfParameters += inputNodes.Length;
        this.inputNodes = inputNodes;
    }

    public InputNode[] InputNodes { 
        get => inputNodes;
    }

    

    public void CollectInput(double[] input) {

        if (input.Length != this.Size) throw new ArgumentException("Invalid number of inputs");
        // fill input layer
        foreach (var index in Enumerable.Range(0, input.Length))
        {
            this.InputNodes[index].SetValue(input[index]);
        }
    }

    public Task BackPropagate(IBackPropagationInput[] backPropagationInput, double gradient) => Task.CompletedTask;

    public void BackpropagateGradients(IPreviousLayerGradientDescentContainer accumulator) { } // No backpropagation on input layer
    public void PerformGradientDescent(IGradientDescentAccumulator accumulator) {


    }  

    public string GetLayerName() => "InputLayer";
}


/// <summary>
/// A neuron layer, could be hidden or output 
/// </summary>
/// <param name="inputLayer">previous layer</param>
/// <param name="size">layer size</param>
/// <param name="activationStrategy">set of action definitions</param>
public abstract class Layer : ILayer
{
    public double Loss {  get; internal set; } = 0;

    private readonly Neuron[] neurons;
    internal Strategies strategies;
    public Layer(ILayer inputLayer, int size, Strategies strategies) {
        this.Id = inputLayer.Id+1;
        this.InputLayer = inputLayer;
        this.Size = size;
        this.strategies = strategies;
        neurons = new Neuron[size];
        foreach (int index in Enumerable.Range(0, size))
        {
            neurons[index] = new Neuron(strategies, this.InputLayer, this, index);
        }
    }

    public Layer(Func<ILayer, Neuron[]> nodeDeferredWrapper, Strategies strategies, ILayer inputLayer, int id)
    {
        var neurons = nodeDeferredWrapper(this);
        this.neurons = neurons;
        this.strategies = strategies;
        InputLayer = inputLayer;
        Id = id;
        Size = this.neurons.Length;
    }

    public ILayer InputLayer { get; init; }

    public Neuron[] Neurons 
    { get => neurons.ToArray(); 
    }

    public int Id { get; init; }
    public int Size { get; init; }

    public INode[] Content => Neurons;

    public virtual async Task Compute() {
        await strategies.ComputeStrategies.ComputeLayer(strategies.ActivationStrategy, this);
    }

    public virtual double[] CollectOutput() =>
        neurons.OrderBy(w => w.Id).Select(w => w.Value).ToArray();

    public abstract Task BackPropagate(IBackPropagationInput[] backPropagationInput, double gradient);

    public void PerformGradientDescent(IGradientDescentAccumulator accumulator)
    {
        foreach (var gradient in accumulator.Gradients[Id])
        {
            var neuron = (gradient.Node as Neuron)!;
            neuron.Bias -= gradient.BiasGradient * accumulator.LeariningRate;
            foreach (var weightGradient in gradient.WeightGradients)
            {
                weightGradient.SetWeightTo(weightGradient.Value - (weightGradient.Gradient * accumulator.LeariningRate));
            }
        }

        InputLayer.PerformGradientDescentWith(accumulator);
    }

    public abstract string GetLayerName();
}

public class HiddenLayer : Layer, IGradientDescentEnabledLayer
{
    public HiddenLayer(ILayer inputLayer, int size, Strategies strategies) : base(inputLayer, size, strategies) { }
    public HiddenLayer(Func<ILayer, Neuron[]> nodeDeferredWrapper, Strategies strategies, ILayer inputLayer, int id)  : base(nodeDeferredWrapper, strategies, inputLayer, id) { }

    public override async Task BackPropagate(IBackPropagationInput[] gradients, double rate)
    {
        if (Globals.DefaultParallelOptions.MaxDegreeOfParallelism != 1)
        {
            await CalculateBackPropagateUsingParallelism(gradients, rate);
            return;
        }

        await CalculateBackPropagateUsingSequential(gradients, rate);
    }

    private async Task CalculateBackPropagateUsingSequential(IBackPropagationInput[] gradients, double rate)
    {
        var hiddenGradients = new List<Tuple<int, HiddenGradientDescentInput>>();
        var neurons = this.Neurons;

        for(int i = 0; i < neurons.Length; i++)
        {
            var neuron = neurons[i];

            double error = neuron.OutputWeights.Select(k => gradients[k.LinksTo.Id].Value * k.Value).ToArray().Fast_Sum();

            var hiddenGradient = new HiddenGradientDescentInput(neuron, error * strategies.ActivationStrategy.ComputeActivationDerivative(neuron.Logit));
            hiddenGradient.ComputeWeights();
            hiddenGradients.Add(new Tuple<int, HiddenGradientDescentInput>(neuron.Id, hiddenGradient));
        };

        await this.InputLayer.BackPropagate(hiddenGradients.OrderBy(t => t.Item1).Select(v => v.Item2).ToArray(), rate); // Ensure results are ordered, since gradients were calculated in parallel.
    }

    private async Task CalculateBackPropagateUsingParallelism(IBackPropagationInput[] gradients, double rate)
    {
        var hiddenGradients = new ConcurrentBag<Tuple<int, HiddenGradientDescentInput>>();
        var neurons = this.Neurons;

        Parallel.ForEach(Enumerable.Range(0, neurons.Length),
            Globals.DefaultParallelOptions
            , (i) =>    // Use the input length to create parallel tasks 
        {
            var neuron = neurons[i];

            double error = neuron.OutputWeights.Select(k => gradients[k.LinksTo.Id].Value * k.Value).ToArray().Fast_Sum();

            var hiddenGradient = new HiddenGradientDescentInput(neuron, error * strategies.ActivationStrategy.ComputeActivationDerivative(neuron.Logit));
            hiddenGradient.ComputeWeights();
            hiddenGradients.Add(new Tuple<int, HiddenGradientDescentInput>(i, hiddenGradient));
        });

        await this.InputLayer.BackPropagate(hiddenGradients.OrderBy(t => t.Item1).Select(v => v.Item2).ToArray(), rate); // Ensure results are ordered, since gradients were calculated in parallel.
    }

    public void BackpropagateGradients(IPreviousLayerGradientDescentContainer accumulator)
    {
        var previousLayerInput = accumulator.PreviousLayerGradientDescent;
        var gradients = new IGradientDescentInput[this.Neurons.Length];

        foreach (var neuron in this.Neurons)
        {
            var errorSum = 0d;
            foreach (var outputWeight in neuron.OutputWeights)
            {
                errorSum += outputWeight.Value * previousLayerInput[outputWeight.LinksTo.Id].Value;
            }
            gradients[neuron.Id] = new HiddenGradientDescentInput(neuron, errorSum * strategies.ActivationStrategy.ComputeActivationDerivative(neuron.Logit)).ComputeGradientDescentWeights();
        }

        accumulator.Add(this, gradients);

        this.InputLayer.PerformGradientBackwardPass(accumulator, gradients);
    }

    public override string GetLayerName() => "HiddenLayer";
}


public class OutputLayer : Layer, IOutputLayer, IGradientDescentEnabledLayer
{
    public OutputLayer(ILayer inputLayer, int size, Strategies strategies) : base(inputLayer, size, strategies) { }
    public OutputLayer(Func<ILayer, Neuron[]> nodeDeferredWrapper, Strategies strategies, ILayer inputLayer, int id) : base(nodeDeferredWrapper, strategies, inputLayer, id) { }
    public override async Task BackPropagate(IBackPropagationInput[] backPropagationInput, double rate) 
    {
        var gradients = new List<IBackPropagationInput>();
        foreach (var input in backPropagationInput)
        {
            var outputGradient = input.CreateBackPropagationInput(this.strategies, rate);
            outputGradient.ComputeWeights();
            gradients.Add(outputGradient);
        }

        // Backpropagate
        await this.InputLayer.BackPropagate(gradients.OrderBy(g => g.Node.Id).ToArray(), rate);
    }

    public Task<IBackPropagationInput[]> CreateBackPropagationInputFromExpected(TrainingData trainingData) => Task.FromResult<IBackPropagationInput[]>(
                    this.Neurons
                        .Index()
                        .Select((e) => new OutputErrorAdjustment(e.Item, trainingData.expectedOutput[e.Index])).ToArray());

    public void BackpropagateGradients(IPreviousLayerGradientDescentContainer accumulator)
    {
        var previousLayerInput = accumulator.PreviousLayerGradientDescent;
        var gradients = new IGradientDescentInput[this.Neurons.Length];
        foreach (var neuron in this.Neurons)
        {
            gradients[neuron.Id] = new OutputGradientDescentAdjustment(neuron, previousLayerInput[neuron.Id].Value)
                                            .ComputeGradientDescentWeights();
        }

        accumulator.Add(this, gradients);

        this.InputLayer.PerformGradientBackwardPass(accumulator, gradients);
    }

    public override string GetLayerName() => "OutputLayer";
    public Task ComputeLoss(double[] expectedOutput) => Task.CompletedTask;

    public async Task<IGradientDescentAccumulator> GetGradientsFromBackPropagation(TrainingData trainingData, double rate)
    {
        var inputs = await CreateBackPropagationInputFromExpected(trainingData);
        await BackPropagate(inputs, rate);
        return new GradientDescentAccumulator(rate); // Empty accumulator since this is not using softmax
    }

}

public class SoftMaxOutputLayer : Layer, IOutputLayer
{
    public SoftMaxOutputLayer(ILayer inputLayer, int size, Strategies strategies) : base(inputLayer, size, strategies) { }


    public Task ComputeLoss(double[] expectedOutput) {

        this.Loss = 0;

        foreach (var item in Neurons)
        {
            this.Loss -= expectedOutput[item.Id] * Math.Log(item.Value + 1e-8); // Avoids Log(0)
        }

        return Task.CompletedTask;
    }

    public override Task BackPropagate(IBackPropagationInput[] backPropagationInput, double rate)
    {
        throw new InvalidOperationException("No use for BackPropagate on a SoftMax layer");
    }

    public override Task Compute()
    {
        var maxValueNode = this.InputLayer.Content.Select(n => (n as Neuron)!.Logit).Max();
        var exps = new List<(INode Node, double Exp)>();
        var sumExps = 0d;
        foreach (var content in this.InputLayer.Content)
        {
            var exp = Math.Exp((content as Neuron)!.Logit - maxValueNode);
            exps.Add((Node: content, Exp: exp));
            sumExps += exp;
        }

        foreach (var item in exps)
        {
            (this.Content[item.Node.Id] as Neuron)!.SetValue(item.Exp / sumExps);
        }

        return Task.CompletedTask;
    }

    public Task<IBackPropagationInput[]> CreateBackPropagationInputFromExpected(TrainingData trainingData)
    {
        var errors = this.Neurons
            .Index()
            .OrderBy(e => e.Index)
            .Select((e) => (IBackPropagationInput)(new SoftMaxErrorAdjustment(e.Item, trainingData.expectedOutput[e.Index])))
            .ToArray();

        return Task.FromResult(errors);
    }

    public async Task<IGradientDescentAccumulator> GetGradientsFromBackPropagation(TrainingData trainingData, double rate)
    {
        var backPropagationInput = await CreateBackPropagationInputFromExpected(trainingData);
        var gradientInputs = backPropagationInput.Cast<IGradientDescentInput>().OrderBy(b => b.Node.Id).ToArray();
        // Backpropagate
        var accumulator = new GradientDescentAccumulator(rate); // Starts new Accumulator
        //accumulator.Add(this, gradientInputs);

        this.InputLayer.PerformGradientBackwardPass(accumulator, gradientInputs);

        return accumulator;
    }

    public void ComputeLoss()
    {
        throw new NotImplementedException();
    }

    public override string GetLayerName() => "SoftMaxOutputLayer";
}

/// <summary>
/// Defines a constant value node that will receive inputs
/// </summary>
/// <param name="value"></param>
public class InputNode(int id, ILayer currentLayer) : INode
{
    public int Id => id;
    public double Value { get; private set; }

    public List<Weight> OutputWeights { get; private set; } = new List<Weight>();

    public List<Weight> InputWeights { get; private set; } = new List<Weight>();

    public ILayer NodeLayer => currentLayer;

    public int CompareTo(object? obj) => (obj as INode)!.Id.CompareTo(this.Id);  

    public Task ComputeValue() => Task.CompletedTask;

    public void SetValue(double value) { 
        this.Value = value;
    }
}

/// <summary>
/// Defines a neuron node with connected weights
/// </summary>
public class Neuron: INode
{

    private readonly Strategies strategies;
    
    public Neuron(Strategies strategies, ILayer inputLayer, ILayer currentLayer, int id)
    {
        this.strategies = strategies;
        this.InputLayer = inputLayer;
        this.NodeLayer = currentLayer;
        this.Id = id;

        InitializeWeights();

        Bias = strategies.ActivationStrategy.GetInitialBias();
    }

    public Neuron(Strategies strategies, List<Weight> inputWeights, List<Weight> outputWeights, double bias, ILayer inputLayer, ILayer currentLayer, int id)
    {
        this.strategies = strategies;
        InputWeights = inputWeights;
        OutputWeights = outputWeights;
        Bias = bias;
        InputLayer = inputLayer;
        this.NodeLayer = currentLayer;
        Id = id;
    }

    private void InitializeWeights()
    {
        foreach (var input in InputLayer.Content)
        {
            this.InputWeights.Add(new Weight(input, this, strategies.ActivationStrategy.GetRandomWeight(InputLayer.Content.Length)));
        }
    }

    public async Task ComputeValue()  {
        var result = await strategies.ComputeStrategies.ComputeNode(strategies.ActivationStrategy, this); // strategies.ActivationStrategy.ComputeActivation(InputWeights.Select(x => x.Compute()).Fast_Sum() + Bias)
        this.Value = result.Activated;
        this.Logit = result.Logit;
    }

    public List<Weight> OutputWeights { get; private set; } = new List<Weight>();

    public List<Weight> InputWeights { get; private set; } = new List<Weight>();

    public double Value { get; private set;}

    public double Logit { get; private set; }

    public void SetValue(double value) => this.Value = value;

    public double Bias { get; set; }    

    public ILayer InputLayer { get; init; }

    public int Id { get; init; }

    public ILayer NodeLayer { get; init; }

    public int CompareTo(object? obj) => (obj as INode)!.Id.CompareTo(this.Id);

}

#region 

#endregion

#region BackPropagation Training

public class OutputErrorAdjustment(Neuron Neuron, double ExpectedValue) : IBackPropagationInput
{
    public INode Node => Neuron;

    public double Value => ExpectedValue;

    public IBackPropagationInput CreateBackPropagationInput(Strategies strategies, double rate)
    {
        var error = ExpectedValue - Neuron.Value;
        return new GradientAdjustment(Neuron, error * strategies.ActivationStrategy.ComputeActivationDerivative(Neuron.Value)*rate);
    }

    public void ComputeWeights()
    {
        // Adjust Weights
        Neuron.InputWeights.Apply(SetWeight);

        // Adjust Bias
        Neuron.Bias = Node.Value;
    }

    private void SetWeight(Weight weight)
    {
        double delta = Node.Value * weight.LinkedFrom.Value;
        weight.SetWeightTo(delta);
    }

    public double SampleLoss() => Value * Math.Log(Neuron.Value) + 1e-8; // Avoids Log(0)
}

public class HiddenGradientDescentInput : IGradientDescentInput
{
    private readonly Neuron neuron;
    private readonly double gradient;

    public HiddenGradientDescentInput(Neuron neuron, double gradient)
    {
        this.neuron = neuron;
        this.gradient = gradient;
        WeightGradients = new Weight[neuron.InputWeights.Count];
    }

    public double Value => gradient;

    public INode Node => neuron;

    public double BiasGradient { get; private set; }

    public Weight[] WeightGradients { get; private set; }

    public int CompareTo(object? obj)
    {
        return Node.Id.CompareTo((obj as IGradientDescentInput)!.Node.Id);
    }

    public void ComputeWeights()
    {
        BiasGradient = gradient;
        //WeightGradients = neuron.InputWeights.Select(n => new WeightGradient(n, gradient * n.LinkedFrom.Value)).ToList();
        foreach (var weight in neuron.InputWeights)
        {
            weight.SetGradient(gradient * weight.LinkedFrom.Value);
            WeightGradients[weight.LinkedFrom.Id] = weight;
        }
    }

    public IBackPropagationInput CreateBackPropagationInput(Strategies strategies, double rate)
    {
        throw new NotImplementedException();
    }
}

public class SoftMaxErrorAdjustment : IGradientDescentInput
{
    private readonly Neuron neuron;
    private readonly double expectedValue;

    public SoftMaxErrorAdjustment(Neuron Neuron, double ExpectedValue)
    {
        neuron = Neuron;
        expectedValue = ExpectedValue;
        var error = neuron.Value - expectedValue;
        Value = error;
        WeightGradients = new Weight[neuron.InputWeights.Count];
    }

    public INode Node => neuron;

    public double Value { get; private set; }

    public double BiasGradient { get; private set; }

    public IBackPropagationInput CreateBackPropagationInput(Strategies strategies, double rate)
    {
        throw new NotSupportedException();
    }

    public int CompareTo(object? obj)
    {
        return Node.Id.CompareTo((obj as IGradientDescentInput)!.Node.Id);
    }

    public void ComputeWeights()
    {
        throw new NotSupportedException();
    }
    public Weight[] WeightGradients { get; private set; }

    public double SampleLoss() => expectedValue * Math.Log(neuron.Value) + 1e-8; // Avoids Log(0)

}

public class OutputGradientDescentAdjustment : IGradientDescentInput
{
    private readonly Neuron neuron;
    private readonly double gradient;
    public OutputGradientDescentAdjustment(Neuron neuron, double gradient)
    {
        this.neuron = neuron;
        this.gradient = gradient;   
        this.Value = gradient;
        WeightGradients = new Weight[neuron.InputWeights.Count];
    }

    public double Value { get; private set; }   

    public INode Node => neuron;

    public double BiasGradient {  get; private set; }

    public Weight[] WeightGradients { get; private set; }

    public int CompareTo(object? obj)
    {
        return Node.Id.CompareTo((obj as IGradientDescentInput)!.Node.Id);
    }

    public void ComputeWeights()
    {
        BiasGradient = gradient;
        //WeightGradients = neuron.InputWeights.Select(n => new WeightGradient(n, gradient * n.LinkedFrom.Value)).ToList();
        foreach (var weight in neuron.InputWeights)
        {
            weight.SetGradient(gradient * weight.LinkedFrom.Value);
            WeightGradients[weight.LinkedFrom.Id] = weight;
        }
    }

    public IBackPropagationInput CreateBackPropagationInput(Strategies strategies, double rate)
    {
        throw new NotSupportedException();
    }

}

public class GradientAdjustment(Neuron Neuron, double Gradient) : IBackPropagationInput
{
    public INode Node => Neuron;
    public double Value => Gradient;

    public void ComputeWeights()
    {
        // Adjust Weights
        Neuron.InputWeights.Apply(SetWeight);

        // Adjust Bias
        Neuron.Bias += Gradient;
    }

    private void SetWeight(Weight weight)
    {
        double delta = Gradient * weight.LinkedFrom.Value;
        weight.SetWeightTo(weight.Value + delta);
    }

    public IBackPropagationInput CreateBackPropagationInput(Strategies strategies, double rate)
    {
        throw new NotImplementedException();
    }
}


public class GradientDescentAccumulator(double learningRate) : IDisposable, IGradientDescentAccumulator, IPreviousLayerGradientDescentContainer
{
    public double LeariningRate => learningRate;

    public Dictionary<int, IEnumerable<IGradientDescentInput>> Gradients { get; internal set; } = new Dictionary<int, IEnumerable<IGradientDescentInput>> ();

    public List<IGradientDescentInput> PreviousLayerGradientDescent { get; internal set; } = new List<IGradientDescentInput>();

    public void Add(ILayer layer, IEnumerable<IGradientDescentInput> gradientDescentInputs)
    {
        this.Gradients.Add(layer.Id, gradientDescentInputs);
    }

    public void AddCurrentAsPreviousLayer(IGradientDescentInput[] gradientDescentInputs)
    {
        PreviousLayerGradientDescent = new List<IGradientDescentInput>(gradientDescentInputs);
    }

    public void AddCurrentAsPreviousLayer(List<IGradientDescentInput> gradientDescentInputs)
    {
        PreviousLayerGradientDescent = gradientDescentInputs;
    }

    public void Dispose()
    {
        Gradients.Clear();
        Gradients = null!;
        GC.SuppressFinalize(this);
    }

    public Task PerformGradientDescent()
    {
        foreach (var gradientSet in Gradients.OrderBy(b => b.Key))
        {
            foreach (var gradient in gradientSet.Value)
            {
                var neuron = (gradient.Node as Neuron)!;
                neuron.Bias -= gradient.BiasGradient * this.LeariningRate;
                foreach (var weightGradient in gradient.WeightGradients)
                {
                    weightGradient.SetWeightTo(weightGradient.Value - (weightGradient.Gradient * this.LeariningRate));
                }
            }
        }

        return Task.CompletedTask;
    }

    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            Gradients.Clear();
            Gradients = null!;
        }
    }
}

#endregion


