// See https://aka.ms/new-console-template for more information

using System.Security.Cryptography;

var network = new NeuralNetwork(new NetworkDefinition(
       InputNodes: 2,
       HiddenLayerNodeDescription: [2, 2],
       OutputNodes: 1,
       ActivationStrategy: new SigmoidActivationStrategy(5)
    ));

var input = new double[] { 0, 0 };

var output = network.Predict(input);

Array.ForEach(output, Console.WriteLine);

#region Base Definitions

public interface ILayer {
    int Size { get; }
    INode[] Content {  get; }
}

public interface INode
{
    public double Value { get; }

    public void ComputeValue();

}

public class Weight(INode linkedFrom, INode linksTo, double initialValue)
{
    public INode LinkedFrom  => linkedFrom;
    public INode LinksTo => linksTo;

    public double Value { get; private set; } = initialValue;

    public double Compute()
    {
        return Value * LinkedFrom.Value;
    }

    public void SetWeightTo(double value)
    {
        Value = value;
    }
};

public interface IActivationStrategy
{
    double ComputeActivation(double input);
    double ComputeActivationDerivative (double input);
    double GetRandomWeight();
}



public record NetworkDefinition(int InputNodes, int[] HiddenLayerNodeDescription, int OutputNodes, IActivationStrategy ActivationStrategy);
public record struct TrainingData(double[] input, double[] expectedOutput);
public record TrainingParameters(TrainingData[] TrainingDataSet, int Epochs, double TrainingRate);
#endregion

#region Activation strategies

public class SigmoidActivationStrategy(int seed) : IActivationStrategy
{
    private Random rand = new(seed);

    public double ComputeActivation(double x) => 1.0 / (1.0 + Math.Exp(-x));

    public double ComputeActivationDerivative (double x) => x * (1 - x);

    public double GetRandomWeight() => rand.NextDouble() *2 -1;
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
    private readonly IActivationStrategy activationStrategy;

    private InputLayer InputLayer => (layers[0] as InputLayer)!;
    private Layer OutputLayer => (layers[^1] as Layer)!;
    private Layer[] HiddenLayer => layers[1..^1].Cast<Layer>().ToArray()!;

    public NeuralNetwork(NetworkDefinition definition) { 
        this.activationStrategy = definition.ActivationStrategy;
        layers = BuildLayers(definition);
    }

    private ILayer[] BuildLayers(NetworkDefinition definition)
    {
        var layerList = new List<ILayer>();

        ILayer previousLayer = new InputLayer(definition.InputNodes);
        layerList.Add(previousLayer);
        foreach (var index in Enumerable.Range(1, definition.HiddenLayerNodeDescription.Length -1)) { 
            var newHiddenLayer = new Layer(previousLayer, definition.HiddenLayerNodeDescription[index], this.activationStrategy);
            layerList.Add(newHiddenLayer);
            previousLayer = newHiddenLayer;
        }
        layerList.Add(new Layer(previousLayer, definition.OutputNodes, activationStrategy));

        return layerList.ToArray();
    }

    public double[] Predict(double[] input)
    {
        // Collect inputs
        InputLayer.CollectInput(input);

        // Compute Hidden Layers
        foreach (var hiddenLayer in this.HiddenLayer)
        {
            hiddenLayer.Compute();
        }

        // Compute Output
        OutputLayer.Compute();

        // Collect Result and  return
        return OutputLayer.CollectOutput();
    }

    /// <summary>
    /// Train the network using backpropagation
    /// </summary>
    /// <param name="trainingParameters">Parameters</param>
    public void Train(TrainingParameters trainingParameters)
    {
        // Backpropagation
        foreach (var epoch in Enumerable.Range(0, trainingParameters.Epochs))
        {
            TraininingCycle(trainingParameters, epoch);
        }      
    }

    public void TraininingCycle(TrainingParameters trainingParameters, int epoch) { 
        // Collect outputs

    }
}

/// <summary>
/// an input layer that holds constant values used as parameters for a network
/// </summary>
public class InputLayer : ILayer
{
    private readonly InputNode[] inputNodes;
    private readonly int numberOfParameters;

    /// <param name="input"></param>
    public InputLayer(int numberOfParameters)
    {
        this.numberOfParameters = numberOfParameters;
        inputNodes = new InputNode[numberOfParameters];
        foreach (var index in Enumerable.Range(0, numberOfParameters))
        {
            inputNodes[index] = new InputNode();
        }
    }
    public int Size => numberOfParameters;

    public InputNode[] InputNodes { 
        get => inputNodes;
    }

    public INode[] Content => inputNodes;

    public void CollectInput(double[] input) {

        if (input.Length != this.Size) throw new ArgumentException("Invalid number of inputs");
        // fill input layer
        foreach (var index in Enumerable.Range(0, input.Length))
        {
            this.InputNodes[index].SetValue(input[index]);
        }
    }
}

/// <summary>
/// A neuron layer, could be hidden or output 
/// </summary>
/// <param name="inputLayer">previous layer</param>
/// <param name="size">layer size</param>
/// <param name="activationStrategy">set of action definitions</param>
public class Layer : ILayer
{
    private readonly Neuron[] neurons;

    public Layer(ILayer inputLayer, int size, IActivationStrategy activationStrategy) {
        this.InputLayer = inputLayer;
        this.Size = size;
        neurons = new Neuron[size];
        foreach (int index in Enumerable.Range(0, size))
        {
            neurons[index] = new Neuron(activationStrategy, this.InputLayer);
        }
    }

    public ILayer InputLayer { get; init; }

    public Neuron[] Neurons 
    { get => neurons.ToArray(); 
    }


    public int Size { get; init; }

    public INode[] Content => Neurons;

    public void Compute() {
        foreach (var neuron in neurons) {
            neuron.ComputeValue();
        }
    }

    public double[] CollectOutput() =>
        neurons.Select(w => w.Value).ToArray();
}

/// <summary>
/// Defines a constant value node that will receive inputs
/// </summary>
/// <param name="value"></param>
public class InputNode() : INode
{
    public double Value { get; private set; }

    public void ComputeValue() { }

    public void SetValue(double value) { 
        this.Value = value;
    }
}

/// <summary>
/// Defines a neuron node with connected weights
/// </summary>
public class Neuron: INode
{

    private readonly IActivationStrategy actions;
    
    public Neuron(IActivationStrategy activationStrategy, ILayer inputLayer)
    {
        this.actions = activationStrategy;
        this.InputLayer = inputLayer;
        this.Weights = new List<Weight>();
        InitializeWeights();
        Bias = activationStrategy.GetRandomWeight();
    }

    private void InitializeWeights()
    {
        foreach (var input in InputLayer.Content)
        {
            this.Weights.Add(new Weight(input, this, actions.GetRandomWeight()));
        }
    }

    public void ComputeValue() => this.Value = actions.ComputeActivation(Weights.Sum(x => x.Compute()) + Bias);

    public List<Weight> Weights { get; private set; }

    public double Value { get; private set;}

    public double Bias { get; private set; }    

    public ILayer InputLayer { get; init; }

}

