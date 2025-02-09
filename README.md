# PerceptrOOn

This is a toy implementation of a Perceptron Neural Network on C# using a Directed Graph approach instead of the classical matrix based code.

This is also a very inneficient and memory/cpu consuming architecture, but IMHO, it's a more intuitive approach for people (like me) that are just starting to learn Neural Networks, as it puts the "network" in Neural Network.

Currently, I'm using the MNIST Dataset to test the concept. On the ConsoleHost, there's an example of training the network and LoadWeightsConsoleHost has a demo on loading training weights from a JSON file. MNIST.json has a network trained on the full dataset , 30 epochs.

There are still many inference errors, so use it at your own risk

If anyone wants to use this, I recommend building on "release", otherwise it will be super slow.

Currently, it supports hardware acceleration via Vector intrinsic support (using HPCSharp). 

Native AOT is also supported (including weight serialization).

Future Enhancements:

- ~A way to serialize / deserialize the network, so weights can be distributed~. --> Done! Added JSONExporter and JSONImporter classes and a test project for it.
- A UI to visualize nodes, weights and (ideally) the training process.
- A UI to visualize the content of the hidden layers 

