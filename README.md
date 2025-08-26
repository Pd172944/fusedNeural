[IN PROGRESS]

A from scratch low-level neural network in C with a focus on performance optimization for resource constrained environments.

Single layer with forward and backward passes
Backprop enginen using MSE loss function
Kernel fusion
Post Training quantization



**How to Run**
1. Clone the repository:
  git clone https://github.com/Pd172944/fusedNeural.git
2. Set synthetic data params
   in main.c, set input_data and target_data dimensions (lines 59 and 63)
    
3. Compile the source files
   gcc -g tensor.c operations.c neuralNet.c loss.c quantized.c main.c -o neuralNet
4. Run the program
   ./neuralNet
  
