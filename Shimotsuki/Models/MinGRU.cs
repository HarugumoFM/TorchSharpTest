using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

namespace Shimotsuki.Models {
    public class MinGRU : Module {
        int inputSize;
        int hiddenSize;
        Linear forgetGate;
        Linear candidateGate;
        
        public MinGRU(int inputSize, int hiddenSize) : base("mingru") {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.forgetGate = Linear(inputSize + hiddenSize, hiddenSize);
            this.candidateGate = Linear(inputSize + hiddenSize, hiddenSize);
            RegisterComponents();
        }
        
        public (Tensor, Tensor) forward(Tensor input, Tensor hidden) {
            if (input.dim() == 1) input = input.unsqueeze(0);
            if (hidden.dim() == 1) hidden = hidden.unsqueeze(0);
            var combined = torch.cat(new List<Tensor> { input, hidden }, dim: 1);
            var forget = torch.sigmoid(forgetGate.forward(combined));
            var candidate = torch.tanh(candidateGate.forward(combined));
            var newHidden = forget * hidden + (1 - forget) * candidate;
            return (newHidden, newHidden);
        }
        
        public Tensor InitHidden(int batchSize = 1) {
            return torch.zeros(batchSize, hiddenSize);
        }
    }
}
