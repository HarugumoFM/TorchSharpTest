using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

namespace Shimotsuki.Models {
    public class MinGRUEncoder : Module {
        int hiddenSize;
        int numLayers;
        Embedding embedding;
        MinGRU mingru1;
        MinGRU mingru2;
        MinGRU mingru3;

        public MinGRUEncoder(int inputSize, int hiddenSize, int numLayers = 3) : base("mingru_encoder") {
            this.hiddenSize = hiddenSize;
            this.numLayers = numLayers;
            this.embedding = Embedding(inputSize, hiddenSize);
            this.mingru1 = new MinGRU(hiddenSize, hiddenSize);
            this.mingru2 = new MinGRU(hiddenSize, hiddenSize);
            this.mingru3 = new MinGRU(hiddenSize, hiddenSize);
            RegisterComponents();
        }

        public MinGRUEncoder(int inputSize, int hiddenSize, Embedding sharedEmbedding, int numLayers = 3) : base("mingru_encoder") {
            this.hiddenSize = hiddenSize;
            this.numLayers = numLayers;
            this.embedding = sharedEmbedding;
            this.mingru1 = new MinGRU(hiddenSize, hiddenSize);
            this.mingru2 = new MinGRU(hiddenSize, hiddenSize);
            this.mingru3 = new MinGRU(hiddenSize, hiddenSize);
            RegisterComponents();
        }

        public (Tensor, List<Tensor>) forward(Tensor input, List<Tensor> hiddenStates, bool debug = false) {
            if (debug)
                Console.WriteLine("入力単語" + input);

            var embed = this.embedding.forward(input).view(new long[] { 1, 1, -1 });
            if (debug)
                Console.WriteLine("埋め込み後" + embed);
            
            var currentInput = embed.squeeze(0);
            var newHiddenStates = new List<Tensor>();
            
            (var output1, var newHidden1) = this.mingru1.forward(currentInput, hiddenStates[0].squeeze(0));
            currentInput = output1;
            newHiddenStates.Add(newHidden1.unsqueeze(0));
            
            (var output2, var newHidden2) = this.mingru2.forward(currentInput, hiddenStates[1].squeeze(0));
            currentInput = output2;
            newHiddenStates.Add(newHidden2.unsqueeze(0));
            
            (var output3, var newHidden3) = this.mingru3.forward(currentInput, hiddenStates[2].squeeze(0));
            currentInput = output3;
            newHiddenStates.Add(newHidden3.unsqueeze(0));
            
            var finalOutput = currentInput.unsqueeze(0);

            if (debug) {
                Console.WriteLine("出力" + finalOutput);
                Console.WriteLine("隠れ状態数" + newHiddenStates.Count);
            }
            return (finalOutput, newHiddenStates);
        }

        public List<Tensor> InitHiddenMultiLayer() {
            var hiddenStates = new List<Tensor>();
            for (int i = 0; i < numLayers; i++) {
                hiddenStates.Add(torch.zeros(1, hiddenSize));
            }
            return hiddenStates;
        }

        public (Tensor, Tensor) forward(Tensor input, Tensor hidden, bool debug = false) {
            var hiddenList = new List<Tensor> { hidden };
            for (int i = 1; i < numLayers; i++) {
                hiddenList.Add(torch.zeros_like(hidden));
            }
            
            var (output, newHiddenList) = forward(input, hiddenList, debug);
            return (output, newHiddenList[numLayers - 1]);
        }

        public Tensor InitHidden() {
            return torch.zeros(1, hiddenSize);
        }
    }
}
