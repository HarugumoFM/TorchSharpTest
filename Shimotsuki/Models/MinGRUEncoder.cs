using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

namespace Shimotsuki.Models {
    public class MinGRUEncoder : Module {
        int hiddenSize;
        int numLayers;
        Embedding embedding;
        List<MinGRU> mingruLayers;

        public MinGRUEncoder(int inputSize, int hiddenSize, int numLayers = 3) : base("mingru_encoder") {
            this.hiddenSize = hiddenSize;
            this.numLayers = numLayers;
            this.embedding = Embedding(inputSize, hiddenSize);
            this.mingruLayers = new List<MinGRU>();
            
            for (int i = 0; i < numLayers; i++) {
                this.mingruLayers.Add(new MinGRU(hiddenSize, hiddenSize));
            }
            RegisterComponents();
        }

        public MinGRUEncoder(int inputSize, int hiddenSize, Embedding sharedEmbedding, int numLayers = 3) : base("mingru_encoder") {
            this.hiddenSize = hiddenSize;
            this.numLayers = numLayers;
            this.embedding = sharedEmbedding;
            this.mingruLayers = new List<MinGRU>();
            
            for (int i = 0; i < numLayers; i++) {
                this.mingruLayers.Add(new MinGRU(hiddenSize, hiddenSize));
            }
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
            
            for (int i = 0; i < numLayers; i++) {
                (var output, var newHidden) = this.mingruLayers[i].forward(currentInput, hiddenStates[i].squeeze(0));
                currentInput = output;
                newHiddenStates.Add(newHidden.unsqueeze(0));
            }
            
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
