using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

namespace Shimotsuki.Models {
    public class MonoMinGRUEncoder : Module {
        int hiddenSize;
        Embedding embedding;
        MinGRU mingru;

        public MonoMinGRUEncoder(int inputSize, int hiddenSize) : base("mono_mingru_encoder") {
            this.hiddenSize = hiddenSize;
            this.embedding = Embedding(inputSize, hiddenSize);
            this.mingru = new MinGRU(hiddenSize, hiddenSize);
            RegisterComponents();
        }

        public MonoMinGRUEncoder(int inputSize, int hiddenSize, Embedding sharedEmbedding) : base("mono_mingru_encoder") {
            this.hiddenSize = hiddenSize;
            this.embedding = sharedEmbedding;
            this.mingru = new MinGRU(hiddenSize, hiddenSize);
            RegisterComponents();
        }

        public (Tensor, Tensor) forward(Tensor input, Tensor hidden, bool debug = false) {
            if (debug)
                Console.WriteLine("入力単語" + input);

            var embed = this.embedding.forward(input).view(new long[] { 1, 1, -1 });
            if (debug)
                Console.WriteLine("埋め込み後" + embed);
            
            var currentInput = embed.squeeze(0);
            var (output, newHidden) = this.mingru.forward(currentInput, hidden.squeeze(0));
            
            var finalOutput = output.unsqueeze(0);
            var finalHidden = newHidden.unsqueeze(0);

            if (debug) {
                Console.WriteLine("出力" + finalOutput);
                Console.WriteLine("隠れ状態" + finalHidden);
            }
            return (finalOutput, finalHidden);
        }

        public Tensor InitHidden() {
            return torch.zeros(1, hiddenSize);
        }
    }
}
