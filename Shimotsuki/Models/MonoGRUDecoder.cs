using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;
using static TorchSharp.torch.nn.functional;

namespace Shimotsuki.Models {
    public class MonoMinGRUDecoder : Module {
        int hiddenSize;
        Embedding embedding;
        MinGRU mingru;
        Linear outLinear;

        public MonoMinGRUDecoder(int hiddenSize, int outputSize) : base("mono_mingru_decoder") {
            this.hiddenSize = hiddenSize;
            this.embedding = Embedding(outputSize, hiddenSize);
            this.mingru = new MinGRU(hiddenSize, hiddenSize);
            this.outLinear = Linear(hiddenSize, outputSize);
            RegisterComponents();
        }

        public MonoMinGRUDecoder(int hiddenSize, int outputSize, Embedding sharedEmbedding) : base("mono_mingru_decoder") {
            this.hiddenSize = hiddenSize;
            this.embedding = sharedEmbedding;
            this.mingru = new MinGRU(hiddenSize, hiddenSize);
            this.outLinear = Linear(hiddenSize, outputSize);
            RegisterComponents();
        }

        public (Tensor, Tensor) forward(Tensor input, Tensor hidden, bool debug = false) {
            if (debug) {
                Console.WriteLine("入力単語" + input.item<long>());
                Console.WriteLine("隠れ状態" + hidden);
            }

            var embed = this.embedding.forward(input).view(new long[] { 1, 1, -1 });
            embed = relu(embed);

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
            
            finalOutput = LogSoftmax(1).forward(outLinear.forward(finalOutput[0]));
            return (finalOutput, finalHidden);
        }

        public Tensor InitHidden() {
            return torch.zeros(1, hiddenSize);
        }
    }
}
