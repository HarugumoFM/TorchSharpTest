using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

namespace Shimotsuki.Models {
    public class Encoder : Module {
        int hiddenSize;
        Embedding embedding;
        GRU gru;

        public Encoder(int inputSize, int hiddenSize) : base("encoder") {
            this.hiddenSize = hiddenSize;
            this.embedding = Embedding(inputSize, hiddenSize);
            this.gru = GRU(hiddenSize, hiddenSize, bidirectional: true);
            RegisterComponents();//パラメータを登録する(optimizer用)
        }

        public (Tensor, Tensor) forward(Tensor input, Tensor hidden, bool debug = false) {
            if (debug)
                Console.WriteLine("入力単語" + input);

            var embed = this.embedding.forward(input).view(new long[] { 1, 1, -1 });
            if (debug)
                Console.WriteLine("埋め込み後" + embed);
            (var output, hidden) = this.gru.forward(embed, hidden);

            if (debug) {
                Console.WriteLine("出力" + output);
                Console.WriteLine("隠れ状態" + hidden);
            }
            return (output, hidden);
        }

        public Tensor InitHidden() {
            return torch.zeros(2, 1, hiddenSize);
        }
    }
}
