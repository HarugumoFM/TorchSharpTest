using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;
using static TorchSharp.torch.nn.functional;

namespace Shimotsuki.Models {
    public class Decoder : Module {
        int hiddenSize;
        Embedding embedding;
        GRU gru;
        Linear outLinear;

        public Decoder(int hiddenSize, int outputSize) : base("encoder") {
            this.hiddenSize = hiddenSize;
            this.embedding = Embedding(outputSize, hiddenSize);
            this.gru = GRU(hiddenSize, hiddenSize);
            this.outLinear = Linear(hiddenSize, outputSize);
            RegisterComponents();//パラメータを登録する(optimizer用)
        }

        public (Tensor, Tensor) forward(Tensor input, Tensor hidden, bool debug = false) {
            if (debug) {
                Console.WriteLine("入力単語" + input.item<long>());
                Console.WriteLine("隠れ層" + hidden + hidden.ToString());
            }

            var embed = this.embedding.forward(input).view(new long[] { 1, 1, -1 });
            embed = relu(embed);

            if (debug)
                Console.WriteLine("埋め込み後" + embed);
            (var output, hidden) = this.gru.forward(embed, hidden);

            if (debug) {
                Console.WriteLine("出力" + output);
                Console.WriteLine("隠れ状態" + hidden);
            }
            output = LogSoftmax(1).forward(outLinear.forward(output[0]));
            return (output, hidden);
        }

        public Tensor InitHidden() {
            return torch.zeros(1, 1, hiddenSize);
        }
    }
}
