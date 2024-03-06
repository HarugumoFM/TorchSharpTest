using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;
using static TorchSharp.torch.nn.functional;

namespace Shimotsuki.Models {
    public class AttnDecoder : Module
    {
        int hiddenSize;
        public Embedding embedding;
        GRU gru;
        Linear outLinear;
        Attn Attn;

        public AttnDecoder(int hiddenSize, int outputSize) : base("decoder")
        {
            this.hiddenSize = hiddenSize;
            this.embedding = Embedding(outputSize, hiddenSize);
            this.gru = GRU(hiddenSize * 2, hiddenSize);
            this.outLinear = Linear(hiddenSize, outputSize);
            this.Attn = new Attn(hiddenSize);
            RegisterComponents();//パラメータを登録する(optimizer用)
        }

        public AttnDecoder(int hiddenSize, int outputSize, Embedding encoder_embedding) : base("decoder")
        {
            this.hiddenSize = hiddenSize;
            this.embedding = encoder_embedding;
            this.gru = GRU(hiddenSize * 2, hiddenSize);
            this.outLinear = Linear(hiddenSize, outputSize);
            this.Attn = new Attn(hiddenSize);
            RegisterComponents();//パラメータを登録する(optimizer用)
        }

        public (Tensor, Tensor) forward(Tensor input, Tensor hidden, Tensor EncoderOutPuts, bool debug = false)
        {
            if (debug)
            {
                Console.WriteLine("入力単語" + input.item<long>());
                Console.WriteLine("隠れ層" + hidden + hidden.ToString());
            }

            var embed = this.embedding.forward(input).view(new long[] { 1, 1, -1 });
            embed = relu(embed);

            //attention
            var query = hidden.permute(new long[] { 1, 0, 2 });
            (var context, _) = Attn.forward(query, EncoderOutPuts.unsqueeze(0));
            var inputGru = torch.cat(new List<Tensor> { embed, context }, dim: 2);

            if (debug)
                Console.WriteLine("埋め込み後" + embed);
            (var output, hidden) = this.gru.forward(inputGru, hidden);
            if (debug)
            {
                Console.WriteLine("出力" + output);
                Console.WriteLine("隠れ状態" + hidden);
            }
            output = outLinear.forward(output[0]);
            output = LogSoftmax(1).forward(output);
            return (output, hidden);
        }

        public Tensor InitHidden()
        {
            return torch.zeros(1, 1, hiddenSize);
        }
    }
}
