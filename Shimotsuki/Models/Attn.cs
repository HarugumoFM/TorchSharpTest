using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

namespace Shimotsuki.Models {
    public class Attn : Module {
        Linear Wa;
        Linear Ua;
        Linear Va;

        public Attn(int hiddenSize) : base("attention") {
            this.Wa = Linear(hiddenSize, hiddenSize);
            this.Ua = Linear(hiddenSize, hiddenSize);
            this.Va = Linear(hiddenSize, 1);
            RegisterComponents();//パラメータを登録する(optimizer用)
        }

        public (Tensor, Tensor) forward(Tensor query, Tensor keys) {
            var scores = Va.forward(torch.tanh(Wa.forward(query) + Ua.forward(keys)));
            scores = scores.squeeze(2).unsqueeze(1);
            var weights = Softmax(-1).forward(scores);
            var context = torch.bmm(weights, keys);

            return (context, weights);
        }  
    }
}
