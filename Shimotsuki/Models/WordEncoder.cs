﻿using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

namespace Shimotsuki.Models {
    public class WordEncoder : Module {
        int hiddenSize;
        Embedding embedding;
        Embedding posEmbedding;
        GRU gru;

        public WordEncoder(int inputSize, int hiddenSize, int maxToken) : base("encoder") {
            this.hiddenSize = hiddenSize;
            this.embedding = Embedding(inputSize, hiddenSize);
            this.posEmbedding = Embedding(maxToken, hiddenSize);
            this.gru = GRU(hiddenSize, hiddenSize);
            RegisterComponents();//パラメータを登録する(optimizer用)
        }

        public WordEncoder(int inputSize, int hiddenSize, Embedding embedding, Embedding posEmbedding) : base("encoder")
        {
            this.hiddenSize = hiddenSize;
            this.embedding = embedding;
            this.posEmbedding = posEmbedding;
            this.gru = GRU(hiddenSize, hiddenSize);
            RegisterComponents();//パラメータを登録する(optimizer用)
        }

        public (Tensor, Tensor) forward(Tensor input, Tensor hidden, Tensor pos,bool debug = false) {
            if (debug)
                Console.WriteLine("入力単語" + input);

            var embed = (this.embedding.forward(input) + this.posEmbedding.forward(pos)).view(new long[] { 1, 1, -1 });
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
            return torch.zeros(1, 1, hiddenSize);
        }
    }
}
