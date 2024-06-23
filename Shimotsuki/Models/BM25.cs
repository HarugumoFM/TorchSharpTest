using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

namespace Shimotsuki.Models {
    /// <summary>
    /// BM25のBM値とIDF値を保存するclass
    /// </summary>
    public class BM25 : Module {
        public Tensor Bm;
        public Tensor Idf;
        public double K1;
        public double B;

        public BM25(int wordSize,int sentenceSize,double k1 =2.0,double b=0.75) : base("bm25") {
            this.K1 = k1;
            this.B = b;
            this.Bm = zeros(new long[] {sentenceSize,wordSize});
            this.Idf = zeros(new long[] { wordSize });

            RegisterComponents();//パラメータを登録する(optimizer用)
        }
    }
}
