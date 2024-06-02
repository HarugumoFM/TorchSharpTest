using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;
using System.Globalization;
using System.Text.RegularExpressions;
using System.Text;

namespace Shimotsuki.Models
{
    public class MonoAttnSeq2Seq : Module
    {

        private Encoder encoder;
        private Encoder Reverse;
        private AttnDecoder decoder;
        private Embedding embedding;
        public double progress = 0;
        public int _accuracy = 0;
        public int epochs = 0;
        public double loss = 0;
        int hiddenSize;
        public Lang Lang;
        public MonoAttnSeq2Seq(int inputSize, int hiddenSize) : base("seq2seq")
        {
            Lang = new Lang();
            this.embedding = Embedding(inputSize, hiddenSize);
            this.hiddenSize = hiddenSize;
            this.encoder = new Encoder(inputSize, hiddenSize, embedding);
            this.Reverse = new Encoder(inputSize, hiddenSize, embedding);
            this.decoder = new AttnDecoder(hiddenSize, inputSize, embedding);
            RegisterComponents();
        }

        public string evaluate(Tensor input, int maxLength)
        {

            this.eval();
            var encoderHidden = encoder.InitHidden();
            var reverseHidden = Reverse.InitHidden();
            var res = new List<string>();
            var inputLength = input.size(0);

            var encoderOutputs = zeros(inputLength, hiddenSize);
            for (int i = 0; i < inputLength; i++)
            {
                (var output, encoderHidden) = encoder.forward(input[i], encoderHidden);
                (var reverseOutput, reverseHidden) = Reverse.forward(input[inputLength - 1 - i], reverseHidden);
                output += reverseOutput;
                encoderOutputs[i] = output[0, 0];
            }
            encoderHidden += reverseHidden;
            encoderHidden = encoderHidden.reshape(new long[] { 1, 1, hiddenSize });
            input = torch.tensor(0);
            for (int i = 0; i < maxLength; i++)
            {
                (var output, encoderHidden) = decoder.forward(input, encoderHidden, encoderOutputs);
                (_, var topi) = output.topk(1);
                input = topi.squeeze().detach();
                if (input.item<long>() == 1)
                    break;
                res.Add(Lang.index2Word[(int)topi.item<long>()]);
            }
            return string.Join(" ", res.ToArray());
        }

        Tensor tensorFromSentence(Lang lang, string sentence)
        {
            var index = indexesFromSentence(lang, sentence);
            index.Add(1);//EOSを追加
            return tensor(index).view(new long[] { -1, 1 });
        }

        List<long> indexesFromSentence(Lang lang, string sentence)
        {
            var res = new List<long>();
            foreach (var word in sentence.Split())
            {
                res.Add(lang.word2Index[word]);
            }
            return res;
        }

        public void trainAll(List<string[]> pairs, int epoch)
        {
            this.train();
            int maxRecall = 0;
            int accuracy = 0;
            var Optim = optim.Adam(this.parameters(), 0.001);

            for (int j = 0; j < epoch; j++)
            {
                loss = trainIter();
                this.epochs++;
            }

            double trainIter()
            {
                double lossTotal = 0;
                accuracy = 0;
                int sum = 0;
                foreach (int i in Enumerable.Range(0, pairs.Count).OrderBy(X => Guid.NewGuid()))
                {
                    var inputTensor = tensorFromSentence(Lang, pairs[i][0]);
                    var targetTensor = tensorFromSentence(Lang, pairs[i][1]);
                    double loss1 = train(inputTensor, targetTensor);
                    lossTotal += loss1;
                    sum++;
                    progress = (double)sum / pairs.Count();
                }
                Console.WriteLine();
                if (accuracy > maxRecall)
                {
                    this.save("model.bin");
                    maxRecall = accuracy;
                }
                Console.WriteLine("recall:" + accuracy + "/" + pairs.Count);
                this._accuracy = accuracy;
                return lossTotal / pairs.Count();

                double train(Tensor input, Tensor target)
                {
                    var encoderHidden = encoder.InitHidden();
                    var reverseHidden = Reverse.InitHidden();
                    Random rand = new Random();
                    Optim.zero_grad();
                    var inputLength = input.size(0);
                    var outputLength = target.size(0);


                    var encoderOutputs = zeros(inputLength, hiddenSize);
                    Tensor? loss = 0;
                    for (int i = 0; i < inputLength; i++)
                    {
                        (var output, encoderHidden) = encoder.forward(input[i], encoderHidden);
                        (var reverseOutput, reverseHidden) = Reverse.forward(input[inputLength - 1 - i], reverseHidden);
                        output += reverseOutput;
                        encoderOutputs[i] = output[0, 0];
                    }
                    encoderHidden += reverseHidden;
                    encoderHidden = encoderHidden.reshape(new long[] { 1, 1, hiddenSize });

                    var answer = new List<long>();
                    var study = new List<long>();
                    input = torch.tensor(0);
                    var criterion = nn.NLLLoss();
                    if (rand.NextDouble() > 0.5)
                    {
                        for (int i = 0; i < outputLength; i++)
                        {
                            (var output, encoderHidden) = decoder.forward(input, encoderHidden, encoderOutputs);
                            loss += criterion.forward(output, target[i]);
                            (_, var topi) = output.topk(1);
                            study.Add(topi.item<long>());
                            answer.Add(target[i].item<long>());

                            input = target[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < outputLength; i++)
                        {
                            (var output, encoderHidden) = decoder.forward(input, encoderHidden, encoderOutputs);
                            loss += criterion.forward(output, target[i]);
                            (_, var topi) = output.topk(1);
                            input = topi.squeeze().detach();
                            if (input.item<long>() == 1)
                                break;
                        }
                    }
                    int length = study.Count;
                    bool res = true;
                    for (int i = 0; i < length; i++)
                    {
                        if (study[i] != answer[i])
                        {
                            res = false;
                            break;
                        }

                    }
                    if (res)
                        accuracy++;
                    loss.backward();

                    var result = loss.item<float>() / outputLength;

                    loss = null;
                    encoderHidden = null;
                    encoderOutputs = null;

                    Optim.step();

                    GC.Collect();


                    return result;

                }
            }
        }
    }
}
