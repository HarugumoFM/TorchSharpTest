using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;
using System.Globalization;
using System.Text.RegularExpressions;
using System.Text;

namespace Shimotsuki.Models {
    public class AttnSeq2Seq : Module {

        private Encoder encoder;
        private AttnDecoder decoder;
        int hiddenSize;
        public Lang LangE;
        public Lang LangF;
        public AttnSeq2Seq(int inputSize, int hiddenSize, int outputSize) : base("seq2seq") {
            LangE = new Lang();
            LangF = new Lang();
            this.hiddenSize = hiddenSize;
            this.encoder = new Encoder(inputSize, hiddenSize);
            this.decoder = new AttnDecoder(hiddenSize * 2, outputSize);
            RegisterComponents();
        }

        public string evaluate(Tensor input, int maxLength) {

            this.eval();
            var encoderHidden = encoder.InitHidden();
            var res = new List<string>();
            var inputLength = input.size(0);

            var encoderOutputs = zeros(inputLength, hiddenSize * 2);
            for (int i = 0; i < inputLength; i++) {
                (var output, encoderHidden) = encoder.forward(input[i], encoderHidden);
                encoderOutputs[i] = output[0, 0];
            }
            encoderHidden = encoderHidden.reshape(new long[] { 1, 1, hiddenSize * 2 });
            input = torch.tensor(0);
            for (int i = 0; i < maxLength; i++) {
                (var output, encoderHidden) = decoder.forward(input, encoderHidden, encoderOutputs);
                (_, var topi) = output.topk(1);
                res.Add(LangF.index2Word[(int)topi.item<long>()]);
                input = topi.squeeze().detach();
                if (input.item<long>() == 1)
                    break;
            }
            return string.Join(" ", res.ToArray());
        }

        Tensor tensorFromSentence(Lang lang, string sentence) {
            var index = indexesFromSentence(lang, sentence);
            index.Add(1);
            return tensor(index).view(new long[] { -1, 1 });
        }

        string UnicodeToAscii(string s) {
            string normalizedString = s.Normalize(NormalizationForm.FormKD);
            StringBuilder stringBuilder = new StringBuilder();

            foreach (char c in normalizedString) {
                UnicodeCategory unicodeCategory = CharUnicodeInfo.GetUnicodeCategory(c);
                if (unicodeCategory != UnicodeCategory.NonSpacingMark) {
                    stringBuilder.Append(c);
                }
            }

            return stringBuilder.ToString();
        }


        string NormalizeString(string s) {
            s = UnicodeToAscii(s.ToLower().Trim());
            s = Regex.Replace(s, @"([.!?])", @" $1");
            s = Regex.Replace(s, @"[^a-zA-Z.!?]+", " ");
            return s;
        }

        List<long> indexesFromSentence(Lang lang, string sentence) {
            var res = new List<long>();
            foreach (var word in sentence.Split()) {
                res.Add(lang.word2Index[word]);
            }
            return res;
        }

        public void trainAll(List<string[]> pairs, int epoch) {
            this.train();
            int maxRecall = 0;

            var encOptim = optim.Adam(encoder.parameters(), 0.001);
            var decOptim = optim.Adam(decoder.parameters(), 0.001);

            for (int j = 0; j < epoch; j++) {
                var loss = trainIter();
                Console.WriteLine(j + 1 + "回目 loss:" + loss);
            }

            double trainIter() {
                double lossTotal = 0;
                int accuracy = 0;
                int sum = 0;
                foreach (int i in Enumerable.Range(0, pairs.Count).OrderBy(X => Guid.NewGuid())) {
                    var inputTensor = tensorFromSentence(LangE, NormalizeString(pairs[i][0]));
                    var targetTensor = tensorFromSentence(LangF, NormalizeString(pairs[i][1]));
                    //Console.WriteLine(pairs[i][1]);
                    double loss1 = train(inputTensor, targetTensor);
                    lossTotal += loss1;
                    sum++;
                    if ((sum * 20) % pairs.Count == 0) {
                        Console.Write('#');
                    }
                }
                Console.WriteLine();
                if (accuracy > maxRecall) {
                    this.save("model.bin");
                    maxRecall = accuracy;
                }
                Console.WriteLine("recall:" + accuracy + "/" + pairs.Count);
                return lossTotal / pairs.Count();

                double train(Tensor input, Tensor target) {
                    var encoderHidden = encoder.InitHidden();
                    Random rand = new Random();
                    encOptim.zero_grad();
                    decOptim.zero_grad();
                    var inputLength = input.size(0);
                    var outputLength = target.size(0);


                    var encoderOutputs = zeros(inputLength, hiddenSize * 2);
                    Tensor loss = 0;
                    for (int i = 0; i < inputLength; i++) {
                        (var output, encoderHidden) = encoder.forward(input[i], encoderHidden);
                        encoderOutputs[i] += output[0, 0];
                    }
                    encoderHidden = encoderHidden.reshape(new long[] { 1, 1, hiddenSize * 2 });

                    var answer = new List<long>();
                    var study = new List<long>();
                    input = torch.tensor(0);
                    var criterion = nn.NLLLoss();
                    if (rand.NextDouble() > 0.5) {
                        for (int i = 0; i < outputLength; i++) {
                            (var output, encoderHidden) = decoder.forward(input, encoderHidden, encoderOutputs);
                            loss += criterion.forward(output, target[i]);
                            //loss.print();
                            (_, var topi) = output.topk(1);
                            study.Add(topi.item<long>());
                            answer.Add(target[i].item<long>());

                            input = target[i];
                        }
                    }
                    else {
                        for (int i = 0; i < outputLength; i++) {
                            (var output, encoderHidden) = decoder.forward(input, encoderHidden, encoderOutputs);
                            loss += criterion.forward(output, target[i]);
                            //loss.print();
                            (_, var topi) = output.topk(1);
                            input = topi.squeeze().detach();
                            if (input.item<long>() == 1)
                                break;
                        }
                    }
                    int length = study.Count;
                    bool res = true;
                    for (int i = 0; i < length; i++) {
                        if (study[i] != answer[i]) {
                            res = false;
                            break;
                        }

                    }
                    if (res)
                        accuracy++;
                    loss.backward();

                    encOptim.step();
                    decOptim.step();
                    var result = loss.item<float>() / outputLength;
                    
                    return result;

                }
            }
        }


    }
}
