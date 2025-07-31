using Shimotsuki.Models;
using TorchSharp;
using static TorchSharp.torch;

namespace Shimotsuki.Example
{
    public class MinGRUChatbot
    {   
        public static void Main()
        {
            void trainChatbot()
            {
                var lang = new Lang();
                
                var chatPairs = new List<string[]>
                {
                    new string[] { "こんにちは", "こんにちは！元気ですか？" },
                    new string[] { "元気です", "それは良かったです！" },
                    new string[] { "ありがとう", "どういたしまして！" },
                    new string[] { "さようなら", "また会いましょう！" },
                    new string[] { "おはよう", "おはようございます！" },
                    new string[] { "こんばんは", "こんばんは！お疲れ様です！" }
                };

                foreach (var pair in chatPairs)
                {
                    lang.addSentence(pair[0]);
                    lang.addSentence(pair[1]);
                }

                int hiddenSize = 128;
                int numLayers = 3;
                var model = new MinGRUSeq2Seq(lang.word2Index.Count, hiddenSize, numLayers);
                model.Lang = lang;

                Console.WriteLine("MinGRU チャットボットの訓練を開始します...");
                model.trainAll(chatPairs, 10);

                Console.WriteLine("テスト:");
                var testInput = tensorFromSentence(lang, "こんにちは");
                var response = model.evaluate(testInput, 10);
                Console.WriteLine("入力: こんにちは");
                Console.WriteLine("応答: " + response);

                var model2 = new MinGRUSeq2Seq(lang.word2Index.Count, hiddenSize, numLayers);
                model2.load("mingru_chatbot_model.bin");
                model2.Lang = lang;

                var testInput2 = tensorFromSentence(lang, "ありがとう");
                var response2 = model2.evaluate(testInput2, 10);
                Console.WriteLine("入力: ありがとう");
                Console.WriteLine("応答: " + response2);
            }

            trainChatbot();
        }

        static Tensor tensorFromSentence(Lang lang, string sentence)
        {
            var index = new List<long>();
            foreach (var word in sentence.Split())
            {
                if (lang.word2Index.ContainsKey(word))
                {
                    index.Add(lang.word2Index[word]);
                }
                else
                {
                    index.Add(lang.word2Index["＊"]);
                }
            }
            index.Add(1);
            return tensor(index).view(new long[] { -1, 1 });
        }
    }
}
