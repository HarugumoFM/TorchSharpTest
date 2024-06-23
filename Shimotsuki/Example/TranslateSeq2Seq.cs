using Shimotsuki.Models;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using static TorchSharp.torch;

namespace Shimotsuki.Example
{
    public class TranslateSeq2Seq
    {   
        public static void Main()
        {
            void trainSeq2Seq()
            {

                var langE = new Lang();
                var langF = new Lang();
                var pairs = new List<string[]>();
                int maxPairs = 500;
                //read English-Francis Pair
                using (var reader = new StreamReader("eng-fra.txt"))
                {
                    string line;
                    int i = 0;
                    while ((line = reader.ReadLine()) != null)
                    {
                        var pair = line.Split('	');

                        if (pair[0].Split().Length < 8 && pair[0].Split().Length > 3 && pair[1].Split().Length < 8)
                        {
                            langE.addSentence(NormalizeString(pair[0]));
                            langF.addSentence(NormalizeString(pair[1]));
                            i++;
                            pairs.Add(pair);
                        }
                        if (maxPairs == i)
                            break;
                    }
                }
                Console.WriteLine("get " + pairs.Count + " pairs");
                Console.WriteLine(langE.word2Index.Count + " English words");
                Console.WriteLine(langF.word2Index.Count + " Frances words");

                int hiddenSize = 128;

                var model = new AttnSeq2Seq(langE.word2Index.Count, hiddenSize, langF.word2Index.Count);

                model.LangE = langE;
                model.LangF = langF;

                model.trainAll(pairs, 10);

                var model2 = new AttnSeq2Seq(langE.word2Index.Count, hiddenSize, langF.word2Index.Count);

                model2.load("model.bin");

                model2.LangE = langE;
                model2.LangF = langF;
                int index = 0;
                foreach (var pair in pairs)
                {
                    Console.WriteLine(string.Join(" ", pair[0]));
                    no_grad();
                    var input = tensorFromSentence(model2.LangE, NormalizeString(pair[0]));
                    Console.WriteLine("answer: " + pair[1]);
                    Console.WriteLine("predict: " + model2.evaluate(input, 10));
                    index++;
                    if (index > 30)
                        break;
                }





                ///Function
                static string UnicodeToAscii(string s)
                {
                    string normalizedString = s.Normalize(NormalizationForm.FormKD);
                    StringBuilder stringBuilder = new StringBuilder();

                    foreach (char c in normalizedString)
                    {
                        UnicodeCategory unicodeCategory = CharUnicodeInfo.GetUnicodeCategory(c);
                        if (unicodeCategory != UnicodeCategory.NonSpacingMark)
                        {
                            stringBuilder.Append(c);
                        }
                    }

                    return stringBuilder.ToString();
                }


                static string NormalizeString(string s)
                {
                    s = UnicodeToAscii(s.ToLower().Trim());
                    s = Regex.Replace(s, @"([.!?])", @" $1");
                    s = Regex.Replace(s, @"[^a-zA-Z.!?]+", " ");
                    return s;
                }

                static List<long> indexesFromSentence(Lang lang, string sentence)
                {
                    var res = new List<long>();
                    foreach (var word in sentence.Split())
                    {
                        res.Add(lang.word2Index[word]);
                    }
                    return res;
                }

                static Tensor tensorFromSentence(Lang lang, string sentence)
                {
                    var index = indexesFromSentence(lang, sentence);
                    index.Add(1);
                    return tensor(index).view(new long[] { -1, 1 });
                }

            }
        }
    }
}
