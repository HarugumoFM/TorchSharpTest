using TorchSharp;
using static TorchSharp.torch.nn;

namespace Shimotsuki.Models
{
    public class Lang
    {

        public Dictionary<string, long> word2Index;
        public Dictionary<string, long> word2Count;
        public Dictionary<int, string> index2Word;
        int nWords = 0;

        public Lang()
        {
            this.word2Index = new Dictionary<string, long>();
            this.word2Count = new Dictionary<string, long>();
            this.index2Word = new Dictionary<int, string>();
            addWord("SOS");
            addWord("EOS");
            addWord(" ");
        }

        /// <summary>
        /// 単語の追加
        /// </summary>
        /// <param name="word"></param>
        public void addWord(string word)
        {
            if (!word2Index.ContainsKey(word))
            {
                this.word2Index.Add(word, nWords);
                this.word2Count.Add(word, 1);
                this.index2Word.Add(nWords, word);
                nWords++;
            }
            else
            {
                this.word2Count[word]++;
            }
        }

        /// <summary>
        /// 文章を登録する
        /// </summary>
        /// <param name="sentence">文章をstringのListにしたもの</param>
        public void addSentence(string sentence)
        {
            foreach (var word in sentence.Split())
            {
                this.addWord(word);
            }
        }
    }
}
