using TorchSharp;
using static TorchSharp.torch.nn;

namespace Shimotsuki.Models
{
    public class Lang
    {

        public Dictionary<string, long> word2Index;
        public Dictionary<string, long> word2Count;
        public Dictionary<long, string> index2Word;
        int nWords = 0;

        public Lang()
        {
            this.word2Index = new Dictionary<string, long>();
            this.word2Count = new Dictionary<string, long>();
            this.index2Word = new Dictionary<long, string>();
            addWord("SOS");
            addWord("EOS");
            addWord("＊");
        }

        public Lang(Dictionary<string, long> word2Index, Dictionary<long,string> index2Word)
        {
            this.word2Index = word2Index;
            this.index2Word = index2Word;
            this.nWords = word2Index.Count;
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
