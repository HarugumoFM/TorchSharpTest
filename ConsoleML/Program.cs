﻿using NMeCab;
using Shimotsuki.Models;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

//FashionMNIST.Main();

var list = new List<string>() {"今日の天気は？",
                                "明日の天気は？",
                                "明後日の天気は？",
                                "あさっての天気は？"
                                };
long wordsCount = 0;


MeCabTagger tagger　= MeCabTagger.Create("ipadic");

//判別単語数計算
int index = 0;
var index2Word = new Dictionary<long,string>();
var word2Index = new Dictionary<string,long>();

foreach(var s in  list)
{
    getWords(s);
}
var bm = new BM25(index, list.Count,2.0,0.75);
var count = new int[list.Count, index];
var sentences = new int[index];
var slongs = new int[list.Count];

//文章数計算
int sIndex = 0;
foreach (var s in list)
{
    calcWords(s,sIndex);
    sIndex++;
}

//IDF値計算
for (int i = 0; i < index;i++)
{
    bm.Idf[i] = Math.Log((sIndex - sentences[i] + 0.5) / (sentences[i] + 0.5) + 1.0);
}
//.Idf.print();

//bm値計算
for(int i=0;i<list.Count;i++)
{
    double bmconst = bm.K1 * (1.0 - bm.B + bm.B * (double)(slongs[i]*list.Count)/wordsCount);
    for(int j=0;j<index;j++)
    {
        if (count[i, j] == 0)
            continue;
        double tf = count[i, j] * (bm.K1 + 1.0) / (count[i, j] + bmconst);
        bm.Bm[i, j] = tf * bm.Idf[j];
    }
}
//bm.Bm.print();
bm.save("bm.bin");
Console.WriteLine("parameter saved");

var bm2 = new BM25(index, list.Count, 2.0, 0.75);
bm2.load("bm.bin");
Console.WriteLine("parameter loaded");
bm2.Idf.print();

Console.WriteLine("calc bm25 a sentence");
//実際に処理すべきこと
var v = calcBM("今日の天気を教えてください");
v.print();
Console.WriteLine("calc Cosine similarity");
//コサイン類似度計算
functional.cosine_similarity(bm.Bm, v).print();


void getWords(string text)
{
    MeCabNode node = tagger.Parse(text)[0];
    while (node.Stat != MeCabNodeStat.Eos)
    {
        wordsCount++;
        var feature = node.Feature.Split(",");
        if (feature[0] == "動詞" || feature[0] == "名詞")
        {
            var word = feature[6];
            if (!word2Index.ContainsKey(word))
            {
                word2Index.Add(word, index);
                index2Word.Add(index, word);
                index++;
            }
        }
        node = node.Next;
    }
}

void calcWords(string text,int sIndex)
{
    MeCabNode node = tagger.Parse(text)[0];
    int length = 0;
    while (node.Stat != MeCabNodeStat.Eos)
    {
        length++;
        var feature = node.Feature.Split(",");
        if (feature[0] == "動詞" || feature[0] == "名詞")
        {
            var word = feature[6];
            count[sIndex, word2Index[word]]++;
        }
        node = node.Next;
    }
    for(int i=0;i<index2Word.Count;i++)
    {
        if (count[sIndex,i] >0)
        {
            sentences[i]++;
        }
    }
    slongs[sIndex] = length;
}

Tensor calcBM(string text)
{
    var vec = zeros(new long[] { index});
    var wCount = new long[index2Word.Count];
    MeCabNode node = tagger.Parse(text)[0];
    long wx = 0;
    while (node.Stat != MeCabNodeStat.Eos)
    {
        wx++;
        var feature = node.Feature.Split(",");
        if (feature[0] == "動詞" || feature[0] == "名詞")
        {
            var word = feature[6];
            if (word2Index.ContainsKey(word))
            {
                wCount[word2Index[word]]++;
            }
        }
        node = node.Next;
    }
    double bmconst = bm.K1 * (1.0 - bm2.B + bm2.B * (double)(wx * list.Count) / wordsCount);
    for (int j = 0; j < index; j++)
    {
        if (wCount[j] == 0)
            continue;
        double tf = wCount[j] * (bm2.K1 + 1.0) / (wCount[j] + bmconst);
        vec[j] = tf * bm2.Idf[j];
    }
    return vec;
}


