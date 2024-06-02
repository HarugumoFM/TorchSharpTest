using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using TorchSharp.Modules;
using System.Text.RegularExpressions;
using System.Globalization;
using System.Text;
using Shimotsuki.Models;
using System.Data;
using static Function;

//nnTest();

//1次元配列
var x = new float[] { 1, 2 };
//2次元配列
var y = new float[,] { {1,0},{0,1}};


Console.WriteLine(y);
trainFashionMNIST();

//train function




//trainFashionMNIST();
//nnTest();


void trainFashionMNIST()
{
    var model = new DNN();
    //starting download
    Console.WriteLine("start download data");
    var datasetPath = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
    var train_data = torchvision.datasets.FashionMNIST(datasetPath, true, download: true);
    var test_data = torchvision.datasets.FashionMNIST(datasetPath, false, download: true);
    Console.WriteLine("success download data");

    var trainloader = torch.utils.data.DataLoader(train_data, 64);
    var testloader = torch.utils.data.DataLoader(test_data, 64);
    var opt = new SGD(model.parameters(), 0.1);

    long accuracy = 0;
    long count = 0;
    model.train();
    for (int i = 0; i < 10; i++)
    {
        float totalLoss = 0;
        accuracy = 0;
        count = 0;
        foreach (var item in trainloader)
        {
            opt.zero_grad();
            var data = item["data"];
            var labels = item["label"];
            var out1 = model.forward(data);
            var loss = nll_loss(out1, labels);


            loss.backward();//誤差逆伝搬
            opt.step();
            totalLoss += loss.ToSingle();
            var (_, pred) = max(out1, dim: 1);
            var act1 = (pred == labels).sum();
            accuracy += (int)act1.ToSingle();//正答率
            count += labels.size(0);
        }
        Console.WriteLine("Epoch {0}, loss: {1} accuracy: {2}", i + 1, totalLoss / count, (double)100 * accuracy / count);
    }


    //モデル評価
    accuracy = 0;
    count = 0;
    no_grad();
    model.eval();
    foreach (var item in testloader)
    {
        var out1 = model.forward(item["data"]);
        var loss = nll_loss(out1, item["label"]);

        var labels = item["label"];
        var (_, pred) = max(out1, dim: 1);
        var act1 = (pred == labels).sum();
        accuracy += (int)act1.ToSingle();//正答率
        count += labels.size(0);
    }
    Console.WriteLine("accuracy:" + (double)accuracy * 100 / count);

}


void trainSeq2Seq() {

    var langE = new Lang();
    var langF = new Lang();
    var pairs = new List<string[]>();
    int maxPairs = 500;
    //read English-Francis Pair
    using (var reader = new StreamReader("eng-fra.txt")) {
        string line;
        int i = 0;
        while ((line = reader.ReadLine()) != null) {
            var pair = line.Split('	');

            if (pair[0].Split().Length < 8 && pair[0].Split().Length > 3 && pair[1].Split().Length < 8) {
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
    foreach (var pair in pairs) {
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
    static string UnicodeToAscii(string s) {
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


    static string NormalizeString(string s) {
        s = UnicodeToAscii(s.ToLower().Trim());
        s = Regex.Replace(s, @"([.!?])", @" $1");
        s = Regex.Replace(s, @"[^a-zA-Z.!?]+", " ");
        return s;
    }

    static List<long> indexesFromSentence(Lang lang, string sentence) {
        var res = new List<long>();
        foreach (var word in sentence.Split()) {
            res.Add(lang.word2Index[word]);
        }
        return res;
    }

    static Tensor tensorFromSentence(Lang lang, string sentence) {
        var index = indexesFromSentence(lang, sentence);
        index.Add(1);
        return tensor(index).view(new long[] { -1, 1 });
    }

}


/// Model

class Net:nn.Module
{
    public Net():base(nameof(Net))
    {
        this.linear1 = Linear(2, 2);
        this.linear2 = Linear(2, 1);
        RegisterComponents();
    }

    public Tensor forward(Tensor x)
    {
        x = linear1.forward(x);
        x = functional.sigmoid(x);
        return linear2.forward(x);
    }

    private Module<Tensor, Tensor> linear1;
    private Module<Tensor, Tensor> linear2;


}

internal class Function
{
   
    /// <summary>
    /// XORの学習とモデルの書き込み・読み込みのテスト関数
    /// </summary>
    public static void nnTest()
    {
        //テストデータ作成
        var trainData = new float[,]
        {
           { 0, 0 },
           { 1, 0 },
           { 0, 1 },
           { 1, 1 },
        };
        //ラベル
        var trainLabel = new float[,]
        {
            {0},
            {1},
            {1},
            {0},
        };
        var model = new Net();
        model.train();
        var x = tensor(trainData);
        var y = tensor(trainLabel);

        const int EPOCH = 10000;
        const double LR = 0.1;
        
        //train
        var optimizer = optim.SGD(model.parameters(), LR);
        var sx = model.parameters();
        for (var ep = 1; ep <= EPOCH; ++ep)
        {
            var eval = model.forward(x);
            var loss = mse_loss(eval, y);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            if (ep % 500 == 0)
            {
                Console.WriteLine($"Epoch:{ep} Loss:{loss.ToSingle()}");
                model.eval();
                Console.Write("x1:0,x2:0 result:");
                model.forward(torch.tensor(new float[] { 0, 0 })).print();
                Console.Write("x1:0,x2:1 result:");
                model.forward(torch.tensor(new float[] { 0, 1 })).print();
                Console.Write("x1:1,x2:0 result:");
                model.forward(torch.tensor(new float[] { 1, 0 })).print();
                Console.Write("x1:1,x2:1 result:");
                model.forward(torch.tensor(new float[] { 1, 1 })).print();
                
                model.train();
            }
        }
        //評価１
        model.eval();
        
        model.forward(torch.tensor(new float[] { 0, 0 })).print();

        model.forward(torch.tensor(new float[] { 0, 1 })).print();

        model.forward(torch.tensor(new float[] { 1, 0 })).print();

        model.forward(torch.tensor(new float[] { 1, 1 })).print();


        //モデルのsave,load,評価2
        model.save("test.bin");
        var model2 = new Net();

        model2.load("test.bin");
        model2.eval();

        model2.forward(torch.tensor(new float[] { 0, 0 })).print();

        model2.forward(torch.tensor(new float[] { 0, 1 })).print();

        model2.forward(torch.tensor(new float[] { 1, 0 })).print();

        model2.forward(torch.tensor(new float[] { 1, 1 })).print();

    }

    public static (float[,], float[,]) makeData(int numDiv, int cycles)
    {
        Random rand = new Random();
        float step = (float)((float)2 * Math.PI / numDiv);
        float[,] res0 = new float[numDiv * cycles, 1];
        float[,] res1 = new float[numDiv * cycles, 1];
        for (int i = 0; i < numDiv; i++)
        {
            float seed = (float)Math.Sin(step * i);
            res0[i, 0] = seed;
            res1[i, 0] = (float)(seed + rand.NextDouble() * 0.04 - 0.02);
        }
        return (res0, res1);
    }
}

class DNN : nn.Module
{
    public DNN() : base(nameof(Net))
    {
        this.linear1 = Linear(784, 256);
        this.linear2 = Linear(256, 10);
        this.flatten = Flatten();
        RegisterComponents();//パラメータの登録
    }

    public Tensor forward(Tensor x)
    {
        var f = flatten.forward(x);
        var a = linear1.forward(f);
        a = functional.sigmoid(a);
        var b = linear2.forward(a);
        return functional.log_softmax(b, 1);//活性化関数
    }

    private Module<Tensor, Tensor> linear1;
    private Module<Tensor, Tensor> linear2;
    private Module<Tensor, Tensor> linear3;
    private Module<Tensor, Tensor> flatten;
}


