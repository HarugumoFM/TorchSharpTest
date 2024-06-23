using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using TorchSharp.Modules;

namespace Shimotsuki.Example
{
    public class FashionMNIST
    {
        public static void Main()
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

        class DNN : nn.Module
        {
            public DNN() : base("DNN")
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
    }
}
