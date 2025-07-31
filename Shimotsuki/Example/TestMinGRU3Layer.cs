using Shimotsuki.Models;
using TorchSharp;
using static TorchSharp.torch;

namespace Shimotsuki.Example
{
    public class TestMinGRU3Layer
    {   
        public static void TestMinGRU3LayerImplementation()
        {
            Console.WriteLine("Testing 3-layer MinGRU implementation...");
            
            try
            {
                var minGRU = new MinGRU(128, 64);
                var input = torch.randn(1, 128);
                var hidden = torch.randn(1, 64);
                var (output, newHidden) = minGRU.forward(input, hidden);
                Console.WriteLine($"MinGRU cell test passed. Output shape: {output.shape}");
                
                var encoder = new MinGRUEncoder(1000, 128, 3);
                var encoderInput = torch.tensor(new long[] { 5 });
                var encoderHidden = encoder.InitHiddenMultiLayer();
                var (encOutput, encHidden) = encoder.forward(encoderInput, encoderHidden);
                Console.WriteLine($"3-layer MinGRU Encoder test passed. Output shape: {encOutput.shape}");
                
                var decoder = new MinGRUDecoder(128, 1000, 3);
                var decoderInput = torch.tensor(new long[] { 10 });
                var decoderHidden = decoder.InitHiddenMultiLayer();
                var (decOutput, decHidden) = decoder.forward(decoderInput, decoderHidden);
                Console.WriteLine($"3-layer MinGRU Decoder test passed. Output shape: {decOutput.shape}");
                
                var seq2seq = new MinGRUSeq2Seq(1000, 128, 3);
                var lang = new Lang();
                lang.addSentence("テスト 文章 です");
                seq2seq.Lang = lang;
                
                var testInput = torch.tensor(new long[] { 0, 1, 2 }).view(new long[] { -1, 1 });
                var response = seq2seq.evaluate(testInput, 5);
                Console.WriteLine($"3-layer MinGRU Seq2Seq test passed. Response: {response}");
                
                Console.WriteLine("All 3-layer MinGRU tests passed successfully!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"3-layer MinGRU test failed: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                throw;
            }
        }
    }
}
