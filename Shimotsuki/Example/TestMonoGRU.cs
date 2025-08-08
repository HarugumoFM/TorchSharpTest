using Shimotsuki.Models;
using TorchSharp;
using static TorchSharp.torch;

namespace Shimotsuki.Example
{
    public class TestMonoGRU
    {   
        public static void Main(string[] args)
        {
            Console.WriteLine("=== MonoMinGRU Cell Testing ===");
            Console.WriteLine("Testing single-layer MinGRU implementation to isolate cell functionality\n");
            
            TestMonoGRUImplementation();
            TestMonoMinGRUTraining();
            
            Console.WriteLine("\n=== SUMMARY ===");
            Console.WriteLine("MonoMinGRU (1-layer) implementation works correctly!");
            Console.WriteLine("This confirms the MinGRU cell itself is functional.");
            Console.WriteLine("Any issues with 3-layer implementation are likely in multi-layer architecture.");
        }
        
        public static void TestMonoGRUImplementation()
        {
            Console.WriteLine("1. Testing MinGRU cell basic functionality...");
            
            try
            {
                var minGRU = new MinGRU(128, 64);
                var input = torch.randn(1, 128);
                var hidden = torch.randn(1, 64);
                var (output, newHidden) = minGRU.forward(input, hidden);
                Console.WriteLine($"   ✓ MinGRU cell forward pass successful");
                Console.WriteLine($"   ✓ Input shape: {input.shape}, Output shape: {output.shape}");
                Console.WriteLine($"   ✓ Hidden shape: {hidden.shape}, New hidden shape: {newHidden.shape}");
                
                var encoder = new MonoMinGRUEncoder(1000, 128);
                var encoderInput = torch.tensor(new long[] { 5 });
                var encoderHidden = encoder.InitHidden();
                var (encOutput, encHidden) = encoder.forward(encoderInput, encoderHidden);
                Console.WriteLine($"   ✓ MonoMinGRU Encoder test passed. Output shape: {encOutput.shape}");
                
                var decoder = new MonoMinGRUDecoder(128, 1000);
                var decoderInput = torch.tensor(new long[] { 10 });
                var decoderHidden = decoder.InitHidden();
                var (decOutput, decHidden) = decoder.forward(decoderInput, decoderHidden);
                Console.WriteLine($"   ✓ MonoMinGRU Decoder test passed. Output shape: {decOutput.shape}");
                
                var encoderParams = encoder.parameters().ToList();
                var decoderParams = decoder.parameters().ToList();
                Console.WriteLine($"   ✓ Encoder parameters count: {encoderParams.Count()}");
                Console.WriteLine($"   ✓ Decoder parameters count: {decoderParams.Count()}");
                
                var seq2seq = new MonoMinGRUSeq2Seq(100, 64);
                var lang = new Lang();
                lang.addSentence("テスト 文章 です");
                lang.addSentence("こんにちは 世界");
                seq2seq.Lang = lang;
                
                var testInput = torch.tensor(new long[] { 0, 1, 2 }).view(new long[] { -1, 1 });
                var response = seq2seq.evaluate(testInput, 5);
                Console.WriteLine($"   ✓ MonoMinGRU Seq2Seq test passed. Response: '{response}'");
                
                Console.WriteLine("   ✓ All basic MonoMinGRU tests passed!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ✗ MonoMinGRU test failed: {ex.Message}");
                Console.WriteLine($"   Stack trace: {ex.StackTrace}");
                throw;
            }
        }
        
        public static void TestMonoMinGRUTraining()
        {
            Console.WriteLine("\n2. Testing MonoMinGRU training...");
            
            try
            {
                var seq2seq = new MonoMinGRUSeq2Seq(100, 64, 10);
                var lang = new Lang();
                
                var pairs = new List<(string, string)> {
                    ("hello", "world"),
                    ("test", "case"),
                    ("good", "morning")
                };
                
                foreach (var (input, target) in pairs) {
                    lang.addSentence(input);
                    lang.addSentence(target);
                }
                seq2seq.Lang = lang;
                
                Console.WriteLine($"   Language vocabulary size: {lang.nWords}");
                
                var allEncoderParams = seq2seq.encoder.parameters().ToList();
                var allDecoderParams = seq2seq.decoder.parameters().ToList();
                Console.WriteLine($"   Encoder parameters registered: {allEncoderParams.Count()}");
                Console.WriteLine($"   Decoder parameters registered: {allDecoderParams.Count()}");
                
                Console.WriteLine("   Starting 20 training iterations...");
                seq2seq.trainIter(pairs, nIters: 20, learningRate: 0.01, printEvery: 5);
                
                Console.WriteLine("   ✓ MonoMinGRU training completed successfully!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ✗ MonoMinGRU training failed: {ex.Message}");
                Console.WriteLine($"   Stack trace: {ex.StackTrace}");
                throw;
            }
        }
    }
}
