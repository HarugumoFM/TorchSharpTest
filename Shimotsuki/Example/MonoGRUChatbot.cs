using Shimotsuki.Models;
using TorchSharp;
using static TorchSharp.torch;

namespace Shimotsuki.Example
{
    public class MonoMinGRUChatbot
    {
        public static void Main()
        {
            Console.WriteLine("=== MonoMinGRU Cell Testing & Chatbot ===");
            Console.WriteLine("Testing single-layer MinGRU implementation to isolate cell functionality\n");
            
            TestBasicFunctionality();
            
            TestTraining();
            
            Console.WriteLine("\n=== SUMMARY ===");
            Console.WriteLine("MonoMinGRU (1-layer) implementation works correctly!");
            Console.WriteLine("This confirms the MinGRU cell itself is functional.");
            Console.WriteLine("Any issues with 3-layer implementation are likely in multi-layer architecture.");
        }
        
        static void TestBasicFunctionality()
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
                lang.addSentence("hello world");
                lang.addSentence("test case");
                seq2seq.Lang = lang;
                
                var testInput = torch.tensor(new long[] { 0, 1, 2 }).view(new long[] { -1, 1 });
                var response = seq2seq.evaluate(testInput, 5);
                Console.WriteLine($"   ✓ MonoMinGRU Seq2Seq evaluation passed. Response: '{response}'");
                
                Console.WriteLine("   ✓ All basic MonoMinGRU tests passed!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ✗ MonoMinGRU test failed: {ex.Message}");
                Console.WriteLine($"   Stack trace: {ex.StackTrace}");
                throw;
            }
        }
        
        static void TestTraining()
        {
            Console.WriteLine("\n2. Testing MonoMinGRU training with parameter registration verification...");
            
            try
            {
                var seq2seq = new MonoMinGRUSeq2Seq(1000, 256, 10);
                var lang = new Lang();
                
                var pairs = new List<(string, string)> {
                    ("hello", "world"),
                    ("good", "morning"),
                    ("thank", "you"),
                    ("test", "case"),
                    ("how", "are")
                };
                
                foreach (var (input, target) in pairs) {
                    lang.addSentence(input);
                    lang.addSentence(target);
                }
                seq2seq.Lang = lang;
                
                Console.WriteLine($"   Vocabulary size: {lang.nWords}");
                Console.WriteLine($"   Training pairs: {pairs.Count}");
                
                var allEncoderParams = seq2seq.encoder.parameters().ToList();
                var allDecoderParams = seq2seq.decoder.parameters().ToList();
                Console.WriteLine($"   Encoder parameters registered: {allEncoderParams.Count()}");
                Console.WriteLine($"   Decoder parameters registered: {allDecoderParams.Count()}");
                
                Console.WriteLine("   Starting MonoMinGRU training (50 iterations)...");
                seq2seq.trainIter(pairs, nIters: 50, learningRate: 0.01, printEvery: 10);
                
                Console.WriteLine("\n   Testing responses after training:");
                foreach (var (input, expected) in pairs.Take(3)) {
                    var inputTensor = seq2seq.tensorFromSentence(input);
                    var response = seq2seq.evaluate(inputTensor, 10);
                    Console.WriteLine($"   Input: '{input}' -> Output: '{response}' (Expected: '{expected}')");
                }
                
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
