# Word-Level-Language-Model RNN

This example train a multi-layer RNN (Elman, GRU or LSTM) on a language modeling task. By default, the training script uses the Wikitext-2 dataset, provided. The trained model can then be used by the generate script to generate new text.

` python main.py --cuda --epochs 6              # Train a LSTM on Wikitext-2 with CUDA `

 ` python main.py --cuda --epochs 6 --tied       # Train a tied LSTM on Wikitext-2 with CUDA
  
  python main.py --cuda --epochs 6 --model Transformer --lr 5   # Train a Transformer model on Wikitext-2 with CUDA
                                           
  python main.py --cuda --tied                   # Train a tied LSTM on Wikitext-2 with CUDA for 40 epochs
  
  python generate.py                             # Generate samples from the trained LSTM model.
  python generate.py --cuda --model Transformer  # Generate samples from the trained Transformer model. `

#### Note: Work on dynamic quantization is on its way.
