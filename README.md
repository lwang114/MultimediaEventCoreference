### Run the code:
  1. Download WASE code, the meta data and bounding boxes for VOA and class vocabulary from https://github.com/limanling/m2e2; Put the files in this repo
     to corresponding directories
  2. Download the Glove embedding
  3. Create a directory called 'data/voa' with subdirectory 'data/voa/rawdata/' and 'data/voa/object\_detect', put the bounding box into 'data/voa/object\_detect' and meta info into 'data/voa/rawdata/'; also create another two folders called 'data/glove/' and 'data/vocab/' and put the embedding and vocab vectors in the respective folders  
  4. Preprocess the vocabulary by running:
		~~~~~~~~~~~~~~~~~~~~~~~~~
		cd m2e2/src/dataflow/numpy
		python prepare_vocab.py
		~~~~~~~~~~~~~~~~~~~~~~~~~
  5. Download and preprocess the VOA dataset by running in m2e2/src/dataflow/numpy/ 
		~~~~~~~~~~~~~~~~~~~~~~~~~
		python prepare_grounding_voa.py
		~~~~~~~~~~~~~~~~~~~~~~~~~
  6. Run the grounding/entity coreference experiment:
		~~~~~~~~~~~~~~~~~~~~~~~~~
		cd m2e2/src/engine
		python Coreferencerunner.py
		~~~~~~~~~~~~~~~~~~~~~~~~~
