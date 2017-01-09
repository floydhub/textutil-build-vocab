from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import csv
import distutils.util
import numpy as np
import re
from tensorflow.contrib import learn

def str2bool(val):
    """
    Helper method to convert string to bool
    """
    if val is None:
        return False
    val = val.lower().strip()
    if val in ['true', 't', 'yes', 'y', '1', 'on']:
        return True
    elif val in ['false', 'f', 'no', 'n', '0', 'off']:
        return False

def main():
    """
    Learn a vocabulary dictionary of all tokens in the raw documents. Then convert text in 
    input to word-ids.
    Uses TFLearn (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/learn/python/learn)
    """

    # Parse command line args
    parser = argparse.ArgumentParser(description='Build vocab and transform input to word-ids')

    parser.add_argument('-i', '--input', required=True,
        help='Path to input file')
    parser.add_argument('-c', '--cols', required=True, type=str, default=0, 
        help='Comma separated list of columns indices to convert')
    parser.add_argument('-d', '--delimiter', required=True, default='\t', 
        help='Column delimiter')
    parser.add_argument('-w', '--maxwords', type=int, default=-1, 
        help='Max number of words in output sentences. Longer sentences are trimmed, shorter are padded')
    parser.add_argument('-f', '--minfreq', type=int, default=0, 
        help='Min frequency for word to be in vocabulary')
    parser.add_argument('-header', '--hasheader', required=False, type=str2bool,
        default='False', help='File has header row?')
    parser.add_argument('-o', '--output', required=True, help='Path to output file')
    parser.add_argument('-ov', '--output_vocabmap', required=True, help='Path to vocab mapping output file')
    parser.add_argument('-of', '--output_vocabfreq', required=True, help='Path to vocab freq output file')
    parser.add_argument('-os', '--output_vocabsize', required=True, help='Path to vocab size output file')

    args = parser.parse_args()
    # Unescape the delimiter
    args.delimiter = args.delimiter.decode('string_escape')
    # Parse cols into list of ints
    args.cols = [int(x) for x in args.cols.split(',')]
    # Transform the maxwords
    args.maxwords = args.maxwords if args.maxwords >= 0 else None

    # Convert args to dict
    vargs = vars(args)

    print("\nArguments:")
    for arg in vargs:
        print("{}={}".format(arg, getattr(args, arg)))

    # Read the input file
    with open(args.input, 'r') as inputfile:
        reader = csv.reader(inputfile, delimiter=args.delimiter)

        # If has header, write it unprocessed
        if args.hasheader:
            headers = next(reader, None)
            if headers:
                writer.writerow(headers)

        print("\nProcessing input")

        # get all documents to build vocab from
        lines = []
        for row in reader:
            for idx, col in enumerate(row):
                if idx in args.cols:
                    lines.append(col) 
        
        # Determine max_document_length from input is not specified 
        if args.maxwords is None:
            max_document_length = max([len(line.split(" ")) for line in lines])
        else:
            max_document_length = args.maxwords

        # Define VocabProcessor
        vocab_processor = learn.preprocessing.VocabularyProcessor(
            max_document_length=max_document_length,
            min_frequency=args.minfreq
        )

        # Generate vocabulary and transform input into word_ids
        print("\nBuilding vocabulary")
        lines_transformed = list(vocab_processor.fit_transform(lines))

        if len(lines) != len(lines_transformed):
            raise AssertionError("Dimensions of lines ({}) and lines_transformed ({}) don't match'".format(
                len(lines), len(lines_transformed)
            ))

        # Output the vocabulary
        print("\nWriting vocabulary mapping to file")
        with open(args.output_vocabmap, 'wb') as file:
            writer = csv.writer(file, delimiter=args.delimiter)
            for key, value in vocab_processor.vocabulary_._mapping.items():
                writer.writerow([key, value])

        print("\nWriting vocabulary freq to file")
        with open(args.output_vocabfreq, 'wb') as file:
            writer = csv.writer(file, delimiter=args.delimiter)
            for key, value in vocab_processor.vocabulary_._freq.items():
                writer.writerow([key, value])

        print("\nWriting vocabulary size to file")
        with open(args.output_vocabsize, 'wb') as file:
            file.write("{}".format(len(vocab_processor.vocabulary_)))
            
        # Map input file to vocab_ids
        print("\nMapping input to vocab_ids")
        inputfile.seek(0) # reset file pointer to beginning
        with open(args.output, 'w') as outputfile:
            writer = csv.writer(outputfile, delimiter=args.delimiter)
            i=0
            for row in reader:
                row_transformed = []
                for idx, col in enumerate(row):
                    if idx in args.cols:
                        # line = np.array2string(lines_transformed[i], max_line_width=np.inf, separator=',') # convert the numpy.ndarray to string
                        line_list = lines_transformed[i].tolist()
                        line = ' '.join(map(str, line_list))
                        row_transformed.append(line)
                        i+=1
                    else:
                        row_transformed.append(col)
                
                writer.writerow(row_transformed)

    print("\nDone. Bye!")

if __name__ == '__main__':
    main()