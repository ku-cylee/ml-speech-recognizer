import parsers.hmm_model as hmm_model
import parsers.transcript as transcript
import parsers.vectors as vectors

from transcript_processor import TranscriptProcessor


def main():
    model = hmm_model.parse()
    transcripts = transcript.parse()

    processor = TranscriptProcessor(model)
    ts = transcripts[0]
    print('Accumulating Transcript: {0}'.format(ts.filename))
    processor.process(ts)


if __name__ == '__main__':
    main()
