import parsers.hmm_model as hmm_model
import parsers.transcript as transcript
import parsers.vectors as vectors

from transcript_processor import TranscriptProcessor


def main():
    model = hmm_model.parse()
    transcripts = transcript.parse()[:40]

    processor = TranscriptProcessor(model)
    for idx, ts in enumerate(transcripts):
        percentage = idx * 100 / len(transcripts)
        print('  Processing Transcript: {0} ({1:.2f}%)'.format(ts.filename, percentage))
        processor.process(ts)
        print('  Likelihood: {}'.format(processor.likelihood))


if __name__ == '__main__':
    main()
