import parsers.hmm_model as hmm_model
import parsers.transcript as transcript
import parsers.vectors as vectors

from params_processor import ParamsProcessor
from transcript_processor import TranscriptProcessor


def main():
    models = hmm_model.parse()
    transcripts = transcript.parse()

    iteration = 1
    while True:
        print('Iteration #{}'.format(iteration))
        ts_processor = TranscriptProcessor(models)
        for idx, ts in enumerate(transcripts):
            percentage = idx * 100 / len(transcripts)
            print('  Processing Transcript: {0} ({1:.2f}%)'.format(ts.filename, percentage))
            ts_processor.process(ts)
            print('  Likelihood: {}'.format(ts_processor.likelihood))

        state_occupancy_table = ts_processor.state_occupancy_table
        params_processor = ParamsProcessor(models,
                                           ts_processor.state_occupancy_table,
                                           len(transcripts))
        params_processor.process()
        iteration += 1


if __name__ == '__main__':
    main()
