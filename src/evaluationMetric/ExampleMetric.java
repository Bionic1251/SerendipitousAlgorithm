package evaluationMetric;

import org.grouplens.lenskit.Recommender;
import org.grouplens.lenskit.eval.Attributed;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.eval.metrics.AbstractMetric;
import org.grouplens.lenskit.eval.metrics.ResultColumn;
import org.grouplens.lenskit.eval.traintest.TestUser;
import org.grouplens.lenskit.util.statistics.MeanAccumulator;


public class ExampleMetric extends AbstractMetric<MeanAccumulator, ExampleMetric.Result, ExampleMetric.Result> {
	public ExampleMetric() {
		super(ExampleMetric.Result.class, ExampleMetric.Result.class);
	}

	@Override
	protected ExampleMetric.Result doMeasureUser(TestUser user, MeanAccumulator context) {
		context.add(2.0);
		return new ExampleMetric.Result(4.0);
	}

	@Override
	protected ExampleMetric.Result getTypedResults(MeanAccumulator context) {
		return new ExampleMetric.Result(context.getMean());
	}

	@Override
	public MeanAccumulator createContext(Attributed algorithm, TTDataSet dataSet, Recommender recommender) {
		return new MeanAccumulator();
	}

	public static class Result {
		@ResultColumn("Serendipity")
		public final double utility;

		public Result(double util) {
			utility = util;
		}
	}
}
