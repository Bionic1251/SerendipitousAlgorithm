package evaluationMetric;

import it.unimi.dsi.fastutil.longs.LongSet;
import org.grouplens.lenskit.Recommender;
import org.grouplens.lenskit.eval.Attributed;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.eval.metrics.AbstractMetric;
import org.grouplens.lenskit.eval.metrics.ResultColumn;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.eval.traintest.TestUser;
import org.grouplens.lenskit.scored.ScoredId;
import org.grouplens.lenskit.util.statistics.MeanAccumulator;

import javax.annotation.Nullable;
import java.util.*;

public class AggregateOPRMetric extends AbstractMetric<MeanAccumulator, AggregateOPRMetric.Result, AggregateOPRMetric.Result> {
	private final ItemSelector goodItems;
	private final ItemSelector candidates;
	private final ItemSelector exclude;

	private MeanAccumulator precisionContext1 = new MeanAccumulator();
	private MeanAccumulator precisionContext5 = new MeanAccumulator();
	private MeanAccumulator precisionContext10 = new MeanAccumulator();
	private MeanAccumulator precisionContext15 = new MeanAccumulator();
	private MeanAccumulator precisionContext20 = new MeanAccumulator();
	private MeanAccumulator precisionContext25 = new MeanAccumulator();
	private MeanAccumulator precisionContext30 = new MeanAccumulator();

	private MeanAccumulator recallContext1 = new MeanAccumulator();
	private MeanAccumulator recallContext5 = new MeanAccumulator();
	private MeanAccumulator recallContext10 = new MeanAccumulator();
	private MeanAccumulator recallContext15 = new MeanAccumulator();
	private MeanAccumulator recallContext20 = new MeanAccumulator();
	private MeanAccumulator recallContext25 = new MeanAccumulator();
	private MeanAccumulator recallContext30 = new MeanAccumulator();

	public AggregateOPRMetric(ItemSelector candidates, ItemSelector exclude, ItemSelector goodItems) {
		super(AggregateOPRMetric.Result.class, AggregateOPRMetric.Result.class);
		this.goodItems = goodItems;
		this.candidates = candidates;
		this.exclude = exclude;
	}

	@Override
	protected Result doMeasureUser(TestUser user, MeanAccumulator context) {
		List<ScoredId> recommendations = user.getRecommendations(1000 + user.getTestHistory().size(), candidates, exclude);
		if (recommendations == null || recommendations.isEmpty()) {
			return new Result(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
		}
		LongSet goodItems = this.goodItems.select(user);
		if (goodItems == null || goodItems.isEmpty()) {
			return null;
		}
		List<Container<Double>> relevantItems = new ArrayList<Container<Double>>();
		List<Container<Double>> irrelevantItems = new ArrayList<Container<Double>>();
		for (ScoredId scoredId : recommendations) {
			if (goodItems.contains(scoredId.getId())) {
				relevantItems.add(new Container<Double>(scoredId.getId(), scoredId.getScore()));
			} else {
				irrelevantItems.add(new Container<Double>(scoredId.getId(), scoredId.getScore()));
			}
		}
		Collections.sort(irrelevantItems);
		Collections.reverse(irrelevantItems);
		Res res1 = measureUser(relevantItems, irrelevantItems, 1);
		Res res5 = measureUser(relevantItems, irrelevantItems, 5);
		Res res10 = measureUser(relevantItems, irrelevantItems, 10);
		Res res15 = measureUser(relevantItems, irrelevantItems, 15);
		Res res20 = measureUser(relevantItems, irrelevantItems, 20);
		Res res25 = measureUser(relevantItems, irrelevantItems, 25);
		Res res30 = measureUser(relevantItems, irrelevantItems, 30);
		precisionContext1.add(res1.precision);
		precisionContext5.add(res5.precision);
		precisionContext10.add(res10.precision);
		precisionContext15.add(res15.precision);
		precisionContext20.add(res20.precision);
		precisionContext25.add(res25.precision);
		precisionContext30.add(res30.precision);
		recallContext1.add(res1.recall);
		recallContext5.add(res5.recall);
		recallContext10.add(res10.recall);
		recallContext15.add(res15.recall);
		recallContext20.add(res20.recall);
		recallContext25.add(res25.recall);
		recallContext30.add(res30.recall);
		return null;
	}

	private Res measureUser(List<Container<Double>> relevantItems, List<Container<Double>> irrelevantItems, int listSize) {
		int hits = 0;
		double borderScore = irrelevantItems.get(listSize - 1).getValue();
		for (Container<Double> container : relevantItems) {
			if (container.getValue() > borderScore) {
				hits++;
			}
		}
		double recall = (double) hits / (double) relevantItems.size();
		double precision = recall / listSize;
		return new Res(precision, recall);
	}

	@Override
	protected Result getTypedResults(MeanAccumulator context) {
		return new Result(precisionContext1.getMean(), precisionContext5.getMean(), precisionContext10.getMean(), precisionContext15.getMean(), precisionContext20.getMean(), precisionContext25.getMean(), precisionContext30.getMean(),
				recallContext1.getMean(), recallContext5.getMean(), recallContext10.getMean(), recallContext15.getMean(), recallContext20.getMean(), recallContext25.getMean(), recallContext30.getMean());
	}

	@Nullable
	@Override
	public MeanAccumulator createContext(Attributed algorithm, TTDataSet dataSet, Recommender recommender) {
		return new MeanAccumulator();
	}

	public static class Result {
		@ResultColumn("OPrecision1")
		public final double precision1;
		@ResultColumn("OPrecision5")
		public final double precision5;
		@ResultColumn("OPrecision10")
		public final double precision10;
		@ResultColumn("OPrecision15")
		public final double precision15;
		@ResultColumn("OPrecision20")
		public final double precision20;
		@ResultColumn("OPrecision25")
		public final double precision25;
		@ResultColumn("OPrecision30")
		public final double precision30;

		@ResultColumn("ORecall1")
		public final double recall1;
		@ResultColumn("ORecall5")
		public final double recall5;
		@ResultColumn("ORecall10")
		public final double recall10;
		@ResultColumn("ORecall15")
		public final double recall15;
		@ResultColumn("ORecall20")
		public final double recall20;
		@ResultColumn("ORecall25")
		public final double recall25;
		@ResultColumn("ORecall30")
		public final double recall30;

		public Result(double precision1, double precision5, double precision10, double precision15, double precision20, double precision25, double precision30, double recall1, double recall5, double recall10, double recall15, double recall20, double recall25, double recall30) {
			this.precision1 = precision1;
			this.precision5 = precision5;
			this.precision10 = precision10;
			this.precision15 = precision15;
			this.precision20 = precision20;
			this.precision25 = precision25;
			this.precision30 = precision30;
			this.recall1 = recall1;
			this.recall5 = recall5;
			this.recall10 = recall10;
			this.recall15 = recall15;
			this.recall20 = recall20;
			this.recall25 = recall25;
			this.recall30 = recall30;
		}
	}

	private class Res {
		private double precision;
		private double recall;

		private Res(double precision, double recall) {
			this.precision = precision;
			this.recall = recall;
		}
	}
}
