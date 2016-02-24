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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class OPRMetric extends AbstractMetric<MeanAccumulator, OPRMetric.Result, OPRMetric.Result> {
	private final int evaluationListSize;
	private final ItemSelector goodItems;
	private final ItemSelector candidates;
	private final ItemSelector exclude;

	private final MeanAccumulator precisionAccumulator = new MeanAccumulator();
	private final MeanAccumulator recallAccumulator = new MeanAccumulator();

	public OPRMetric(int listSize, ItemSelector candidates, ItemSelector exclude, ItemSelector goodItems) {
		super(OPRMetric.Result.class, OPRMetric.Result.class);
		this.evaluationListSize = listSize;
		this.goodItems = goodItems;
		this.candidates = candidates;
		this.exclude = exclude;
	}

	@Override
	protected Result doMeasureUser(TestUser user, MeanAccumulator context) {
		List<ScoredId> recommendations = user.getRecommendations(1000 + user.getTestHistory().size(), candidates, exclude);
		if (recommendations == null || recommendations.isEmpty()) {
			return new Result(0.0, 0.0);
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
		int hits = 0;
		Collections.sort(irrelevantItems);
		Collections.reverse(irrelevantItems);
		double borderScore = irrelevantItems.get(evaluationListSize - 1).getValue();
		for(Container<Double> container : relevantItems){
			if(container.getValue() > borderScore){
				hits++;
			}
		}
		double recall = (double) hits / (double) relevantItems.size();
		double precision = recall / evaluationListSize;
		precisionAccumulator.add(precision);
		recallAccumulator.add(recall);
		return new Result(precision, recall);
	}

	@Override
	protected Result getTypedResults(MeanAccumulator context) {
		return new Result(precisionAccumulator.getMean(), recallAccumulator.getMean());
	}

	@Nullable
	@Override
	public MeanAccumulator createContext(Attributed algorithm, TTDataSet dataSet, Recommender recommender) {
		return new MeanAccumulator();
	}

	public static class Result {
		@ResultColumn("OPrecision")
		public final double precision;

		@ResultColumn("ORecall")
		public final double recall;

		public Result(double precision, double recall) {
			this.precision = precision;
			this.recall = recall;
		}
	}
}
